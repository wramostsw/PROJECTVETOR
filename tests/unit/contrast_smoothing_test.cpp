// tests/unit/contrast_smoothing_test.cpp — o suavizador sabe o contraste de cada fronteira (docs §1.29).
//
// Entre duas faixas parecidas de um degradê a posição da fronteira é incerta, e os degraus de 8 px dos blocos do JPEG
// parecem cantos vivos: o canto perfurado os preservava, e a escada sobrevivia à suavização. Dois mecanismos, ambos
// pelo contraste entre as duas faces: (1) nenhum canto perfurado abaixo de `punctureMinContrast`; (2) mola de
// fidelidade com peso ∝ contraste² (`contrastFullLevels`, desligada por padrão). Estes testes medem cada um contra uma
// escada de bloco 8×8 de verdade conhecida.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/segment/PixelSegmenter.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/vector/ContourSmoother.h"
#include "cnx/vector/EnergyTerms.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::SmoothParams;
using cnx::VectorImage;
using sla::Array2D;
using sla::Rgba8;
using sla::Vec2;

namespace {

constexpr int kW = 160;
constexpr int kH = 64;
constexpr double kSlope = 0.15;
constexpr double kY0 = 24.0;
constexpr int kBlock = 16;  // largura do degrau (a do bloco de JPEG que o produz)

double idealY(double x) { return kY0 + kSlope * x; }

/// Duas faixas de cor separadas por uma fronteira inclinada da qual só se conhece a versão em blocos de kBlock colunas
/// (como o JPEG): em cada bloco a fronteira fica na linha mais próxima da reta no centro do bloco (degraus de 2,4 px).
Array2D<Rgba8> blockStairImage(Rgba8 top, Rgba8 bottom) {
    Array2D<Rgba8> img(kW, kH);
    for (int x = 0; x < kW; ++x) {
        const int block = x / kBlock;
        const int boundary = static_cast<int>(std::lround(idealY(static_cast<double>(kBlock) * block + kBlock / 2.0)));
        for (int y = 0; y < kH; ++y) img(x, y) = y < boundary ? top : bottom;
    }
    return img;
}

/// Os parâmetros de suavização da foto/degradê (mola só na 1ª fase), com os campos de contraste editáveis.
SmoothParams photoSmooth() {
    cnx::EngineParameters e;
    e.applyProfile(cnx::ImageProfile::PhotoMed);
    return e.smooth;
}

struct Smoothed {
    VectorImage image;
    cnx::SmoothResult result;
    VectorImage staircase;
};

Smoothed smooth(const Array2D<Rgba8>& img, const SmoothParams& params) {
    cnx::Palette palette;
    palette.colors = {sla::toRgba(img(0, 0)), sla::toRgba(img(0, kH - 1))};
    cnx::SegmentationResult seg = cnx::PixelSegmenter(img, cnx::SegmentParams{}).run({}, &palette);
    cnx::TraceResult traced = cnx::BoundaryTracer().trace(seg.map);
    Smoothed out;
    out.staircase = traced.image;
    out.image = std::move(traced.image);
    out.result = cnx::ContourSmoother(params, nullptr).run(out.image);
    return out;
}

/// Nós interiores das arestas entre duas faces que não são o exterior, longe da moldura (x em [16, 144]).
std::vector<Vec2> boundaryNodes(const VectorImage& v) {
    std::vector<Vec2> out;
    for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(v.edgeCount()); ++e) {
        const cnx::Edge& edge = v.edge(e);
        if (edge.left == cnx::kExteriorFace || edge.right == cnx::kExteriorFace) continue;
        for (const cnx::NodeId n : v.edgeNodes(e)) {
            const Vec2 p = v.position(n);
            if (p.x >= 16.0 && p.x <= kW - 16.0) out.push_back(p);
        }
    }
    return out;
}

double rmsFromIdealLine(const std::vector<Vec2>& pts) {
    double s = 0.0;
    for (const Vec2& p : pts) s += (p.y - idealY(p.x)) * (p.y - idealY(p.x));
    return std::sqrt(s / static_cast<double>(pts.size()));
}

double maxPositionDifference(const VectorImage& a, const VectorImage& b) {
    double worst = 0.0;
    for (std::size_t n = 0; n < a.nodeCount(); ++n) {
        worst = std::max(worst, sla::norm(a.position(static_cast<cnx::NodeId>(n)) - b.position(static_cast<cnx::NodeId>(n))));
    }
    return worst;
}

constexpr Rgba8 kTop{250, 200, 20, 255};
constexpr Rgba8 kLowBottom{250, 184, 20, 255};   // contraste 16 níveis
constexpr Rgba8 kHighBottom{20, 60, 200, 255};   // contraste de dezenas de níveis (um logo)

}  // namespace

CNX_TEST(the_contrast_of_each_boundary_node_is_measured_from_its_two_faces) {
    const Array2D<Rgba8> img = blockStairImage(kTop, kLowBottom);
    cnx::Palette palette;
    palette.colors = {sla::toRgba(kTop), sla::toRgba(kLowBottom)};
    cnx::SegmentationResult seg = cnx::PixelSegmenter(img, cnx::SegmentParams{}).run({}, &palette);
    const cnx::TraceResult traced = cnx::BoundaryTracer().trace(seg.map);
    const cnx::SmoothTopology t = cnx::SmoothTopology::build(traced.image, 8);
    CHECK(t.nodeContrast.size() == traced.image.nodeCount());
    CHECK(t.fidelityScale.empty());  // o peso por contraste só existe se pedido
    int between = 0;
    int onFrame = 0;
    for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(traced.image.edgeCount()); ++e) {
        const cnx::Edge& edge = traced.image.edge(e);
        const bool exterior = edge.left == cnx::kExteriorFace || edge.right == cnx::kExteriorFace;
        for (const cnx::NodeId n : traced.image.edgeNodes(e)) {
            const double c = t.nodeContrast[static_cast<std::size_t>(n)];
            if (exterior) {
                CHECK(c >= 255.0);  // a face 0 é transparente: contraste máximo
                ++onFrame;
            } else if (traced.image.position(n).x > 0.0 && traced.image.position(n).x < kW) {
                CHECK(std::abs(c - 16.0) < 0.5);  // 250,200,20 contra 250,184,20
                ++between;
            }
        }
    }
    CHECK(between > 10 && onFrame > 4);
    // As pontas da fronteira (junções com a moldura) herdam o maior contraste das arestas que passam por elas.
    for (const Vec2& p : {Vec2{0.0, idealY(0)}, Vec2{static_cast<double>(kW), idealY(kW)}}) {
        for (std::size_t n = 0; n < traced.image.nodeCount(); ++n) {
            const Vec2 q = traced.image.position(static_cast<cnx::NodeId>(n));
            if (std::abs(q.x - p.x) < 0.01 && std::abs(q.y - std::round(p.y)) < 1.5 && (traced.image.flags(static_cast<cnx::NodeId>(n)) & cnx::NodeFlag::Junction)) {
                CHECK(t.nodeContrast[n] >= 255.0);
            }
        }
    }
}

CNX_TEST(the_fidelity_scale_follows_the_square_of_the_contrast) {
    const Array2D<Rgba8> img = blockStairImage(kTop, kLowBottom);
    cnx::Palette palette;
    palette.colors = {sla::toRgba(kTop), sla::toRgba(kLowBottom)};
    cnx::SegmentationResult seg = cnx::PixelSegmenter(img, cnx::SegmentParams{}).run({}, &palette);
    const cnx::TraceResult traced = cnx::BoundaryTracer().trace(seg.map);
    const cnx::SmoothTopology t = cnx::SmoothTopology::build(traced.image, 8, {60.0, 0.05});
    CHECK(t.fidelityScale.size() == traced.image.nodeCount());
    for (std::size_t n = 0; n < t.fidelityScale.size(); ++n) {
        const double r = t.nodeContrast[n] / 60.0;
        CHECK_CTX(std::abs(t.fidelityScale[n] - std::clamp(r * r, 0.05, 1.0)) < 1e-12, "escala = clamp((c/cheio)², piso, 1)");
    }
    // Contraste 16 com referência 60: (16/60)² = 0,0711, acima do piso 0,05 e abaixo de 1; com o piso em 0,2 o piso manda.
    bool sawLow = false;
    for (std::size_t n = 0; n < t.fidelityScale.size(); ++n) {
        if (std::abs(t.nodeContrast[n] - 16.0) < 0.5) {
            sawLow = true;
            CHECK(std::abs(t.fidelityScale[n] - 0.0711) < 0.002);
        }
    }
    CHECK(sawLow);
    const cnx::SmoothTopology floored = cnx::SmoothTopology::build(traced.image, 8, {60.0, 0.2});
    for (std::size_t n = 0; n < floored.fidelityScale.size(); ++n) CHECK(floored.fidelityScale[n] >= 0.2 - 1e-12);
    // Piso inválido só importa com o peso ligado.
    CHECK_THROWS(std::invalid_argument, cnx::SmoothTopology::build(traced.image, 8, {60.0, 0.0}));
    CHECK_THROWS(std::invalid_argument, cnx::SmoothTopology::build(traced.image, 8, {60.0, 1.5}));
    CHECK(cnx::SmoothTopology::build(traced.image, 8, {0.0, 0.0}).fidelityScale.empty());
}

CNX_TEST(the_spring_scale_multiplies_energy_and_gradient_per_node) {
    const Array2D<Rgba8> img = blockStairImage(kTop, kLowBottom);
    cnx::Palette palette;
    palette.colors = {sla::toRgba(kTop), sla::toRgba(kLowBottom)};
    cnx::SegmentationResult seg = cnx::PixelSegmenter(img, cnx::SegmentParams{}).run({}, &palette);
    cnx::TraceResult traced = cnx::BoundaryTracer().trace(seg.map);
    VectorImage& v = traced.image;
    cnxtest::SplitMix64 rng{5};
    for (std::size_t n = 0; n < v.nodeCount(); ++n) {
        const auto id = static_cast<cnx::NodeId>(n);
        v.setPosition(id, v.position(id) + Vec2{cnxtest::symmetric(rng, 0.6), cnxtest::symmetric(rng, 0.6)});
    }
    std::vector<double> scale(v.nodeCount());
    for (double& s : scale) s = 0.05 + cnxtest::uniform01(rng);
    const std::vector<double> ones(v.nodeCount(), 1.0);
    std::vector<Vec2> g1(v.nodeCount());
    std::vector<Vec2> gs(v.nodeCount());
    const double plain = cnx::springEnergy(v, 2.0, 0.0, g1);
    const double unit = cnx::springEnergy(v, 2.0, 0.0, gs, ones);
    CHECK(plain > 0.0 && std::abs(plain - unit) < 1e-12);  // escala 1 em todos = a mola de sempre
    double sumUnit = 0.0;
    for (std::size_t n = 0; n < g1.size(); ++n) sumUnit += sla::norm(g1[n] - gs[n]);
    CHECK(sumUnit < 1e-12);
    std::fill(gs.begin(), gs.end(), Vec2{});
    const double scaled = cnx::springEnergy(v, 2.0, 0.0, gs, scale);
    CHECK(scaled > 0.0 && std::abs(scaled - plain) > 1e-6);
    // Gradiente por diferenças finitas (nó a nó, em x e em y).
    const double h = 1e-6;
    for (std::size_t n = 0; n < v.nodeCount(); n += 7) {
        const auto id = static_cast<cnx::NodeId>(n);
        const Vec2 p = v.position(id);
        v.setPosition(id, p + Vec2{h, 0.0});
        const double ex = cnx::springEnergy(v, 2.0, 0.0, {}, scale);
        v.setPosition(id, p - Vec2{h, 0.0});
        const double ex2 = cnx::springEnergy(v, 2.0, 0.0, {}, scale);
        v.setPosition(id, p + Vec2{0.0, h});
        const double ey = cnx::springEnergy(v, 2.0, 0.0, {}, scale);
        v.setPosition(id, p - Vec2{0.0, h});
        const double ey2 = cnx::springEnergy(v, 2.0, 0.0, {}, scale);
        v.setPosition(id, p);
        CHECK_CTX(std::abs((ex - ex2) / (2.0 * h) - gs[n].x) < 1e-5 && std::abs((ey - ey2) / (2.0 * h) - gs[n].y) < 1e-5, "gradiente da mola pesada");
    }
    // Escala zero num nó: ele não custa nada. Tamanho errado lança.
    std::vector<double> zero(v.nodeCount(), 0.0);
    CHECK(cnx::springEnergy(v, 2.0, 0.0, {}, zero) == 0.0);
    const std::vector<double> shortScale(v.nodeCount() - 1, 1.0);
    CHECK_THROWS(std::invalid_argument, cnx::springEnergy(v, 2.0, 0.0, {}, shortScale));
}

CNX_TEST(corners_are_not_punctured_between_similar_bands_and_the_stairs_go_away) {
    const Array2D<Rgba8> img = blockStairImage(kTop, kLowBottom);
    SmoothParams off = photoSmooth();
    off.punctureMinContrast = 0.0;  // o comportamento de antes: todo degrau de 90° com braços retos é canto
    SmoothParams on = photoSmooth();
    CHECK(on.punctureMinContrast > 16.0);
    const Smoothed without = smooth(img, off);
    const Smoothed with = smooth(img, on);
    const double rmsStair = rmsFromIdealLine(boundaryNodes(without.staircase));
    const double rmsOff = rmsFromIdealLine(boundaryNodes(without.image));
    const double rmsOn = rmsFromIdealLine(boundaryNodes(with.image));
    const int punctOff = without.result.phases[0].puncturedNodes;
    const int punctOn = with.result.phases[0].puncturedNodes;
    std::printf("  fronteira de 16 níveis: escadinha %.3f px da reta; sem o portão %.3f (%d cantos); com o portão %.3f (%d cantos)\n", rmsStair, rmsOff, punctOff,
                rmsOn, punctOn);
    CHECK(punctOff > 0);   // sem o portão os degraus viram cantos
    CHECK(punctOn == 0);   // com ele, nenhum
    CHECK(rmsOff > 0.5 * rmsStair);  // sem o portão os 8 cantos congelados deixam a escada quase como estava
    CHECK(rmsOn < 0.3 * rmsOff);     // com o portão a fronteira chega muito mais perto da reta
}

CNX_TEST(corners_between_high_contrast_faces_are_kept_exactly_as_before) {
    const Array2D<Rgba8> img = blockStairImage(kTop, kHighBottom);
    SmoothParams off = photoSmooth();
    off.punctureMinContrast = 0.0;
    const Smoothed without = smooth(img, off);
    const Smoothed with = smooth(img, photoSmooth());
    CHECK(without.result.phases[0].puncturedNodes > 0);
    CHECK(with.result.phases[0].puncturedNodes == without.result.phases[0].puncturedNodes);
    CHECK(maxPositionDifference(without.image, with.image) == 0.0);  // um logo não muda em nada
}

CNX_TEST(the_puncture_gate_is_an_inclusive_threshold_on_the_node_contrast) {
    const Array2D<Rgba8> img = blockStairImage(kTop, kLowBottom);  // contraste 16
    SmoothParams below = photoSmooth();
    below.punctureMinContrast = 15.5;  // abaixo do contraste da fronteira: pode haver canto
    SmoothParams above = photoSmooth();
    above.punctureMinContrast = 16.5;  // acima: nenhum
    CHECK(smooth(img, below).result.phases[0].puncturedNodes > 0);
    CHECK(smooth(img, above).result.phases[0].puncturedNodes == 0);
}

CNX_TEST(the_weighted_spring_frees_low_contrast_nodes_and_leaves_sharp_ones_alone) {
    SmoothParams plain = photoSmooth();
    plain.measurementTypes = {cnx::MeasurementType::Spring, cnx::MeasurementType::Spring, cnx::MeasurementType::Spring};
    plain.punctureMinContrast = 0.0;  // isola a mola: sem o portão do canto
    plain.contrastFullLevels = 0.0;
    SmoothParams weighted = plain;
    weighted.contrastFullLevels = 60.0;
    weighted.contrastFloor = 0.05;

    const auto meanShift = [](const Smoothed& s) {
        double sum = 0.0;
        int count = 0;
        for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(s.image.edgeCount()); ++e) {
            const cnx::Edge& edge = s.image.edge(e);
            if (edge.left == cnx::kExteriorFace || edge.right == cnx::kExteriorFace) continue;
            for (const cnx::NodeId n : s.image.edgeNodes(e)) {
                sum += sla::norm(s.image.position(n) - s.image.originalPosition(n));
                ++count;
            }
        }
        return sum / count;
    };
    const Array2D<Rgba8> low = blockStairImage(kTop, kLowBottom);
    const double lowPlain = meanShift(smooth(low, plain));
    const double lowWeighted = meanShift(smooth(low, weighted));
    std::printf("  deslocamento médio da fronteira: 16 níveis, mola plena %.3f px, pesada %.3f px\n", lowPlain, lowWeighted);
    CHECK(lowWeighted > 1.2 * lowPlain);  // com a mola fraca os nós se afastam mais da escadinha

    const Array2D<Rgba8> high = blockStairImage(kTop, kHighBottom);  // contraste ≥ 60: escala 1
    const Smoothed a = smooth(high, plain);
    const Smoothed b = smooth(high, weighted);
    CHECK(maxPositionDifference(a.image, b.image) == 0.0);  // fronteira nítida: o peso por contraste não faz nada
}

CNX_TEST(the_defaults_gate_the_corners_and_leave_the_spring_unweighted) {
    // A mola pesada por contraste não ajudou no logo em degradê medido (docs §1.29): fica desligada até haver caso que a peça.
    const SmoothParams d;
    CHECK(d.punctureMinContrast == 24.0);
    CHECK(d.contrastFullLevels == 0.0);
    CHECK(d.contrastFloor == 0.05);
    for (int i = 0; i <= 9; ++i) {  // nenhum perfil mexe nesses campos
        cnx::EngineParameters e;
        e.applyProfile(static_cast<cnx::ImageProfile>(i));
        CHECK(e.smooth.punctureMinContrast == d.punctureMinContrast && e.smooth.contrastFullLevels == d.contrastFullLevels && e.smooth.contrastFloor == d.contrastFloor);
    }
}

CNX_TEST(contrast_parameters_are_validated_by_the_smoother) {
    SmoothParams p = photoSmooth();
    p.punctureMinContrast = -1.0;
    CHECK_THROWS(std::invalid_argument, cnx::ContourSmoother(p, nullptr));
    p = photoSmooth();
    p.punctureMinContrast = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(std::invalid_argument, cnx::ContourSmoother(p, nullptr));
    p = photoSmooth();
    p.contrastFullLevels = -3.0;
    CHECK_THROWS(std::invalid_argument, cnx::ContourSmoother(p, nullptr));
    p = photoSmooth();
    p.contrastFloor = 0.0;
    CHECK_THROWS(std::invalid_argument, cnx::ContourSmoother(p, nullptr));
    p = photoSmooth();
    p.contrastFloor = 1.01;
    CHECK_THROWS(std::invalid_argument, cnx::ContourSmoother(p, nullptr));
    Smoothed ok = smooth(blockStairImage(kTop, kLowBottom), photoSmooth());
    CHECK(cnx::ContourSmoother(photoSmooth(), nullptr).run(ok.staircase).phases[0].freeNodes > 0);  // parâmetros válidos rodam
}

int main() { return cnxtest::runAll(); }
