// tests/unit/block_jog_test.cpp — degraus de bloco 8×8 do JPEG não são cantos do desenho (docs §1.32).
//
// Num JPEG o bloco de 8×8 pixels pode deslocar uma borda reta de 1 a 2 px na fronteira do bloco: a borda sai e logo
// volta, e a faixa fina de halo ao lado dela termina ali (uma junção). O canto perfurado preservava a saliência, e o
// ajuste de Bézier quebrava a curva nela. Dois cantos próximos, na fronteira de uma mesma face, de convexidades opostas,
// são esse degrau: nenhum dos dois é perfurado. Um canto de verdade (os dois do mesmo sentido, ou longe um do outro) fica.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/segment/PixelSegmenter.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/vector/ContourSmoother.h"
#include "cnx/vector/EnergyTerms.h"
#include "test_harness.h"

using cnx::SmoothParams;
using cnx::VectorImage;
using sla::Array2D;
using sla::Rgba8;
using sla::Vec2;

namespace {

constexpr int kW = 48;
constexpr int kH = 48;
constexpr Rgba8 kWhite{255, 255, 255, 255};
constexpr Rgba8 kCyan{150, 240, 255, 255};
constexpr Rgba8 kBlue{0, 110, 190, 255};

/// Haste azul à direita de uma borda vertical que, na linha `jogRow`, salta `ledge` px para a esquerda. `strip` põe uma
/// faixa de halo de 1 px (ciano) contra a borda de cima, como a que termina nas junções de uma letra de JPEG; `mirror`
/// espelha a imagem (a haste passa a ficar à esquerda e o degrau, à direita).
Array2D<Rgba8> stemImage(int edgeTop, int ledge, bool strip, bool mirror, int jogRow = 24) {
    Array2D<Rgba8> img(kW, kH);
    for (int y = 0; y < kH; ++y) {
        const int edge = y < jogRow ? edgeTop : edgeTop - ledge;  // x onde começa o azul nesta linha
        for (int x = 0; x < kW; ++x) {
            Rgba8 c = x >= edge ? kBlue : kWhite;
            if (strip && y < jogRow && x == edge - 1) c = kCyan;
            img(mirror ? kW - 1 - x : x, y) = c;
        }
    }
    return img;
}

/// As 8 simetrias do quadrado (bit 0: espelha x, bit 1: espelha y, bit 2: transpõe): a haste vira horizontal, e cada aresta
/// passa a ser percorrida em outro sentido em relação às faces.
Array2D<Rgba8> symmetric(const Array2D<Rgba8>& img, int k) {
    Array2D<Rgba8> out(kW, kH);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            int sx = (k & 1) != 0 ? kW - 1 - x : x;
            int sy = (k & 2) != 0 ? kH - 1 - y : y;
            if ((k & 4) != 0) std::swap(sx, sy);
            out(x, y) = img(sx, sy);
        }
    }
    return out;
}

struct Traced {
    VectorImage image;
    cnx::SmoothTopology topology;
};

Traced trace(const Array2D<Rgba8>& img, bool strip) {
    cnx::Palette palette;
    palette.colors = {sla::toRgba(kWhite), sla::toRgba(kBlue)};
    if (strip) palette.colors.push_back(sla::toRgba(kCyan));
    cnx::SegmentationResult seg = cnx::PixelSegmenter(img, cnx::SegmentParams{}).run({}, &palette);
    cnx::TraceResult t = cnx::BoundaryTracer().trace(seg.map);
    Traced out;
    out.image = std::move(t.image);
    out.topology = cnx::SmoothTopology::build(out.image, 8);
    return out;
}

SmoothParams withLedge(double ledge) {
    SmoothParams p;
    p.jogMaxLedgePx = ledge;
    return p;
}

int cornerCount(const VectorImage& v) {
    int n = 0;
    for (cnx::NodeId i = 0; i < static_cast<cnx::NodeId>(v.nodeCount()); ++i) n += (v.flags(i) & cnx::NodeFlag::Corner) != 0 ? 1 : 0;
    return n;
}

bool isCornerAt(const VectorImage& v, Vec2 p) {
    for (cnx::NodeId i = 0; i < static_cast<cnx::NodeId>(v.nodeCount()); ++i) {
        if (sla::norm(v.position(i) - p) < 1e-9 && (v.flags(i) & cnx::NodeFlag::Corner) != 0) return true;
    }
    return false;
}

}  // namespace

CNX_TEST(a_jog_through_the_junction_of_a_halo_strip_is_not_punctured_as_two_corners) {
    // Faixa de halo de 1 px (ciano) contra a borda de cima: ela termina na linha 24, numa junção entre o ciano, o branco e o
    // azul, e a borda salta 2 px para a esquerda ali: um canto de cada lado da junção, de sentidos opostos, a 2 px.
    const Array2D<Rgba8> img = stemImage(24, 2, true, false);
    {
        Traced t = trace(img, true);
        int jogs = -1;
        const int marked = cnx::ContourSmoother::puncture(t.image, t.topology, withLedge(0.0), &jogs);  // filtro desligado
        CHECK(marked == 2 && jogs == 0);
        CHECK(isCornerAt(t.image, {24, 24}) && isCornerAt(t.image, {22, 24}));  // o que o filtro existe para evitar
    }
    Traced t = trace(img, true);
    bool junction = false;
    for (cnx::NodeId i = 0; i < static_cast<cnx::NodeId>(t.image.nodeCount()); ++i) {
        junction = junction || ((t.image.flags(i) & cnx::NodeFlag::Junction) != 0 && sla::norm(t.image.position(i) - Vec2{23, 24}) < 1e-9);
    }
    CHECK(junction);  // a premissa da cena: a junção está entre os dois cantos
    int jogs = -1;
    CHECK(cnx::ContourSmoother::puncture(t.image, t.topology, withLedge(2.5), &jogs) == 0);
    CHECK(jogs == 2);
    CHECK(!isCornerAt(t.image, {24, 24}) && !isCornerAt(t.image, {22, 24}));
}

CNX_TEST(the_ledge_limit_decides_what_is_a_jog) {
    // Sem faixa de halo, só um degrau de 3 px ou mais vira par de cantos (o filtro de arco apaga os menores). Com o limite
    // padrão (2,5) os de 3 e 4 px são cantos; subindo o limite para 3,5 e 4,5 cada um passa a ser degrau.
    for (const int ledge : {3, 4}) {
        {
            Traced t = trace(stemImage(26, ledge, false, false), false);
            int jogs = -1;
            const int marked = cnx::ContourSmoother::puncture(t.image, t.topology, withLedge(2.5), &jogs);
            CHECK_CTX(jogs == 0 && marked == 2, "ledge " + std::to_string(ledge) + ", limite 2,5");
        }
        Traced t = trace(stemImage(26, ledge, false, false), false);
        int jogs = -1;
        const int marked = cnx::ContourSmoother::puncture(t.image, t.topology, withLedge(ledge + 0.5), &jogs);
        CHECK_CTX(jogs == 2 && marked == 0, "ledge " + std::to_string(ledge) + ", limite " + std::to_string(ledge + 0.5));
    }
    // O limite é de distância: 3 px de ledge com o limite exatamente em 3 ainda é degrau, e com 2,99 não.
    Traced a = trace(stemImage(26, 3, false, false), false);
    int jogsA = -1;
    CHECK(cnx::ContourSmoother::puncture(a.image, a.topology, withLedge(3.0), &jogsA) == 0 && jogsA == 2);
    Traced b = trace(stemImage(26, 3, false, false), false);
    int jogsB = -1;
    CHECK(cnx::ContourSmoother::puncture(b.image, b.topology, withLedge(2.99), &jogsB) == 2 && jogsB == 0);
}

CNX_TEST(the_filter_does_not_depend_on_which_side_of_the_edge_the_face_is) {
    // A cena espelhada: a haste fica à esquerda, o degrau à direita, e as faces trocam de lado em cada aresta (o giro de
    // um canto é medido no sentido da aresta, e a convexidade em relação à face comum depende de qual lado ela está).
    for (const bool strip : {false, true}) {
        Traced t = trace(stemImage(26, strip ? 2 : 3, strip, true), strip);
        int jogs = -1;
        const int marked = cnx::ContourSmoother::puncture(t.image, t.topology, withLedge(3.5), &jogs);
        CHECK_CTX(marked == 0 && jogs == 2, strip ? "com faixa" : "sem faixa");
    }
}

CNX_TEST(the_jog_is_found_in_every_orientation_of_the_edge) {
    // O giro de um canto é medido no sentido da aresta e a convexidade depende de em que lado dela a face comum está: com
    // a cena girada e espelhada, cada aresta é percorrida em um sentido diferente em relação ao azul.
    // As posições da borda e da linha do degrau também variam: o sentido em que o rastreador percorre cada aresta depende
    // de onde começa, e a face comum cai à esquerda de uma aresta e à direita da outra em parte dos casos.
    for (int k = 0; k < 8; ++k) {
        for (const int top : {22, 25}) {
            for (const int row : {21, 24, 27}) {
                Traced t = trace(symmetric(stemImage(top, 2, true, false, row), k), true);
                int jogs = -1;
                const int marked = cnx::ContourSmoother::puncture(t.image, t.topology, withLedge(2.5), &jogs);
                const std::string where = "simetria " + std::to_string(k) + ", borda " + std::to_string(top) + ", linha " + std::to_string(row);
                CHECK_CTX(jogs == 2 && marked == 0, where);
            }
        }
    }
}

CNX_TEST(two_corners_of_the_same_turn_close_together_stay_corners) {
    // Um traço de 3 px de largura e 32 de comprimento: as pontas têm dois cantos convexos a 3 px, do mesmo sentido (é o
    // fim de um traço, não um degrau), mesmo com o limite acima dessa distância.
    Array2D<Rgba8> img(kW, kH);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) img(x, y) = (x >= 20 && x < 23 && y >= 8 && y < 40) ? kBlue : kWhite;
    }
    Traced t = trace(img, false);
    const int without = cnx::ContourSmoother::puncture(t.image, t.topology, withLedge(0.0));
    CHECK(without == 4);
    Traced u = trace(img, false);
    int jogs = -1;
    const int marked = cnx::ContourSmoother::puncture(u.image, u.topology, withLedge(3.5), &jogs);
    CHECK(jogs == 0);
    CHECK(marked == without);
}

CNX_TEST(a_corner_with_no_partner_near_it_stays_a_corner) {
    // O degrau com faixa de halo (linha 24) e, 20 linhas adiante, o fim da haste (linha 44): o canto do fim não tem par
    // por perto e continua canto.
    Array2D<Rgba8> img = stemImage(24, 2, true, false);
    for (int y = 44; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) img(x, y) = kWhite;
    }
    Traced t = trace(img, true);
    int jogs = -1;
    const int marked = cnx::ContourSmoother::puncture(t.image, t.topology, withLedge(2.5), &jogs);
    CHECK(jogs == 2);
    CHECK(marked == 1 && isCornerAt(t.image, {22, 44}));
    CHECK(!isCornerAt(t.image, {24, 24}) && !isCornerAt(t.image, {22, 24}));
}

CNX_TEST(after_smoothing_the_edge_runs_through_the_ledge_without_a_step) {
    // Maior giro, em radianos, entre segmentos consecutivos das fronteiras perto do degrau, depois do suavizador.
    const auto sharpestTurn = [&](double ledgeLimit, int* jogNodes) {
        Traced t = trace(stemImage(24, 2, true, false), true);
        SmoothParams p = withLedge(ledgeLimit);
        p.measurementTypes = {cnx::MeasurementType::Spring, cnx::MeasurementType::Spring, cnx::MeasurementType::Spring};
        const cnx::SmoothResult result = cnx::ContourSmoother(p, nullptr).run(t.image);
        *jogNodes = result.phases[0].jogNodes;
        double worst = 0.0;
        for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(t.image.edgeCount()); ++e) {
            const cnx::Edge& edge = t.image.edge(e);
            if (edge.left == cnx::kExteriorFace || edge.right == cnx::kExteriorFace) continue;
            const auto nodes = t.image.edgeNodes(e);
            for (std::size_t i = 1; i + 1 < nodes.size(); ++i) {
                const Vec2 a = t.image.position(nodes[i]) - t.image.position(nodes[i - 1]);
                const Vec2 b = t.image.position(nodes[i + 1]) - t.image.position(nodes[i]);
                if (std::abs(t.image.position(nodes[i]).y - 24.0) > 6.0) continue;
                worst = std::max(worst, std::abs(std::atan2(sla::cross(a, b), sla::dot(a, b))));
            }
        }
        return worst;
    };
    int keptJogs = -1;
    int flatJogs = -1;
    const double kept = sharpestTurn(0.0, &keptJogs);
    const double flattened = sharpestTurn(2.5, &flatJogs);
    CHECK(keptJogs == 0 && flatJogs == 2);  // o que o suavizador reporta por fase (PhaseStats::jogNodes)
    std::printf("  maior giro perto do degrau: %.2f rad com os cantos perfurados, %.2f rad sem eles\n", kept, flattened);
    CHECK(kept > 1.2);       // dois cantos de ~90° (1,57 rad) preservados
    CHECK(flattened < 0.8);  // alisado
}
CNX_TEST(the_jog_ledge_parameter_is_validated_fingerprinted_and_filed) {
    CHECK(SmoothParams{}.jogMaxLedgePx == 2.5);
    CHECK(withLedge(1.0).fingerprint({}) != withLedge(2.0).fingerprint({}));
    CHECK_THROWS(std::invalid_argument, cnx::ContourSmoother(withLedge(-0.1), nullptr));
    CHECK_THROWS(std::invalid_argument, cnx::ContourSmoother(withLedge(std::nan("")), nullptr));
    cnx::ContourSmoother ok(withLedge(0.0), nullptr);  // 0 desliga e é válido
    (void)ok;
    cnx::EngineParameters params;
    std::istringstream apf("ContourSmoother::jog_max_ledge_px 3.25\n");
    cnx::ParameterFiler::readAPF(apf, params);
    CHECK(params.smooth.jogMaxLedgePx == 3.25);
    cnx::EngineParameters bad;
    std::istringstream negative("ContourSmoother::jog_max_ledge_px -1\n");
    bool rejected = false;
    try {
        cnx::ParameterFiler::readAPF(negative, bad);
    } catch (const std::exception&) {
        rejected = true;
    }
    CHECK(rejected);
}

int main() { return cnxtest::runAll(); }
