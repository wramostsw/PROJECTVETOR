// tests/unit/edge_anchors_test.cpp — posição sub-pixel das bordas pela cobertura dos pixels de transição (docs §1.31).
//
// A verdade de cada cena é conhecida: o antialiasing é a cobertura EXATA (superamostragem), então a borda verdadeira (uma
// reta inclinada, um círculo) está a uma distância medível de cada nó da escadinha. Medimos a distância dos nós (ou das
// âncoras) à borda verdadeira antes e depois, e o que sobra no resultado final do suavizador.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

#include "cnx/engine/CoreEngine.h"
#include "cnx/engine/Parameters.h"
#include "cnx/preprocess/Preprocessor.h"
#include "cnx/vector/ContourSmoother.h"
#include "cnx/vector/EdgeAnchors.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::SmoothParams;
using cnx::VectorImage;
using sla::Array2D;
using sla::Rgba;
using sla::Rgba8;
using sla::Vec2;

namespace {

constexpr Rgba kWhite{1, 1, 1, 1};
constexpr Rgba kBlack{0, 0, 0, 1};

Rgba8 gray(double v01) { return {static_cast<std::uint8_t>(std::lround(255.0 * v01)), static_cast<std::uint8_t>(std::lround(255.0 * v01)), static_cast<std::uint8_t>(std::lround(255.0 * v01)), 255}; }

/// Imagem 8 bits do alvo com antialiasing exato: preto onde a cobertura é 1, branco onde é 0.
Array2D<Rgba8> toImage(const cnxtest::AaScene& s) {
    Array2D<Rgba8> img(s.coverage.width(), s.coverage.height());
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) img(x, y) = gray(1.0 - s.coverage(x, y));
    }
    return img;
}

SmoothParams anchorParams() {
    SmoothParams p;
    p.subPixelAnchors = true;
    return p;
}

// ---- cenas

constexpr int kW = 64;
constexpr int kH = 44;
constexpr double kSlope = 0.31;
constexpr double kY0 = 12.0;

double lineDistance(Vec2 p) { return std::abs(p.y - (kY0 + kSlope * p.x)) / std::sqrt(1.0 + kSlope * kSlope); }
bool belowLine(double x, double y) { return y > kY0 + kSlope * x; }

constexpr double kCx = 32.0;
constexpr double kCy = 22.0;
constexpr double kR = 14.0;
double circleDistance(Vec2 p) { return std::abs(std::hypot(p.x - kCx, p.y - kCy) - kR); }
bool inCircle(double x, double y) { return std::hypot(x - kCx, y - kCy) < kR; }

/// Nós interiores de arestas entre duas faces visíveis (longe da moldura), com a posição escolhida por `pos`.
template <class PosFn>
std::vector<Vec2> boundaryNodes(const VectorImage& v, PosFn&& pos, double margin = 6.0) {
    std::vector<Vec2> out;
    for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(v.edgeCount()); ++e) {
        const cnx::Edge& edge = v.edge(e);
        if (edge.left == cnx::kExteriorFace || edge.right == cnx::kExteriorFace) continue;
        for (const cnx::NodeId n : v.edgeNodes(e)) {
            if ((v.flags(n) & (cnx::NodeFlag::ImageBorder | cnx::NodeFlag::Junction | cnx::NodeFlag::Pinned)) != 0) continue;
            const Vec2 p = pos(n);
            if (p.x >= margin && p.x <= kW - margin) out.push_back(p);
        }
    }
    return out;
}

template <class DistFn>
double rms(const std::vector<Vec2>& pts, DistFn&& dist) {
    double s = 0.0;
    for (const Vec2& p : pts) s += dist(p) * dist(p);
    return std::sqrt(s / static_cast<double>(pts.size()));
}

/// Imagem de uma borda vertical em x0 (cobertura exata): preto à direita.
Array2D<Rgba8> verticalEdge(double x0, int blur = 0) {
    constexpr int w = 20;
    constexpr int h = 8;
    std::vector<double> col(w);
    for (int x = 0; x < w; ++x) col[static_cast<std::size_t>(x)] = std::clamp(x + 1.0 - x0, 0.0, 1.0);
    if (blur > 0) {  // média móvel de largura 2·blur+1 colunas
        std::vector<double> b(w);
        for (int x = 0; x < w; ++x) {
            double s = 0.0;
            for (int k = -blur; k <= blur; ++k) s += col[static_cast<std::size_t>(std::clamp(x + k, 0, w - 1))];
            b[static_cast<std::size_t>(x)] = s / (2 * blur + 1);
        }
        col = b;
    }
    Array2D<Rgba8> img(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) img(x, y) = gray(1.0 - col[static_cast<std::size_t>(x)]);
    }
    return img;
}

}  // namespace

// ================================================================ o perfil de uma borda

CNX_TEST(the_profile_integral_recovers_the_subpixel_position_of_an_edge) {
    for (const double x0 : {10.0, 10.2, 10.5, 10.8, 11.3, 9.65}) {
        const Array2D<Rgba8> img = verticalEdge(x0);
        const auto s = cnx::edgeOffset(img, {10.0, 4.0}, {1.0, 0.0}, kWhite, kBlack, 2.5, 0.5);
        CHECK_CTX(s.has_value(), "x0 = " + std::to_string(x0));
        if (s) CHECK_CTX(std::abs(*s - (x0 - 10.0)) < 0.02, "x0 = " + std::to_string(x0) + ", medido " + std::to_string(*s));
    }
    // Centro fora da grade de pixels: o trapézio deixa de coincidir com os cantos da interpolação, o erro continua pequeno.
    const Array2D<Rgba8> img = verticalEdge(10.35);
    const auto off = cnx::edgeOffset(img, {10.37, 4.0}, {1.0, 0.0}, kWhite, kBlack, 2.5, 0.5);
    CHECK(off.has_value() && std::abs(*off - (10.35 - 10.37)) < 0.05);
}

CNX_TEST(the_offset_flips_sign_with_the_normal_and_the_order_of_the_colors) {
    const Array2D<Rgba8> img = verticalEdge(10.3);
    const auto fwd = cnx::edgeOffset(img, {10.0, 4.0}, {1.0, 0.0}, kWhite, kBlack, 2.5, 0.5);
    // Normal invertida: o lado negativo passa a ser o escuro (a direita) e a borda fica a −0,3 ao longo dela.
    const auto rev = cnx::edgeOffset(img, {10.0, 4.0}, {-1.0, 0.0}, kBlack, kWhite, 2.5, 0.5);
    CHECK(fwd && rev);
    if (fwd && rev) {
        CHECK(std::abs(*fwd - 0.3) < 0.02);
        CHECK(std::abs(*rev + 0.3) < 0.02);
    }
    // Cores na ordem errada para esta normal: não é a transição de a para b.
    CHECK(!cnx::edgeOffset(img, {10.0, 4.0}, {1.0, 0.0}, kBlack, kWhite, 2.5, 0.5).has_value());
}

CNX_TEST(a_blurred_edge_keeps_its_position_because_the_integral_is_symmetric) {
    for (const double x0 : {10.0, 10.4, 10.75}) {
        const Array2D<Rgba8> img = verticalEdge(x0, 1);  // desfoque de 3 colunas (borda de JPEG)
        const auto s = cnx::edgeOffset(img, {10.0, 4.0}, {1.0, 0.0}, kWhite, kBlack, 2.5, 0.5);
        CHECK_CTX(s.has_value(), "x0 = " + std::to_string(x0));
        if (s) CHECK_CTX(std::abs(*s - (x0 - 10.0)) < 0.06, "x0 = " + std::to_string(x0) + ", medido " + std::to_string(*s));
    }
}

CNX_TEST(a_profile_that_is_not_a_clean_transition_between_the_two_faces_is_rejected) {
    // Uma faixa vermelha de 1 px no meio da transição: a cor sai da reta branco–preto.
    Array2D<Rgba8> img = verticalEdge(10.0);
    for (int y = 0; y < img.height(); ++y) img(10, y) = {255, 0, 0, 255};
    CHECK(!cnx::edgeOffset(img, {10.0, 4.0}, {1.0, 0.0}, kWhite, kBlack, 2.5, 0.5).has_value());
    // Um lado que não é puro: a janela chega a outra borda antes de acabar a transição.
    Array2D<Rgba8> two(20, 8);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 20; ++x) two(x, y) = x < 9 ? gray(1.0) : (x < 11 ? gray(0.0) : gray(1.0));
    }
    CHECK(!cnx::edgeOffset(two, {10.0, 4.0}, {1.0, 0.0}, kWhite, kBlack, 2.5, 0.5).has_value());
    // Duas cores iguais: nada a medir.
    CHECK(!cnx::edgeOffset(img, {10.0, 4.0}, {1.0, 0.0}, kBlack, kBlack, 2.5, 0.5).has_value());
}

CNX_TEST(jpeg_ringing_beyond_the_two_colors_is_not_coverage) {
    // Borda exata em x = 10 entre dois cinzas médios, com um pixel mais claro que o lado claro logo depois dela (o anel do
    // JPEG, 30 % do contraste). Cobertura não passa de 100 %: sem o corte de t em 0..1 o anel empurraria a borda 0,3 px.
    Array2D<Rgba8> img(20, 8);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 20; ++x) img(x, y) = x < 10 ? gray(0.3) : (x == 10 ? gray(0.82) : gray(0.7));
    }
    const auto s = cnx::edgeOffset(img, {10.0, 4.0}, {1.0, 0.0}, sla::Rgba{0.3f, 0.3f, 0.3f, 1.0f}, sla::Rgba{0.7f, 0.7f, 0.7f, 1.0f}, 2.5, 0.5);
    CHECK(s.has_value());
    // Com o corte o anel ainda deixa um resto (a amostra entre o pixel do anel e o vizinho já cai dentro de 0..1): 0,075 px.
    // Sem o corte seriam 0,3 px.
    if (s) CHECK(std::abs(*s) < 0.12);
}

CNX_TEST(edge_offset_validates_its_arguments) {
    const Array2D<Rgba8> img = verticalEdge(10.0);
    CHECK_THROWS(std::invalid_argument, cnx::edgeOffset(img, {10, 4}, {1, 0}, kWhite, kBlack, 0.0, 0.5));
    CHECK_THROWS(std::invalid_argument, cnx::edgeOffset(img, {10, 4}, {1, 0}, kWhite, kBlack, 2.5, 0.0));
    CHECK_THROWS(std::invalid_argument, cnx::edgeOffset(img, {10, 4}, {1, 0}, kWhite, kBlack, 0.4, 0.5));
    CHECK_THROWS(std::invalid_argument, cnx::edgeOffset(Array2D<Rgba8>(0, 0), {10, 4}, {1, 0}, kWhite, kBlack, 2.5, 0.5));
}

// ================================================================ âncoras de uma aresta inteira

CNX_TEST(anchors_move_from_the_staircase_onto_a_slanted_edge) {
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, belowLine, 8);
    const Array2D<Rgba8> img = toImage(s);
    const auto original = [&](cnx::NodeId n) { return s.image.originalPosition(n); };
    const std::vector<Vec2> before = boundaryNodes(s.image, original);
    CHECK(before.size() > 10);
    const double stairRms = rms(before, lineDistance);

    const cnx::AnchorStats st = cnx::refineEdgeAnchors(s.image, img, anchorParams());
    CHECK(st.moved > 10);
    const std::vector<Vec2> after = boundaryNodes(s.image, original);
    const double anchorRms = rms(after, lineDistance);
    std::printf("  reta: escadinha %.3f px, âncoras %.3f px (%d movidas, deslocamento médio %.3f)\n", stairRms, anchorRms, st.moved,
                st.meanShiftPx);
    CHECK(stairRms > 0.25);   // a escadinha tem de estar longe da reta, senão o teste não prova nada
    CHECK(anchorRms < 0.1);
    CHECK(anchorRms < 0.4 * stairRms);
}

CNX_TEST(only_the_anchors_move_never_the_node_positions) {
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, belowLine, 8);
    std::vector<Vec2> positions;
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) positions.push_back(s.image.position(n));
    (void)cnx::refineEdgeAnchors(s.image, toImage(s), anchorParams());
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) {
        CHECK(s.image.position(n) == positions[static_cast<std::size_t>(n)]);
    }
}

CNX_TEST(the_shift_of_each_anchor_is_capped) {
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, belowLine, 8);
    SmoothParams p = anchorParams();
    p.anchorMaxShiftPx = 0.1;
    std::vector<Vec2> before;
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) before.push_back(s.image.originalPosition(n));
    const cnx::AnchorStats st = cnx::refineEdgeAnchors(s.image, toImage(s), p);
    CHECK(st.moved > 10);
    double worst = 0.0;
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) {
        worst = std::max(worst, sla::norm(s.image.originalPosition(n) - before[static_cast<std::size_t>(n)]));
    }
    CHECK(worst <= 0.1 + 1e-9);
    CHECK(worst > 0.09);  // e o teto foi atingido: o deslocamento medido é maior que 0,1
    CHECK(st.meanShiftPx <= 0.1 + 1e-9);
}

CNX_TEST(low_contrast_edges_are_left_on_the_staircase) {
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, belowLine, 8);
    SmoothParams p = anchorParams();
    p.anchorMinContrast = 1000.0;  // acima do contraste preto–branco (441 níveis)
    std::vector<Vec2> before;
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) before.push_back(s.image.originalPosition(n));
    const cnx::AnchorStats st = cnx::refineEdgeAnchors(s.image, toImage(s), p);
    CHECK(st.moved == 0 && st.candidates == 0 && st.lowContrast > 10);
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) {
        CHECK(s.image.originalPosition(n) == before[static_cast<std::size_t>(n)]);
    }
}

// Retângulo de cantos em posições fracionárias, longe da moldura: 4 nós, todos cantos de 90°, e lados retos sem nó nenhum
// entre os cantos (o tracer tira os colineares), de modo que só mover os cantos move os lados.
// Os pixels dos cantos têm cobertura 0,8 × 0,8 = 0,64 ≥ ½: entram na segmentação, e a escadinha tem só 4 nós.
constexpr double kRx0 = 20.2, kRx1 = 44.8, kRy0 = 12.2, kRy1 = 30.8;
bool inRect(double x, double y) { return x > kRx0 && x < kRx1 && y > kRy0 && y < kRy1; }

double cornerError(const VectorImage& v, const std::vector<cnx::NodeId>& corners, bool anchors) {
    double worst = 0.0;
    for (const cnx::NodeId n : corners) {
        const Vec2 p = anchors ? v.originalPosition(n) : v.position(n);
        const double dx = std::min(std::abs(p.x - kRx0), std::abs(p.x - kRx1));
        const double dy = std::min(std::abs(p.y - kRy0), std::abs(p.y - kRy1));
        worst = std::max(worst, std::hypot(dx, dy));
    }
    return worst;
}

std::vector<cnx::NodeId> rectCorners(const VectorImage& v) {
    std::vector<cnx::NodeId> out;
    for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(v.edgeCount()); ++e) {
        const cnx::Edge& edge = v.edge(e);
        if (edge.left == cnx::kExteriorFace || edge.right == cnx::kExteriorFace) continue;
        for (const cnx::NodeId n : v.edgeNodes(e)) {
            if (std::find(out.begin(), out.end(), n) == out.end()) out.push_back(n);
        }
    }
    return out;
}

CNX_TEST(a_corner_goes_to_the_intersection_of_its_two_subpixel_edges) {
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, inRect, 8);
    const std::vector<cnx::NodeId> corners = rectCorners(s.image);
    CHECK(corners.size() == 4);
    const double before = cornerError(s.image, corners, true);
    const cnx::AnchorStats st = cnx::refineEdgeAnchors(s.image, toImage(s), anchorParams());
    const double after = cornerError(s.image, corners, true);
    std::printf("  retângulo: pior canto %.3f px na escadinha, %.3f px nas âncoras (%d cantos movidos)\n", before, after, st.cornersMoved);
    CHECK(before > 0.25);  // a escadinha erra o canto em 0,2 px por eixo (0,28 px)
    CHECK(st.cornersMoved == 4);
    CHECK(after < 0.1);
}

CNX_TEST(a_corner_with_a_short_or_curved_arm_stays_on_the_staircase) {
    // Retângulo de 5 px de largura: os lados menores têm menos de 6 px de arco, não há braço de 2 a 6 px a medir.
    const auto thin = [](double x, double y) { return x > 20.2 && x < 24.8 && y > 12.2 && y < 30.8; };
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, thin, 8);
    std::vector<Vec2> before;
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) before.push_back(s.image.originalPosition(n));
    const cnx::AnchorStats st = cnx::refineEdgeAnchors(s.image, toImage(s), anchorParams());
    CHECK(st.cornersMoved == 0);
    CHECK(st.skippedCorner >= 4);
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) {
        CHECK(s.image.originalPosition(n) == before[static_cast<std::size_t>(n)]);
    }
}

CNX_TEST(the_straight_sides_between_corners_follow_the_corner_anchors_after_smoothing) {
    const auto run = [&](bool anchors) {
        cnxtest::AaScene s = cnxtest::aaScene(kW, kH, inRect, 8);
        cnx::EngineParameters e;
        e.applyProfile(cnx::ImageProfile::PhotoMed);
        SmoothParams p = e.smooth;
        if (anchors) (void)cnx::refineEdgeAnchors(s.image, toImage(s), p);
        (void)cnx::ContourSmoother(p, nullptr).run(s.image);
        double worst = 0.0;  // pior distância de um nó do retângulo à sua posição verdadeira
        for (const cnx::NodeId n : rectCorners(s.image)) {
            const Vec2 q = s.image.position(n);
            worst = std::max(worst, std::hypot(std::min(std::abs(q.x - kRx0), std::abs(q.x - kRx1)),
                                               std::min(std::abs(q.y - kRy0), std::abs(q.y - kRy1))));
        }
        return worst;
    };
    const double plain = run(false);
    const double refined = run(true);
    std::printf("  retângulo suavizado: pior canto %.3f px sem âncoras, %.3f px com âncoras\n", plain, refined);
    CHECK(refined < 0.5 * plain);
}

CNX_TEST(the_frame_nodes_are_never_touched) {
    // Cena em que a figura encosta na moldura: os nós sobre a moldura continuam onde o tracer os pôs.
    const auto touching = [](double x, double y) { return x < 30.4 && y > 10.2; };
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, touching, 8);
    std::vector<Vec2> before;
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) before.push_back(s.image.originalPosition(n));
    (void)cnx::refineEdgeAnchors(s.image, toImage(s), anchorParams());
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(s.image.nodeCount()); ++n) {
        if ((s.image.flags(n) & (cnx::NodeFlag::ImageBorder | cnx::NodeFlag::Junction | cnx::NodeFlag::Pinned)) != 0) {
            CHECK(s.image.originalPosition(n) == before[static_cast<std::size_t>(n)]);
        }
    }
}

CNX_TEST(anchors_follow_a_curved_edge_too) {
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, inCircle, 8);
    const auto original = [&](cnx::NodeId n) { return s.image.originalPosition(n); };
    const double stairRms = rms(boundaryNodes(s.image, original, 0.0), circleDistance);
    const cnx::AnchorStats st = cnx::refineEdgeAnchors(s.image, toImage(s), anchorParams());
    const double anchorRms = rms(boundaryNodes(s.image, original, 0.0), circleDistance);
    std::printf("  círculo: escadinha %.3f px, âncoras %.3f px (%d movidas, %d em canto)\n", stairRms, anchorRms, st.moved,
                st.skippedCorner);
    CHECK(st.moved > 20);
    CHECK(anchorRms < stairRms);
    CHECK(anchorRms < 0.7 * stairRms);
}

CNX_TEST(a_thin_stroke_is_measured_with_a_shorter_reach) {
    // Traço inclinado de 2,2 px: a janela de ±2,5 px passa da outra borda e o perfil deixa de ser uma transição limpa;
    // o alcance cai (até 1,5 px) até as duas pontas voltarem a ser puras.
    const auto stroke = [](double x, double y) {
        const double d = (y - (kY0 + kSlope * x)) / std::sqrt(1.0 + kSlope * kSlope);
        return d > 0.0 && d < 2.2;
    };
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, stroke, 8);
    const cnx::AnchorStats st = cnx::refineEdgeAnchors(s.image, toImage(s), anchorParams());
    std::printf("  traço de 2,2 px: %d candidatos, %d movidos, %d de perfil inválido\n", st.candidates, st.moved, st.skippedProfile);
    CHECK(st.candidates > 20);
    CHECK(st.moved > 0.5 * st.candidates);  // sem a redução do alcance nenhum perfil é limpo (ver o mutante E8)
    // Com um alcance de 1,5 px desde o começo o resultado é o mesmo (não há o que reduzir mais).
    cnxtest::AaScene t = cnxtest::aaScene(kW, kH, stroke, 8);
    SmoothParams shorter = anchorParams();
    shorter.anchorReachPx = 1.5;
    const cnx::AnchorStats st2 = cnx::refineEdgeAnchors(t.image, toImage(t), shorter);
    CHECK(st2.moved == st.moved);
}

CNX_TEST(an_image_of_another_size_is_rejected) {
    cnxtest::AaScene s = cnxtest::aaScene(kW, kH, belowLine, 4);
    CHECK_THROWS(std::invalid_argument, cnx::refineEdgeAnchors(s.image, Array2D<Rgba8>(kW + 1, kH), anchorParams()));
}

// ================================================================ no resultado do suavizador

CNX_TEST(the_smoothed_contour_ends_closer_to_the_true_edge_with_subpixel_anchors) {
    const auto run = [&](bool anchors) {
        cnxtest::AaScene s = cnxtest::aaScene(kW, kH, belowLine, 8);
        cnx::EngineParameters e;
        e.applyProfile(cnx::ImageProfile::PhotoMed);
        SmoothParams p = e.smooth;
        if (anchors) (void)cnx::refineEdgeAnchors(s.image, toImage(s), p);
        (void)cnx::ContourSmoother(p, nullptr).run(s.image);
        const auto pos = [&](cnx::NodeId n) { return s.image.position(n); };
        return rms(boundaryNodes(s.image, pos), lineDistance);
    };
    const double plain = run(false);
    const double refined = run(true);
    std::printf("  suavizado: sem âncoras %.3f px, com âncoras %.3f px\n", plain, refined);
    CHECK(refined < plain);
    CHECK(refined < 0.8 * plain);
}

// ================================================================ no motor inteiro

namespace {

class MemoryLoader final : public cnx::IImageLoader {
public:
    explicit MemoryLoader(Array2D<Rgba8> img) : image_(std::move(img)) {}
    cnx::SourceImage load(const std::filesystem::path&, const std::stop_token&) override {
        cnx::SourceImage s;
        s.pixels = image_;
        s.formatHint = ".png";
        return s;
    }

private:
    Array2D<Rgba8> image_;
};

/// Duas tintas (laranja e azul) com a borda inclinada de antialiasing exato.
Array2D<Rgba8> colorLineImage() {
    const cnxtest::AaScene s = cnxtest::aaScene(kW, kH, belowLine, 8);
    Array2D<Rgba8> img(kW, kH);
    const auto mix = [](double a, double b, double c) { return static_cast<std::uint8_t>(std::lround(a + (b - a) * c)); };
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const double c = s.coverage(x, y);
            img(x, y) = {mix(250, 20, c), mix(184, 60, c), mix(20, 200, c), 255};
        }
    }
    return img;
}

double engineBoundaryRms(bool anchors) {
    cnx::CoreEngine engine(std::make_shared<MemoryLoader>(colorLineImage()));
    engine.setInput("cena.png");
    cnx::EngineParameters p = engine.parameters();
    p.classifier.autoClassify = false;
    p.classifier.forcedProfile = cnx::ImageProfile::Custom;  // o perfil não reescreve nada: só vale o que está aqui
    p.palette.forcedNumColors = 2;
    p.segment.paletteSnapMax = p.segment.mergeThreshMax / 9.0;
    p.segment.absorbRamps = true;
    p.smooth.subPixelAnchors = anchors;
    engine.setParameters(p);
    engine.start(cnx::Stage::SmoothContours);
    engine.wait();
    const auto snap = engine.snapshot();
    CHECK(snap->smoothed != nullptr);
    CHECK(snap->preprocessed->scale == 1.0);  // sem reamostragem: as coordenadas continuam as da cena
    const auto pos = [&](cnx::NodeId n) { return snap->smoothed->position(n); };
    return rms(boundaryNodes(*snap->smoothed, pos), lineDistance);
}

}  // namespace

CNX_TEST(the_engine_places_the_boundary_with_subpixel_anchors_when_asked) {
    const double plain = engineBoundaryRms(false);
    const double refined = engineBoundaryRms(true);
    std::printf("  motor, borda inclinada: sem âncoras %.3f px, com âncoras %.3f px\n", plain, refined);
    CHECK(refined < plain);
    CHECK(refined < 0.5 * plain);
}

// ================================================================ parâmetros

CNX_TEST(the_photo_and_gradient_profiles_turn_the_anchors_on_the_logo_profiles_do_not) {
    for (const auto profile : {cnx::ImageProfile::PhotoHigh, cnx::ImageProfile::PhotoMed, cnx::ImageProfile::PhotoLow}) {
        cnx::EngineParameters e;
        e.applyProfile(profile);
        CHECK(e.smooth.subPixelAnchors);
    }
    for (const auto profile : {cnx::ImageProfile::LogoHighAA, cnx::ImageProfile::LogoMedAA, cnx::ImageProfile::LogoLowAA,
                               cnx::ImageProfile::LogoHigh, cnx::ImageProfile::LogoMed, cnx::ImageProfile::LogoLow,
                               cnx::ImageProfile::Custom}) {
        cnx::EngineParameters e;
        e.applyProfile(profile);
        CHECK(!e.smooth.subPixelAnchors);
    }
    CHECK(!SmoothParams{}.subPixelAnchors);
}

CNX_TEST(the_anchor_parameters_are_in_the_fingerprint_and_in_the_apf_file) {
    const SmoothParams a;
    for (int field = 0; field < 4; ++field) {
        SmoothParams b;
        if (field == 0) b.subPixelAnchors = true;
        if (field == 1) b.anchorReachPx = 3.0;
        if (field == 2) b.anchorMaxShiftPx = 0.5;
        if (field == 3) b.anchorMinContrast = 40.0;
        CHECK_CTX(b.fingerprint({}) != a.fingerprint({}), "campo " + std::to_string(field));
    }
    cnx::EngineParameters params;
    std::istringstream apf("ContourSmoother::sub_pixel_anchors true\nContourSmoother::anchor_reach_px 3.5\n"
                           "ContourSmoother::anchor_max_shift_px 0.75\nContourSmoother::anchor_min_contrast 30\n");
    cnx::ParameterFiler::readAPF(apf, params);
    CHECK(params.smooth.subPixelAnchors);
    CHECK(params.smooth.anchorReachPx == 3.5 && params.smooth.anchorMaxShiftPx == 0.75 && params.smooth.anchorMinContrast == 30.0);
    cnx::EngineParameters bad;
    std::istringstream tooShort("ContourSmoother::anchor_reach_px 0.2\n");
    bool rejected = false;
    try {
        cnx::ParameterFiler::readAPF(tooShort, bad);
    } catch (const std::exception&) {
        rejected = true;
    }
    CHECK(rejected);
}

int main() { return cnxtest::runAll(); }
