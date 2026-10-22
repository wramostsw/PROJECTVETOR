// tests/unit/energy_terms_test.cpp — termos da energia de suavização: valores em forma fechada,
// gradientes contra diferenças finitas centradas e o detector de cantos perfurados.
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <string>
#include <vector>

#include "cnx/vector/EnergyTerms.h"
#include "cnx/vector/GenerativeModel.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::EnergyBreakdown;
using cnx::EnergyWeights;
using cnx::NodeId;
using cnx::SmoothTopology;
using cnx::VectorImage;
using cnxtest::SplitMix64;
using sla::Vec2;

namespace {

constexpr double kPi = std::numbers::pi;

/// Imagem 12×10 com uma ilha 4×5 (x 3..6, y 2..6): arestas fechadas e nenhuma junção.
cnxtest::VectorScene islandScene() {
    cnx::PixelMap map(12, 10);
    const cnx::RegionId bg = map.createRegion();
    const cnx::RegionId island = map.createRegion();
    sla::Array2D<sla::Rgba8> image(12, 10);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 12; ++x) {
            const bool in = x >= 3 && x < 7 && y >= 2 && y < 7;
            image(x, y) = in ? sla::Rgba8{0, 0, 0, 255} : sla::Rgba8{255, 255, 255, 255};
            map.setLabel(x, y, in ? island : bg, image(x, y));
        }
    }
    map.assignMedianColors();
    (void)map.compact();
    cnx::TraceResult t = cnx::BoundaryTracer().trace(map);
    return {std::move(t.image), std::move(image), std::move(t.pixelFaces)};
}

/// Imagem 10×6 dividida em duas regiões por x = 5: os nós (5,0) e (5,6) são junções sobre a moldura.
cnxtest::VectorScene splitScene() {
    cnx::PixelMap map(10, 6);
    const cnx::RegionId a = map.createRegion();
    const cnx::RegionId b = map.createRegion();
    sla::Array2D<sla::Rgba8> image(10, 6);
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 10; ++x) {
            image(x, y) = x < 5 ? sla::Rgba8{255, 0, 0, 255} : sla::Rgba8{0, 0, 255, 255};
            map.setLabel(x, y, x < 5 ? a : b, image(x, y));
        }
    }
    map.assignMedianColors();
    (void)map.compact();
    cnx::TraceResult t = cnx::BoundaryTracer().trace(map);
    return {std::move(t.image), std::move(image), std::move(t.pixelFaces)};
}

using EnergyFn = std::function<double(const VectorImage&, std::span<Vec2>)>;

/// Maior desvio relativo entre o gradiente analítico e a diferença finita centrada, em todos os nós.
double worstGradientError(const VectorImage& base, const EnergyFn& energy, double h = 1e-6) {
    VectorImage image = base;
    std::vector<Vec2> grad(image.nodeCount());
    (void)energy(image, grad);
    double worst = 0.0;
    for (NodeId n = 0; n < static_cast<NodeId>(image.nodeCount()); ++n) {
        const Vec2 p = image.position(n);
        for (int axis = 0; axis < 2; ++axis) {
            const Vec2 d = axis == 0 ? Vec2{h, 0.0} : Vec2{0.0, h};
            image.setPosition(n, p + d);
            const double ep = energy(image, {});
            image.setPosition(n, p - d);
            const double em = energy(image, {});
            image.setPosition(n, p);
            const double fd = (ep - em) / (2.0 * h);
            const double g = axis == 0 ? grad[static_cast<std::size_t>(n)].x : grad[static_cast<std::size_t>(n)].y;
            worst = std::max(worst, std::abs(g - fd) / (1.0 + std::abs(fd)));
        }
    }
    return worst;
}

double shoelaceOf(const VectorImage& image, const std::vector<NodeId>& nodes) {
    double a = 0.0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        a += sla::cross(image.position(nodes[i]), image.position(nodes[(i + 1) % nodes.size()]));
    }
    return 0.5 * a;
}

}  // namespace

// ================================================================ topologia

CNX_TEST(topology_matches_an_independent_count) {
    SplitMix64 rng{301};
    for (int scene = 0; scene < 5; ++scene) {
        const cnxtest::VectorScene s = cnxtest::vectorScene(16 + scene, 12, 3, 14, rng);
        const SmoothTopology topo = SmoothTopology::build(s.image, 8);
        const std::string tag = "cena " + std::to_string(scene);

        std::size_t expectedTriples = 0;
        for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(s.image.edgeCount()); ++e) {
            const auto nodes = s.image.edgeNodes(e);
            const bool closed = nodes.front() == nodes.back();
            expectedTriples += closed ? (nodes.size() >= 4 ? nodes.size() - 1 : 0) : nodes.size() - 2;
        }
        CHECK_CTX(topo.triples.size() == expectedTriples, tag);
        for (const SmoothTopology::Triple& t : topo.triples) {
            // Uma tripla nunca tem junção no meio, e os vizinhos são os da cadeia.
            CHECK_CTX((s.image.flags(t.mid) & cnx::NodeFlag::Junction) == 0, tag);
            const auto nodes = s.image.edgeNodes(t.edge);
            CHECK_CTX(nodes[t.index] == t.mid && nodes[t.index + 1] == t.next, tag);
        }

        // Laços pequenos das faces interiores (externo + furos) contados à parte.
        std::size_t expectedLoops = 0;
        const auto count = [&](cnx::HalfEdgeId first) {
            std::size_t n = 0;
            cnx::HalfEdgeId h = first;
            do {
                n += s.image.edgeNodes(cnx::edgeOf(h)).size() - 1;
                h = s.image.next(h);
            } while (h != first);
            expectedLoops += (n >= 3 && n < 8) ? 1 : 0;
        };
        for (cnx::FaceId f = 1; f < static_cast<cnx::FaceId>(s.image.faceCount()); ++f) {
            if (s.image.face(f).outerLoop != cnx::kInvalidId) count(s.image.face(f).outerLoop);
            for (const cnx::HalfEdgeId h : s.image.holeLoops(f)) count(h);
        }
        CHECK_CTX(topo.airLoops.size() == expectedLoops, tag);
        for (const SmoothTopology::AirLoop& loop : topo.airLoops) {
            CHECK_CTX(loop.nodes.size() >= 3 && loop.nodes.size() < 8, tag);
            std::vector<Vec2> origin;
            double a = 0.0;
            for (std::size_t i = 0; i < loop.nodes.size(); ++i) {
                a += sla::cross(s.image.originalPosition(loop.nodes[i]), s.image.originalPosition(loop.nodes[(i + 1) % loop.nodes.size()]));
            }
            CHECK_CTX(std::abs(0.5 * a - loop.originalArea) < 1e-12, tag);
        }

        std::size_t expectedBorder = 0;
        for (NodeId n = 0; n < static_cast<NodeId>(s.image.nodeCount()); ++n) {
            const auto fl = s.image.flags(n);
            expectedBorder += ((fl & cnx::NodeFlag::ImageBorder) != 0 && (fl & cnx::NodeFlag::Pinned) == 0) ? 1 : 0;
        }
        CHECK_CTX(topo.borderAnchors.size() == expectedBorder, tag);
        for (const SmoothTopology::BorderAnchor& b : topo.borderAnchors) {
            const Vec2 o = s.image.originalPosition(b.node);
            const double c = b.axis == 0 ? o.x : o.y;
            CHECK_CTX(c == b.coordinate, tag);
            CHECK_CTX(b.coordinate == 0.0 || b.coordinate == (b.axis == 0 ? s.image.width() : s.image.height()), tag);
        }
    }
}

// ================================================================ valores em forma fechada

CNX_TEST(spring_energy_and_free_zone_closed_form) {
    cnxtest::VectorScene s = islandScene();
    const NodeId n = 3;
    const Vec2 p0 = s.image.position(n);
    s.image.setPosition(n, p0 + Vec2{0.3, 0.4});  // |d| = 0,5
    std::vector<Vec2> grad(s.image.nodeCount());
    CHECK(std::abs(cnx::springEnergy(s.image, 2.0, 0.0, grad) - 2.0 * 0.25) < 1e-12);
    CHECK(std::abs(grad[3].x - 2.0 * 2.0 * 0.3) < 1e-12 && std::abs(grad[3].y - 2.0 * 2.0 * 0.4) < 1e-12);
    for (std::size_t i = 0; i < grad.size(); ++i) {
        if (i != 3) CHECK(grad[i].x == 0.0 && grad[i].y == 0.0);
    }
    // Zona livre de raio 0,2: custa (0,5 − 0,2)²; dentro do raio custa zero e o gradiente some.
    std::vector<Vec2> g2(s.image.nodeCount());
    CHECK(std::abs(cnx::springEnergy(s.image, 1.0, 0.2, g2) - 0.09) < 1e-12);
    CHECK(std::abs(g2[3].x - 2.0 * 0.3 * 0.6) < 1e-12);  // 2·(d−r)·dir = 2·0,3·(0,6; 0,8)
    std::vector<Vec2> g3(s.image.nodeCount());
    CHECK(cnx::springEnergy(s.image, 1.0, 0.6, g3) == 0.0);
    CHECK(g3[3].x == 0.0 && g3[3].y == 0.0);
}

CNX_TEST(angular_energy_of_a_rectangle_is_four_right_angles) {
    cnxtest::VectorScene s = islandScene();
    const SmoothTopology topo = SmoothTopology::build(s.image, 8);
    CHECK(topo.triples.size() == 4);
    CHECK(std::abs(cnx::angularEnergy(s.image, topo, 1.0, {}) - kPi * kPi) < 1e-12);  // 4·(π/2)²
    CHECK(std::abs(cnx::angularEnergy(s.image, topo, 3.0, {}) - 3.0 * kPi * kPi) < 1e-12);
    // Canto perfurado: o nó com a flag Corner sai da soma.
    s.image.setFlags(topo.triples[0].mid, static_cast<std::uint8_t>(s.image.flags(topo.triples[0].mid) | cnx::NodeFlag::Corner));
    CHECK(std::abs(cnx::angularEnergy(s.image, topo, 1.0, {}) - 0.75 * kPi * kPi) < 1e-12);
    // Segmento colapsado é ignorado (sem NaN).
    s.image.setPosition(topo.triples[1].next, s.image.position(topo.triples[1].mid));
    CHECK(std::isfinite(cnx::angularEnergy(s.image, topo, 1.0, {})));
}

CNX_TEST(spacing_energy_of_a_rectangle_and_of_a_uniform_chain) {
    cnxtest::VectorScene s = islandScene();
    const SmoothTopology topo = SmoothTopology::build(s.image, 8);
    // Lados 4, 5, 4, 5: em cada nó (|b| − |a|)/(|b| + |a|) = ±1/9 → 4 termos de 1/81 (diferença RELATIVA, sem escala).
    CHECK(std::abs(cnx::spacingEnergy(s.image, topo, 1.0, {}) - 4.0 / 81.0) < 1e-12);
    // Quadrado: todos os lados iguais → energia zero e gradiente zero.
    cnx::PixelMap map(8, 8);
    const cnx::RegionId bg = map.createRegion();
    const cnx::RegionId sq = map.createRegion();
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) map.setLabel(x, y, (x >= 2 && x < 6 && y >= 2 && y < 6) ? sq : bg, {0, 0, 0, 255});
    }
    map.assignMedianColors();
    (void)map.compact();
    const cnx::TraceResult t = cnx::BoundaryTracer().trace(map);
    const SmoothTopology topo2 = SmoothTopology::build(t.image, 8);
    std::vector<Vec2> grad(t.image.nodeCount());
    CHECK(cnx::spacingEnergy(t.image, topo2, 1.0, grad) == 0.0);
    for (const Vec2& g : grad) CHECK(std::abs(g.x) < 1e-12 && std::abs(g.y) < 1e-12);
}

CNX_TEST(spacing_energy_is_scale_free_and_bounded) {
    // A forma absoluta (|b| − |a|)² custava 961 por canto num fio de 34 × 3 px e achatava o fio num quadrado.
    // Agora s = (|b| − |a|)/(|b| + |a|) ∈ [−1, 1]: não depende da escala e cada nó custa no máximo o peso.
    cnxtest::VectorScene s = islandScene();
    const SmoothTopology topo = SmoothTopology::build(s.image, 8);
    const double base = cnx::spacingEnergy(s.image, topo, 1.0, {});
    for (const double k : {0.1, 7.0, 1234.5}) {
        VectorImage scaled = s.image;
        for (NodeId n = 0; n < static_cast<NodeId>(scaled.nodeCount()); ++n) scaled.setPosition(n, s.image.position(n) * k);
        CHECK_CTX(std::abs(cnx::spacingEnergy(scaled, topo, 1.0, {}) - base) < 1e-12, "escala " + std::to_string(k));
    }
    // Fio de 34 × 3 montado à mão (uma aresta fechada de 4 cantos): lados 34, 3, 34, 3 → s = ±31/37 nos 4 nós.
    VectorImage thin(40.0, 10.0);
    const cnx::FaceId face = thin.addFace(sla::Rgba{0.0f, 0.0f, 0.0f, 1.0f}, 0);
    const NodeId a = thin.addNode({3, 3}, cnx::NodeFlag::LoopAnchor);
    const std::array<NodeId, 5> loop{a, thin.addNode({37, 3}), thin.addNode({37, 6}), thin.addNode({3, 6}), a};
    (void)thin.addEdge(loop, face, cnx::kExteriorFace);
    thin.linkFaceLoops();
    const SmoothTopology thinTopo = SmoothTopology::build(thin, 8);
    CHECK(thinTopo.triples.size() == 4);
    const double s1 = 31.0 / 37.0;
    const double thinEnergy = cnx::spacingEnergy(thin, thinTopo, 1.0, {});
    std::printf("  fio 34 × 3: energia de espaçamento %.4f (a forma absoluta daria %.0f)\n", thinEnergy, 4.0 * 31.0 * 31.0);
    CHECK(std::abs(thinEnergy - 4.0 * s1 * s1) < 1e-12);
    CHECK(thinEnergy <= static_cast<double>(thinTopo.triples.size()));  // cada tripla custa no máximo o peso
    // O mesmo fio com o peso 5 (o da fase 0): 4 · 5 · s² ≈ 14, contra 4 · 5 · 961 = 19 220 antes.
    CHECK(std::abs(cnx::spacingEnergy(thin, thinTopo, 5.0, {}) - 5.0 * 4.0 * s1 * s1) < 1e-12);
    // Nós coincidentes (o fiscal só os recusa na entrada do suavizador, mas a energia é chamada com a posição de teste
    // da busca em linha): la = lb = 0 é 0/0 e não pode contaminar a energia nem o gradiente com NaN.
    VectorImage collapsed = thin;
    for (NodeId n = 0; n < static_cast<NodeId>(collapsed.nodeCount()); ++n) collapsed.setPosition(n, {7.0, 7.0});
    std::vector<Vec2> g(collapsed.nodeCount());
    const double e0 = cnx::spacingEnergy(collapsed, thinTopo, 1.0, g);
    CHECK(std::isfinite(e0) && e0 == 0.0);
    for (const Vec2& v : g) CHECK(std::isfinite(v.x) && std::isfinite(v.y));
}

CNX_TEST(the_arc_length_box_filter_removes_stairs_but_keeps_corners) {
    using P = std::vector<Vec2>;
    // h ≤ 0 ou menos de 3 pontos: cópia.
    const P two{{0, 0}, {5, 5}};
    CHECK(cnx::smoothChain(two, false, 1.0) == two);
    const P any{{0, 0}, {3, 1}, {6, 0}, {9, 4}};
    CHECK(cnx::smoothChain(any, false, 0.0) == any && cnx::smoothChain(any, true, -1.0) == any);
    // Valor exato: cadeia aberta (0,0) → (10,0) → (10,10), h = 1. O nó 1 (canto, s = 10) vira a média sobre [9, 11]:
    // metade em (9,5; 0) e metade em (10; 0,5) → (9,75; 0,25). Os extremos não se mexem.
    const P ell{{0, 0}, {10, 0}, {10, 10}};
    const P sm = cnx::smoothChain(ell, false, 1.0);
    CHECK(sm[0] == ell[0] && sm[2] == ell[2]);
    CHECK(std::abs(sm[1].x - 9.75) < 1e-12 && std::abs(sm[1].y - 0.25) < 1e-12);
    // Reta de nós desiguais: a média sobre uma janela simétrica de uma reta é o próprio ponto (o nó volta a si mesmo,
    // qualquer h e qualquer espaçamento), inclusive perto das pontas, onde a janela encolhe.
    const P line{{0, 0}, {1, 0}, {4, 0}, {4.5, 0}, {10, 0}, {13, 0}};
    for (const double h : {0.3, 1.0, 2.5}) {
        const P out = cnx::smoothChain(line, false, h);
        for (std::size_t i = 0; i < line.size(); ++i) {
            CHECK_CTX(std::abs(out[i].y) < 1e-12 && std::abs(out[i].x - line[i].x) < 1e-12, "reta h = " + std::to_string(h) + ", nó " + std::to_string(i));
        }
    }
    // Escadinha de 45°: (0,0) (1,0) (1,1) (2,1) (2,2) …: os pontos crus ficam a até 0,71 da reta y = x; filtrados (h = 1),
    // os interiores ficam a menos de 0,2.
    P stair;
    for (int i = 0; i < 12; ++i) {
        stair.push_back({static_cast<double>(i), static_cast<double>(i)});
        stair.push_back({static_cast<double>(i + 1), static_cast<double>(i)});
    }
    const P filtered = cnx::smoothChain(stair, false, 1.0);
    double rawWorst = 0.0;
    double filteredWorst = 0.0;
    for (std::size_t i = 4; i + 4 < stair.size(); ++i) {
        rawWorst = std::max(rawWorst, std::abs(stair[i].x - stair[i].y) / std::sqrt(2.0));
        filteredWorst = std::max(filteredWorst, std::abs(filtered[i].x - filtered[i].y - 0.5) / std::sqrt(2.0));  // a escadinha oscila em torno de y = x − 0,5
    }
    std::printf("  escadinha 45°: desvio da reta %.3f px cru, %.3f px filtrado\n", rawWorst, filteredWorst);
    CHECK(rawWorst > 0.3 && filteredWorst < 0.2);
    // Cadeia fechada: periódica. Girar o nó inicial gira o resultado, e o quadrado fica simétrico.
    const P square{{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    const P ring = cnx::smoothChain(square, true, 1.0);
    for (std::size_t k = 0; k < 4; ++k) {
        CHECK(std::abs(sla::norm(ring[k] - Vec2{5, 5}) - sla::norm(ring[0] - Vec2{5, 5})) < 1e-12);
        const P rotated{square[(k + 1) % 4], square[(k + 2) % 4], square[(k + 3) % 4], square[k]};
        const P rr = cnx::smoothChain(rotated, true, 1.0);
        CHECK(std::abs(rr[3].x - ring[k].x) < 1e-12 && std::abs(rr[3].y - ring[k].y) < 1e-12);
    }
    // Cada canto do quadrado vai para dentro: (0,0) → média sobre [−1, 1] do arco = metade em (0, 0,5) e metade em (0,5; 0).
    CHECK(std::abs(ring[0].x - 0.25) < 1e-12 && std::abs(ring[0].y - 0.25) < 1e-12);
    // Janela que atravessa o fechamento nos dois sentidos: o último segmento (0,5 px) é menor que a meia janela, então
    // o nó antes dele enxerga o início da cadeia à frente, e o primeiro enxerga o fim para trás. Qualquer rotação do nó
    // inicial tem de dar o mesmo contorno.
    const P shortClose{{0, 0}, {10, 0}, {10, 10}, {0, 10}, {0, 0.5}};
    const P base = cnx::smoothChain(shortClose, true, 1.0);
    for (std::size_t r = 1; r < shortClose.size(); ++r) {
        P rotated;
        for (std::size_t i = 0; i < shortClose.size(); ++i) rotated.push_back(shortClose[(i + r) % shortClose.size()]);
        const P rr = cnx::smoothChain(rotated, true, 1.0);
        for (std::size_t i = 0; i < shortClose.size(); ++i) {
            const Vec2 want = base[(i + r) % shortClose.size()];
            CHECK_CTX(std::abs(rr[i].x - want.x) < 1e-12 && std::abs(rr[i].y - want.y) < 1e-12, "rotação " + std::to_string(r) + ", nó " + std::to_string(i));
        }
    }
    // Cadeia de comprimento nulo ou janela maior que a cadeia: não explode.
    const P degenerate{{1, 1}, {1, 1}, {1, 1}};
    CHECK(cnx::smoothChain(degenerate, true, 1.0) == degenerate);
    CHECK(cnx::smoothChain(P{{0, 0}, {0.5, 0}, {1, 0}}, false, 5.0).size() == 3);
}

CNX_TEST(air_pressure_vanishes_at_the_original_shape_and_follows_the_area_change) {
    cnxtest::VectorScene s = islandScene();
    const SmoothTopology topo = SmoothTopology::build(s.image, 8);
    CHECK(topo.airLoops.size() == 3);  // moldura (fundo), furo do fundo e a ilha; o exterior não conta
    CHECK(cnx::airPressureEnergy(s.image, topo, 5.0, {}) == 0.0);
    // Move um canto da ilha em (1, 0): as áreas da ilha e do furo mudam em ±ΔA.
    const NodeId corner = topo.triples[0].mid;
    s.image.setPosition(corner, s.image.position(corner) + Vec2{1.0, 0.0});
    double expected = 0.0;
    for (const SmoothTopology::AirLoop& loop : topo.airLoops) {
        const double d = shoelaceOf(s.image, loop.nodes) - loop.originalArea;
        expected += 5.0 * d * d;
    }
    CHECK(expected > 0.0);
    CHECK(std::abs(cnx::airPressureEnergy(s.image, topo, 5.0, {}) - expected) < 1e-12);
    // Contornos com 8 ou mais nós não entram.
    CHECK(SmoothTopology::build(s.image, 4).airLoops.empty());  // todos têm exatamente 4 nós
}

CNX_TEST(border_energy_pulls_frame_nodes_back_but_lets_them_slide) {
    cnxtest::VectorScene s = splitScene();
    const SmoothTopology topo = SmoothTopology::build(s.image, 8);
    CHECK(topo.borderAnchors.size() == 2);
    CHECK(cnx::borderEnergy(s.image, topo, 10.0, {}) == 0.0);
    for (const SmoothTopology::BorderAnchor& b : topo.borderAnchors) {
        CHECK(b.axis == 1);  // nós no topo e na base: ancorados em y
        s.image.setPosition(b.node, s.image.position(b.node) + Vec2{0.7, 0.3});  // deslizar em x é grátis
    }
    std::vector<Vec2> grad(s.image.nodeCount());
    CHECK(std::abs(cnx::borderEnergy(s.image, topo, 10.0, grad) - 10.0 * 2 * 0.09) < 1e-12);
    for (const SmoothTopology::BorderAnchor& b : topo.borderAnchors) {
        CHECK(grad[static_cast<std::size_t>(b.node)].x == 0.0);
        CHECK(std::abs(grad[static_cast<std::size_t>(b.node)].y - 10.0 * 2.0 * 0.3) < 1e-12);
    }
}

// ================================================================ gradientes = diferenças finitas

CNX_TEST(each_term_gradient_matches_finite_differences) {
    SplitMix64 rng{302};
    for (int scene = 0; scene < 4; ++scene) {
        // Regiões de 1 pixel fundidas ao acaso: contornos irregulares, com muitos nós interiores.
        cnxtest::VectorScene s = cnxtest::vectorScene(15 + scene, 11, 1, 90, rng, 3);
        cnxtest::jitterNodes(s.image, 0.35, false, rng);
        const SmoothTopology topo = SmoothTopology::build(s.image, 8);
        // Alguns cantos perfurados, para o gradiente respeitar a flag.
        for (std::size_t i = 0; i < topo.triples.size(); i += 3) {
            const NodeId n = topo.triples[i].mid;
            s.image.setFlags(n, static_cast<std::uint8_t>(s.image.flags(n) | cnx::NodeFlag::Corner));
        }
        const std::string tag = "cena " + std::to_string(scene);
        CHECK_CTX(!topo.airLoops.empty(), tag + ": sem contornos pequenos");
        CHECK_CTX(!topo.borderAnchors.empty(), tag + ": sem nós de moldura");
        CHECK_CTX(topo.triples.size() > 10, tag + ": poucas triplas (" + std::to_string(topo.triples.size()) + ")");

        const auto check = [&](const char* name, const EnergyFn& fn) {
            const double worst = worstGradientError(s.image, fn);
            CHECK_CTX(worst < 1e-6, tag + " " + name + ": " + std::to_string(worst));
        };
        check("mola", [&](const VectorImage& im, std::span<Vec2> g) { return cnx::springEnergy(im, 1.7, 0.0, g); });
        check("mola com zona livre", [&](const VectorImage& im, std::span<Vec2> g) { return cnx::springEnergy(im, 1.7, 0.25, g); });
        check("angular", [&](const VectorImage& im, std::span<Vec2> g) { return cnx::angularEnergy(im, topo, 10.0, g); });
        check("espaçamento", [&](const VectorImage& im, std::span<Vec2> g) { return cnx::spacingEnergy(im, topo, 0.5, g); });
        check("pressão do ar", [&](const VectorImage& im, std::span<Vec2> g) { return cnx::airPressureEnergy(im, topo, 5.0, g); });
        check("âncora de margem", [&](const VectorImage& im, std::span<Vec2> g) { return cnx::borderEnergy(im, topo, 10.0, g); });
    }
}

CNX_TEST(total_energy_is_the_sum_of_the_terms_and_its_gradient_matches_finite_differences) {
    SplitMix64 rng{303};
    for (int scene = 0; scene < 3; ++scene) {
        cnxtest::VectorScene s = cnxtest::vectorScene(15, 11, 3, 12, rng);
        cnxtest::jitterNodes(s.image, 0.35, false, rng);
        const SmoothTopology topo = SmoothTopology::build(s.image, 8);
        const cnx::GenerativeModel model(cnx::GenerativeModel::targetFrom(s.source));
        const std::string tag = "cena " + std::to_string(scene);
        for (const cnx::MeasurementType m : {cnx::MeasurementType::Spring, cnx::MeasurementType::Generative, cnx::MeasurementType::None}) {
            EnergyWeights w;
            w.measurement = m;
            w.fidelity = 1.3;
            w.freeRadius = 0.1;
            w.spacing = 0.7;
            EnergyBreakdown parts;
            std::vector<Vec2> grad(s.image.nodeCount());
            const double e = cnx::evaluateEnergy(s.image, topo, &model, w, grad, &parts);
            CHECK_CTX(std::abs(e - parts.total()) < 1e-12 * (1.0 + e), tag);
            CHECK_CTX((m == cnx::MeasurementType::Spring) == (parts.spring > 0.0), tag);
            CHECK_CTX((m == cnx::MeasurementType::Generative) == (parts.generative > 0.0), tag);
            CHECK_CTX(std::abs(cnx::evaluateEnergy(s.image, topo, &model, w, {}, nullptr) - e) < 1e-9 * (1.0 + e), tag);
            const double worst = worstGradientError(s.image, [&](const VectorImage& im, std::span<Vec2> g) {
                return cnx::evaluateEnergy(im, topo, &model, w, g, nullptr);
            }, 1e-5);
            CHECK_CTX(worst < 1e-5, tag + " medição " + std::to_string(static_cast<int>(m)) + ": " + std::to_string(worst));
        }
    }
}

CNX_TEST(evaluate_energy_and_phase_weights_validate_arguments) {
    cnxtest::VectorScene s = islandScene();
    const SmoothTopology topo = SmoothTopology::build(s.image, 8);
    EnergyWeights w;
    w.measurement = cnx::MeasurementType::Generative;
    CHECK_THROWS(std::invalid_argument, (void)cnx::evaluateEnergy(s.image, topo, nullptr, w, {}, nullptr));
    std::vector<Vec2> shortGrad(s.image.nodeCount() - 1);
    w.measurement = cnx::MeasurementType::Spring;
    CHECK_THROWS(std::invalid_argument, (void)cnx::evaluateEnergy(s.image, topo, nullptr, w, shortGrad, nullptr));

    cnx::SmoothParams p;
    CHECK(cnx::phaseWeights(p, 0).spacing == 5.0 && cnx::phaseWeights(p, 1).spacing == 0.5 && cnx::phaseWeights(p, 2).spacing == 0.5);
    CHECK(cnx::phaseWeights(p, 1).angular == p.priorStrength && cnx::phaseWeights(p, 1).air == p.airPressureWeight);
    CHECK_THROWS(std::invalid_argument, (void)cnx::phaseWeights(p, 3));
    CHECK_THROWS(std::invalid_argument, (void)cnx::phaseWeights(p, -1));
    p.priorType = 2;
    CHECK_THROWS(std::invalid_argument, (void)cnx::phaseWeights(p, 0));
}

// ================================================================ canto perfurado

CNX_TEST(corner_detector_on_polylines) {
    using P = std::vector<Vec2>;
    const double arm = 4.0;
    // L aberto: (0,0) → (10,0) → (10,10). O nó 1 é um canto reto; os extremos não têm braço de um lado.
    const P ell{{0, 0}, {10, 0}, {10, 10}};
    const cnx::PunctureMeasure m = cnx::measureCorner(ell, 1, false, arm);
    CHECK(std::abs(std::abs(m.turn) - kPi / 2) < 1e-12);
    CHECK(m.deviationBack == 0.0 && m.deviationFwd == 0.0 && m.armBack == arm && m.armFwd == arm);
    CHECK(cnx::isProbablyPuncturedNode(ell, 1, false, arm, 0.8, 0.3));
    CHECK(!cnx::isProbablyPuncturedNode(ell, 1, false, arm, 0.0, 0.3));  // limiar ≤ 0 desativa
    CHECK(!cnx::isProbablyPuncturedNode(ell, 1, false, arm, 2.0, 0.3));  // giro menor que o limiar

    // Reta: giro nulo.
    const P line{{0, 0}, {5, 0}, {10, 0}, {15, 0}};
    CHECK(std::abs(cnx::measureCorner(line, 2, false, arm).turn) < 1e-12);
    CHECK(!cnx::isProbablyPuncturedNode(line, 2, false, arm, 0.8, 0.3));
    // Extremo de cadeia aberta: braço de um lado nulo → nunca é canto.
    CHECK(cnx::measureCorner(line, 0, false, arm).armBack == 0.0);
    CHECK(!cnx::isProbablyPuncturedNode(line, 0, false, arm, 0.8, 0.3));

    // Arco suave de raio 20: giro em 4 px de braço ≈ 0,2 rad < 0,8.
    P arc;
    for (int i = -10; i <= 10; ++i) arc.push_back({20.0 * std::sin(i * 0.05), 20.0 * (1.0 - std::cos(i * 0.05))});
    CHECK(!cnx::isProbablyPuncturedNode(arc, 10, false, arm, 0.8, 0.3));

    // Quina com braço torto: o nó 2 do zigue-zague tem desvio > 0,3 px num dos lados.
    const P zig{{0, 0}, {2, 1.0}, {4, 0}, {10, 0}, {10, 10}};
    // Com braço de 4 px o nó 3 vê só trechos retos ((10,0)→(6,0) e (10,0)→(10,4)): é canto.
    CHECK(cnx::measureCorner(zig, 3, false, arm).deviationBack == 0.0);
    CHECK(cnx::isProbablyPuncturedNode(zig, 3, false, arm, 0.8, 0.3));
    // Com braço de 9 px o lado anterior passa pelo zigue-zague (desvio ≈ 0,8 px > 0,3): deixa de ser canto.
    CHECK(cnx::measureCorner(zig, 3, false, 9.0).deviationBack > 0.3);
    CHECK(!cnx::isProbablyPuncturedNode(zig, 3, false, 9.0, 0.8, 0.3));

    // O mesmo zigue-zague visto do outro lado: braço anterior reto, braço da frente torto → não é canto.
    const P zigFwd{{10, 10}, {10, 0}, {4, 0}, {2, 1.0}, {0, 0}};
    CHECK(cnx::isProbablyPuncturedNode(zigFwd, 1, false, arm, 0.8, 0.3));  // braços de 4 px: retos
    CHECK(cnx::measureCorner(zigFwd, 1, false, 9.0).deviationBack == 0.0);
    CHECK(cnx::measureCorner(zigFwd, 1, false, 9.0).deviationFwd > 0.3);
    CHECK(!cnx::isProbablyPuncturedNode(zigFwd, 1, false, 9.0, 0.8, 0.3));

    // Quadrado fechado de lado 10: os 4 cantos são detectados; com braço 12 > lado, nenhum.
    const P sq{{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    for (std::size_t k = 0; k < 4; ++k) CHECK(cnx::isProbablyPuncturedNode(sq, k, true, arm, 0.8, 0.3));
    for (std::size_t k = 0; k < 4; ++k) CHECK(!cnx::isProbablyPuncturedNode(sq, k, true, 12.0, 0.8, 0.3));

    CHECK_THROWS(std::out_of_range, (void)cnx::measureCorner(sq, 4, true, arm));
}

int main() { return cnxtest::runAll(); }
