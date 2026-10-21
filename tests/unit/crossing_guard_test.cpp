// tests/unit/crossing_guard_test.cpp — fiscal anti-cruzamento: casos construídos e grade contra força bruta.
#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "cnx/vector/CrossingGuard.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::Crossing;
using cnx::CrossingGuard;
using cnx::NodeId;
using cnx::VectorImage;
using cnxtest::SplitMix64;

namespace {

/// VectorImage com arestas soltas (cada uma uma poligonal aberta); só a geometria importa para o fiscal.
VectorImage polylines(const std::vector<std::vector<sla::Vec2>>& chains) {
    VectorImage v(100.0, 100.0);
    const cnx::FaceId a = v.addFace(sla::Rgba{1, 0, 0, 1}, 0);
    const cnx::FaceId b = v.addFace(sla::Rgba{0, 0, 1, 1}, 1);
    for (const auto& chain : chains) {
        std::vector<NodeId> ids;
        for (const sla::Vec2 p : chain) ids.push_back(v.addNode(p));
        (void)v.addEdge(ids, a, b);
    }
    return v;
}

}  // namespace

CNX_TEST(detects_a_proper_crossing) {
    const VectorImage v = polylines({{{0, 0}, {10, 10}}, {{0, 10}, {10, 0}}});
    const std::vector<Crossing> c = CrossingGuard::find(v);
    CHECK(c.size() == 1);
    CHECK(c == CrossingGuard::findBruteForce(v));
    const std::vector<NodeId> nodes = CrossingGuard::offendingNodes(c);
    CHECK(nodes == (std::vector<NodeId>{0, 1, 2, 3}));
}

CNX_TEST(touching_parallel_and_collinear_segments_are_not_crossings) {
    // Compartilhar um nó (cadeia contínua), tocar com a ponta, paralelas e colineares sobrepostas.
    const VectorImage chain = polylines({{{0, 0}, {5, 5}, {10, 0}}});
    CHECK(CrossingGuard::find(chain).empty());
    const VectorImage tee = polylines({{{0, 0}, {10, 0}}, {{5, 0}, {5, 10}}});  // nó sobre outro segmento
    CHECK(CrossingGuard::find(tee).empty());
    const VectorImage parallel = polylines({{{0, 0}, {10, 0}}, {{0, 1}, {10, 1}}});
    CHECK(CrossingGuard::find(parallel).empty());
    const VectorImage collinear = polylines({{{0, 0}, {10, 0}}, {{5, 0}, {15, 0}}});
    CHECK(CrossingGuard::find(collinear).empty());
    const VectorImage apart = polylines({{{0, 0}, {4, 4}}, {{6, 6}, {10, 0}}});
    CHECK(CrossingGuard::find(apart).empty());
    // Dois segmentos da mesma cadeia que se cruzam (laço em forma de 8).
    const VectorImage bow = polylines({{{0, 0}, {10, 10}, {10, 0}, {0, 10}}});
    CHECK(CrossingGuard::find(bow).size() == 1);
}

CNX_TEST(staircase_contours_do_not_cross) {
    SplitMix64 rng{401};
    for (int scene = 0; scene < 5; ++scene) {
        const cnxtest::VectorScene s = cnxtest::vectorScene(20 + scene, 15, 3, 25, rng);
        CHECK_CTX(CrossingGuard::find(s.image).empty(), "cena " + std::to_string(scene));
        CHECK_CTX(CrossingGuard::findBruteForce(s.image).empty(), "cena " + std::to_string(scene));
    }
}

CNX_TEST(grid_search_equals_brute_force_on_perturbed_scenes) {
    SplitMix64 rng{402};
    std::size_t withCrossings = 0;
    std::size_t total = 0;
    for (int scene = 0; scene < 12; ++scene) {
        cnxtest::VectorScene s = cnxtest::vectorScene(18 + scene, 14, 3, 20, rng);
        // De pouco a muito deslocamento: dos poucos aos muitos cruzamentos.
        cnxtest::jitterNodes(s.image, 0.2 + 0.25 * scene, false, rng);
        const std::vector<Crossing> grid = CrossingGuard::find(s.image);
        const std::vector<Crossing> brute = CrossingGuard::findBruteForce(s.image);
        CHECK_CTX(grid == brute, "cena " + std::to_string(scene) + ": " + std::to_string(grid.size()) + " vs " + std::to_string(brute.size()));
        withCrossings += grid.empty() ? 0 : 1;
        total += grid.size();
    }
    CHECK(withCrossings >= 6);  // o teste só vale se há cruzamentos de verdade
    CHECK(total > 20);
}

CNX_TEST(offending_nodes_are_sorted_and_unique) {
    const std::vector<Crossing> c{{5, 6, 1, 2}, {5, 6, 2, 3}, {1, 2, 9, 8}};
    const std::vector<NodeId> nodes = CrossingGuard::offendingNodes(c);
    CHECK(nodes == (std::vector<NodeId>{1, 2, 3, 5, 6, 8, 9}));
    CHECK(CrossingGuard::offendingNodes({}).empty());
}

CNX_TEST(short_segments_are_reported_with_both_endpoints) {
    // Segmentos de 5, 0,02 e 0,5 px; só o do meio é mais curto que 0,05.
    const VectorImage v = polylines({{{0, 0}, {5, 0}, {5.0, 0.02}, {5.0, 0.52}}});
    CHECK(CrossingGuard::shortSegmentNodes(v, 0.05) == (std::vector<NodeId>{1, 2}));
    CHECK(CrossingGuard::shortSegmentNodes(v, 0.01).empty());
    CHECK(CrossingGuard::shortSegmentNodes(v, 1.0) == (std::vector<NodeId>{1, 2, 3}));
    // Nós coincidentes (comprimento 0).
    const VectorImage same = polylines({{{3, 3}, {3, 3}, {4, 4}}});
    CHECK(CrossingGuard::shortSegmentNodes(same, 1e-9) == (std::vector<NodeId>{0, 1}));
    // A staircase tem segmentos de 1 px: nada abaixo de 0,05.
    SplitMix64 rng{403};
    const cnxtest::VectorScene s = cnxtest::vectorScene(18, 12, 3, 20, rng);
    CHECK(CrossingGuard::shortSegmentNodes(s.image, 0.05).empty());
}

CNX_TEST(handles_degenerate_inputs) {
    VectorImage empty(10.0, 10.0);
    CHECK(CrossingGuard::find(empty).empty());
    const VectorImage single = polylines({{{0, 0}, {1, 1}}});
    CHECK(CrossingGuard::find(single).empty());
    // Segmentos de comprimento nulo e nós coincidentes não geram falsos positivos nem divisão por zero.
    const VectorImage zero = polylines({{{3, 3}, {3, 3}}, {{0, 3}, {6, 3}}});
    CHECK(CrossingGuard::find(zero) == CrossingGuard::findBruteForce(zero));
}

int main() { return cnxtest::runAll(); }
