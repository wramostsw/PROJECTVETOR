// tests/unit/vector_image_test.cpp — VectorImage com mapas montados à mão: ligação dos laços por
// ordem angular, áreas (nós e cúbicas), pais, variáveis do otimizador e detecção de violações de I1–I4.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnx/vector/VectorImage.h"
#include "sla/BezierCurve.h"
#include "test_harness.h"

namespace {

using cnx::EdgeId;
using cnx::FaceId;
using cnx::HalfEdgeId;
using cnx::NodeId;
using cnx::VectorImage;

constexpr FaceId kExt = cnx::kExteriorFace;
constexpr auto kFramePin =
    static_cast<std::uint8_t>(cnx::NodeFlag::Pinned | cnx::NodeFlag::Corner | cnx::NodeFlag::ImageBorder);

EdgeId addEdge(VectorImage& vi, const std::vector<NodeId>& nodes, FaceId left, FaceId right) {
    return vi.addEdge(nodes, left, right);
}

std::size_t loopLength(const VectorImage& vi, HalfEdgeId first) {
    std::size_t n = 0;
    HalfEdgeId h = first;
    do {
        ++n;
        h = vi.next(h);
    } while (h != first);
    return n;
}

bool contains(const std::string& text, const char* needle) { return text.find(needle) != std::string::npos; }

/// Quadrado [0,2]² dividido em 4 triângulos pelas diagonais até o centro (direções não axiais).
struct Fan {
    VectorImage vi;
    FaceId bottom = 0, right = 0, top = 0, left = 0;
    NodeId center = 0;
};

Fan makeFan() {
    Fan f;
    f.vi = VectorImage(2.0, 2.0);
    VectorImage& vi = f.vi;
    const NodeId c0 = vi.addNode({0, 0}, kFramePin);
    const NodeId c1 = vi.addNode({2, 0}, kFramePin);
    const NodeId c2 = vi.addNode({2, 2}, kFramePin);
    const NodeId c3 = vi.addNode({0, 2}, kFramePin);
    f.center = vi.addNode({1, 1}, cnx::NodeFlag::Junction);
    f.bottom = vi.addFace({1, 0, 0}, 0);
    f.right = vi.addFace({0, 1, 0}, 1);
    f.top = vi.addFace({0, 0, 1}, 2);
    f.left = vi.addFace({1, 1, 0}, 3);
    // Moldura: a face interna fica do lado de área positiva (n = (Δy, −Δx) aponta para fora dela).
    addEdge(vi, {c0, c1}, f.bottom, kExt);
    addEdge(vi, {c1, c2}, f.right, kExt);
    addEdge(vi, {c2, c3}, f.top, kExt);
    addEdge(vi, {c3, c0}, f.left, kExt);
    // Diagonais do centro para os cantos.
    addEdge(vi, {f.center, c0}, f.bottom, f.left);
    addEdge(vi, {f.center, c1}, f.right, f.bottom);
    addEdge(vi, {f.center, c2}, f.top, f.right);
    addEdge(vi, {f.center, c3}, f.left, f.top);
    vi.linkFaceLoops();
    return f;
}

/// Moldura W×H (face `outer`, aresta fechada) com uma ilha quadrada [1,2]² (face `island`).
struct Island {
    VectorImage vi;
    FaceId outer = 0, island = 0;
    EdgeId frameEdge = 0, islandEdge = 0;
};

Island makeIsland(double frameW = 4.0, double imageW = 4.0) {
    Island s;
    s.vi = VectorImage(imageW, 4.0);
    VectorImage& vi = s.vi;
    const NodeId f0 = vi.addNode({0, 0}, kFramePin);
    const NodeId f1 = vi.addNode({frameW, 0}, kFramePin);
    const NodeId f2 = vi.addNode({frameW, 4}, kFramePin);
    const NodeId f3 = vi.addNode({0, 4}, kFramePin);
    const NodeId i0 = vi.addNode({1, 1}, cnx::NodeFlag::LoopAnchor);
    const NodeId i1 = vi.addNode({2, 1});
    const NodeId i2 = vi.addNode({2, 2});
    const NodeId i3 = vi.addNode({1, 2});
    s.outer = vi.addFace({1, 1, 1}, 0);
    s.island = vi.addFace({0, 0, 0}, 1);
    s.frameEdge = addEdge(vi, {f0, f1, f2, f3, f0}, s.outer, kExt);
    s.islandEdge = addEdge(vi, {i0, i1, i2, i3, i0}, s.island, s.outer);
    vi.linkFaceLoops();
    return s;
}

CNX_TEST(fan_links_loops_by_angular_order) {
    const Fan f = makeFan();
    const auto report = f.vi.validate();
    CHECK_CTX(report.ok, report.message);
    for (const FaceId t : {f.bottom, f.right, f.top, f.left}) {
        CHECK(std::abs(f.vi.face(t).area - 1.0) < 1e-12);
        CHECK(loopLength(f.vi, f.vi.face(t).outerLoop) == 3);
        CHECK(f.vi.face(t).holeCount == 0);
        CHECK(f.vi.face(t).parent == kExt);
    }
    CHECK(f.vi.face(kExt).outerLoop == cnx::kInvalidId);
    CHECK(f.vi.face(kExt).holeCount == 1);
    CHECK(std::abs(f.vi.face(kExt).area + 4.0) < 1e-12);
}

CNX_TEST(island_is_hole_of_outer_face_and_child_in_draw_order) {
    const Island s = makeIsland();
    const auto report = s.vi.validate();
    CHECK_CTX(report.ok, report.message);
    CHECK(s.vi.face(s.outer).holeCount == 1);
    CHECK(s.vi.face(s.island).holeCount == 0);
    CHECK(s.vi.face(s.island).parent == s.outer);
    CHECK(s.vi.face(s.outer).parent == kExt);
    CHECK(std::abs(s.vi.face(s.outer).area - 15.0) < 1e-12);
    CHECK(std::abs(s.vi.face(s.island).area - 1.0) < 1e-12);
    // Laço fechado: a meia-aresta é o próprio next.
    const HalfEdgeId h = cnx::halfEdge(s.islandEdge, false);
    CHECK(s.vi.next(h) == h);
    CHECK(s.vi.next(cnx::twin(h)) == cnx::twin(h));
}

CNX_TEST(add_edge_rejects_invalid_input) {
    VectorImage vi(2.0, 2.0);
    const NodeId a = vi.addNode({0, 0});
    const NodeId b = vi.addNode({1, 0});
    const NodeId c = vi.addNode({1, 1});
    const FaceId f = vi.addFace({}, 0);
    CHECK_THROWS(std::invalid_argument, addEdge(vi, {a}, f, kExt));
    CHECK_THROWS(std::invalid_argument, addEdge(vi, {a, 99}, f, kExt));
    CHECK_THROWS(std::invalid_argument, addEdge(vi, {a, b}, f, f));
    CHECK_THROWS(std::invalid_argument, addEdge(vi, {a, b}, f, 7));
    CHECK_THROWS(std::invalid_argument, addEdge(vi, {a, b, b, c}, f, kExt));
    CHECK_THROWS(std::invalid_argument, addEdge(vi, {a, b, a}, f, kExt));
    CHECK(vi.edgeCount() == 0);
    // Aresta construída a partir dos nós de outra (span apontando para o próprio armazenamento).
    const EdgeId e0 = addEdge(vi, {a, b, c}, f, kExt);
    const EdgeId e1 = vi.addEdge(vi.edgeNodes(e0), kExt, f);
    CHECK(vi.edgeNodes(e1).size() == 3 && vi.edgeNodes(e1)[2] == c);
}

CNX_TEST(disconnected_face_is_rejected) {
    VectorImage vi(4.0, 4.0);
    const NodeId f0 = vi.addNode({0, 0}, kFramePin);
    const NodeId f1 = vi.addNode({4, 0}, kFramePin);
    const NodeId f2 = vi.addNode({4, 4}, kFramePin);
    const NodeId f3 = vi.addNode({0, 4}, kFramePin);
    const FaceId outer = vi.addFace({}, 0);
    const FaceId split = vi.addFace({}, 1);
    addEdge(vi, {f0, f1, f2, f3, f0}, outer, kExt);
    for (const double x : {0.5, 2.5}) {
        const NodeId a = vi.addNode({x, 1});
        const NodeId b = vi.addNode({x + 1, 1});
        const NodeId c = vi.addNode({x + 1, 2});
        const NodeId d = vi.addNode({x, 2});
        addEdge(vi, {a, b, c, d, a}, split, outer);  // duas ilhas na mesma face
    }
    CHECK_THROWS(std::logic_error, vi.linkFaceLoops());
}

CNX_TEST(validate_detects_orphan_node_I1) {
    Island s = makeIsland();
    s.vi.addNode({3, 3});
    const auto report = s.vi.validate();
    CHECK(!report.ok);
    CHECK_CTX(contains(report.message, "I1"), report.message);
}

CNX_TEST(validate_detects_junction_flag_on_interior_node_I1) {
    Island s = makeIsland();
    const NodeId interior = s.vi.edgeNodes(s.islandEdge)[1];
    s.vi.setFlags(interior, cnx::NodeFlag::Junction);
    const auto report = s.vi.validate();
    CHECK(!report.ok);
    CHECK_CTX(contains(report.message, "I1"), report.message);
}

CNX_TEST(validate_detects_uncovered_image_I4) {
    const Island s = makeIsland(4.0, 5.0);  // moldura 4×4 numa imagem 5×4
    const auto report = s.vi.validate();
    CHECK(!report.ok);
    CHECK_CTX(contains(report.message, "I4"), report.message);
}

CNX_TEST(validate_detects_unlinked_edges) {
    Fan f = makeFan();
    const NodeId a = f.vi.addNode({0.5, 0.2});
    const NodeId b = f.vi.addNode({1.5, 0.2});
    addEdge(f.vi, {a, b}, f.bottom, kExt);
    const auto report = f.vi.validate();
    CHECK(!report.ok);
    CHECK_CTX(contains(report.message, "linkFaceLoops"), report.message);
}

CNX_TEST(curves_change_area_exactly_and_keep_coverage_I4) {
    Island s = makeIsland();
    VectorImage& vi = s.vi;
    const auto nodes = vi.edgeNodes(s.islandEdge);
    std::vector<sla::CurveSegment> chain;
    for (std::size_t k = 0; k + 1 < nodes.size(); ++k) {
        chain.push_back({sla::SegmentKind::Line,
                         sla::BezierCurve::fromLine(vi.position(nodes[k]), vi.position(nodes[k + 1]))});
    }
    vi.setEdgeCurves(s.islandEdge, chain);
    const HalfEdgeId islandLoop = vi.face(s.island).outerLoop;
    CHECK(std::abs(vi.loopArea(islandLoop) - 1.0) < 1e-12);  // retas como cúbicas: mesma área

    // Arqueia o lado de baixo da ilha para fora (y < 1).
    chain[0] = {sla::SegmentKind::Cubic, sla::BezierCurve{{1, 1}, {1.3, 0.4}, {1.8, 0.6}, {2, 1}}};
    vi.setEdgeCurves(s.islandEdge, chain);
    double bulge = 0.0;  // área entre a curva e a corda, por polilinha fina
    constexpr int kSteps = 200000;
    sla::Vec2 prev = chain[0].curve.eval(0.0);
    for (int i = 1; i <= kSteps; ++i) {
        const sla::Vec2 p = chain[0].curve.eval(static_cast<double>(i) / kSteps);
        bulge += 0.5 * sla::cross(prev, p);
        prev = p;
    }
    bulge += 0.5 * sla::cross(chain[0].curve.p[3], chain[0].curve.p[0]);  // volta pela corda
    CHECK(bulge > 0.0);
    CHECK_CTX(std::abs(vi.loopArea(islandLoop) - (1.0 + bulge)) < 1e-9,
              std::to_string(vi.loopArea(islandLoop)) + " vs " + std::to_string(1.0 + bulge));

    // A fronteira é compartilhada: o furo da face externa muda pelo mesmo valor e I4 continua valendo.
    vi.updateFaceAreas();
    CHECK(std::abs(vi.face(s.outer).area - (15.0 - bulge)) < 1e-9);
    const auto report = vi.validate();
    CHECK_CTX(report.ok, report.message);

    std::vector<sla::CurveSegment> broken = chain;
    broken[1].curve.p[0] = {2.5, 1};
    CHECK_THROWS(std::invalid_argument, vi.setEdgeCurves(s.islandEdge, broken));
    broken = chain;
    broken[0].curve.p[0] = {0, 0};
    CHECK_THROWS(std::invalid_argument, vi.setEdgeCurves(s.islandEdge, broken));
}

CNX_TEST(free_nodes_quarantine_gather_scatter) {
    Fan f = makeFan();
    VectorImage& vi = f.vi;
    std::vector<NodeId> free = vi.collectFreeNodes();
    CHECK(free.size() == 1 && free[0] == f.center);  // cantos da moldura são Pinned

    vi.quarantine(f.center, 2);
    CHECK(vi.collectFreeNodes().empty());
    vi.tickQuarantine();
    CHECK(vi.collectFreeNodes().empty());
    vi.tickQuarantine();
    CHECK(vi.collectFreeNodes().size() == 1);

    std::vector<double> x(2);
    vi.gather(free, x);
    CHECK(x[0] == 1.0 && x[1] == 1.0);
    x = {1.25, 0.75};
    vi.scatter(free, x);
    CHECK(vi.position(f.center) == (sla::Vec2{1.25, 0.75}));
    CHECK(vi.originalPosition(f.center) == (sla::Vec2{1.0, 1.0}));
    vi.updateFaceAreas();
    CHECK(std::abs(vi.face(f.bottom).area - 0.75) < 1e-12);
    CHECK(std::abs(vi.face(f.top).area - 1.25) < 1e-12);
    CHECK(std::abs(vi.face(f.right).area - 0.75) < 1e-12);
    CHECK(std::abs(vi.face(f.left).area - 1.25) < 1e-12);
    const auto report = vi.validate();
    CHECK_CTX(report.ok, report.message);

    std::vector<double> wrong(3);
    CHECK_THROWS(std::invalid_argument, vi.gather(free, wrong));
    CHECK_THROWS(std::invalid_argument, vi.scatter(free, wrong));
}

}  // namespace

int main() { return cnxtest::runAll(); }
