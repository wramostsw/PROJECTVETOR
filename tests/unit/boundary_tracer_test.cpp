// tests/unit/boundary_tracer_test.cpp — BoundaryTracer e invariantes I1–I4 em casos construídos
// (ilhas, anéis, xadrez, selas, laço preso a junção) e em varreduras aleatórias.
//
// Além de VectorImage::validate(), cada traçado passa por verificações independentes:
//   * área de cada face = número de pixels (exato: coordenadas inteiras);
//   * número de voltas de cada face em torno de cada centro de pixel = [pixel ∈ face] (exterior: −1);
//   * modo "bolo em camadas": pintando os laços externos por profundidade (pai antes do filho),
//     a última face que cobre o centro do pixel é a face do pixel.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "cnx/engine/TaskContext.h"
#include "cnx/segment/PixelMap.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/vector/VectorImage.h"
#include "test_harness.h"
#include "test_support.h"

namespace {

using cnx::EdgeId;
using cnx::FaceId;
using cnx::HalfEdgeId;
using cnx::NodeId;
using cnx::RegionId;
using cnx::TraceResult;
using cnx::VectorImage;

constexpr FaceId kExt = cnx::kExteriorFace;

// ------------------------------------------------------------------ geradores
sla::Array2D<RegionId> labelsFromRows(std::initializer_list<std::string_view> rows) {
    const int h = static_cast<int>(rows.size());
    const int w = static_cast<int>(rows.begin()->size());
    sla::Array2D<RegionId> labels(w, h, 0);
    int y = 0;
    for (const std::string_view row : rows) {
        if (static_cast<int>(row.size()) != w) throw std::invalid_argument("linhas de tamanhos diferentes");
        for (int x = 0; x < w; ++x) labels(x, y) = row[static_cast<std::size_t>(x)] - 'a';
        ++y;
    }
    return labels;
}

RegionId labelCount(const sla::Array2D<RegionId>& labels) {
    RegionId m = 0;
    for (int y = 0; y < labels.height(); ++y) {
        for (int x = 0; x < labels.width(); ++x) m = std::max(m, labels(x, y) + 1);
    }
    return m;
}

std::vector<sla::Rgba> colorsFor(RegionId n) {
    std::vector<sla::Rgba> colors;
    for (RegionId i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(std::max<RegionId>(1, n));
        colors.emplace_back(t, 1.0f - t, 0.5f * t, 1.0f);
    }
    return colors;
}

TraceResult traceLabels(const sla::Array2D<RegionId>& labels, cnx::BoundaryTracer::Options options = {}) {
    const auto colors = colorsFor(labelCount(labels));
    return cnx::BoundaryTracer(options).trace(labels, colors);
}

using cnxtest::SplitMix64;

/// Ruído puro: muitas faces de 1 pixel, selas e beliscões.
sla::Array2D<RegionId> noiseLabels(int w, int h, int k, SplitMix64& rng) {
    sla::Array2D<RegionId> labels(w, h, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) labels(x, y) = rng.below(k);
    }
    return labels;
}

/// Discos sobrepostos (aninhamento, furos, anéis) com 1/16 dos pixels trocados ao acaso.
sla::Array2D<RegionId> discLabels(int w, int h, int k, int discs, SplitMix64& rng) {
    sla::Array2D<RegionId> labels(w, h, rng.below(k));
    for (int i = 0; i < discs; ++i) {
        const int cx = rng.below(w);
        const int cy = rng.below(h);
        const int r = 1 + rng.below(std::max(1, std::min(w, h) / 2));
        const RegionId label = rng.below(k);
        for (int y = std::max(0, cy - r); y <= std::min(h - 1, cy + r); ++y) {
            for (int x = std::max(0, cx - r); x <= std::min(w - 1, cx + r); ++x) {
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) labels(x, y) = label;
            }
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (rng.below(16) == 0) labels(x, y) = rng.below(k);
        }
    }
    return labels;
}

// ------------------------------------------------------------------ verificações independentes
/// +1 / −1 se o segmento a→b cruza a semirreta horizontal à direita de p (Sunday, 2001).
int crossingSign(sla::Vec2 a, sla::Vec2 b, sla::Vec2 p) {
    const double side = sla::cross(b - a, p - a);
    if (a.y <= p.y) return (b.y > p.y && side > 0.0) ? 1 : 0;
    return (b.y <= p.y && side < 0.0) ? -1 : 0;
}

struct Segment {
    sla::Vec2 a, b;
};

std::vector<Segment> halfEdgeSegments(const VectorImage& vi, HalfEdgeId h) {
    std::vector<Segment> out;
    const auto nodes = vi.edgeNodes(cnx::edgeOf(h));
    const std::size_t n = nodes.size();
    for (std::size_t k = 0; k + 1 < n; ++k) {
        const NodeId a = cnx::isReversed(h) ? nodes[n - 1 - k] : nodes[k];
        const NodeId b = cnx::isReversed(h) ? nodes[n - 2 - k] : nodes[k + 1];
        out.push_back({vi.position(a), vi.position(b)});
    }
    return out;
}

void checkTrace(const TraceResult& r, const std::string& context) {
    const VectorImage& vi = r.image;
    const auto report = vi.validate();
    CHECK_CTX(report.ok, context + ": " + report.message);
    if (!report.ok) return;

    const int w = r.pixelFaces.width();
    const int h = r.pixelFaces.height();
    const std::size_t faceCount = vi.faceCount();

    // Área de cada face = pixels da face.
    std::vector<double> pixels(faceCount, 0.0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) pixels[static_cast<std::size_t>(r.pixelFaces(x, y))] += 1.0;
    }
    std::size_t areaMismatches = 0;
    for (std::size_t f = 1; f < faceCount; ++f) areaMismatches += vi.face(static_cast<FaceId>(f)).area != pixels[f] ? 1 : 0;
    CHECK_CTX(areaMismatches == 0, context + ": faces com área ≠ nº de pixels: " + std::to_string(areaMismatches));
    CHECK_CTX(vi.face(kExt).area == -static_cast<double>(w) * h, context);

    // Segmentos de todas as arestas (para o número de voltas por face) e dos laços externos.
    struct EdgeSeg {
        Segment s;
        FaceId left, right;
    };
    std::vector<EdgeSeg> all;
    for (std::size_t e = 0; e < vi.edgeCount(); ++e) {
        const cnx::Edge& ed = vi.edge(static_cast<EdgeId>(e));
        for (const Segment& s : halfEdgeSegments(vi, cnx::halfEdge(static_cast<EdgeId>(e), false))) {
            all.push_back({s, ed.left, ed.right});
        }
    }
    std::vector<std::vector<Segment>> outer(faceCount);
    std::vector<int> depth(faceCount, 0);
    for (std::size_t f = 1; f < faceCount; ++f) {
        const HalfEdgeId first = vi.face(static_cast<FaceId>(f)).outerLoop;
        HalfEdgeId hh = first;
        do {
            for (const Segment& s : halfEdgeSegments(vi, hh)) outer[f].push_back(s);
            hh = vi.next(hh);
        } while (hh != first);
        for (FaceId p = vi.face(static_cast<FaceId>(f)).parent; p != kExt; p = vi.face(p).parent) ++depth[f];
    }
    std::vector<FaceId> order(faceCount - 1);
    std::iota(order.begin(), order.end(), FaceId{1});
    std::stable_sort(order.begin(), order.end(),
                     [&](FaceId a, FaceId b) { return depth[static_cast<std::size_t>(a)] < depth[static_cast<std::size_t>(b)]; });

    std::size_t windingMismatches = 0;
    std::size_t painterMismatches = 0;
    std::vector<int> wn(faceCount);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const sla::Vec2 p{x + 0.5, y + 0.5};
            const FaceId own = r.pixelFaces(x, y);
            std::fill(wn.begin(), wn.end(), 0);
            for (const EdgeSeg& es : all) {
                const int s = crossingSign(es.s.a, es.s.b, p);
                wn[static_cast<std::size_t>(es.left)] += s;
                wn[static_cast<std::size_t>(es.right)] -= s;
            }
            for (std::size_t f = 0; f < faceCount; ++f) {
                const int expected = f == 0 ? -1 : (static_cast<FaceId>(f) == own ? 1 : 0);
                windingMismatches += wn[f] != expected ? 1 : 0;
            }
            FaceId top = cnx::kInvalidId;
            for (const FaceId f : order) {
                int cover = 0;
                for (const Segment& s : outer[static_cast<std::size_t>(f)]) cover += crossingSign(s.a, s.b, p);
                if (cover != 0) top = f;
            }
            painterMismatches += top != own ? 1 : 0;
        }
    }
    CHECK_CTX(windingMismatches == 0, context + ": voltas incorretas: " + std::to_string(windingMismatches));
    CHECK_CTX(painterMismatches == 0, context + ": pintura por camadas incorreta em " +
                                          std::to_string(painterMismatches) + " pixels");
}

std::size_t countFlag(const VectorImage& vi, std::uint8_t flag) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < vi.nodeCount(); ++i) n += (vi.flags(static_cast<NodeId>(i)) & flag) != 0 ? 1 : 0;
    return n;
}

// ------------------------------------------------------------------ casos construídos
CNX_TEST(single_pixel) {
    const TraceResult r = traceLabels(labelsFromRows({"a"}));
    checkTrace(r, "1×1");
    CHECK(r.image.faceCount() == 2);
    CHECK(r.image.edgeCount() == 4);
    CHECK(r.image.nodeCount() == 4);
    CHECK(countFlag(r.image, cnx::NodeFlag::Pinned) == 4);
}

CNX_TEST(uniform_image_has_only_the_frame) {
    const TraceResult r = traceLabels(sla::Array2D<RegionId>(5, 3, 0));
    checkTrace(r, "5×3 uniforme");
    CHECK(r.image.faceCount() == 2);
    CHECK(r.image.edgeCount() == 4);
    CHECK(r.image.face(1).area == 15.0);
    CHECK(r.image.face(1).parent == kExt);
}

CNX_TEST(vertical_split) {
    const TraceResult r = traceLabels(labelsFromRows({"aabb", "aabb"}));
    checkTrace(r, "divisão vertical");
    CHECK(r.image.faceCount() == 3);
    CHECK(r.image.edgeCount() == 7);  // moldura em 6 trechos + divisória
    CHECK(countFlag(r.image, cnx::NodeFlag::Junction) == 2);
    CHECK(r.image.nodeCount() == 6);  // 4 cantos + 2 junções; trechos retos sem nós internos
}

CNX_TEST(nested_square_is_a_loop_with_anchor) {
    const TraceResult r = traceLabels(labelsFromRows({"aaaaa", "abbba", "abbba", "abbba", "aaaaa"}));
    checkTrace(r, "quadrado aninhado");
    const VectorImage& vi = r.image;
    CHECK(vi.faceCount() == 3);
    const FaceId ring = r.pixelFaces(0, 0);
    const FaceId inner = r.pixelFaces(2, 2);
    CHECK(vi.face(ring).holeCount == 1);
    CHECK(vi.face(inner).parent == ring);
    CHECK(vi.face(ring).parent == kExt);
    CHECK(countFlag(vi, cnx::NodeFlag::LoopAnchor) == 1);
    bool found = false;
    for (std::size_t e = 0; e < vi.edgeCount(); ++e) {
        const auto nodes = vi.edgeNodes(static_cast<EdgeId>(e));
        if (nodes.front() != nodes.back()) continue;
        found = true;
        CHECK(nodes.size() == 5);  // âncora + 3 dobras + âncora
        CHECK(vi.position(nodes.front()) == (sla::Vec2{1, 1}));
        CHECK((vi.flags(nodes.front()) & cnx::NodeFlag::LoopAnchor) != 0);
    }
    CHECK(found);
}

CNX_TEST(concentric_rings_nest_three_levels) {
    const TraceResult r = traceLabels(labelsFromRows(
        {"aaaaaaa", "abbbbba", "abcccba", "abcdcba", "abcccba", "abbbbba", "aaaaaaa"}));
    checkTrace(r, "anéis concêntricos");
    const VectorImage& vi = r.image;
    const FaceId a = r.pixelFaces(0, 0);
    const FaceId b = r.pixelFaces(1, 1);
    const FaceId c = r.pixelFaces(2, 2);
    const FaceId d = r.pixelFaces(3, 3);
    CHECK(vi.face(d).parent == c);
    CHECK(vi.face(c).parent == b);
    CHECK(vi.face(b).parent == a);
    CHECK(vi.face(a).parent == kExt);
    CHECK(vi.face(a).holeCount == 1 && vi.face(b).holeCount == 1 && vi.face(c).holeCount == 1);
}

CNX_TEST(checkerboard_saddle_is_degree_four_junction) {
    const TraceResult r = traceLabels(labelsFromRows({"ab", "ba"}));
    checkTrace(r, "xadrez 2×2");
    const VectorImage& vi = r.image;
    CHECK(vi.faceCount() == 5);  // cada pixel é uma componente 4-conexa própria
    CHECK(vi.edgeCount() == 12);
    std::size_t centerSpokes = 0;
    for (std::size_t e = 0; e < vi.edgeCount(); ++e) {
        const auto nodes = vi.edgeNodes(static_cast<EdgeId>(e));
        for (const NodeId n : {nodes.front(), nodes.back()}) centerSpokes += vi.position(n) == sla::Vec2{1, 1} ? 1 : 0;
    }
    CHECK(centerSpokes == 4);
}

CNX_TEST(face_touching_itself_diagonally_is_pinched) {
    // A (7 px) é 4-conexa e toca a si mesma na quina (2,2), entre b(1,1) e b(2,2).
    const TraceResult r = traceLabels(labelsFromRows({"aaa", "aba", "aab"}));
    checkTrace(r, "beliscão");
    const VectorImage& vi = r.image;
    CHECK(vi.faceCount() == 4);
    const FaceId a = r.pixelFaces(0, 0);
    CHECK(vi.face(a).area == 7.0);
    CHECK(vi.face(a).holeCount == 0);  // contorno externo e o "furo" viram um único passeio beliscado
    CHECK(vi.face(r.pixelFaces(1, 1)).parent == kExt);
}

CNX_TEST(loop_attached_to_junction_counts_two_components) {
    // b e c se tocam só pela quina (2,2): cada uma é uma aresta fechada nessa junção (não LoopAnchor).
    const TraceResult r = traceLabels(labelsFromRows({"aaaa", "abaa", "aaca", "aaaa"}));
    checkTrace(r, "laço preso a junção");
    const VectorImage& vi = r.image;
    const FaceId a = r.pixelFaces(0, 0);
    const FaceId b = r.pixelFaces(1, 1);
    const FaceId c = r.pixelFaces(2, 2);
    CHECK(vi.face(a).holeCount == 1);  // um furo em forma de 8 em torno de b e c
    CHECK(vi.face(b).parent == a && vi.face(c).parent == a);
    CHECK(countFlag(vi, cnx::NodeFlag::LoopAnchor) == 0);
    std::size_t closedAtJunction = 0;
    for (std::size_t e = 0; e < vi.edgeCount(); ++e) {
        const auto nodes = vi.edgeNodes(static_cast<EdgeId>(e));
        if (nodes.front() == nodes.back() && (vi.flags(nodes.front()) & cnx::NodeFlag::Junction) != 0) ++closedAtJunction;
    }
    CHECK(closedAtJunction == 2);
}

CNX_TEST(diagonal_staircase_splits_into_components) {
    const TraceResult r = traceLabels(labelsFromRows({"abbb", "babb", "bbab", "bbba"}));
    checkTrace(r, "diagonal");
    CHECK(r.image.faceCount() == 1 + 4 + 2);  // 4 pixels 'a' isolados + 2 triângulos 'b'
    CHECK(countFlag(r.image, cnx::NodeFlag::Junction) >= 3);
}

CNX_TEST(colors_and_palette_indices_follow_regions) {
    const auto labels = labelsFromRows({"aab", "cab"});
    const std::vector<sla::Rgba> colors{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const std::vector<std::int32_t> palette{7, 8, 9};
    const TraceResult r = cnx::BoundaryTracer().trace(labels, colors, palette);
    checkTrace(r, "cores");
    for (std::size_t f = 1; f < r.image.faceCount(); ++f) {
        const auto region = static_cast<std::size_t>(r.faceRegion[f]);
        CHECK(r.image.face(static_cast<FaceId>(f)).color == colors[region]);
        CHECK(r.image.face(static_cast<FaceId>(f)).paletteIndex == palette[region]);
    }
    CHECK(r.faceRegion[0] == cnx::kNoRegion);
}

CNX_TEST(keeping_collinear_nodes_preserves_invariants) {
    const auto labels = labelsFromRows({"aaaaa", "abbba", "abbba", "aaaaa"});
    const TraceResult compact = traceLabels(labels);
    const TraceResult dense = traceLabels(labels, cnx::BoundaryTracer::Options{false});
    checkTrace(dense, "colineares mantidos");
    CHECK(dense.image.nodeCount() > compact.image.nodeCount());
    // Moldura (4·(5+4) = 18 quinas) + contorno do bloco 3×2 (10 quinas).
    CHECK(dense.image.nodeCount() == 28);
}

CNX_TEST(tracing_is_deterministic) {
    SplitMix64 rng{42};
    const auto labels = discLabels(23, 17, 4, 9, rng);
    const TraceResult r1 = traceLabels(labels);
    const TraceResult r2 = traceLabels(labels);
    CHECK(r1.image.nodeCount() == r2.image.nodeCount() && r1.image.edgeCount() == r2.image.edgeCount());
    bool same = r1.image.nodeCount() == r2.image.nodeCount();
    for (std::size_t n = 0; same && n < r1.image.nodeCount(); ++n) {
        same = r1.image.position(static_cast<NodeId>(n)) == r2.image.position(static_cast<NodeId>(n));
    }
    for (std::size_t e = 0; same && e < r1.image.edgeCount(); ++e) {
        const auto a = r1.image.edgeNodes(static_cast<EdgeId>(e));
        const auto b = r2.image.edgeNodes(static_cast<EdgeId>(e));
        same = std::equal(a.begin(), a.end(), b.begin(), b.end());
    }
    CHECK(same);
}

CNX_TEST(invalid_input_is_rejected) {
    const auto colors = colorsFor(2);
    const cnx::BoundaryTracer tracer;
    auto labels = labelsFromRows({"ab"});
    labels(1, 0) = 2;
    CHECK_THROWS(std::invalid_argument, (void)tracer.trace(labels, colors));
    labels(1, 0) = -1;
    CHECK_THROWS(std::invalid_argument, (void)tracer.trace(labels, colors));
    CHECK_THROWS(std::invalid_argument, (void)tracer.trace(sla::Array2D<RegionId>(0, 3), colors));
    const std::vector<std::int32_t> shortPalette{0};
    CHECK_THROWS(std::invalid_argument, (void)tracer.trace(labelsFromRows({"ab"}), colors, shortPalette));
}

CNX_TEST(cancellation_is_cooperative) {
    std::stop_source source;
    source.request_stop();
    SplitMix64 rng{3};
    const auto labels = noiseLabels(8, 8, 3, rng);
    const auto colors = colorsFor(3);
    CHECK_THROWS(cnx::CancelTaskException, (void)cnx::BoundaryTracer().trace(labels, colors, {}, source.get_token()));
}

// ------------------------------------------------------------------ varreduras aleatórias
CNX_TEST(random_noise_images_satisfy_invariants) {
    SplitMix64 rng{2026};
    for (int i = 0; i < 120; ++i) {
        const int w = 1 + rng.below(14);
        const int h = 1 + rng.below(14);
        const int k = 1 + rng.below(4);
        const auto labels = noiseLabels(w, h, k, rng);
        checkTrace(traceLabels(labels), "ruído #" + std::to_string(i) + " (" + std::to_string(w) + "×" +
                                            std::to_string(h) + ", k=" + std::to_string(k) + ")");
    }
}

CNX_TEST(random_disc_images_satisfy_invariants) {
    SplitMix64 rng{1987};
    for (int i = 0; i < 120; ++i) {
        const int w = 2 + rng.below(18);
        const int h = 2 + rng.below(18);
        const int k = 2 + rng.below(4);
        const auto labels = discLabels(w, h, k, 1 + rng.below(8), rng);
        const bool keepCollinear = rng.below(4) == 0;
        checkTrace(traceLabels(labels, cnx::BoundaryTracer::Options{!keepCollinear}),
                   "discos #" + std::to_string(i) + " (" + std::to_string(w) + "×" + std::to_string(h) + ")");
    }
}

CNX_TEST(large_image_timing_and_invariants) {
    SplitMix64 rng{7};
    const auto labels = discLabels(1024, 768, 6, 400, rng);
    const auto colors = colorsFor(6);
    const auto t0 = std::chrono::steady_clock::now();
    const TraceResult r = cnx::BoundaryTracer().trace(labels, colors);
    const auto t1 = std::chrono::steady_clock::now();
    const auto report = r.image.validate();
    const auto t2 = std::chrono::steady_clock::now();
    CHECK_CTX(report.ok, report.message);
    std::vector<double> pixels(r.image.faceCount(), 0.0);
    for (int y = 0; y < labels.height(); ++y) {
        for (int x = 0; x < labels.width(); ++x) pixels[static_cast<std::size_t>(r.pixelFaces(x, y))] += 1.0;
    }
    std::size_t mismatches = 0;
    for (std::size_t f = 1; f < pixels.size(); ++f) mismatches += r.image.face(static_cast<FaceId>(f)).area != pixels[f] ? 1 : 0;
    CHECK(mismatches == 0);
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("  1024×768: %zu faces, %zu arestas, %zu nós; traçado %.1f ms, validate %.1f ms\n",
                r.image.faceCount(), r.image.edgeCount(), r.image.nodeCount(), ms(t0, t1), ms(t1, t2));
}

// ------------------------------------------------------------------ a partir do PixelMap
CNX_TEST(trace_from_pixel_map_after_merges_and_moves) {
    SplitMix64 rng{31};
    for (int scene = 0; scene < 4; ++scene) {
        cnxtest::Scene s = cnxtest::blockScene(16 + scene, 12, 2, rng, 4);
        cnxtest::randomMerges(s.map, 30, rng);
        (void)cnxtest::randomMoves(s.map, s.image, 400, rng);
        s.map.assignMedianColors();
        const RegionId regions = s.map.compact();
        const TraceResult r = cnx::BoundaryTracer().trace(s.map);
        const std::string tag = "PixelMap #" + std::to_string(scene);
        checkTrace(r, tag);
        // Blocos iniciais, fusões entre vizinhas e trocas de pontos simples mantêm cada região
        // 4-conexa: exatamente uma face por região.
        CHECK_CTX(r.image.faceCount() == static_cast<std::size_t>(regions) + 1, tag);
        std::size_t mismatches = 0;
        for (std::size_t f = 1; f < r.image.faceCount(); ++f) {
            const RegionId region = r.faceRegion[f];
            const cnx::Face& face = r.image.face(static_cast<FaceId>(f));
            mismatches += face.color == s.map.regionColor(region) ? 0 : 1;
            mismatches += face.area == static_cast<double>(s.map.forest().pixelCount(region)) ? 0 : 1;
        }
        CHECK_CTX(mismatches == 0, tag);
    }
}

CNX_TEST(trace_from_pixel_map_requires_compact) {
    SplitMix64 rng{32};
    cnxtest::Scene s = cnxtest::blockScene(6, 6, 2, rng, 3);
    CHECK_THROWS(std::logic_error, (void)cnx::BoundaryTracer().trace(s.map));
    (void)s.map.compact();
    (void)s.map.forest().unite(0, 1);
    CHECK_THROWS(std::logic_error, (void)cnx::BoundaryTracer().trace(s.map));
}

CNX_TEST(disconnected_region_becomes_several_faces) {
    cnx::PixelMap map(3, 1);
    const RegionId a = map.createRegion();
    const RegionId b = map.createRegion();
    map.setLabel(0, 0, a, {255, 0, 0, 255});
    map.setLabel(1, 0, b, {0, 255, 0, 255});
    map.setLabel(2, 0, a, {255, 0, 0, 255});
    map.assignMedianColors(1);
    (void)map.compact();
    const TraceResult r = cnx::BoundaryTracer().trace(map);
    checkTrace(r, "região desconexa");
    CHECK(r.image.faceCount() == 4);
    CHECK(r.faceRegion[static_cast<std::size_t>(r.pixelFaces(0, 0))] == 0);
    CHECK(r.faceRegion[static_cast<std::size_t>(r.pixelFaces(2, 0))] == 0);
    CHECK(r.pixelFaces(0, 0) != r.pixelFaces(2, 0));
    CHECK(r.image.face(r.pixelFaces(0, 0)).color == sla::toRgba({255, 0, 0, 255}));
}

}  // namespace

int main() { return cnxtest::runAll(); }
