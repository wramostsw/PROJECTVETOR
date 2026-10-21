// src/vector/CrossingGuard.cpp — detecção de cruzamentos entre segmentos por grade uniforme.
#include "cnx/vector/CrossingGuard.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

namespace cnx {
namespace {

using sla::Vec2;

struct Seg {
    NodeId a = kInvalidId;
    NodeId b = kInvalidId;
    Vec2 p;
    Vec2 q;
    double minX = 0, minY = 0, maxX = 0, maxY = 0;
};

std::vector<Seg> collectSegments(const VectorImage& image) {
    std::vector<Seg> segs;
    for (EdgeId e = 0; e < static_cast<EdgeId>(image.edgeCount()); ++e) {
        const auto nodes = image.edgeNodes(e);
        for (std::size_t i = 0; i + 1 < nodes.size(); ++i) {
            Seg s;
            s.a = nodes[i];
            s.b = nodes[i + 1];
            s.p = image.position(s.a);
            s.q = image.position(s.b);
            s.minX = std::min(s.p.x, s.q.x);
            s.maxX = std::max(s.p.x, s.q.x);
            s.minY = std::min(s.p.y, s.q.y);
            s.maxY = std::max(s.p.y, s.q.y);
            segs.push_back(s);
        }
    }
    return segs;
}

bool sharesNode(const Seg& s, const Seg& t) noexcept { return s.a == t.a || s.a == t.b || s.b == t.a || s.b == t.b; }

double orient(Vec2 a, Vec2 b, Vec2 c) noexcept { return sla::cross(b - a, c - a); }

/// Interseção própria: as pontas de cada segmento ficam em lados estritamente opostos do outro.
bool properlyIntersect(const Seg& s, const Seg& t) noexcept {
    const double d1 = orient(s.p, s.q, t.p);
    const double d2 = orient(s.p, s.q, t.q);
    const double d3 = orient(t.p, t.q, s.p);
    const double d4 = orient(t.p, t.q, s.q);
    return ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0));
}

Crossing makeCrossing(const Seg& s, const Seg& t) {
    Crossing c{s.a, s.b, t.a, t.b};
    // Forma canônica: o par com o menor (a, b) primeiro.
    if (std::tie(c.b0, c.b1) < std::tie(c.a0, c.a1)) {
        std::swap(c.a0, c.b0);
        std::swap(c.a1, c.b1);
    }
    return c;
}

bool crossingLess(const Crossing& x, const Crossing& y) noexcept {
    return std::tie(x.a0, x.a1, x.b0, x.b1) < std::tie(y.a0, y.a1, y.b0, y.b1);
}

}  // namespace

std::vector<Crossing> CrossingGuard::findBruteForce(const VectorImage& image) {
    const std::vector<Seg> segs = collectSegments(image);
    std::vector<Crossing> out;
    for (std::size_t i = 0; i < segs.size(); ++i) {
        for (std::size_t j = i + 1; j < segs.size(); ++j) {
            if (!sharesNode(segs[i], segs[j]) && properlyIntersect(segs[i], segs[j])) out.push_back(makeCrossing(segs[i], segs[j]));
        }
    }
    std::sort(out.begin(), out.end(), crossingLess);
    return out;
}

std::vector<Crossing> CrossingGuard::find(const VectorImage& image) {
    const std::vector<Seg> segs = collectSegments(image);
    std::vector<Crossing> out;
    if (segs.size() < 2) return out;

    double minX = std::numeric_limits<double>::infinity(), minY = minX;
    double maxX = -minX, maxY = -minX;
    double totalLength = 0.0;
    for (const Seg& s : segs) {
        minX = std::min(minX, s.minX);
        minY = std::min(minY, s.minY);
        maxX = std::max(maxX, s.maxX);
        maxY = std::max(maxY, s.maxY);
        totalLength += sla::norm(s.q - s.p);
    }
    // Células com ~2× o comprimento médio dos segmentos; teto de 1024 células por eixo.
    double cell = std::max(2.0 * totalLength / static_cast<double>(segs.size()), 1e-6);
    cell = std::max({cell, (maxX - minX) / 1024.0, (maxY - minY) / 1024.0});
    const int nx = std::max(1, static_cast<int>(std::floor((maxX - minX) / cell)) + 1);
    const int ny = std::max(1, static_cast<int>(std::floor((maxY - minY) / cell)) + 1);
    const auto cellX = [&](double x) { return std::clamp(static_cast<int>(std::floor((x - minX) / cell)), 0, nx - 1); };
    const auto cellY = [&](double y) { return std::clamp(static_cast<int>(std::floor((y - minY) / cell)), 0, ny - 1); };

    // Listas por célula em formato CSR (contagem + preenchimento).
    std::vector<std::uint32_t> start(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) + 1, 0);
    const auto forCells = [&](const Seg& s, auto&& fn) {
        for (int cy = cellY(s.minY); cy <= cellY(s.maxY); ++cy) {
            for (int cx = cellX(s.minX); cx <= cellX(s.maxX); ++cx) fn(static_cast<std::size_t>(cy) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(cx));
        }
    };
    for (const Seg& s : segs) forCells(s, [&](std::size_t c) { ++start[c + 1]; });
    for (std::size_t c = 0; c + 1 < start.size(); ++c) start[c + 1] += start[c];
    std::vector<std::uint32_t> items(start.back());
    std::vector<std::uint32_t> fill(start.begin(), start.end() - 1);
    for (std::size_t i = 0; i < segs.size(); ++i) forCells(segs[i], [&](std::size_t c) { items[fill[c]++] = static_cast<std::uint32_t>(i); });

    for (std::size_t c = 0; c + 1 < start.size(); ++c) {
        const int cx = static_cast<int>(c % static_cast<std::size_t>(nx));
        const int cy = static_cast<int>(c / static_cast<std::size_t>(nx));
        for (std::uint32_t u = start[c]; u < start[c + 1]; ++u) {
            for (std::uint32_t v = u + 1; v < start[c + 1]; ++v) {
                const Seg& s = segs[items[u]];
                const Seg& t = segs[items[v]];
                // Cada par é testado só na célula do canto inferior-esquerdo da interseção das caixas.
                if (cellX(std::max(s.minX, t.minX)) != cx || cellY(std::max(s.minY, t.minY)) != cy) continue;
                if (!sharesNode(s, t) && properlyIntersect(s, t)) out.push_back(makeCrossing(s, t));
            }
        }
    }
    std::sort(out.begin(), out.end(), crossingLess);
    return out;
}

std::vector<NodeId> CrossingGuard::shortSegmentNodes(const VectorImage& image, double minLength) {
    std::vector<NodeId> nodes;
    for (EdgeId e = 0; e < static_cast<EdgeId>(image.edgeCount()); ++e) {
        const auto ns = image.edgeNodes(e);
        for (std::size_t i = 0; i + 1 < ns.size(); ++i) {
            if (sla::norm(image.position(ns[i + 1]) - image.position(ns[i])) < minLength) {
                nodes.push_back(ns[i]);
                nodes.push_back(ns[i + 1]);
            }
        }
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
}

std::vector<NodeId> CrossingGuard::offendingNodes(const std::vector<Crossing>& crossings) {
    std::vector<NodeId> nodes;
    for (const Crossing& c : crossings) {
        nodes.insert(nodes.end(), {c.a0, c.a1, c.b0, c.b1});
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
}

}  // namespace cnx
