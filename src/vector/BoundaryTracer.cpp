// src/vector/BoundaryTracer.cpp — rotulação 4-conexa, grade de fendas e cadeias entre vértices.
#include "cnx/vector/BoundaryTracer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cnx/engine/TaskContext.h"

namespace cnx {
namespace {

// Direções na grade de quinas (x → direita, y → baixo).
constexpr int kEast = 0;
constexpr int kSouth = 1;
constexpr int kWest = 2;
constexpr int kNorth = 3;
constexpr std::array<int, 4> kDx{1, 0, -1, 0};
constexpr std::array<int, 4> kDy{0, 1, 0, -1};
constexpr int opposite(int d) noexcept { return (d + 2) & 3; }

void throwIfStopped(const std::stop_token& stop) {
    if (stop.stop_requested()) throw CancelTaskException{};
}

std::uint8_t borderFlag(int cx, int cy, int w, int h) noexcept {
    return (cx == 0 || cy == 0 || cx == w || cy == h) ? NodeFlag::ImageBorder : std::uint8_t{0};
}

/// Componentes 4-conexas de cada rótulo → faces 1..F−1 em ordem de varredura.
void labelComponents(const sla::Array2D<RegionId>& labels, sla::Array2D<FaceId>& faces,
                     std::vector<RegionId>& faceRegion, const std::stop_token& stop) {
    const int w = labels.width();
    const int h = labels.height();
    faces = sla::Array2D<FaceId>(w, h, kInvalidId);
    faceRegion.assign(1, kNoRegion);  // face 0 = exterior
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h; ++y) {
        throwIfStopped(stop);
        for (int x = 0; x < w; ++x) {
            if (faces(x, y) != kInvalidId) continue;
            if (faceRegion.size() > static_cast<std::size_t>(std::numeric_limits<FaceId>::max())) {
                throw std::length_error("BoundaryTracer: excesso de faces");
            }
            const auto face = static_cast<FaceId>(faceRegion.size());
            const RegionId region = labels(x, y);
            faceRegion.push_back(region);
            faces(x, y) = face;
            stack.emplace_back(x, y);
            while (!stack.empty()) {
                const auto [px, py] = stack.back();
                stack.pop_back();
                for (int d = 0; d < 4; ++d) {
                    const int nx = px + kDx[static_cast<std::size_t>(d)];
                    const int ny = py + kDy[static_cast<std::size_t>(d)];
                    if (labels.inBounds(nx, ny) && faces(nx, ny) == kInvalidId && labels(nx, ny) == region) {
                        faces(nx, ny) = face;
                        stack.emplace_back(nx, ny);
                    }
                }
            }
        }
    }
}

/// Fendas: lados unitários da grade de quinas (W+1)×(H+1) que separam faces diferentes
/// (fora da imagem = exterior). Índices: horizontais [0, W·(H+1)), verticais depois.
class CrackGrid {
public:
    explicit CrackGrid(const sla::Array2D<FaceId>& faces)
        : faces_(faces),
          w_(faces.width()),
          h_(faces.height()),
          visited_(static_cast<std::size_t>(w_) * static_cast<std::size_t>(h_ + 1) +
                       static_cast<std::size_t>(w_ + 1) * static_cast<std::size_t>(h_),
                   0) {}

    FaceId face(int x, int y) const noexcept { return faces_.inBounds(x, y) ? faces_(x, y) : kExteriorFace; }

    bool hasCrack(int cx, int cy, int d) const noexcept {
        switch (d) {
            case kEast: return cx < w_ && face(cx, cy - 1) != face(cx, cy);
            case kSouth: return cy < h_ && face(cx - 1, cy) != face(cx, cy);
            case kWest: return cx > 0 && face(cx - 1, cy - 1) != face(cx - 1, cy);
            default: return cy > 0 && face(cx - 1, cy - 1) != face(cx, cy - 1);
        }
    }

    int degree(int cx, int cy) const noexcept {
        int n = 0;
        for (int d = 0; d < 4; ++d) n += hasCrack(cx, cy, d) ? 1 : 0;
        return n;
    }

    /// (left, right) ao percorrer a fenda d a partir da quina (cx, cy), na convenção de VectorImage:
    /// left fica do lado de área positiva, isto é, n = (Δy, −Δx) aponta para fora de left.
    std::pair<FaceId, FaceId> sides(int cx, int cy, int d) const noexcept {
        switch (d) {
            case kEast: return {face(cx, cy), face(cx, cy - 1)};
            case kSouth: return {face(cx - 1, cy), face(cx, cy)};
            case kWest: return {face(cx - 1, cy - 1), face(cx - 1, cy)};
            default: return {face(cx, cy - 1), face(cx - 1, cy - 1)};
        }
    }

    bool visited(int cx, int cy, int d) const noexcept { return visited_[index(cx, cy, d)] != 0; }
    void markVisited(int cx, int cy, int d) noexcept { visited_[index(cx, cy, d)] = 1; }

private:
    std::size_t index(int cx, int cy, int d) const noexcept {
        const auto w = static_cast<std::size_t>(w_);
        const std::size_t verticalBase = w * static_cast<std::size_t>(h_ + 1);
        switch (d) {
            case kEast: return static_cast<std::size_t>(cy) * w + static_cast<std::size_t>(cx);
            case kWest: return static_cast<std::size_t>(cy) * w + static_cast<std::size_t>(cx - 1);
            case kSouth: return verticalBase + static_cast<std::size_t>(cy) * (w + 1) + static_cast<std::size_t>(cx);
            default: return verticalBase + static_cast<std::size_t>(cy - 1) * (w + 1) + static_cast<std::size_t>(cx);
        }
    }

    const sla::Array2D<FaceId>& faces_;
    int w_;
    int h_;
    std::vector<std::uint8_t> visited_;
};

}  // namespace

TraceResult BoundaryTracer::trace(const sla::Array2D<RegionId>& labels, std::span<const sla::Rgba> regionColors,
                                  std::span<const std::int32_t> regionPaletteIndex,
                                  const std::stop_token& stop) const {
    const int w = labels.width();
    const int h = labels.height();
    if (w <= 0 || h <= 0) throw std::invalid_argument("BoundaryTracer::trace: imagem vazia");
    if (!regionPaletteIndex.empty() && regionPaletteIndex.size() != regionColors.size()) {
        throw std::invalid_argument("BoundaryTracer::trace: regionPaletteIndex deve ser vazio ou do tamanho de regionColors");
    }
    if (static_cast<std::size_t>(w + 1) * static_cast<std::size_t>(h + 1) >
        static_cast<std::size_t>(std::numeric_limits<NodeId>::max())) {
        throw std::length_error("BoundaryTracer::trace: imagem grande demais para ids de 32 bits");
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const RegionId r = labels(x, y);
            if (r < 0 || static_cast<std::size_t>(r) >= regionColors.size()) {
                throw std::invalid_argument("BoundaryTracer::trace: rótulo " + std::to_string(r) + " fora de [0, " +
                                            std::to_string(regionColors.size()) + ")");
            }
        }
    }

    TraceResult result;
    labelComponents(labels, result.pixelFaces, result.faceRegion, stop);

    VectorImage image(static_cast<double>(w), static_cast<double>(h));
    for (std::size_t f = 1; f < result.faceRegion.size(); ++f) {
        const auto r = static_cast<std::size_t>(result.faceRegion[f]);
        const FaceId id = image.addFace(regionColors[r], regionPaletteIndex.empty() ? -1 : regionPaletteIndex[r]);
        if (static_cast<std::size_t>(id) != f) throw std::logic_error("BoundaryTracer: ids de face fora de ordem");
    }

    CrackGrid grid(result.pixelFaces);
    sla::Array2D<std::uint8_t> degree(w + 1, h + 1, 0);
    for (int cy = 0; cy <= h; ++cy) {
        for (int cx = 0; cx <= w; ++cx) degree(cx, cy) = static_cast<std::uint8_t>(grid.degree(cx, cy));
    }

    // 1. Vértices: junções (grau ≥ 3, inclusive selas) e cantos da moldura.
    sla::Array2D<NodeId> cornerNode(w + 1, h + 1, kInvalidId);
    for (int cy = 0; cy <= h; ++cy) {
        throwIfStopped(stop);
        for (int cx = 0; cx <= w; ++cx) {
            const sla::Vec2 p{static_cast<double>(cx), static_cast<double>(cy)};
            if (degree(cx, cy) >= 3) {
                cornerNode(cx, cy) = image.addNode(p, static_cast<std::uint8_t>(NodeFlag::Junction | borderFlag(cx, cy, w, h)));
            } else if ((cx == 0 || cx == w) && (cy == 0 || cy == h)) {
                cornerNode(cx, cy) =
                    image.addNode(p, static_cast<std::uint8_t>(NodeFlag::Pinned | NodeFlag::Corner | NodeFlag::ImageBorder));
            }
        }
    }

    // 2. Cadeias: de um vértice, segue as fendas pelas quinas de grau 2 até o próximo vértice.
    std::vector<NodeId> chain;
    const auto walk = [&](int cx, int cy, int dir, NodeId start) {
        const std::pair<FaceId, FaceId> sides = grid.sides(cx, cy, dir);
        chain.clear();
        chain.push_back(start);
        for (;;) {
            grid.markVisited(cx, cy, dir);
            cx += kDx[static_cast<std::size_t>(dir)];
            cy += kDy[static_cast<std::size_t>(dir)];
            const NodeId end = cornerNode(cx, cy);
            if (end != kInvalidId) {
                chain.push_back(end);
                break;
            }
            int nextDir = -1;
            for (int d = 0; d < 4; ++d) {
                if (d != opposite(dir) && grid.hasCrack(cx, cy, d)) {
                    nextDir = d;
                    break;
                }
            }
            if (nextDir < 0 || degree(cx, cy) != 2) throw std::logic_error("BoundaryTracer: quina interna de grau ≠ 2");
            if (grid.sides(cx, cy, nextDir) != sides) throw std::logic_error("BoundaryTracer: faces mudam ao longo da cadeia");
            if (nextDir != dir || !options_.removeCollinearNodes) {
                chain.push_back(image.addNode({static_cast<double>(cx), static_cast<double>(cy)}, borderFlag(cx, cy, w, h)));
            }
            dir = nextDir;
        }
        image.addEdge(chain, sides.first, sides.second);
    };

    for (int cy = 0; cy <= h; ++cy) {
        throwIfStopped(stop);
        for (int cx = 0; cx <= w; ++cx) {
            const NodeId v = cornerNode(cx, cy);
            if (v == kInvalidId) continue;
            for (int d = 0; d < 4; ++d) {
                if (grid.hasCrack(cx, cy, d) && !grid.visited(cx, cy, d)) walk(cx, cy, d, v);
            }
        }
    }

    // 3. Laços sem junção. Na varredura, a primeira quina de um laço ainda não visitado é a mais alta
    //    à esquerda: uma dobra com fendas Leste e Sul, que vira o LoopAnchor.
    for (int cy = 0; cy <= h; ++cy) {
        throwIfStopped(stop);
        for (int cx = 0; cx <= w; ++cx) {
            if (cornerNode(cx, cy) != kInvalidId || degree(cx, cy) != 2) continue;
            bool pending = false;
            for (int d = 0; d < 4; ++d) pending = pending || (grid.hasCrack(cx, cy, d) && !grid.visited(cx, cy, d));
            if (!pending) continue;
            if (!grid.hasCrack(cx, cy, kEast) || !grid.hasCrack(cx, cy, kSouth)) {
                throw std::logic_error("BoundaryTracer: âncora de laço sem fendas Leste e Sul");
            }
            const NodeId anchor = image.addNode({static_cast<double>(cx), static_cast<double>(cy)},
                                                static_cast<std::uint8_t>(NodeFlag::LoopAnchor | borderFlag(cx, cy, w, h)));
            cornerNode(cx, cy) = anchor;
            walk(cx, cy, kEast, anchor);
        }
    }

    image.linkFaceLoops();
    result.image = std::move(image);
    return result;
}

TraceResult BoundaryTracer::trace(const PixelMap& map, const std::stop_token& stop) const {
    if (!map.isCompact()) throw std::logic_error("BoundaryTracer::trace: PixelMap não compactado (chame compact())");
    const std::size_t regions = map.forest().capacity();
    std::vector<sla::Rgba> colors(regions);
    std::vector<std::int32_t> palette(regions);
    for (std::size_t r = 0; r < regions; ++r) {
        const auto id = static_cast<RegionId>(r);
        colors[r] = map.regionColor(id);
        palette[r] = map.forest().stats(id).paletteIndex;
    }
    return trace(map.compactLabels(), colors, palette, stop);
}

}  // namespace cnx
