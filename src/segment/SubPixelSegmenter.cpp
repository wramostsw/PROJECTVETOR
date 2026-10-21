// src/segment/SubPixelSegmenter.cpp — micro-regiões, praias, sombras e realinhamento à paleta.
#include "cnx/segment/SubPixelSegmenter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "cnx/segment/PixelSegmenter.h"

namespace cnx {
namespace {

using sla::Rgba;

constexpr std::uint32_t kMeanFallbackBelow = 3;  // §4: a média só é fallback para regiões minúsculas
constexpr std::array<int, 4> kDx4{1, 0, -1, 0};
constexpr std::array<int, 4> kDy4{0, 1, 0, -1};

constexpr std::size_t ux(std::int32_t i) noexcept { return static_cast<std::size_t>(i); }

/// Cor de uma região: mediana por canal (média abaixo de 3 pixels), em cache até a região mudar.
class ColorCache {
public:
    explicit ColorCache(const PixelMap& map) : map_(map) {}

    Rgba get(RegionId root) {
        const std::size_t i = ux(root);
        if (i >= valid_.size()) {
            const std::size_t n = std::max(i + 1, map_.forest().capacity());
            color_.resize(n);
            valid_.resize(n, 0);
        }
        if (!valid_[i]) {
            const MedianAccumulator& s = map_.forest().stats(root).samples;
            color_[i] = s.count() < kMeanFallbackBelow ? s.mean() : sla::toRgba(s.median());
            valid_[i] = 1;
        }
        return color_[i];
    }
    void invalidate(RegionId root) {
        const std::size_t i = ux(root);
        if (i < valid_.size()) valid_[i] = 0;
    }

private:
    const PixelMap& map_;
    std::vector<Rgba> color_;
    std::vector<std::uint8_t> valid_;
};

/// Vizinha de `r` com a cor mais próxima (empate → maior contato → menor id). kNoRegion se não há vizinha.
template <class ColorFn>
RegionId closestNeighbor(RegionForest& forest, RegionId r, ColorFn&& color) {
    RegionId best = kNoRegion;
    double bestD = std::numeric_limits<double>::infinity();
    std::uint32_t bestContact = 0;
    const Rgba c = color(r);
    for (const RegionId n : forest.neighbors(r)) {
        const double d = sla::distance2(c, color(n));
        const std::uint32_t contact = forest.contactCount(r, n);
        if (d < bestD || (d == bestD && contact > bestContact)) {
            bestD = d;
            bestContact = contact;
            best = n;
        }
    }
    return best;
}

}  // namespace

SubPixelSegmenter::SubPixelSegmenter(const sla::Array2D<sla::Rgba8>& image, SegmentParams params, Palette palette)
    : image_(image), params_(params), palette_(std::move(palette)) {
    if (palette_.colors.empty()) throw std::invalid_argument("SubPixelSegmenter: paleta vazia");
}

std::size_t SubPixelSegmenter::nearestPalette(const Rgba& c) const {
    std::size_t best = 0;
    double bestD = sla::distance2(c, palette_.colors[0]);
    for (std::size_t k = 1; k < palette_.colors.size(); ++k) {
        const double d = sla::distance2(c, palette_.colors[k]);
        if (d < bestD) {
            bestD = d;
            best = k;
        }
    }
    return best;
}

const sla::Array2D<Rgba>& SubPixelSegmenter::blurred() const {
    if (!blurred_.empty()) return blurred_;
    const int w = image_.width();
    const int h = image_.height();
    const int r = std::max(0, params_.shadowBlurRadius);
    const double inv = 1.0 / static_cast<double>(2 * r + 1);
    sla::Array2D<Rgba> tmp(w, h);
    for (int y = 0; y < h; ++y) {  // horizontal, bordas replicadas
        for (int x = 0; x < w; ++x) {
            std::array<double, 4> s{};
            for (int k = -r; k <= r; ++k) {
                const Rgba c = sla::toRgba(image_(std::clamp(x + k, 0, w - 1), y));
                s[0] += c.r; s[1] += c.g; s[2] += c.b; s[3] += c.a;
            }
            tmp(x, y) = {static_cast<float>(s[0] * inv), static_cast<float>(s[1] * inv), static_cast<float>(s[2] * inv), static_cast<float>(s[3] * inv)};
        }
    }
    blurred_ = sla::Array2D<Rgba>(w, h);
    for (int y = 0; y < h; ++y) {  // vertical
        for (int x = 0; x < w; ++x) {
            std::array<double, 4> s{};
            for (int k = -r; k <= r; ++k) {
                const Rgba c = tmp(x, std::clamp(y + k, 0, h - 1));
                s[0] += c.r; s[1] += c.g; s[2] += c.b; s[3] += c.a;
            }
            blurred_(x, y) = {static_cast<float>(s[0] * inv), static_cast<float>(s[1] * inv), static_cast<float>(s[2] * inv), static_cast<float>(s[3] * inv)};
        }
    }
    return blurred_;
}

// ---------------------------------------------------------------- 1. micro-regiões

int SubPixelSegmenter::mergeMicroRegions(PixelMap& map) const {
    RegionForest& forest = map.forest();
    ColorCache cc(map);
    int total = 0;
    for (int size = 1; size <= params_.microRegionMaxPixels; ++size) {
        for (bool changed = true; changed;) {
            changed = false;
            for (std::size_t i = 0; i < forest.capacity(); ++i) {
                const auto r = static_cast<RegionId>(i);
                if (!forest.isRoot(r)) continue;
                const std::uint32_t n = forest.pixelCount(r);
                if (n == 0 || n > static_cast<std::uint32_t>(size)) continue;
                const RegionId best = closestNeighbor(forest, r, [&](RegionId q) { return cc.get(q); });
                if (best == kNoRegion) continue;  // região única da imagem
                cc.invalidate(r);
                cc.invalidate(best);
                cc.invalidate(forest.unite(r, best));
                ++total;
                changed = true;
            }
        }
    }
    return total;
}

// ---------------------------------------------------------------- 2. praias

SubPixelSegmenter::BeachResult SubPixelSegmenter::splitAndMergeBeaches(PixelMap& map) const {
    BeachResult out;
    RegionForest& forest = map.forest();
    ColorCache cc(map);
    const int w = map.width();
    const int h = map.height();
    const std::size_t cap = forest.capacity();
    std::vector<std::pair<std::uint32_t, RegionId>> nb;
    std::vector<std::pair<int, int>> pixels;
    std::vector<RegionId> sides;
    std::vector<std::uint8_t> done;
    for (std::size_t i = 0; i < cap; ++i) {
        const auto r = static_cast<RegionId>(i);
        if (!forest.isRoot(r) || forest.pixelCount(r) == 0) continue;

        // As duas vizinhas dominantes (maior contato) e a fração do contorno que elas cobrem.
        nb.clear();
        std::uint64_t total = 0;
        for (const RegionId n : forest.neighbors(r)) {
            const std::uint32_t c = forest.contactCount(r, n);
            nb.emplace_back(c, n);
            total += c;
        }
        if (nb.size() < 2) continue;
        std::sort(nb.begin(), nb.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        const RegionId A = nb[0].second;
        const RegionId B = nb[1].second;
        if (static_cast<double>(nb[0].first + nb[1].first) < params_.beachNeighborShare * static_cast<double>(total)) continue;

        // A cor da região tem de estar na reta A–B, longe das pontas.
        const Rgba ca = cc.get(A);
        const Rgba cb = cc.get(B);
        const Rgba cr = cc.get(r);
        const Rgba v = cb - ca;
        const double vv = sla::dot(v, v);
        if (vv < params_.beachMinContrast) continue;
        const double t = sla::dot(cr - ca, v) / vv;
        if (!(t > 0.02 && t < 0.98)) continue;
        const Rgba off = (cr - ca) - v * static_cast<float>(t);
        if (std::sqrt(sla::dot(off, off)) > params_.beachLineTolerance * std::sqrt(vv)) continue;

        // Fina: nenhum pixel interior (todo pixel toca outra região ou a borda da imagem).
        const RegionStats& st = forest.stats(r);
        pixels.clear();
        bool interior = false;
        for (int y = st.minY; y <= st.maxY && !interior; ++y) {
            for (int x = st.minX; x <= st.maxX; ++x) {
                if (map.regionAtConst(x, y) != r) continue;
                pixels.emplace_back(x, y);
                bool inside = true;
                for (std::size_t d = 0; d < 4 && inside; ++d) {
                    const int qx = x + kDx4[d];
                    const int qy = y + kDy4[d];
                    inside = qx >= 0 && qy >= 0 && qx < w && qy < h && map.regionAtConst(qx, qy) == r;
                }
                if (inside) {
                    interior = true;
                    break;
                }
            }
        }
        if (interior) continue;
        ++out.beaches;

        // Cada pixel vai para o lado de sua própria cor: t < ½ → A, senão B.
        sides.assign(pixels.size(), A);
        for (std::size_t k = 0; k < pixels.size(); ++k) {
            const Rgba c = sla::toRgba(image_(pixels[k].first, pixels[k].second));
            sides[k] = sla::dot(c - ca, v) / vv < 0.5 ? A : B;
        }
        done.assign(pixels.size(), 0);
        for (int sweep = 0; sweep < 8; ++sweep) {
            bool progress = false;
            for (std::size_t k = 0; k < pixels.size(); ++k) {
                if (done[k]) continue;
                const auto [x, y] = pixels[k];
                const RegionId target = sides[k];
                bool adjacent = false;
                for (std::size_t d = 0; d < 4 && !adjacent; ++d) {
                    const int qx = x + kDx4[d];
                    const int qy = y + kDy4[d];
                    adjacent = qx >= 0 && qy >= 0 && qx < w && qy < h && map.regionAt(qx, qy) == target;
                }
                if (!adjacent) continue;
                if (map.tryMovePixel(x, y, target, image_(x, y))) {
                    done[k] = 1;
                    ++out.moved;
                    progress = true;
                }
            }
            if (!progress) break;
        }
        cc.invalidate(r);
        cc.invalidate(A);
        cc.invalidate(B);
        if (forest.pixelCount(r) > 0) {  // resto que a proteção topológica não deixou mover: funde inteiro
            const RegionId side = sla::dot(cc.get(r) - ca, v) / vv < 0.5 ? A : B;
            cc.invalidate(forest.unite(r, side));
            ++out.merged;
        }
    }
    return out;
}

// ---------------------------------------------------------------- 3. sombras

int SubPixelSegmenter::flagShadowSegments(PixelMap& map) const {
    RegionForest& forest = map.forest();
    const sla::Array2D<Rgba>& bl = blurred();
    struct Acc {
        double n = 0.0;
        std::array<double, 4> s{}, q{}, bs{}, bq{};
    };
    std::vector<Acc> acc(forest.capacity());
    for (int y = 0; y < map.height(); ++y) {
        for (int x = 0; x < map.width(); ++x) {
            const RegionId r = map.regionAt(x, y);
            if (r == kNoRegion) continue;
            Acc& a = acc[ux(r)];
            const Rgba c = sla::toRgba(image_(x, y));
            const Rgba b = bl(x, y);
            const std::array<double, 4> cv{c.r, c.g, c.b, c.a};
            const std::array<double, 4> bv{b.r, b.g, b.b, b.a};
            a.n += 1.0;
            for (std::size_t k = 0; k < 4; ++k) {
                a.s[k] += cv[k];
                a.q[k] += cv[k] * cv[k];
                a.bs[k] += bv[k];
                a.bq[k] += bv[k] * bv[k];
            }
        }
    }
    int flagged = 0;
    for (std::size_t i = 0; i < acc.size(); ++i) {
        const auto r = static_cast<RegionId>(i);
        const Acc& a = acc[i];
        if (!forest.isRoot(r) || a.n < static_cast<double>(params_.shadowMinPixels)) continue;
        double varRaw = 0.0;
        double varBlur = 0.0;
        for (std::size_t k = 0; k < 4; ++k) {
            varRaw += a.q[k] / a.n - (a.s[k] / a.n) * (a.s[k] / a.n);
            varBlur += a.bq[k] / a.n - (a.bs[k] / a.n) * (a.bs[k] / a.n);
        }
        // Sombra/desfoque = variação grande (não é cor chapada) e suave (o desfoque a preserva; o ruído não).
        if (varRaw >= params_.shadowVariance && varBlur >= params_.shadowSmoothRatio * varRaw) {
            forest.stats(r).flags = static_cast<std::uint8_t>(forest.stats(r).flags | RegionFlag::Shadow);
            ++flagged;
        }
    }
    return flagged;
}

int SubPixelSegmenter::rebuildShadowSegments(PixelMap& map) const {
    RegionForest& forest = map.forest();
    const sla::Array2D<Rgba>& bl = blurred();
    const std::size_t cap = forest.capacity();
    int created = 0;
    for (std::size_t i = 0; i < cap; ++i) {
        const auto r = static_cast<RegionId>(i);
        if (!forest.isRoot(r) || forest.pixelCount(r) == 0) continue;
        if ((forest.stats(r).flags & RegionFlag::Shadow) == 0) continue;
        struct Box {
            int minX, minY, maxX, maxY;
        };
        const Box st{forest.stats(r).minX, forest.stats(r).minY, forest.stats(r).maxX, forest.stats(r).maxY};  // cópia: a caixa muda ao mover pixels
        const int bw = st.maxX - st.minX + 1;
        const int bh = st.maxY - st.minY + 1;
        const auto at = [&](int x, int y) { return static_cast<std::size_t>(y - st.minY) * static_cast<std::size_t>(bw) + static_cast<std::size_t>(x - st.minX); };
        std::vector<std::int32_t> pal(static_cast<std::size_t>(bw) * static_cast<std::size_t>(bh), -1);
        for (int y = st.minY; y <= st.maxY; ++y) {
            for (int x = st.minX; x <= st.maxX; ++x) {
                if (map.regionAtConst(x, y) == r) pal[at(x, y)] = static_cast<std::int32_t>(nearestPalette(bl(x, y)));
            }
        }
        // Componentes 4-conexas de mesma cor da paleta dentro da região.
        std::vector<std::uint8_t> seen(pal.size(), 0);
        std::vector<std::vector<std::pair<int, int>>> comps;
        std::vector<std::pair<int, int>> stack;
        for (int y = st.minY; y <= st.maxY; ++y) {
            for (int x = st.minX; x <= st.maxX; ++x) {
                if (pal[at(x, y)] < 0 || seen[at(x, y)]) continue;
                std::vector<std::pair<int, int>> comp;
                const std::int32_t label = pal[at(x, y)];
                stack.assign(1, {x, y});
                seen[at(x, y)] = 1;
                while (!stack.empty()) {
                    const auto [cx, cy] = stack.back();
                    stack.pop_back();
                    comp.emplace_back(cx, cy);
                    for (std::size_t d = 0; d < 4; ++d) {
                        const int nx = cx + kDx4[d];
                        const int ny = cy + kDy4[d];
                        if (nx < st.minX || nx > st.maxX || ny < st.minY || ny > st.maxY) continue;
                        if (pal[at(nx, ny)] != label || seen[at(nx, ny)]) continue;
                        seen[at(nx, ny)] = 1;
                        stack.emplace_back(nx, ny);
                    }
                }
                comps.push_back(std::move(comp));
            }
        }
        if (comps.size() <= 1) continue;  // já uniforme em termos de paleta
        std::stable_sort(comps.begin(), comps.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
        for (std::size_t c = 1; c < comps.size(); ++c) {  // a maior fica com o id da região
            const RegionId fresh = map.createRegion();
            for (const auto& [x, y] : comps[c]) map.reassignPixel(x, y, fresh, image_(x, y));
            forest.stats(fresh).flags = RegionFlag::Shadow;
            ++created;
        }
    }
    return created;
}

// ---------------------------------------------------------------- 4. paleta

int SubPixelSegmenter::mergeSamePaletteNeighbors(PixelMap& map) const {
    RegionForest& forest = map.forest();
    int merges = 0;
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i = 0; i < forest.capacity(); ++i) {
            RegionId cur = static_cast<RegionId>(i);
            if (!forest.isRoot(cur) || forest.pixelCount(cur) == 0) continue;
            const std::int32_t pal = forest.stats(cur).paletteIndex;
            if (pal < 0) continue;
            const std::vector<RegionId> nb(forest.neighbors(cur).begin(), forest.neighbors(cur).end());
            for (const RegionId n0 : nb) {
                const RegionId n = forest.find(n0);
                if (n == cur || forest.stats(n).paletteIndex != pal) continue;
                cur = forest.unite(cur, n);
                ++merges;
                changed = true;
            }
        }
    }
    return merges;
}

int SubPixelSegmenter::enforceMinNumPixels(PixelMap& map) const {
    RegionForest& forest = map.forest();
    int merges = 0;
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i = 0; i < forest.capacity(); ++i) {
            const auto r = static_cast<RegionId>(i);
            if (!forest.isRoot(r)) continue;
            const std::uint32_t n = forest.pixelCount(r);
            if (n == 0 || static_cast<int>(n) >= params_.minRegionPixels) continue;
            const RegionId best = closestNeighbor(forest, r, [&](RegionId q) { return forest.stats(q).color; });
            if (best == kNoRegion) continue;
            const Rgba color = forest.stats(best).color;  // a vizinha prevalece (a união pode escolher outra raiz)
            const std::int32_t pal = forest.stats(best).paletteIndex;
            const RegionId root = forest.unite(r, best);
            forest.stats(root).color = color;
            forest.stats(root).paletteIndex = pal;
            ++merges;
            changed = true;
        }
    }
    return merges;
}

int SubPixelSegmenter::splitDisconnected(PixelMap& map, const sla::Array2D<sla::Rgba8>& image) {
    const int w = map.width();
    const int h = map.height();
    struct Comp {
        RegionId root = kNoRegion;
        std::vector<std::pair<int, int>> px;
    };
    std::vector<Comp> comps;
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
            if (seen[idx]) continue;
            Comp comp;
            comp.root = map.regionAtConst(x, y);
            if (comp.root == kNoRegion) throw std::logic_error("splitDisconnected: pixel sem região");
            stack.assign(1, {x, y});
            seen[idx] = 1;
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                comp.px.emplace_back(cx, cy);
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + kDx4[d];
                    const int ny = cy + kDy4[d];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    const std::size_t nidx = static_cast<std::size_t>(ny) * static_cast<std::size_t>(w) + static_cast<std::size_t>(nx);
                    if (seen[nidx] || map.regionAtConst(nx, ny) != comp.root) continue;
                    seen[nidx] = 1;
                    stack.emplace_back(nx, ny);
                }
            }
            comps.push_back(std::move(comp));
        }
    }
    std::vector<int> keeper(map.forest().capacity(), -1);  // maior componente de cada raiz (1ª em empate)
    for (std::size_t c = 0; c < comps.size(); ++c) {
        int& k = keeper[ux(comps[c].root)];
        if (k < 0 || comps[c].px.size() > comps[static_cast<std::size_t>(k)].px.size()) k = static_cast<int>(c);
    }
    int created = 0;
    for (std::size_t c = 0; c < comps.size(); ++c) {
        if (keeper[ux(comps[c].root)] == static_cast<int>(c)) continue;
        const RegionId fresh = map.createRegion();
        for (const auto& [x, y] : comps[c].px) map.reassignPixel(x, y, fresh, image(x, y));
        ++created;
    }
    return created;
}

// ---------------------------------------------------------------- pipeline

SubPixelResult SubPixelSegmenter::run(PixelMap map, TaskContext ctx) const {
    if (map.width() != image_.width() || map.height() != image_.height()) {
        throw std::invalid_argument("SubPixelSegmenter::run: mapa com dimensões diferentes da imagem");
    }
    for (int y = 0; y < map.height(); ++y) {
        for (int x = 0; x < map.width(); ++x) {
            if (map.regionAtConst(x, y) == kNoRegion) throw std::invalid_argument("SubPixelSegmenter::run: pixel sem região");
        }
    }
    SubPixelStats stats;
    ctx.checkpoint(0.0);
    stats.microMerges = mergeMicroRegions(map);
    ctx.checkpoint(0.2);
    const BeachResult beach = splitAndMergeBeaches(map);
    stats.beaches = beach.beaches;
    stats.beachPixelsMoved = beach.moved;
    stats.beachesMerged = beach.merged;
    ctx.checkpoint(0.4);
    stats.shadowFlagged = flagShadowSegments(map);
    if (stats.shadowFlagged > 0) stats.shadowNewRegions = rebuildShadowSegments(map);
    ctx.checkpoint(0.6);
    stats.splitRegions = splitDisconnected(map, image_);  // garantia: as etapas acima preservam a 4-conexidade
    snapColorsToPalette(map, palette_);
    stats.paletteMerges = mergeSamePaletteNeighbors(map);
    stats.smallMerges = enforceMinNumPixels(map);
    ctx.checkpoint(0.9);
    stats.finalRegions = static_cast<int>(map.compact());
    SubPixelResult result;
    result.map = std::move(map);
    result.stats = stats;
    ctx.report(1.0);
    return result;
}

}  // namespace cnx
