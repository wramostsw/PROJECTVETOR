// src/segment/PixelSegmenter.cpp — QuadTree + juntar/trocar + passada diagonal.
#include "cnx/segment/PixelSegmenter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "cnx/segment/QuadTree.h"

namespace cnx {
namespace {

constexpr std::uint32_t kMeanFallbackBelow = 3;  // §4: a média só é fallback para regiões minúsculas

constexpr std::size_t ux(std::int32_t i) noexcept { return static_cast<std::size_t>(i); }

constexpr std::array<int, 4> kDx4{1, 0, -1, 0};
constexpr std::array<int, 4> kDy4{0, 1, 0, -1};

}  // namespace

// ---------------------------------------------------------------- paleta

void snapColorsToPalette(PixelMap& map, const Palette& palette) {
    if (palette.colors.empty()) throw std::invalid_argument("snapColorsToPalette: paleta vazia");
    RegionForest& forest = map.forest();
    for (std::size_t i = 0; i < forest.capacity(); ++i) {
        const auto r = static_cast<RegionId>(i);
        if (!forest.isRoot(r)) continue;
        RegionStats& s = forest.stats(r);
        if (s.samples.count() == 0) continue;
        const sla::Rgba mean = s.samples.mean();
        std::size_t best = 0;
        double bestD = sla::distance2(mean, palette.colors[0]);
        for (std::size_t k = 1; k < palette.colors.size(); ++k) {
            const double d = sla::distance2(mean, palette.colors[k]);
            if (d < bestD) {
                bestD = d;
                best = k;
            }
        }
        s.paletteIndex = static_cast<std::int32_t>(best);
        s.color = palette.colors[best];
    }
}

// ---------------------------------------------------------------- construção

PixelSegmenter::PixelSegmenter(const sla::Array2D<sla::Rgba8>& image, SegmentParams params)
    : image_(image), params_(params), map_(image.width(), image.height()) {
    (void)thresholdSchedule(params_);  // valida cedo
    if (params_.quadTreeBlock < 1) throw std::invalid_argument("PixelSegmenter: quadTreeBlock < 1");
}

PixelSegmenter::PixelSegmenter(const sla::Array2D<sla::Rgba8>& image, SegmentParams params, PixelMap initial)
    : image_(image), params_(params), map_(std::move(initial)) {
    (void)thresholdSchedule(params_);
    if (map_.width() != image.width() || map_.height() != image.height()) {
        throw std::invalid_argument("PixelSegmenter: mapa inicial com dimensões diferentes da imagem");
    }
    for (int y = 0; y < map_.height(); ++y) {
        for (int x = 0; x < map_.width(); ++x) {
            if (map_.regionAtConst(x, y) == kNoRegion) {
                throw std::invalid_argument("PixelSegmenter: mapa inicial com pixel sem região");
            }
        }
    }
    stats_.superpixels = static_cast<int>(map_.forest().rootCount());
    initialized_ = true;
}

std::vector<double> PixelSegmenter::thresholdSchedule(const SegmentParams& p) {
    if (!(p.mergeThreshStart > 0.0) || !(p.mergeThreshGrowth > 1.0) || !(p.mergeThreshMax >= p.mergeThreshStart)) {
        throw std::invalid_argument("PixelSegmenter: requer 0 < mergeThreshStart ≤ mergeThreshMax e mergeThreshGrowth > 1");
    }
    std::vector<double> t;
    for (double v = p.mergeThreshStart; v < p.mergeThreshMax; v *= p.mergeThreshGrowth) t.push_back(v);
    t.push_back(p.mergeThreshMax);
    return t;
}

// ---------------------------------------------------------------- cache de cores

sla::Rgba PixelSegmenter::regionColor(RegionId root) const {
    const MedianAccumulator& s = map_.forest().stats(root).samples;
    return s.count() < kMeanFallbackBelow ? s.mean() : sla::toRgba(s.median());
}

void PixelSegmenter::ensureCache() {
    const std::size_t n = map_.forest().capacity();
    if (color_.size() != n) {
        color_.assign(n, sla::Rgba{});
        dirty_.assign(n, 1);
    }
}

const sla::Rgba& PixelSegmenter::colorOf(RegionId root) {
    const std::size_t i = ux(root);
    if (dirty_[i]) {
        color_[i] = regionColor(root);
        dirty_[i] = 0;
    }
    return color_[i];
}

void PixelSegmenter::markDirty(RegionId root) { dirty_[ux(root)] = 1; }

bool PixelSegmenter::colorCacheConsistent() const {
    const RegionForest& forest = map_.forest();
    for (std::size_t i = 0; i < color_.size(); ++i) {
        const auto r = static_cast<RegionId>(i);
        if (dirty_[i] || !forest.isRoot(r) || forest.pixelCount(r) == 0) continue;
        if (!(color_[i] == regionColor(r))) return false;
    }
    return true;
}

void PixelSegmenter::refreshAll() {
    ensureCache();
    for (std::size_t i = 0; i < dirty_.size(); ++i) {
        const auto r = static_cast<RegionId>(i);
        if (dirty_[i] && map_.forest().isRoot(r) && map_.forest().pixelCount(r) > 0) {
            color_[i] = regionColor(r);
        }
        dirty_[i] = 0;
    }
}

// ---------------------------------------------------------------- 1. QuadTree

void PixelSegmenter::initializeFromQuadTree() {
    if (initialized_) throw std::logic_error("PixelSegmenter::initializeFromQuadTree: mapa já inicializado");
    const std::vector<QuadBlock> leaves = quadTreeSplit(image_, params_.quadTreeBlock, params_.quadTreeVariance);
    for (const QuadBlock& b : leaves) {
        const RegionId id = map_.createRegion();
        for (int y = b.y; y < b.y + b.h; ++y) {
            for (int x = b.x; x < b.x + b.w; ++x) map_.setLabel(x, y, id, image_(x, y));
        }
    }
    stats_.superpixels = static_cast<int>(leaves.size());
    initialized_ = true;
}

// ---------------------------------------------------------------- 2a. juntar

bool PixelSegmenter::whatIfAccepts(RegionId a, RegionId b, double threshold) {
    const RegionForest& forest = map_.forest();
    const MedianAccumulator& sa = forest.stats(a).samples;
    const MedianAccumulator& sb = forest.stats(b).samples;
    const sla::Rgba ca = colorOf(a);
    const sla::Rgba cb = colorOf(b);

    // Cor da união pela mesma regra de regionColor (média abaixo de 3 pixels, mediana acima).
    const std::uint32_t na = sa.count();
    const std::uint32_t nb = sb.count();
    sla::Rgba center;
    if (na + nb < kMeanFallbackBelow) {
        const double wa = static_cast<double>(na) / static_cast<double>(na + nb);
        center = sa.mean() * static_cast<float>(wa) + sb.mean() * static_cast<float>(1.0 - wa);
    } else {
        center = sla::toRgba(MedianAccumulator::unionMedian(sa, sb));
    }

    // Orçamento: erro das partes + folga proporcional ao menor lado. O aborto antecipado de
    // unionSquaredDeviation corta o cálculo assim que o acumulador passa do orçamento.
    const double budget = sa.squaredDeviation(ca) + sb.squaredDeviation(cb) +
                          params_.whatIfFactor * threshold * static_cast<double>(std::min(na, nb));
    return MedianAccumulator::unionSquaredDeviation(sa, sb, center, budget) <= budget;
}

int PixelSegmenter::joinSweep(double threshold) {
    if (!initialized_) throw std::logic_error("PixelSegmenter::joinSweep: mapa não inicializado");
    ensureCache();
    RegionForest& forest = map_.forest();
    int merges = 0;
    std::vector<std::pair<double, RegionId>> candidates;
    for (std::size_t i = 0; i < forest.capacity(); ++i) {
        RegionId cur = static_cast<RegionId>(i);
        if (!forest.isRoot(cur) || forest.pixelCount(cur) == 0) continue;
        for (;;) {
            candidates.clear();
            const sla::Rgba cc = colorOf(cur);
            for (const RegionId n : forest.neighbors(cur)) {
                const double d = sla::distance2(cc, colorOf(n));
                if (d <= threshold) candidates.emplace_back(d, n);  // rejeição barata antes do "e se"
            }
            std::sort(candidates.begin(), candidates.end());  // mais parecidas primeiro; empate → menor id
            bool merged = false;
            for (const auto& [d, n] : candidates) {
                (void)d;
                if (!whatIfAccepts(cur, n, threshold)) continue;
                cur = forest.unite(cur, n);
                markDirty(cur);
                ++merges;
                merged = true;
                break;
            }
            if (!merged) break;
        }
    }
    stats_.joins += merges;
    return merges;
}

// ---------------------------------------------------------------- 2b. trocar

int PixelSegmenter::swapSweep() {
    if (!initialized_) throw std::logic_error("PixelSegmenter::swapSweep: mapa não inicializado");
    refreshAll();
    int moves = 0;
    const int w = map_.width();
    const int h = map_.height();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const RegionId own = map_.regionAt(x, y);
            const sla::Rgba c = sla::toRgba(image_(x, y));
            double bestD = sla::distance2(c, color_[ux(own)]);
            if (bestD == 0.0) continue;
            RegionId best = kNoRegion;
            for (std::size_t d = 0; d < 4; ++d) {
                const int qx = x + kDx4[d];
                const int qy = y + kDy4[d];
                if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
                const RegionId s = map_.regionAt(qx, qy);
                if (s == own) continue;
                const double ds = sla::distance2(c, color_[ux(s)]);
                if (ds < bestD) {
                    bestD = ds;
                    best = s;
                }
            }
            if (best != kNoRegion && map_.tryMovePixel(x, y, best, image_(x, y))) {
                markDirty(own);
                markDirty(best);
                ++moves;
            }
        }
    }
    refreshAll();
    stats_.swaps += moves;
    return moves;
}

// ---------------------------------------------------------------- 3. passada diagonal

bool PixelSegmenter::saddleAt(int x, int y) {
    const RegionId r00 = map_.regionAt(x, y);
    const RegionId r10 = map_.regionAt(x + 1, y);
    const RegionId r01 = map_.regionAt(x, y + 1);
    const RegionId r11 = map_.regionAt(x + 1, y + 1);
    // Uma diagonal da mesma região e as outras duas células diferentes dela: a região toca a si
    // mesma só pela quina. (Num mapa válido as duas diagonais nunca são ambas "iguais": o caminho
    // 4-conexo de uma delas separaria as células da outra.)
    return (r00 == r11 && r10 != r00 && r01 != r00) || (r10 == r01 && r00 != r10 && r11 != r10);
}

int PixelSegmenter::countSaddles() {
    int n = 0;
    for (int y = 0; y + 1 < map_.height(); ++y) {
        for (int x = 0; x + 1 < map_.width(); ++x) n += saddleAt(x, y) ? 1 : 0;
    }
    return n;
}

int PixelSegmenter::diagonalSweep() {
    if (!initialized_) throw std::logic_error("PixelSegmenter::diagonalSweep: mapa não inicializado");
    refreshAll();
    int moves = 0;
    struct Move {
        double cost;
        int x, y;
        RegionId to;
    };
    // Cada célula da janela tem dois 4-vizinhos dentro dela: (dx, dy) de cada um, por célula.
    // Células na ordem 00, 10, 01, 11.
    static constexpr std::array<std::array<std::array<int, 2>, 2>, 4> kWindowNeighbors{{
        {{{1, 0}, {0, 1}}}, {{{0, 0}, {1, 1}}}, {{{0, 0}, {1, 1}}}, {{{1, 0}, {0, 1}}}}};
    static constexpr std::array<std::array<int, 2>, 4> kWindowCell{{{0, 0}, {1, 0}, {0, 1}, {1, 1}}};
    std::vector<Move> cand;
    for (int y = 0; y + 1 < map_.height(); ++y) {
        for (int x = 0; x + 1 < map_.width(); ++x) {
            if (!saddleAt(x, y)) continue;
            // Qualquer movimento de uma célula para a região de um 4-vizinho dentro da janela
            // desfaz a sela (a quina passa a ter 3 regiões). Custo = quanto o pixel se afasta da cor
            // da sua região ao ir para a nova: vence a ligação de cores mais parecidas (§4).
            cand.clear();
            for (std::size_t k = 0; k < 4; ++k) {
                const int px = x + kWindowCell[k][0];
                const int py = y + kWindowCell[k][1];
                const RegionId own = map_.regionAt(px, py);
                const sla::Rgba c = sla::toRgba(image_(px, py));
                RegionId previous = kNoRegion;
                for (const auto& d : kWindowNeighbors[k]) {
                    const RegionId to = map_.regionAt(x + d[0], y + d[1]);
                    if (to == own || to == previous) continue;
                    previous = to;
                    cand.push_back({sla::distance2(c, color_[ux(to)]) - sla::distance2(c, color_[ux(own)]), px, py, to});
                }
            }
            std::stable_sort(cand.begin(), cand.end(), [](const Move& a, const Move& b) { return a.cost < b.cost; });
            for (const Move& m : cand) {
                const RegionId own = map_.regionAt(m.x, m.y);
                if (map_.tryMovePixel(m.x, m.y, m.to, image_(m.x, m.y))) {
                    markDirty(own);
                    markDirty(m.to);
                    ++moves;
                    break;
                }
            }
        }
    }
    refreshAll();
    stats_.diagonalMoves += moves;
    return moves;
}

// ---------------------------------------------------------------- enforceMinNumPixels

int PixelSegmenter::mergeSmallRegions() {
    if (!initialized_) throw std::logic_error("PixelSegmenter::mergeSmallRegions: mapa não inicializado");
    ensureCache();
    RegionForest& forest = map_.forest();
    int total = 0;
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i = 0; i < forest.capacity(); ++i) {
            const auto r = static_cast<RegionId>(i);
            if (!forest.isRoot(r) || forest.pixelCount(r) == 0) continue;
            if (static_cast<int>(forest.pixelCount(r)) >= params_.minRegionPixels) continue;
            const sla::Rgba c = colorOf(r);
            RegionId best = kNoRegion;
            double bestD = std::numeric_limits<double>::infinity();
            std::uint32_t bestContact = 0;
            for (const RegionId n : forest.neighbors(r)) {
                const double d = sla::distance2(c, colorOf(n));
                const std::uint32_t contact = forest.contactCount(r, n);
                if (d < bestD || (d == bestD && contact > bestContact)) {
                    bestD = d;
                    bestContact = contact;
                    best = n;
                }
            }
            if (best == kNoRegion) continue;  // região única da imagem
            markDirty(forest.unite(r, best));
            ++total;
            changed = true;
        }
    }
    stats_.smallMerges += total;
    return total;
}

// ---------------------------------------------------------------- fim e pipeline

PixelMap PixelSegmenter::finish(const Palette* palette) {
    if (!initialized_) throw std::logic_error("PixelSegmenter::finish: mapa não inicializado");
    map_.assignMedianColors(kMeanFallbackBelow);
    if (palette != nullptr) snapColorsToPalette(map_, *palette);
    stats_.finalRegions = static_cast<int>(map_.compact());
    return std::move(map_);
}

SegmentationResult PixelSegmenter::run(TaskContext ctx, const Palette* palette) {
    if (!initialized_) initializeFromQuadTree();
    const std::vector<double> schedule = thresholdSchedule(params_);
    const double total = static_cast<double>(schedule.size()) + 2.0;

    // Juntar e trocar com limiar crescente: regiões quase idênticas se unem primeiro.
    for (std::size_t k = 0; k < schedule.size(); ++k) {
        ctx.checkpoint(static_cast<double>(k) / total);
        joinSweep(schedule[k]);
        swapSweep();
        ++stats_.rounds;
    }
    // No teto, repete até estabilizar.
    bool stable = false;
    for (int k = 0; k < params_.finalSweepCap && !stable; ++k) {
        ctx.throwIfCancelled();
        const int j = joinSweep(schedule.back());
        const int s = swapSweep();
        stable = j == 0 && s == 0;
    }
    bool settled = stable;
    // Regiões minúsculas e selas se influenciam: alterna até nenhuma das duas mudar.
    ctx.checkpoint(static_cast<double>(schedule.size()) / total);
    bool diagonalDone = false;
    for (int k = 0; k < params_.finalSweepCap && !diagonalDone; ++k) {
        ctx.throwIfCancelled();
        const int m = mergeSmallRegions();
        const int d = diagonalSweep();
        diagonalDone = m == 0 && d == 0;
    }
    // Trocas e diagonais podem deixar um mapa com selas só por falta de movimento simples: isso
    // não é falha de convergência; stable só cai se o laço acima acabou pelo teto.
    stats_.stable = settled && diagonalDone;
    ctx.checkpoint((static_cast<double>(schedule.size()) + 1.0) / total);
    SegmentationResult result;
    result.map = finish(palette);
    result.stats = stats_;
    ctx.report(1.0);
    return result;
}

}  // namespace cnx
