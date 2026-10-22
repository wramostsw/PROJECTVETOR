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

/// Cor de uma região: mediana por canal (média abaixo de 3 pixels).
Rgba regionColor(const RegionForest& forest, RegionId root) {
    const MedianAccumulator& s = forest.stats(root).samples;
    return s.count() < kMeanFallbackBelow ? s.mean() : sla::toRgba(s.median());
}

/// Cor de uma região, em cache até a região mudar.
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
            color_[i] = regionColor(map_.forest(), root);
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

/// Raiz da região de cada pixel (índice y·w + x).
std::vector<RegionId> rootLabels(const PixelMap& map) {
    std::vector<RegionId> labels(static_cast<std::size_t>(map.width()) * static_cast<std::size_t>(map.height()));
    for (int y = 0; y < map.height(); ++y) {
        for (int x = 0; x < map.width(); ++x) labels[static_cast<std::size_t>(y) * static_cast<std::size_t>(map.width()) + static_cast<std::size_t>(x)] = map.regionAtConst(x, y);
    }
    return labels;
}

/// Média da imagem na janela (2·radius+1)² em torno de (x, y), só sobre os pixels da região `r`: o desfoque não
/// mistura a tinta da região com a das vizinhas (nas bordas, uma média comum cria cores que não existem em lugar nenhum).
Rgba maskedBlur(const sla::Array2D<sla::Rgba8>& image, const std::vector<RegionId>& labels, int x, int y, int radius, RegionId r) {
    const int w = image.width();
    const int h = image.height();
    std::array<double, 4> s{};
    double n = 0.0;
    for (int j = std::max(0, y - radius); j <= std::min(h - 1, y + radius); ++j) {
        for (int i = std::max(0, x - radius); i <= std::min(w - 1, x + radius); ++i) {
            if (labels[static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + static_cast<std::size_t>(i)] != r) continue;
            const Rgba c = sla::toRgba(image(i, j));
            s[0] += c.r;
            s[1] += c.g;
            s[2] += c.b;
            s[3] += c.a;
            n += 1.0;
        }
    }
    if (n == 0.0) return sla::toRgba(image(x, y));
    return {static_cast<float>(s[0] / n), static_cast<float>(s[1] / n), static_cast<float>(s[2] / n), static_cast<float>(s[3] / n)};
}

/// Quadrado da distância de p ao segmento a–b.
double segmentDistance2(const Rgba& p, const Rgba& a, const Rgba& b) {
    const Rgba v = b - a;
    const double vv = sla::dot(v, v);
    const double t = vv > 0.0 ? std::clamp(static_cast<double>(sla::dot(p - a, v)) / vv, 0.0, 1.0) : 0.0;
    const Rgba off = (p - a) - v * static_cast<float>(t);
    return sla::dot(off, off);
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

std::vector<std::size_t> SubPixelSegmenter::shadowInks(RegionForest& forest, RegionId r, const std::vector<Rgba>& colors) const {
    // As tintas de uma região de sombra são as das regiões que ela toca: um tom intermediário entre duas delas é
    // mistura (a sombra ou o halo de uma borda), não uma terceira tinta que por acaso fica mais perto em RGB.
    std::vector<std::size_t> inks;
    for (const RegionId n : forest.neighbors(r)) {
        const std::size_t k = nearestPalette(regionColor(forest, n));
        if (std::find(inks.begin(), inks.end(), k) == inks.end()) inks.push_back(k);
    }
    if (inks.empty()) {  // região única: sem vizinhas para dizer quais são as tintas
        for (std::size_t k = 0; k < palette_.colors.size(); ++k) inks.push_back(k);
        return inks;
    }
    if (inks.size() < 2) return inks;
    // Com mais de duas vizinhas, a sombra vai de uma tinta a outra: fica o par de extremos cuja reta melhor ajusta as
    // cores da região. Sem isto, uma vizinha que só toca a borda (o vermelho debaixo de um degradê branco–preto) entra
    // na disputa, e o cinza do meio, que de fato está mais perto do vermelho em RGB, vira uma faixa vermelha.
    if (inks.size() > 2) {
        double bestFit = std::numeric_limits<double>::infinity();
        std::size_t bi = 0;
        std::size_t bj = 1;
        for (std::size_t i = 0; i < inks.size(); ++i) {
            for (std::size_t j = i + 1; j < inks.size(); ++j) {
                double fit = 0.0;
                for (const Rgba& c : colors) fit += segmentDistance2(c, palette_.colors[inks[i]], palette_.colors[inks[j]]);
                if (fit < bestFit) {
                    bestFit = fit;
                    bi = i;
                    bj = j;
                }
            }
        }
        inks = {inks[bi], inks[bj]};
    }
    // Uma cor da paleta que está sobre a reta entre os dois extremos é uma mistura legítima deles (o cinza de uma
    // sombra de preto sobre branco) e também vale; a mesma tolerância da praia (`beachLineTolerance`). Uma cor fora da
    // reta (o amarelo entre o vermelho e o branco) é uma terceira tinta e fica de fora.
    const Rgba& a = palette_.colors[inks[0]];
    const Rgba& b = palette_.colors[inks[1]];
    const double tol = params_.beachLineTolerance;
    for (std::size_t k = 0; k < palette_.colors.size(); ++k) {
        if (k == inks[0] || k == inks[1]) continue;
        if (segmentDistance2(palette_.colors[k], a, b) <= tol * tol * sla::distance2(a, b)) inks.push_back(k);
    }
    return inks;
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
    const std::vector<RegionId> labels = rootLabels(map);
    const int w = map.width();
    const int h = map.height();
    const int radius = std::max(0, params_.shadowBlurRadius);
    struct Acc {
        double n = 0.0;
        std::array<double, 4> s{}, q{}, bs{}, bq{};
    };
    std::vector<Acc> acc(forest.capacity());
    const auto at = [&](int x, int y) { return static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x); };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const RegionId r = labels[at(x, y)];
            if (r == kNoRegion) continue;
            // Só pixels interiores (os 4 vizinhos na mesma região): o pixel de borda de uma região é a mistura com a
            // vizinha (o halo que as praias entregam a cada lado), e sozinho dá a um fundo chapado a variância de um gradiente.
            bool interior = true;
            for (std::size_t d = 0; d < 4 && interior; ++d) {
                const int qx = x + kDx4[d];
                const int qy = y + kDy4[d];
                interior = qx >= 0 && qy >= 0 && qx < w && qy < h && labels[at(qx, qy)] == r;
            }
            if (!interior) continue;
            Acc& a = acc[ux(r)];
            const Rgba c = sla::toRgba(image_(x, y));
            const Rgba b = maskedBlur(image_, labels, x, y, radius, r);
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
    // Os pixels de uma região só mudam de dono quando a própria região é reconstruída, então o mapa de rótulos do
    // início serve para todas.
    const std::vector<RegionId> labels = rootLabels(map);
    const int radius = std::max(0, params_.shadowBlurRadius);
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
        std::vector<Rgba> blurred(pal.size());
        std::vector<Rgba> colors;
        for (int y = st.minY; y <= st.maxY; ++y) {
            for (int x = st.minX; x <= st.maxX; ++x) {
                if (map.regionAtConst(x, y) != r) continue;
                blurred[at(x, y)] = maskedBlur(image_, labels, x, y, radius, r);
                colors.push_back(blurred[at(x, y)]);
            }
        }
        const std::vector<std::size_t> inks = shadowInks(forest, r, colors);
        for (int y = st.minY; y <= st.maxY; ++y) {
            for (int x = st.minX; x <= st.maxX; ++x) {
                if (map.regionAtConst(x, y) != r) continue;
                const Rgba c = blurred[at(x, y)];
                std::size_t best = inks.front();
                double bestD = sla::distance2(c, palette_.colors[best]);
                for (std::size_t k = 1; k < inks.size(); ++k) {
                    const double d = sla::distance2(c, palette_.colors[inks[k]]);
                    if (d < bestD) {
                        bestD = d;
                        best = inks[k];
                    }
                }
                pal[at(x, y)] = static_cast<std::int32_t>(best);
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
