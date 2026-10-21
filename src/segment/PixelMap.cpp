// src/segment/PixelMap.cpp — MedianAccumulator, RegionForest (Union-Find + contatos) e PixelMap
// (atribuição, troca com proteção topológica, compactação).
#include "cnx/segment/PixelMap.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cnx {
namespace {

constexpr std::size_t kBins = 256;
constexpr std::size_t kSmallSortLimit = 32;  ///< median(): até aqui ordena as amostras em vez de varrer os bins

constexpr std::size_t ux(std::int32_t i) noexcept { return static_cast<std::size_t>(i); }

constexpr std::uint8_t channel(sla::Rgba8 c, std::size_t ch) noexcept {
    switch (ch) {
        case 0: return c.r;
        case 1: return c.g;
        case 2: return c.b;
        default: return c.a;
    }
}

/// Menor valor v cuja contagem acumulada passa de k: com k = ⌊(n−1)/2⌋, a mediana inferior.
std::uint8_t lowerMedian(const std::uint32_t* bins, std::uint32_t k) noexcept {
    std::uint64_t cumulative = 0;
    for (std::size_t v = 0; v < kBins; ++v) {
        cumulative += bins[v];
        if (cumulative > k) return static_cast<std::uint8_t>(v);
    }
    return 255;
}

RegionId toRegionId(std::size_t index) {
    if (index > static_cast<std::size_t>(std::numeric_limits<RegionId>::max())) {
        throw std::length_error("cnx: excesso de regiões para ids de 32 bits");
    }
    return static_cast<RegionId>(index);
}

// ---------------------------------------------------------------- tabela de pontos simples
// N8*(p) em ordem circular: E, NE, N, NW, W, SW, S, SE. Índices pares são os 4-vizinhos.
constexpr std::array<int, 8> kRingDx{1, 1, 0, -1, -1, -1, 0, 1};
constexpr std::array<int, 8> kRingDy{0, -1, -1, -1, 0, 1, 1, 1};

constexpr bool ringAdjacent(int i, int j, bool eight) noexcept {
    const int dx = kRingDx[ux(i)] - kRingDx[ux(j)];
    const int dy = kRingDy[ux(i)] - kRingDy[ux(j)];
    const int ax = dx < 0 ? -dx : dx;
    const int ay = dy < 0 ? -dy : dy;
    return eight ? (ax <= 1 && ay <= 1 && ax + ay > 0) : (ax + ay == 1);
}

/// Componentes de `members` (máscara sobre o anel) na 4- ou 8-adjacência. Com
/// `onlyTouchingCenter4`, conta só as que contêm um 4-vizinho de p (definição de T4).
constexpr int ringComponents(unsigned members, bool eight, bool onlyTouchingCenter4) noexcept {
    unsigned seen = 0;
    int count = 0;
    for (int s = 0; s < 8; ++s) {
        if (((members >> s) & 1u) == 0 || ((seen >> s) & 1u) != 0) continue;
        std::array<int, 8> stack{};
        int top = 0;
        stack[ux(top++)] = s;
        seen |= 1u << s;
        bool touches = false;
        while (top > 0) {
            const int i = stack[ux(--top)];
            touches = touches || i % 2 == 0;
            for (int j = 0; j < 8; ++j) {
                if (((members >> j) & 1u) != 0 && ((seen >> j) & 1u) == 0 && ringAdjacent(i, j, eight)) {
                    seen |= 1u << j;
                    stack[ux(top++)] = j;
                }
            }
        }
        if (!onlyTouchingCenter4 || touches) ++count;
    }
    return count;
}

/// kSimple[m]: p é simples quando R ∩ N8*(p) tem a máscara m (bit k = vizinho k em R).
constexpr std::array<bool, 256> makeSimpleTable() noexcept {
    std::array<bool, 256> table{};
    for (unsigned m = 0; m < 256; ++m) {
        const int t4 = ringComponents(m, false, true);         // T4(p, R)
        const int t8 = ringComponents(~m & 0xFFu, true, false);  // T8(p, R̄)
        table[m] = t4 == 1 && t8 == 1;
    }
    return table;
}

constexpr std::array<bool, 256> kSimple = makeSimpleTable();

static_assert(!kSimple[0x00], "pixel isolado: removê-lo apaga uma componente");
static_assert(!kSimple[0xFF], "pixel interior: removê-lo cria um furo");
static_assert(kSimple[0x01], "ponta de linha (só o vizinho E)");
static_assert(!kSimple[0x11], "meio de linha horizontal (E e W): removê-lo divide a região");
static_assert(kSimple[0xC1], "canto de bloco 2×2 (E, SE, S)");
static_assert(!kSimple[0x02], "só o vizinho diagonal NE: não há 4-vizinho");

constexpr std::array<int, 4> kDx4{1, 0, -1, 0};
constexpr std::array<int, 4> kDy4{0, 1, 0, -1};

}  // namespace

// ================================================================ MedianAccumulator

void MedianAccumulator::add(sla::Rgba8 c) {
    if (count_ == std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("MedianAccumulator: excesso de amostras");
    ++count_;
    for (std::size_t ch = 0; ch < 4; ++ch) sum_[ch] += channel(c, ch);
    if (!histogram_.empty()) {
        for (std::size_t ch = 0; ch < 4; ++ch) ++histogram_[ch * kBins + channel(c, ch)];
        return;
    }
    samples_.push_back(c);
    if (samples_.size() > kSwitchToHistogram) promoteToHistogram();
}

void MedianAccumulator::remove(sla::Rgba8 c) {
    if (count_ == 0) throw std::logic_error("MedianAccumulator::remove: acumulador vazio");
    if (!histogram_.empty()) {
        for (std::size_t ch = 0; ch < 4; ++ch) {
            if (histogram_[ch * kBins + channel(c, ch)] == 0) throw std::logic_error("MedianAccumulator::remove: cor não registrada");
        }
        for (std::size_t ch = 0; ch < 4; ++ch) --histogram_[ch * kBins + channel(c, ch)];
    } else {
        const auto it = std::find(samples_.begin(), samples_.end(), c);
        if (it == samples_.end()) throw std::logic_error("MedianAccumulator::remove: cor não registrada");
        *it = samples_.back();
        samples_.pop_back();
    }
    --count_;
    for (std::size_t ch = 0; ch < 4; ++ch) sum_[ch] -= channel(c, ch);
}

void MedianAccumulator::merge(MedianAccumulator&& other) {
    if (&other == this) throw std::invalid_argument("MedianAccumulator::merge: fusão consigo mesmo");
    if (other.count_ == 0) return;
    if (std::numeric_limits<std::uint32_t>::max() - count_ < other.count_) {
        throw std::overflow_error("MedianAccumulator::merge: excesso de amostras");
    }
    if (histogram_.empty() && !other.histogram_.empty()) {
        // Adota o histograma do outro e despeja nele as amostras deste.
        const std::vector<sla::Rgba8> mine = std::move(samples_);
        samples_ = {};
        histogram_ = std::move(other.histogram_);
        for (const sla::Rgba8 s : mine) {
            for (std::size_t ch = 0; ch < 4; ++ch) ++histogram_[ch * kBins + channel(s, ch)];
        }
    } else if (!histogram_.empty()) {
        if (!other.histogram_.empty()) {
            for (std::size_t i = 0; i < histogram_.size(); ++i) histogram_[i] += other.histogram_[i];
        } else {
            for (const sla::Rgba8 s : other.samples_) {
                for (std::size_t ch = 0; ch < 4; ++ch) ++histogram_[ch * kBins + channel(s, ch)];
            }
        }
    } else {
        samples_.insert(samples_.end(), other.samples_.begin(), other.samples_.end());
        if (samples_.size() > kSwitchToHistogram) promoteToHistogram();
    }
    count_ += other.count_;
    for (std::size_t ch = 0; ch < 4; ++ch) sum_[ch] += other.sum_[ch];
    other = MedianAccumulator{};
}

sla::Rgba8 MedianAccumulator::median() const {
    if (count_ == 0) throw std::logic_error("MedianAccumulator::median: acumulador vazio");
    const std::uint32_t k = (count_ - 1) / 2;
    std::array<std::uint8_t, 4> m{};
    if (!histogram_.empty()) {
        for (std::size_t ch = 0; ch < 4; ++ch) m[ch] = lowerMedian(histogram_.data() + ch * kBins, k);
    } else if (samples_.size() <= kSmallSortLimit) {
        // Regiões minúsculas (superpixels 1×1, 2×2…) dominam a contagem: ordenar ≤ 32 valores é
        // bem mais barato que varrer 4 × 256 contadores. Mesma definição: posição ⌊(n−1)/2⌋.
        std::array<std::uint8_t, kSmallSortLimit> v{};
        for (std::size_t ch = 0; ch < 4; ++ch) {
            for (std::size_t i = 0; i < samples_.size(); ++i) v[i] = channel(samples_[i], ch);
            std::sort(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(samples_.size()));
            m[ch] = v[k];
        }
    } else {
        std::array<std::uint32_t, kBins> bins{};
        for (std::size_t ch = 0; ch < 4; ++ch) {
            bins.fill(0);
            for (const sla::Rgba8 s : samples_) ++bins[channel(s, ch)];
            m[ch] = lowerMedian(bins.data(), k);
        }
    }
    return {m[0], m[1], m[2], m[3]};
}

sla::Rgba MedianAccumulator::mean() const noexcept {
    if (count_ == 0) return {};
    const double inv = 1.0 / (255.0 * static_cast<double>(count_));
    return {static_cast<float>(static_cast<double>(sum_[0]) * inv), static_cast<float>(static_cast<double>(sum_[1]) * inv),
            static_cast<float>(static_cast<double>(sum_[2]) * inv), static_cast<float>(static_cast<double>(sum_[3]) * inv)};
}

void MedianAccumulator::addToBins(std::uint32_t* bins) const {
    if (!histogram_.empty()) {
        for (std::size_t i = 0; i < histogram_.size(); ++i) bins[i] += histogram_[i];
        return;
    }
    for (const sla::Rgba8 s : samples_) {
        for (std::size_t ch = 0; ch < 4; ++ch) ++bins[ch * kBins + channel(s, ch)];
    }
}

sla::Rgba8 MedianAccumulator::unionMedian(const MedianAccumulator& a, const MedianAccumulator& b) {
    if (a.count_ == 0 && b.count_ == 0) throw std::logic_error("MedianAccumulator::unionMedian: ambos vazios");
    if (std::numeric_limits<std::uint32_t>::max() - a.count_ < b.count_) {
        throw std::overflow_error("MedianAccumulator::unionMedian: excesso de amostras");
    }
    std::array<std::uint32_t, 4 * kBins> bins{};
    a.addToBins(bins.data());
    b.addToBins(bins.data());
    const std::uint32_t k = (a.count_ + b.count_ - 1) / 2;
    return {lowerMedian(bins.data(), k), lowerMedian(bins.data() + kBins, k), lowerMedian(bins.data() + 2 * kBins, k),
            lowerMedian(bins.data() + 3 * kBins, k)};
}

double MedianAccumulator::squaredDeviation(const sla::Rgba& center, double abortAbove) const {
    constexpr double inv = 1.0 / 255.0;
    const std::array<double, 4> c{center.r, center.g, center.b, center.a};
    double acc = 0.0;
    if (!histogram_.empty()) {
        for (std::size_t ch = 0; ch < 4; ++ch) {
            for (std::size_t v = 0; v < kBins; ++v) {
                const std::uint32_t n = histogram_[ch * kBins + v];
                if (n == 0) continue;
                const double d = static_cast<double>(v) * inv - c[ch];
                acc += static_cast<double>(n) * d * d;
                if (acc > abortAbove) return acc;
            }
        }
        return acc;
    }
    for (const sla::Rgba8 s : samples_) {
        for (std::size_t ch = 0; ch < 4; ++ch) {
            const double d = static_cast<double>(channel(s, ch)) * inv - c[ch];
            acc += d * d;
        }
        if (acc > abortAbove) return acc;
    }
    return acc;
}

double MedianAccumulator::unionSquaredDeviation(const MedianAccumulator& a, const MedianAccumulator& b,
                                                const sla::Rgba& center, double abortAbove) {
    const double first = a.squaredDeviation(center, abortAbove);
    if (first > abortAbove) return first;
    return first + b.squaredDeviation(center, abortAbove - first);
}

void MedianAccumulator::promoteToHistogram() {
    histogram_.assign(4 * kBins, 0);
    for (const sla::Rgba8 s : samples_) {
        for (std::size_t ch = 0; ch < 4; ++ch) ++histogram_[ch * kBins + channel(s, ch)];
    }
    samples_ = {};
}

// ================================================================ RegionForest

RegionForest::RegionForest(std::size_t reserve) {
    parent_.reserve(reserve);
    size_.reserve(reserve);
    stats_.reserve(reserve);
    neighbors_.reserve(reserve);
}

RegionId RegionForest::create() {
    const RegionId id = toRegionId(parent_.size());
    parent_.push_back(id);
    size_.push_back(0);
    stats_.emplace_back();
    neighbors_.emplace_back();
    ++roots_;
    return id;
}

void RegionForest::checkId(RegionId r, const char* where) const {
    if (r < 0 || ux(r) >= parent_.size()) {
        throw std::out_of_range(std::string(where) + ": região " + std::to_string(r) + " inexistente");
    }
}

RegionId RegionForest::unite(RegionId a, RegionId b) {
    checkId(a, "RegionForest::unite");
    checkId(b, "RegionForest::unite");
    RegionId root = find(a);
    RegionId child = find(b);
    if (root == child) return root;
    if (size_[ix(child)] > size_[ix(root)] || (size_[ix(child)] == size_[ix(root)] && child < root)) std::swap(root, child);

    parent_[ix(child)] = root;
    size_[ix(root)] += size_[ix(child)];
    --roots_;

    RegionStats& rs = stats_[ix(root)];
    RegionStats& cs = stats_[ix(child)];
    rs.samples.merge(std::move(cs.samples));
    rs.minX = std::min(rs.minX, cs.minX);
    rs.minY = std::min(rs.minY, cs.minY);
    rs.maxX = std::max(rs.maxX, cs.maxX);
    rs.maxY = std::max(rs.maxY, cs.maxY);
    rs.flags = static_cast<std::uint8_t>(rs.flags | cs.flags);

    // Re-indexa os contatos da raiz absorvida: (child, X) → (root, X); (child, root) é interno.
    std::vector<RegionId> childNeighbors = std::move(neighbors_[ix(child)]);
    neighbors_[ix(child)] = {};
    for (const RegionId entry : childNeighbors) {
        const RegionId x = find(entry);
        if (x == root) {
            contacts_.erase(contactKey(child, root));
            continue;
        }
        const auto it = contacts_.find(contactKey(child, x));
        if (it == contacts_.end()) continue;  // entrada duplicada ou contato zerado
        const std::uint32_t n = it->second;
        contacts_.erase(it);
        contacts_[contactKey(root, x)] += n;
    }
    contacts_.erase(contactKey(child, root));  // caso a lista do filho não tivesse a raiz
    std::vector<RegionId>& rootNeighbors = neighbors_[ix(root)];
    rootNeighbors.insert(rootNeighbors.end(), childNeighbors.begin(), childNeighbors.end());
    return root;
}

void RegionForest::addContacts(RegionId a, RegionId b, std::uint32_t n) {
    checkId(a, "RegionForest::addContacts");
    checkId(b, "RegionForest::addContacts");
    const RegionId ra = find(a);
    const RegionId rb = find(b);
    if (ra == rb || n == 0) return;
    std::uint32_t& count = contacts_[contactKey(ra, rb)];
    if (count == 0) {  // novo par: entra nas listas preguiçosas dos dois lados
        neighbors_[ix(ra)].push_back(rb);
        neighbors_[ix(rb)].push_back(ra);
    }
    count += n;
}

void RegionForest::removeContacts(RegionId a, RegionId b, std::uint32_t n) {
    checkId(a, "RegionForest::removeContacts");
    checkId(b, "RegionForest::removeContacts");
    const RegionId ra = find(a);
    const RegionId rb = find(b);
    if (ra == rb || n == 0) return;
    const auto it = contacts_.find(contactKey(ra, rb));
    if (it == contacts_.end() || it->second < n) throw std::logic_error("RegionForest::removeContacts: contagem insuficiente");
    it->second -= n;
    if (it->second == 0) contacts_.erase(it);
}

std::uint32_t RegionForest::contactCount(RegionId a, RegionId b) const {
    checkId(a, "RegionForest::contactCount");
    checkId(b, "RegionForest::contactCount");
    const RegionId ra = findNoCompress(a);
    const RegionId rb = findNoCompress(b);
    if (ra == rb) return 0;
    const auto it = contacts_.find(contactKey(ra, rb));
    return it == contacts_.end() ? 0 : it->second;
}

std::span<const RegionId> RegionForest::neighbors(RegionId region) {
    checkId(region, "RegionForest::neighbors");
    const RegionId r = find(region);
    std::vector<RegionId>& list = neighbors_[ix(r)];
    for (RegionId& e : list) e = find(e);
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    list.erase(std::remove_if(list.begin(), list.end(),
                              [&](RegionId e) { return e == r || contacts_.find(contactKey(r, e)) == contacts_.end(); }),
               list.end());
    return list;
}

// ================================================================ PixelMap

PixelMap::PixelMap(int width, int height) : labels_(width, height, kNoRegion) {}

void PixelMap::setLabel(int x, int y, RegionId region, sla::Rgba8 color) {
    if (!labels_.inBounds(x, y)) throw std::out_of_range("PixelMap::setLabel: pixel fora da imagem");
    forest_.checkId(region, "PixelMap::setLabel");
    if (labels_(x, y) != kNoRegion) throw std::logic_error("PixelMap::setLabel: pixel já tem região (use tryMovePixel)");
    const RegionId r = forest_.find(region);
    labels_(x, y) = r;
    RegionStats& s = forest_.stats(r);
    s.samples.add(color);
    s.minX = std::min(s.minX, x);
    s.minY = std::min(s.minY, y);
    s.maxX = std::max(s.maxX, x);
    s.maxY = std::max(s.maxY, y);
    ++forest_.size_[ux(r)];
    for (std::size_t d = 0; d < 4; ++d) {
        const int qx = x + kDx4[d];
        const int qy = y + kDy4[d];
        if (!labels_.inBounds(qx, qy)) continue;
        const RegionId q = regionAt(qx, qy);
        if (q != kNoRegion && q != r) forest_.addContacts(r, q, 1);
    }
    compact_ = false;
}

bool PixelMap::isSimplePoint(int x, int y) const noexcept {
    if (!labels_.inBounds(x, y)) return false;
    const RegionId r = regionAtConst(x, y);
    if (r == kNoRegion) return false;
    unsigned mask = 0;
    for (std::size_t k = 0; k < 8; ++k) {
        const int qx = x + kRingDx[k];
        const int qy = y + kRingDy[k];
        if (labels_.inBounds(qx, qy) && regionAtConst(qx, qy) == r) mask |= 1u << k;
    }
    return kSimple[mask];
}

bool PixelMap::tryMovePixel(int x, int y, RegionId to, sla::Rgba8 color) {
    if (!labels_.inBounds(x, y)) throw std::out_of_range("PixelMap::tryMovePixel: pixel fora da imagem");
    forest_.checkId(to, "PixelMap::tryMovePixel");
    const RegionId from = regionAt(x, y);
    if (from == kNoRegion) throw std::logic_error("PixelMap::tryMovePixel: pixel sem região");
    const RegionId target = forest_.find(to);
    if (target == from) throw std::invalid_argument("PixelMap::tryMovePixel: o pixel já pertence à região de destino");

    std::array<RegionId, 4> around{kNoRegion, kNoRegion, kNoRegion, kNoRegion};
    bool touchesTarget = false;
    for (std::size_t d = 0; d < 4; ++d) {
        const int qx = x + kDx4[d];
        const int qy = y + kDy4[d];
        if (!labels_.inBounds(qx, qy)) continue;
        around[d] = regionAt(qx, qy);
        touchesTarget = touchesTarget || around[d] == target;
    }
    if (!touchesTarget) throw std::invalid_argument("PixelMap::tryMovePixel: destino não é 4-vizinho do pixel");
    if (!isSimplePoint(x, y)) return false;
    moveUnchecked(x, y, from, target, color, around);
    return true;
}

void PixelMap::moveUnchecked(int x, int y, RegionId from, RegionId target, sla::Rgba8 color,
                             const std::array<RegionId, 4>& around) {
    // Contatos: cada 4-vizinho q (região Q) deixa de tocar `from` por p e passa a tocar `target`.
    for (const RegionId q : around) {
        if (q == kNoRegion) continue;
        if (q != from) forest_.removeContacts(from, q, 1);
        if (q != target) forest_.addContacts(target, q, 1);
    }
    forest_.stats(from).samples.remove(color);
    --forest_.size_[ux(from)];
    RegionStats& t = forest_.stats(target);
    t.samples.add(color);
    t.minX = std::min(t.minX, x);
    t.minY = std::min(t.minY, y);
    t.maxX = std::max(t.maxX, x);
    t.maxY = std::max(t.maxY, y);
    ++forest_.size_[ux(target)];
    labels_(x, y) = target;
    compact_ = false;
}

bool PixelMap::reassignPixel(int x, int y, RegionId to, sla::Rgba8 color) {
    if (!labels_.inBounds(x, y)) throw std::out_of_range("PixelMap::reassignPixel: pixel fora da imagem");
    forest_.checkId(to, "PixelMap::reassignPixel");
    const RegionId from = regionAt(x, y);
    if (from == kNoRegion) throw std::logic_error("PixelMap::reassignPixel: pixel sem região");
    const RegionId target = forest_.find(to);
    if (target == from) return false;
    std::array<RegionId, 4> around{kNoRegion, kNoRegion, kNoRegion, kNoRegion};
    for (std::size_t d = 0; d < 4; ++d) {
        const int qx = x + kDx4[d];
        const int qy = y + kDy4[d];
        if (labels_.inBounds(qx, qy)) around[d] = regionAt(qx, qy);
    }
    moveUnchecked(x, y, from, target, color, around);
    return true;
}

void PixelMap::assignMedianColors(std::uint32_t meanFallbackBelow) {
    for (std::size_t i = 0; i < forest_.capacity(); ++i) {
        const auto r = static_cast<RegionId>(i);
        if (!forest_.isRoot(r)) continue;
        RegionStats& s = forest_.stats(r);
        if (s.samples.count() == 0) continue;
        s.color = s.samples.count() < meanFallbackBelow ? s.samples.mean() : sla::toRgba(s.samples.median());
    }
}

RegionId PixelMap::compact() {
    const int w = width();
    const int h = height();
    std::vector<RegionId> newId(forest_.capacity(), kNoRegion);
    std::vector<RegionId> oldRoot;  // por novo id
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const RegionId l = labels_(x, y);
            if (l == kNoRegion) {
                throw std::logic_error("PixelMap::compact: pixel (" + std::to_string(x) + ", " + std::to_string(y) +
                                       ") sem região");
            }
            const RegionId r = forest_.find(l);
            if (newId[ux(r)] == kNoRegion) {
                newId[ux(r)] = toRegionId(oldRoot.size());
                oldRoot.push_back(r);
            }
            labels_(x, y) = newId[ux(r)];
        }
    }

    RegionForest fresh(oldRoot.size());
    for (std::size_t n = 0; n < oldRoot.size(); ++n) {
        (void)fresh.create();
        RegionStats& s = fresh.stats_[n];
        s = std::move(forest_.stats_[ux(oldRoot[n])]);
        s.minX = INT_MAX;
        s.minY = INT_MAX;
        s.maxX = INT_MIN;
        s.maxY = INT_MIN;
        fresh.size_[n] = forest_.size_[ux(oldRoot[n])];
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const RegionId r = labels_(x, y);
            RegionStats& s = fresh.stats_[ux(r)];
            s.minX = std::min(s.minX, x);
            s.minY = std::min(s.minY, y);
            s.maxX = std::max(s.maxX, x);
            s.maxY = std::max(s.maxY, y);
            if (x + 1 < w && labels_(x + 1, y) != r) fresh.addContacts(r, labels_(x + 1, y), 1);
            if (y + 1 < h && labels_(x, y + 1) != r) fresh.addContacts(r, labels_(x, y + 1), 1);
        }
    }
    forest_ = std::move(fresh);
    compact_ = true;
    return toRegionId(oldRoot.size());
}

const sla::Array2D<RegionId>& PixelMap::compactLabels() const {
    if (!isCompact()) throw std::logic_error("PixelMap::compactLabels: chame compact() antes");
    return labels_;
}

const sla::Rgba& PixelMap::regionColor(RegionId region) const {
    if (!isCompact()) throw std::logic_error("PixelMap::regionColor: chame compact() antes");
    forest_.checkId(region, "PixelMap::regionColor");
    return forest_.stats(region).color;
}

}  // namespace cnx
