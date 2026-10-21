// tests/unit/pixel_map_test.cpp — MedianAccumulator, RegionForest (contatos exatos) e PixelMap
// (ponto simples, troca, compactação), sempre comparados com força bruta.
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cnx/segment/PixelMap.h"
#include "test_harness.h"
#include "test_support.h"

namespace {

using cnx::kNoRegion;
using cnx::MedianAccumulator;
using cnx::PixelMap;
using cnx::RegionId;
using cnxtest::SplitMix64;

std::uint8_t channel(sla::Rgba8 c, int ch) { return ch == 0 ? c.r : ch == 1 ? c.g : ch == 2 ? c.b : c.a; }

sla::Rgba8 bruteMedian(const std::vector<sla::Rgba8>& px) {
    std::array<std::uint8_t, 4> m{};
    for (int ch = 0; ch < 4; ++ch) {
        std::vector<std::uint8_t> v;
        for (const sla::Rgba8 c : px) v.push_back(channel(c, ch));
        std::sort(v.begin(), v.end());
        m[static_cast<std::size_t>(ch)] = v[(v.size() - 1) / 2];
    }
    return {m[0], m[1], m[2], m[3]};
}

bool meanMatches(const MedianAccumulator& acc, const std::vector<sla::Rgba8>& px) {
    std::array<double, 4> s{};
    for (const sla::Rgba8 c : px) {
        for (int ch = 0; ch < 4; ++ch) s[static_cast<std::size_t>(ch)] += channel(c, ch);
    }
    const sla::Rgba m = acc.mean();
    const double n = 255.0 * static_cast<double>(px.size());
    return std::abs(m.r - s[0] / n) < 1e-6 && std::abs(m.g - s[1] / n) < 1e-6 && std::abs(m.b - s[2] / n) < 1e-6 &&
           std::abs(m.a - s[3] / n) < 1e-6;
}

MedianAccumulator accumulate(const std::vector<sla::Rgba8>& px) {
    MedianAccumulator acc;
    for (const sla::Rgba8 c : px) acc.add(c);
    return acc;
}

std::vector<sla::Rgba8> randomColors(std::size_t n, SplitMix64& rng, int levels) {
    std::vector<sla::Rgba8> v;
    for (std::size_t i = 0; i < n; ++i) v.push_back(cnxtest::randomColor(rng, levels));
    return v;
}

// ------------------------------------------------------------------ força bruta sobre PixelMap
/// Compara contagens, medianas, contatos e vizinhos do PixelMap com uma varredura direta.
void checkAgainstBruteForce(PixelMap& map, const sla::Array2D<sla::Rgba8>& image, const std::string& context) {
    const int w = map.width();
    const int h = map.height();
    std::map<RegionId, std::vector<sla::Rgba8>> members;
    std::map<std::pair<RegionId, RegionId>, std::uint32_t> contacts;
    const auto key = [](RegionId a, RegionId b) { return a < b ? std::pair{a, b} : std::pair{b, a}; };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const RegionId r = map.regionAt(x, y);
            members[r].push_back(image(x, y));
            if (x + 1 < w && map.regionAt(x + 1, y) != r) ++contacts[key(r, map.regionAt(x + 1, y))];
            if (y + 1 < h && map.regionAt(x, y + 1) != r) ++contacts[key(r, map.regionAt(x, y + 1))];
        }
    }
    cnx::RegionForest& forest = map.forest();
    std::size_t badRegions = 0;
    std::size_t badNeighbors = 0;
    for (const auto& [r, px] : members) {
        const cnx::RegionStats& s = forest.stats(r);
        if (!forest.isRoot(r) || forest.pixelCount(r) != px.size() || s.samples.count() != px.size() ||
            !(s.samples.median() == bruteMedian(px))) {
            ++badRegions;
        }
        std::vector<RegionId> expected;
        for (const auto& [pair, n] : contacts) {
            if (pair.first == r) expected.push_back(pair.second);
            if (pair.second == r) expected.push_back(pair.first);
        }
        std::sort(expected.begin(), expected.end());
        const auto got = forest.neighbors(r);
        if (!std::equal(got.begin(), got.end(), expected.begin(), expected.end())) ++badNeighbors;
    }
    std::size_t badContacts = 0;
    for (const auto& [pair, n] : contacts) badContacts += forest.contactCount(pair.first, pair.second) != n ? 1 : 0;
    CHECK_CTX(badRegions == 0, context + ": regiões com contagem/mediana divergente: " + std::to_string(badRegions));
    CHECK_CTX(badNeighbors == 0, context + ": listas de vizinhos divergentes: " + std::to_string(badNeighbors));
    CHECK_CTX(badContacts == 0, context + ": contatos divergentes: " + std::to_string(badContacts));
    CHECK_CTX(forest.rootCount() == members.size(), context + ": rootCount ≠ regiões com pixels");
}

/// Número de regiões com mais de uma componente 4-conexa.
std::size_t disconnectedRegions(PixelMap& map) {
    const int w = map.width();
    const int h = map.height();
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
    std::map<RegionId, int> components;
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (seen[static_cast<std::size_t>(y * w + x)] != 0) continue;
            const RegionId r = map.regionAt(x, y);
            ++components[r];
            seen[static_cast<std::size_t>(y * w + x)] = 1;
            stack.emplace_back(x, y);
            while (!stack.empty()) {
                const auto [px, py] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int qx = px + cnxtest::kDx4[d];
                    const int qy = py + cnxtest::kDy4[d];
                    if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
                    const auto i = static_cast<std::size_t>(qy * w + qx);
                    if (seen[i] == 0 && map.regionAt(qx, qy) == r) {
                        seen[i] = 1;
                        stack.emplace_back(qx, qy);
                    }
                }
            }
        }
    }
    std::size_t bad = 0;
    for (const auto& [r, n] : components) bad += n > 1 ? 1 : 0;
    return bad;
}

/// Componentes de uma máscara binária w×h (fora da máscara = não membro).
int countComponents(const std::vector<std::uint8_t>& member, int w, int h, bool eight) {
    std::vector<std::uint8_t> seen(member.size(), 0);
    int count = 0;
    std::vector<int> stack;
    for (int i = 0; i < w * h; ++i) {
        if (member[static_cast<std::size_t>(i)] == 0 || seen[static_cast<std::size_t>(i)] != 0) continue;
        ++count;
        seen[static_cast<std::size_t>(i)] = 1;
        stack.push_back(i);
        while (!stack.empty()) {
            const int c = stack.back();
            stack.pop_back();
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx == 0 && dy == 0) || (!eight && dx != 0 && dy != 0)) continue;
                    const int x = c % w + dx;
                    const int y = c / w + dy;
                    if (x < 0 || y < 0 || x >= w || y >= h) continue;
                    const auto j = static_cast<std::size_t>(y * w + x);
                    if (member[j] != 0 && seen[j] == 0) {
                        seen[j] = 1;
                        stack.push_back(static_cast<int>(j));
                    }
                }
            }
        }
    }
    return count;
}

/// Definição global: remover p de R preserva o número de 4-componentes de R e o de 8-componentes
/// do complemento (a grade tem moldura de não membros, que representa o exterior conexo).
bool globallySimple(const std::vector<std::uint8_t>& inR, int w, int h, int p) {
    std::vector<std::uint8_t> without = inR;
    without[static_cast<std::size_t>(p)] = 0;
    std::vector<std::uint8_t> complement(inR.size());
    std::vector<std::uint8_t> complementWith(inR.size());
    for (std::size_t i = 0; i < inR.size(); ++i) {
        complement[i] = inR[i] == 0 ? 1 : 0;
        complementWith[i] = without[i] == 0 ? 1 : 0;
    }
    return countComponents(inR, w, h, false) == countComponents(without, w, h, false) &&
           countComponents(complement, w, h, true) == countComponents(complementWith, w, h, true);
}

/// PixelMap com duas regiões: 0 onde inR = 1, 1 no resto.
PixelMap twoRegionMap(const std::vector<std::uint8_t>& inR, int w, int h) {
    PixelMap map(w, h);
    const RegionId a = map.createRegion();
    const RegionId b = map.createRegion();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) map.setLabel(x, y, inR[static_cast<std::size_t>(y * w + x)] != 0 ? a : b, {});
    }
    return map;
}

PixelMap mapFromRows(std::initializer_list<const char*> rows, sla::Array2D<sla::Rgba8>* image = nullptr) {
    const int h = static_cast<int>(rows.size());
    const int w = static_cast<int>(std::char_traits<char>::length(*rows.begin()));
    PixelMap map(w, h);
    std::map<char, RegionId> ids;
    int y = 0;
    for (const char* row : rows) {
        for (int x = 0; x < w; ++x) {
            const char c = row[x];
            if (ids.find(c) == ids.end()) ids[c] = map.createRegion();
            const sla::Rgba8 color{static_cast<std::uint8_t>(c), 0, 0, 255};
            if (image != nullptr) (*image)(x, y) = color;
            map.setLabel(x, y, ids[c], color);
        }
        ++y;
    }
    return map;
}

// ================================================================== MedianAccumulator
CNX_TEST(median_is_lower_median_per_channel) {
    const MedianAccumulator odd = accumulate({{10, 200, 0, 255}, {30, 100, 0, 255}, {20, 150, 9, 255}});
    CHECK(odd.median() == (sla::Rgba8{20, 150, 0, 255}));
    const MedianAccumulator even = accumulate({{10, 1, 1, 1}, {40, 2, 2, 2}, {20, 3, 3, 3}, {30, 4, 4, 4}});
    CHECK(even.median() == (sla::Rgba8{20, 2, 2, 2}));  // inferior dos dois centrais
    CHECK(!even.usesHistogram());
    MedianAccumulator empty;
    CHECK_THROWS(std::logic_error, (void)empty.median());
    CHECK(empty.mean() == sla::Rgba{});
}

CNX_TEST(median_survives_promotion_to_histogram) {
    SplitMix64 rng{11};
    const auto px = randomColors(MedianAccumulator::kSwitchToHistogram + 1, rng, 7);
    const MedianAccumulator acc = accumulate(px);
    CHECK(acc.usesHistogram());
    CHECK(acc.count() == px.size());
    CHECK(acc.median() == bruteMedian(px));
    CHECK(meanMatches(acc, px));
}

CNX_TEST(remove_in_both_modes) {
    SplitMix64 rng{12};
    for (const std::size_t n : {std::size_t{40}, std::size_t{3000}}) {
        auto px = randomColors(n, rng, 5);
        MedianAccumulator acc = accumulate(px);
        for (int i = 0; i < static_cast<int>(n) / 2; ++i) {
            const auto k = static_cast<std::size_t>(rng.below(static_cast<int>(px.size())));
            acc.remove(px[k]);
            px.erase(px.begin() + static_cast<std::ptrdiff_t>(k));
        }
        CHECK(acc.count() == px.size());
        CHECK(acc.median() == bruteMedian(px));
        CHECK(meanMatches(acc, px));
    }
    MedianAccumulator small = accumulate({{1, 2, 3, 4}});
    CHECK_THROWS(std::logic_error, small.remove({9, 9, 9, 9}));
    MedianAccumulator big = accumulate(std::vector<sla::Rgba8>(2000, sla::Rgba8{5, 5, 5, 5}));
    CHECK_THROWS(std::logic_error, big.remove({6, 5, 5, 5}));
    MedianAccumulator none;
    CHECK_THROWS(std::logic_error, none.remove({}));
}

CNX_TEST(merge_all_mode_combinations) {
    SplitMix64 rng{13};
    const std::array<std::size_t, 4> sizes{0, 7, 1000, 1700};
    for (const std::size_t na : sizes) {
        for (const std::size_t nb : sizes) {
            const auto a = randomColors(na, rng, 6);
            const auto b = randomColors(nb, rng, 6);
            MedianAccumulator accA = accumulate(a);
            MedianAccumulator accB = accumulate(b);
            accA.merge(std::move(accB));
            std::vector<sla::Rgba8> all = a;
            all.insert(all.end(), b.begin(), b.end());
            const std::string tag = std::to_string(na) + "+" + std::to_string(nb);
            CHECK_CTX(accA.count() == all.size(), tag);
            CHECK_CTX(accB.count() == 0, tag);  // o absorvido fica vazio
            if (!all.empty()) {
                CHECK_CTX(accA.median() == bruteMedian(all), tag);
                CHECK_CTX(meanMatches(accA, all), tag);
            }
            CHECK_CTX(accA.usesHistogram() == (all.size() > MedianAccumulator::kSwitchToHistogram || na > 1024 || nb > 1024), tag);
        }
    }
    MedianAccumulator self = accumulate({{1, 1, 1, 1}});
    CHECK_THROWS(std::invalid_argument, self.merge(std::move(self)));
}

// ================================================================== ponto simples
CNX_TEST(simple_point_table_matches_global_definition_exhaustively) {
    // As 256 configurações de N8*(p), embutidas numa grade 5×5 com moldura fora de R.
    int mismatches = 0;
    for (unsigned mask = 0; mask < 256; ++mask) {
        constexpr std::array<int, 8> dx{1, 1, 0, -1, -1, -1, 0, 1};
        constexpr std::array<int, 8> dy{0, -1, -1, -1, 0, 1, 1, 1};
        std::vector<std::uint8_t> inR(25, 0);
        inR[12] = 1;
        for (std::size_t k = 0; k < 8; ++k) {
            if (((mask >> k) & 1u) != 0) inR[static_cast<std::size_t>((2 + dy[k]) * 5 + 2 + dx[k])] = 1;
        }
        const PixelMap map = twoRegionMap(inR, 5, 5);
        mismatches += map.isSimplePoint(2, 2) != globallySimple(inR, 5, 5, 12) ? 1 : 0;
    }
    CHECK(mismatches == 0);
}

CNX_TEST(simple_point_matches_global_definition_on_random_maps) {
    SplitMix64 rng{14};
    int mismatches = 0;
    int simpleCount = 0;
    int tested = 0;
    for (int trial = 0; trial < 80; ++trial) {
        constexpr int n = 9;  // 7×7 aleatório dentro de uma moldura fora de R
        std::vector<std::uint8_t> inR(n * n, 0);
        for (int y = 1; y < n - 1; ++y) {
            for (int x = 1; x < n - 1; ++x) inR[static_cast<std::size_t>(y * n + x)] = rng.below(100) < 55 ? 1 : 0;
        }
        const PixelMap map = twoRegionMap(inR, n, n);
        for (int p = 0; p < n * n; ++p) {
            if (inR[static_cast<std::size_t>(p)] == 0) continue;
            const bool local = map.isSimplePoint(p % n, p / n);
            mismatches += local != globallySimple(inR, n, n, p) ? 1 : 0;
            simpleCount += local ? 1 : 0;
            ++tested;
        }
    }
    CHECK(mismatches == 0);
    CHECK(simpleCount > 0 && simpleCount < tested);  // os dois casos foram exercitados
}

// ================================================================== troca de pixel
CNX_TEST(move_refuses_cutting_a_line) {
    sla::Array2D<sla::Rgba8> image(5, 3);
    PixelMap map = mapFromRows({"bbbbb", "aaaaa", "ccccc"}, &image);
    const RegionId a = map.regionAt(0, 1);
    const RegionId b = map.regionAt(0, 0);
    const std::uint32_t before = map.forest().contactCount(a, b);
    CHECK(!map.isSimplePoint(2, 1));
    CHECK(!map.tryMovePixel(2, 1, b, image(2, 1)));
    CHECK(map.regionAt(2, 1) == a);
    CHECK(map.forest().pixelCount(a) == 5);
    CHECK(map.forest().contactCount(a, b) == before);

    CHECK(map.isSimplePoint(0, 1));  // ponta da linha
    CHECK(map.tryMovePixel(0, 1, b, image(0, 1)));
    CHECK(map.regionAt(0, 1) == b);
    CHECK(map.forest().pixelCount(a) == 4 && map.forest().pixelCount(b) == 6);
    checkAgainstBruteForce(map, image, "linha");
}

CNX_TEST(move_refuses_erasing_a_region) {
    sla::Array2D<sla::Rgba8> image(3, 3);
    PixelMap map = mapFromRows({"bbb", "bab", "bbb"}, &image);
    CHECK(!map.tryMovePixel(1, 1, map.regionAt(0, 0), image(1, 1)));
    CHECK(map.forest().rootCount() == 2);
}

CNX_TEST(move_rejects_invalid_requests) {
    sla::Array2D<sla::Rgba8> image(4, 1);
    PixelMap map = mapFromRows({"abcc"}, &image);
    const RegionId a = map.regionAt(0, 0);
    const RegionId c = map.regionAt(3, 0);
    CHECK_THROWS(std::invalid_argument, map.tryMovePixel(0, 0, a, image(0, 0)));  // já está em a
    CHECK_THROWS(std::invalid_argument, map.tryMovePixel(0, 0, c, image(0, 0)));  // c não é 4-vizinho
    CHECK_THROWS(std::out_of_range, map.tryMovePixel(4, 0, a, image(0, 0)));
    CHECK_THROWS(std::out_of_range, map.tryMovePixel(0, 0, 99, image(0, 0)));
    PixelMap blank(2, 1);
    const RegionId r = blank.createRegion();
    blank.setLabel(1, 0, r, {});
    CHECK_THROWS(std::logic_error, blank.tryMovePixel(0, 0, r, {}));
    CHECK_THROWS(std::logic_error, blank.setLabel(1, 0, r, {}));
    CHECK_THROWS(std::out_of_range, blank.setLabel(0, 0, 5, {}));
}

CNX_TEST(random_merges_and_moves_match_brute_force) {
    SplitMix64 rng{15};
    for (int scene = 0; scene < 6; ++scene) {
        cnxtest::Scene s = cnxtest::blockScene(20 + scene, 16, 2, rng, 3);
        const std::string tag = "cena " + std::to_string(scene);
        checkAgainstBruteForce(s.map, s.image, tag + " inicial");
        cnxtest::MoveStats total;
        for (int round = 0; round < 5; ++round) {
            cnxtest::randomMerges(s.map, 10, rng);
            const cnxtest::MoveStats m = cnxtest::randomMoves(s.map, s.image, 300, rng);
            total.accepted += m.accepted;
            total.refused += m.refused;
            total.inconsistent += m.inconsistent;
            checkAgainstBruteForce(s.map, s.image, tag + " rodada " + std::to_string(round));
            // Proteção topológica: blocos iniciais e fusões entre vizinhas são conexos, e trocas
            // de pontos simples preservam isso.
            CHECK_CTX(disconnectedRegions(s.map) == 0, tag);
        }
        CHECK_CTX(total.inconsistent == 0, tag);
        CHECK_CTX(total.accepted > 0 && total.refused > 0, tag + ": " + std::to_string(total.accepted) + " aceitas, " +
                                                                std::to_string(total.refused) + " recusadas");
    }
}

// ================================================================== compactação
CNX_TEST(compact_renumbers_in_scan_order_and_rebuilds_exactly) {
    SplitMix64 rng{16};
    cnxtest::Scene s = cnxtest::blockScene(12, 9, 3, rng, 4);
    cnxtest::randomMerges(s.map, 6, rng);  // 12 blocos − 6 fusões: restam ≥ 6 regiões
    (void)cnxtest::randomMoves(s.map, s.image, 200, rng);
    (void)s.map.createRegion();  // região vazia: deve ser descartada
    CHECK(!s.map.isCompact());

    std::map<RegionId, std::uint32_t> countsBefore;
    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 12; ++x) ++countsBefore[s.map.regionAt(x, y)];
    }
    const RegionId f = s.map.compact();
    CHECK(s.map.isCompact());
    CHECK(static_cast<std::size_t>(f) == countsBefore.size());
    CHECK(s.map.forest().capacity() == static_cast<std::size_t>(f));
    CHECK(s.map.forest().rootCount() == static_cast<std::size_t>(f));

    // Ids em ordem de primeira aparição na varredura.
    RegionId nextNew = 0;
    bool scanOrder = true;
    std::vector<int> minX(static_cast<std::size_t>(f), INT_MAX), minY(minX), maxX(static_cast<std::size_t>(f), INT_MIN),
        maxY(maxX);
    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 12; ++x) {
            const RegionId r = s.map.compactLabels()(x, y);
            if (r == nextNew) ++nextNew;
            scanOrder = scanOrder && r < nextNew && s.map.rawLabel(x, y) == r;
            const auto i = static_cast<std::size_t>(r);
            minX[i] = std::min(minX[i], x);
            minY[i] = std::min(minY[i], y);
            maxX[i] = std::max(maxX[i], x);
            maxY[i] = std::max(maxY[i], y);
        }
    }
    CHECK(scanOrder);
    bool exactBoxes = true;
    for (RegionId r = 0; r < f; ++r) {
        const cnx::RegionStats& st = s.map.forest().stats(r);
        const auto i = static_cast<std::size_t>(r);
        exactBoxes = exactBoxes && st.minX == minX[i] && st.minY == minY[i] && st.maxX == maxX[i] && st.maxY == maxY[i];
    }
    CHECK(exactBoxes);
    checkAgainstBruteForce(s.map, s.image, "após compact");

    (void)s.map.forest().unite(0, 1);
    CHECK(!s.map.isCompact());
    CHECK_THROWS(std::logic_error, (void)s.map.regionColor(0));
    CHECK_THROWS(std::logic_error, (void)s.map.compactLabels());
}

CNX_TEST(compact_shrinks_boxes_that_moves_left_as_supersets) {
    sla::Array2D<sla::Rgba8> image(4, 1);
    PixelMap map = mapFromRows({"caab"}, &image);
    const RegionId a = map.regionAt(1, 0);
    CHECK(map.tryMovePixel(1, 0, map.regionAt(0, 0), image(1, 0)));  // a perde seu pixel mais à esquerda
    CHECK(map.forest().stats(a).minX == 1);                            // superconjunto, documentado
    (void)map.compact();
    const cnx::RegionStats& s = map.forest().stats(map.regionAt(2, 0));
    CHECK(s.minX == 2 && s.maxX == 2 && s.minY == 0 && s.maxY == 0);
    checkAgainstBruteForce(map, image, "caixas");
}

CNX_TEST(compact_requires_every_pixel_assigned) {
    PixelMap map(2, 2);
    const RegionId r = map.createRegion();
    map.setLabel(0, 0, r, {});
    map.setLabel(1, 0, r, {});
    map.setLabel(0, 1, r, {});
    CHECK(map.regionAt(1, 1) == kNoRegion);
    CHECK(!map.isSimplePoint(1, 1));
    CHECK_THROWS(std::logic_error, (void)map.compact());
}

CNX_TEST(median_colors_with_mean_fallback) {
    PixelMap map(5, 1);
    const RegionId a = map.createRegion();
    const RegionId b = map.createRegion();
    const std::array<sla::Rgba8, 5> colors{
        {{0, 0, 0, 255}, {100, 0, 0, 255}, {200, 0, 0, 255}, {0, 0, 0, 255}, {255, 0, 0, 255}}};
    for (int x = 0; x < 5; ++x) map.setLabel(x, 0, x < 3 ? a : b, colors[static_cast<std::size_t>(x)]);
    map.assignMedianColors(3);  // a: 3 pixels → mediana; b: 2 pixels → média
    CHECK(map.forest().stats(a).color == sla::toRgba({100, 0, 0, 255}));
    CHECK(std::abs(map.forest().stats(b).color.r - 0.5f) < 1e-6f);
}

}  // namespace

int main() { return cnxtest::runAll(); }
