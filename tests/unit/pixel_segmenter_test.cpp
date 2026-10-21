// tests/unit/pixel_segmenter_test.cpp — QuadTree, helpers "e se" da mediana, estágios do
// PixelSegmenter e o pipeline completo (contra verdade conhecida e contra força bruta).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <stop_token>
#include <string>
#include <vector>

#include "cnx/segment/PixelMap.h"
#include "cnx/segment/PixelSegmenter.h"
#include "cnx/segment/QuadTree.h"
#include "cnx/vector/BoundaryTracer.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::PixelMap;
using cnx::PixelSegmenter;
using cnx::RegionId;
using cnx::SegmentParams;
using cnxtest::SplitMix64;
using sla::Array2D;
using sla::Rgba8;

namespace {

constexpr Rgba8 kBlack{0, 0, 0, 255};
constexpr Rgba8 kWhite{255, 255, 255, 255};
constexpr Rgba8 kRed{255, 0, 0, 255};
constexpr Rgba8 kGreen{0, 255, 0, 255};
constexpr Rgba8 kBlue{0, 0, 255, 255};

// ---------------------------------------------------------------- utilidades

/// Cada região do rótulo `label(x, y)` é 4-conexa? (flood fill por força bruta)
bool allRegionsConnected(int w, int h, const std::function<RegionId(int, int)>& label) {
    std::vector<char> seen(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
    std::map<RegionId, int> components;
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (seen[static_cast<std::size_t>(y * w + x)]) continue;
            const RegionId r = label(x, y);
            if (++components[r] > 1) return false;
            stack.assign(1, {x, y});
            seen[static_cast<std::size_t>(y * w + x)] = 1;
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + cnxtest::kDx4[d];
                    const int ny = cy + cnxtest::kDy4[d];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    if (seen[static_cast<std::size_t>(ny * w + nx)] || label(nx, ny) != r) continue;
                    seen[static_cast<std::size_t>(ny * w + nx)] = 1;
                    stack.emplace_back(nx, ny);
                }
            }
        }
    }
    return true;
}

bool mapConnected(PixelMap& map) {
    return allRegionsConnected(map.width(), map.height(), [&](int x, int y) { return map.regionAt(x, y); });
}

/// Os rótulos finais são uma bijeção sobre a verdade?
bool sameAsTruth(const Array2D<RegionId>& labels, const Array2D<int>& truth, int* regions = nullptr) {
    std::map<RegionId, int> forward;
    std::map<int, RegionId> backward;
    for (int y = 0; y < labels.height(); ++y) {
        for (int x = 0; x < labels.width(); ++x) {
            const auto [fi, fnew] = forward.emplace(labels(x, y), truth(x, y));
            const auto [bi, bnew] = backward.emplace(truth(x, y), labels(x, y));
            (void)fnew;
            (void)bnew;
            if (fi->second != truth(x, y) || bi->second != labels(x, y)) return false;
        }
    }
    if (regions != nullptr) *regions = static_cast<int>(forward.size());
    return true;
}

struct Picture {
    Array2D<Rgba8> image;
    PixelMap map;
};

/// Mapa e imagem a partir de desenhos ASCII: `labels` define as regiões (um caractere = uma
/// região; caracteres iguais devem ser 4-conexos) e `colors` a cor de cada pixel (via `table`).
Picture picture(const std::vector<std::string>& labels, const std::vector<std::string>& colors,
                const std::map<char, Rgba8>& table) {
    const int h = static_cast<int>(labels.size());
    const int w = static_cast<int>(labels[0].size());
    Picture p{Array2D<Rgba8>(w, h), PixelMap(w, h)};
    std::map<char, RegionId> ids;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const char l = labels[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
            if (!ids.count(l)) ids[l] = p.map.createRegion();
            p.image(x, y) = table.at(colors[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)]);
            p.map.setLabel(x, y, ids[l], p.image(x, y));
        }
    }
    return p;
}

char labelChar(const std::vector<std::string>& rows, int x, int y) {
    return rows[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
}

Rgba8 jitter(Rgba8 c, int amplitude, SplitMix64& rng) {
    auto j = [&](std::uint8_t v) {
        const int d = rng.below(2 * amplitude + 1) - amplitude;
        return static_cast<std::uint8_t>(std::clamp(static_cast<int>(v) + d, 0, 255));
    };
    return {j(c.r), j(c.g), j(c.b), c.a};
}

/// Logo sintético: fundo branco, retângulo vermelho, retângulo azul e um verde pequeno.
struct Logo {
    Array2D<Rgba8> image;
    Array2D<int> truth;
};

Logo makeLogo(int w, int h, int noise, SplitMix64& rng) {
    Logo l{Array2D<Rgba8>(w, h), Array2D<int>(w, h)};
    const auto inRect = [&](int x, int y, double x0, double y0, double x1, double y1) {
        return x >= static_cast<int>(x0 * w) && x < static_cast<int>(x1 * w) && y >= static_cast<int>(y0 * h) &&
               y < static_cast<int>(y1 * h);
    };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int id = 0;
            Rgba8 c = kWhite;
            if (inRect(x, y, 0.62, 0.50, 0.92, 0.92)) { id = 2; c = kBlue; }
            if (inRect(x, y, 0.10, 0.12, 0.48, 0.42)) { id = 1; c = kRed; }
            if (inRect(x, y, 0.64, 0.14, 0.80, 0.34)) { id = 3; c = kGreen; }
            l.truth(x, y) = id;
            l.image(x, y) = noise > 0 ? jitter(c, noise, rng) : c;
        }
    }
    return l;
}

}  // namespace

// ================================================================ helpers "e se" da mediana

CNX_TEST(median_small_sort_path_matches_brute_force) {
    SplitMix64 rng{101};
    for (std::size_t n = 1; n <= 40; ++n) {  // cruza o limite de 32 do caminho rápido
        for (int trial = 0; trial < 20; ++trial) {
            cnx::MedianAccumulator acc;
            std::vector<Rgba8> v;
            for (std::size_t i = 0; i < n; ++i) {
                v.push_back(cnxtest::randomColor(rng, 1 + trial % 6));
                acc.add(v.back());
            }
            std::array<std::uint8_t, 4> expect{};
            for (std::size_t ch = 0; ch < 4; ++ch) {
                std::vector<int> c;
                for (const Rgba8 s : v) c.push_back(ch == 0 ? s.r : ch == 1 ? s.g : ch == 2 ? s.b : s.a);
                std::sort(c.begin(), c.end());
                expect[ch] = static_cast<std::uint8_t>(c[(n - 1) / 2]);
            }
            const Rgba8 m = acc.median();
            CHECK_CTX(m.r == expect[0] && m.g == expect[1] && m.b == expect[2] && m.a == expect[3],
                      "n=" + std::to_string(n));
        }
    }
}

CNX_TEST(median_union_and_deviation_match_brute_force) {
    SplitMix64 rng{102};
    const std::array<std::size_t, 8> sizes{1, 2, 5, 33, 400, 1024, 1025, 3000};  // modo pequeno e histograma
    for (const std::size_t na : sizes) {
        for (const std::size_t nb : sizes) {
            cnx::MedianAccumulator a;
            cnx::MedianAccumulator b;
            cnx::MedianAccumulator both;
            std::vector<Rgba8> all;
            for (std::size_t i = 0; i < na; ++i) {
                const Rgba8 c = cnxtest::randomColor(rng, 5);
                a.add(c);
                both.add(c);
                all.push_back(c);
            }
            for (std::size_t i = 0; i < nb; ++i) {
                const Rgba8 c = cnxtest::randomColor(rng, 7);
                b.add(c);
                both.add(c);
                all.push_back(c);
            }
            const std::string tag = std::to_string(na) + "+" + std::to_string(nb);
            CHECK_CTX(cnx::MedianAccumulator::unionMedian(a, b) == both.median(), tag);
            // As entradas não são alteradas.
            CHECK_CTX(a.count() == na && b.count() == nb, tag);

            const sla::Rgba center{0.3f, 0.6f, 0.1f, 0.9f};
            double exact = 0.0;  // em double, como o acumulador (distance2 em float diverge em 1e-7)
            for (const Rgba8 s : all) {
                const double dr = s.r / 255.0 - center.r;
                const double dg = s.g / 255.0 - center.g;
                const double db = s.b / 255.0 - center.b;
                const double da = s.a / 255.0 - center.a;
                exact += dr * dr + dg * dg + db * db + da * da;
            }
            const double got = cnx::MedianAccumulator::unionSquaredDeviation(a, b, center);
            CHECK_CTX(std::abs(got - exact) <= 1e-9 * (1.0 + exact), tag);
            CHECK_CTX(std::abs(both.squaredDeviation(center) - exact) <= 1e-9 * (1.0 + exact), tag);

            // Aborto antecipado: abaixo do total devolve algo > limite; acima devolve o total exato.
            const double low = exact * 0.5;
            CHECK_CTX(cnx::MedianAccumulator::unionSquaredDeviation(a, b, center, low) > low, tag);
            CHECK_CTX(std::abs(cnx::MedianAccumulator::unionSquaredDeviation(a, b, center, exact + 1.0) - exact) <=
                          1e-9 * (1.0 + exact),
                      tag);
        }
    }
    cnx::MedianAccumulator empty;
    CHECK_THROWS(std::logic_error, (void)cnx::MedianAccumulator::unionMedian(empty, empty));
}

// ================================================================ QuadTree

CNX_TEST(quadtree_leaves_partition_the_image) {
    SplitMix64 rng{103};
    const std::array<std::array<int, 3>, 6> shapes{{{37, 21, 16}, {16, 16, 16}, {1, 1, 16}, {5, 40, 16}, {33, 33, 8}, {20, 7, 1}}};
    for (const auto& [w, h, block] : shapes) {
        for (const double variance : {0.0, 1e-3, 0.05, 10.0}) {
            Array2D<Rgba8> image(w, h);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    // Metade esquerda ruidosa, metade direita uniforme.
                    image(x, y) = x < w / 2 ? cnxtest::randomColor(rng, 3) : kBlue;
                }
            }
            const std::vector<cnx::QuadBlock> leaves = cnx::quadTreeSplit(image, block, variance);
            const std::string tag = std::to_string(w) + "x" + std::to_string(h) + " b=" + std::to_string(block) +
                                    " v=" + std::to_string(variance);
            Array2D<int> cover(w, h);
            for (const cnx::QuadBlock& b : leaves) {
                CHECK_CTX(b.w >= 1 && b.h >= 1 && b.w <= block && b.h <= block, tag);
                CHECK_CTX(b.x >= 0 && b.y >= 0 && b.x + b.w <= w && b.y + b.h <= h, tag);
                // Folha: homogênea ou do tamanho de um pixel.
                CHECK_CTX((b.w == 1 && b.h == 1) || cnx::blockColorVariance(image, b) <= variance, tag);
                for (int y = b.y; y < b.y + b.h; ++y) {
                    for (int x = b.x; x < b.x + b.w; ++x) ++cover(x, y);
                }
            }
            int bad = 0;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) bad += cover(x, y) == 1 ? 0 : 1;
            }
            CHECK_CTX(bad == 0, tag);
            CHECK_CTX(cnx::quadTreeSplit(image, block, variance) == leaves, tag);  // determinismo
        }
    }
}

CNX_TEST(quadtree_uniform_image_gives_one_leaf_per_block) {
    Array2D<Rgba8> image(40, 35);
    for (int y = 0; y < 35; ++y) {
        for (int x = 0; x < 40; ++x) image(x, y) = kRed;
    }
    const auto leaves = cnx::quadTreeSplit(image, 16, 0.0);
    CHECK(leaves.size() == 3 * 3);  // ⌈40/16⌉ × ⌈35/16⌉
    CHECK(cnx::blockColorVariance(image, {0, 0, 40, 35}) == 0.0);
    // Variância conhecida: dois valores 0 e 255 em partes iguais no canal R → (255/2)² / 255² = 1/4.
    Array2D<Rgba8> two(2, 1);
    two(0, 0) = {0, 0, 0, 255};
    two(1, 0) = {255, 0, 0, 255};
    CHECK(std::abs(cnx::blockColorVariance(two, {0, 0, 2, 1}) - 0.25) < 1e-12);
}

CNX_TEST(quadtree_rejects_bad_arguments) {
    Array2D<Rgba8> image(4, 4);
    CHECK_THROWS(std::invalid_argument, (void)cnx::quadTreeSplit(image, 0, 0.1));
    CHECK_THROWS(std::invalid_argument, (void)cnx::quadTreeSplit(Array2D<Rgba8>(), 16, 0.1));
}

// ================================================================ parâmetros e construção

CNX_TEST(threshold_schedule_is_geometric_and_ends_at_the_cap) {
    SegmentParams p;
    const std::vector<double> t = PixelSegmenter::thresholdSchedule(p);
    CHECK(t.size() >= 2);
    CHECK(t.front() == p.mergeThreshStart);
    CHECK(t.back() == p.mergeThreshMax);
    for (std::size_t i = 1; i + 1 < t.size(); ++i) CHECK(std::abs(t[i] / t[i - 1] - p.mergeThreshGrowth) < 1e-12);
    CHECK(t[t.size() - 2] < p.mergeThreshMax);

    p.mergeThreshStart = p.mergeThreshMax;  // início = teto: uma única rodada
    CHECK(PixelSegmenter::thresholdSchedule(p).size() == 1);

    SegmentParams bad;
    bad.mergeThreshGrowth = 1.0;
    CHECK_THROWS(std::invalid_argument, (void)PixelSegmenter::thresholdSchedule(bad));
    bad = SegmentParams{};
    bad.mergeThreshStart = 0.0;
    CHECK_THROWS(std::invalid_argument, (void)PixelSegmenter::thresholdSchedule(bad));
    bad = SegmentParams{};
    bad.mergeThreshMax = bad.mergeThreshStart / 2;
    CHECK_THROWS(std::invalid_argument, (void)PixelSegmenter::thresholdSchedule(bad));
}

CNX_TEST(constructor_validates_the_initial_map) {
    Array2D<Rgba8> image(3, 3);
    PixelMap wrongSize(4, 3);
    CHECK_THROWS(std::invalid_argument, PixelSegmenter(image, SegmentParams{}, std::move(wrongSize)));
    PixelMap incomplete(3, 3);
    const RegionId r = incomplete.createRegion();
    incomplete.setLabel(0, 0, r, kBlack);
    CHECK_THROWS(std::invalid_argument, PixelSegmenter(image, SegmentParams{}, std::move(incomplete)));
    PixelSegmenter fresh(image, SegmentParams{});
    CHECK_THROWS(std::logic_error, (void)fresh.joinSweep(0.1));  // exige initializeFromQuadTree()
    fresh.initializeFromQuadTree();
    CHECK_THROWS(std::logic_error, fresh.initializeFromQuadTree());
}

// ================================================================ juntar

CNX_TEST(join_threshold_and_what_if_on_flat_regions) {
    // Duas regiões chapadas (10 e 6 pixels) que diferem de 5/255 em um canal: dist² ≈ 3,8e-4.
    // A segunda linha tem 6 pixels de B e 4 de C (C é bem diferente e serve de controle).
    const std::vector<std::string> shape{"AAAAAAAAAA", "BBBBBBCCCC"};
    const std::vector<std::string> colors{"aaaaaaaaaa", "bbbbbbcccc"};
    const std::map<char, Rgba8> table{{'a', {100, 100, 100, 255}}, {'b', {105, 100, 100, 255}}, {'c', {200, 0, 0, 255}}};

    {   // Limiar abaixo da distância: nada se funde.
        Picture p = picture(shape, colors, table);
        PixelSegmenter seg(p.image, SegmentParams{}, std::move(p.map));
        CHECK(seg.joinSweep(1e-4) == 0);
    }
    {   // Limiar acima: A e B se fundem; C (longe) fica.
        Picture p = picture(shape, colors, table);
        PixelSegmenter seg(p.image, SegmentParams{}, std::move(p.map));
        CHECK(seg.joinSweep(1e-3) == 1);
        CHECK(seg.map().forest().rootCount() == 2);
        CHECK(seg.map().regionAt(0, 0) == seg.map().regionAt(0, 1));
        CHECK(seg.map().regionAt(0, 0) != seg.map().regionAt(9, 1));
    }
    {   // O orçamento "e se" é proporcional ao fator: com 0, nenhuma fusão com erro > 0 passa.
        SegmentParams params;
        params.whatIfFactor = 0.0;
        Picture p = picture(shape, colors, table);
        PixelSegmenter seg(p.image, params, std::move(p.map));
        CHECK(seg.joinSweep(1e-3) == 0);
    }
}

CNX_TEST(join_never_merges_non_adjacent_regions) {
    // A e C têm a mesma cor mas não se tocam; só B (diferente) as separa.
    const std::vector<std::string> shape{"ABC"};
    const std::vector<std::string> colors{"aba"};
    const std::map<char, Rgba8> table{{'a', kRed}, {'b', kBlue}};
    Picture p = picture(shape, colors, table);
    PixelSegmenter seg(p.image, SegmentParams{}, std::move(p.map));
    CHECK(seg.joinSweep(1.0) == 0);  // limiar enorme, mas dist²(A,B) = 2 > 1
    CHECK(seg.map().forest().rootCount() == 3);
}

// ================================================================ trocar

CNX_TEST(swap_moves_misassigned_boundary_pixels) {
    // Verdade: esquerda preta (x < 6), direita branca. O mapa inicial erra dois pixels simples.
    const std::vector<std::string> labels{
        "AAAAAABBBBBB",
        "AAAAABBBBBBB",  // (5,1) preto rotulado B
        "AAAAAAABBBBB",  // (6,2) branco rotulado A
        "AAAAAABBBBBB"};
    const std::vector<std::string> colors{"kkkkkkwwwwww", "kkkkkkwwwwww", "kkkkkkwwwwww", "kkkkkkwwwwww"};
    const std::map<char, Rgba8> table{{'k', kBlack}, {'w', kWhite}};
    Picture p = picture(labels, colors, table);
    PixelSegmenter seg(p.image, SegmentParams{}, std::move(p.map));
    CHECK(seg.swapSweep() == 2);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 12; ++x) {
            CHECK_CTX((seg.map().regionAt(x, y) == seg.map().regionAt(0, 0)) == (x < 6),
                      std::to_string(x) + "," + std::to_string(y));
        }
    }
    CHECK(seg.swapSweep() == 0);  // ponto fixo
    CHECK(mapConnected(seg.map()));
}

CNX_TEST(swap_refuses_a_pixel_whose_removal_splits_its_region) {
    // Região do meio é uma coluna de 3 pixels; o do centro tem a cor da esquerda, mas removê-lo parte a coluna.
    const std::vector<std::string> shape{"PQR", "PQR", "PQR"};
    const std::vector<std::string> colors{"pqr", "ppr", "pqr"};
    const std::map<char, Rgba8> table{{'p', kRed}, {'q', kGreen}, {'r', kBlue}};
    Picture p = picture(shape, colors, table);
    PixelSegmenter seg(p.image, SegmentParams{}, std::move(p.map));
    CHECK(seg.swapSweep() == 0);
    CHECK(seg.map().regionAt(1, 1) == seg.map().regionAt(1, 0));
    CHECK(seg.map().regionAt(1, 1) == seg.map().regionAt(1, 2));
}

CNX_TEST(swap_sweeps_keep_regions_connected_on_random_scenes) {
    SplitMix64 rng{104};
    for (int scene = 0; scene < 6; ++scene) {
        cnxtest::Scene s = cnxtest::blockScene(23 + scene, 17, 3, rng, 3);
        cnxtest::randomMerges(s.map, 25, rng);
        PixelSegmenter seg(s.image, SegmentParams{}, std::move(s.map));
        for (int sweep = 0; sweep < 6; ++sweep) {
            (void)seg.swapSweep();
            CHECK_CTX(mapConnected(seg.map()), "cena " + std::to_string(scene) + " varredura " + std::to_string(sweep));
        }
        // Soma dos tamanhos das raízes = área da imagem.
        std::uint64_t total = 0;
        for (std::size_t r = 0; r < seg.map().forest().capacity(); ++r) {
            if (seg.map().forest().isRoot(static_cast<RegionId>(r))) total += seg.map().forest().pixelCount(static_cast<RegionId>(r));
        }
        CHECK(total == static_cast<std::uint64_t>(seg.map().width()) * static_cast<std::uint64_t>(seg.map().height()));
    }
}

// ================================================================ diagonal

CNX_TEST(diagonal_pass_resolves_a_saddle_choosing_the_cheapest_move) {
    // X toca a si mesmo na quina da janela (1,1): (1,1) e (2,2) são X; (2,1) é Y; (1,2) é Z.
    const std::vector<std::string> shape{
        "ZXXX",
        "ZXYX",
        "ZZXX",
        "ZZZZ"};
    // O pixel (1,2), rotulado Z, tem cor de X: mover (1,2) → X custa menos que qualquer outro.
    const std::vector<std::string> colors{
        "zxxx",
        "zxyx",
        "zxxx",
        "zzzz"};
    const std::map<char, Rgba8> table{{'x', kRed}, {'y', kBlue}, {'z', kGreen}};
    Picture p = picture(shape, colors, table);
    PixelSegmenter seg(p.image, SegmentParams{}, std::move(p.map));
    CHECK(seg.countSaddles() == 1);
    CHECK(seg.diagonalSweep() == 1);
    CHECK(seg.countSaddles() == 0);
    CHECK(seg.map().regionAt(1, 2) == seg.map().regionAt(1, 1));  // (1,2) foi para X
    CHECK(seg.map().regionAt(1, 2) == seg.map().regionAt(2, 2));
    CHECK(mapConnected(seg.map()));
    CHECK(seg.diagonalSweep() == 0);
}

/// Mapa cujas regiões são as componentes 4-conexas de cor igual (ruído de poucos níveis gera muitas selas).
static PixelMap mapFromColorComponents(const Array2D<Rgba8>& image) {
    PixelMap map(image.width(), image.height());
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (map.regionAtConst(x, y) != cnx::kNoRegion) continue;
            const RegionId id = map.createRegion();
            map.setLabel(x, y, id, image(x, y));
            stack.assign(1, {x, y});
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + cnxtest::kDx4[d];
                    const int ny = cy + cnxtest::kDy4[d];
                    if (!image.inBounds(nx, ny) || map.regionAtConst(nx, ny) != cnx::kNoRegion || !(image(nx, ny) == image(x, y))) continue;
                    map.setLabel(nx, ny, id, image(nx, ny));
                    stack.emplace_back(nx, ny);
                }
            }
        }
    }
    return map;
}

CNX_TEST(diagonal_pass_reaches_a_fixed_point_where_no_saddle_has_a_simple_move) {
    SplitMix64 rng{105};
    int saddlesBefore = 0;
    int saddlesAfter = 0;
    for (int scene = 0; scene < 8; ++scene) {
        Array2D<Rgba8> image(20 + scene, 18);
        for (int y = 0; y < image.height(); ++y) {
            // 65% de branco: acima do limiar de percolação (≈ 0,593), então há uma componente branca
            // gigante e muitas quinas onde ela toca a si mesma em diagonal.
            for (int x = 0; x < image.width(); ++x) image(x, y) = rng.below(100) < 65 ? kWhite : kBlack;
        }
        PixelSegmenter seg(image, SegmentParams{}, mapFromColorComponents(image));
        saddlesBefore += seg.countSaddles();
        bool converged = false;
        for (int sweep = 0; sweep < 60 && !converged; ++sweep) converged = seg.diagonalSweep() == 0;
        const std::string tag = "cena " + std::to_string(scene);
        CHECK_CTX(converged, tag);
        CHECK_CTX(mapConnected(seg.map()), tag);
        saddlesAfter += seg.countSaddles();
        // No ponto fixo, toda sela restante tem as 4 células como pontos não simples.
        for (int y = 0; y + 1 < seg.map().height(); ++y) {
            for (int x = 0; x + 1 < seg.map().width(); ++x) {
                const RegionId a = seg.map().regionAt(x, y);
                const RegionId b = seg.map().regionAt(x + 1, y);
                const RegionId c = seg.map().regionAt(x, y + 1);
                const RegionId d = seg.map().regionAt(x + 1, y + 1);
                const bool saddle = (a == d && b != a && c != a) || (b == c && a != b && d != b);
                if (!saddle) continue;
                const bool anySimple = seg.map().isSimplePoint(x, y) || seg.map().isSimplePoint(x + 1, y) ||
                                       seg.map().isSimplePoint(x, y + 1) || seg.map().isSimplePoint(x + 1, y + 1);
                CHECK_CTX(!anySimple, tag);
            }
        }
    }
    CHECK(saddlesBefore > 0);  // a cena de fato tinha selas
    CHECK(saddlesAfter < saddlesBefore);
    std::printf("  selas: %d -> %d\n", saddlesBefore, saddlesAfter);
}

CNX_TEST(color_cache_stays_consistent_through_every_stage) {
    SplitMix64 rng{113};
    for (int scene = 0; scene < 4; ++scene) {
        const int w = 40 + scene;
        const int h = 30;
        Array2D<Rgba8> image(w, h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // Quatro faixas de cor + ruído moderado: há fusões, trocas e selas de verdade.
                image(x, y) = jitter(rng.below(8) == 0 ? kBlue : (x / 10 + y / 15) % 2 == 0 ? kRed : kGreen, 12, rng);
            }
        }
        PixelSegmenter seg(image, SegmentParams{});
        seg.initializeFromQuadTree();
        const std::string tag = "cena " + std::to_string(scene);
        int joins = 0;
        int swaps = 0;
        for (const double t : PixelSegmenter::thresholdSchedule(SegmentParams{})) {
            joins += seg.joinSweep(t);
            CHECK_CTX(seg.colorCacheConsistent(), tag + " após juntar");
            swaps += seg.swapSweep();
            CHECK_CTX(seg.colorCacheConsistent(), tag + " após trocar");
        }
        (void)seg.diagonalSweep();
        CHECK_CTX(seg.colorCacheConsistent(), tag + " após diagonal");
        (void)seg.mergeSmallRegions();
        CHECK_CTX(seg.colorCacheConsistent(), tag + " após minúsculas");
        CHECK_CTX(joins > 0 && swaps > 0, tag);  // os estágios realmente agiram
    }
}

// ================================================================ regiões minúsculas

CNX_TEST(small_regions_merge_into_the_closest_color_neighbor) {
    const std::vector<std::string> shape{"AAAAAA", "AAxAAB", "AAAAAB", "AAAAAB"};
    const std::vector<std::string> colors{"kkkkkk", "kkwkkw", "kkkkkw", "kkkkkw"};
    const std::map<char, Rgba8> table{{'k', kBlack}, {'w', kWhite}};
    Picture p = picture(shape, colors, table);
    SegmentParams params;
    params.minRegionPixels = 2;  // x tem 1 pixel; B (3 pixels) fica
    PixelSegmenter seg(p.image, params, std::move(p.map));
    CHECK(seg.mergeSmallRegions() == 1);
    CHECK(seg.map().regionAt(2, 1) == seg.map().regionAt(0, 0));  // vizinho único: A
    CHECK(seg.map().forest().rootCount() == 2);
    CHECK(seg.mergeSmallRegions() == 0);
}

CNX_TEST(small_region_prefers_closest_color_then_longest_contact) {
    // Região pequena S (cinza) entre A (preta, 1 lado de contato) e B (branca, 3 lados); cor mais
    // próxima de A → funde com A apesar do contato menor.
    const std::vector<std::string> shape{"BBB", "BSB", "BAB", "BAB"};
    const std::vector<std::string> colors{"www", "wgw", "wkw", "wkw"};
    const std::map<char, Rgba8> table{{'k', kBlack}, {'w', kWhite}, {'g', {40, 40, 40, 255}}};
    {
        Picture p = picture(shape, colors, table);
        SegmentParams params;
        params.minRegionPixels = 2;
        PixelSegmenter seg(p.image, params, std::move(p.map));
        CHECK(seg.mergeSmallRegions() == 1);
        CHECK(seg.map().regionAt(1, 1) == seg.map().regionAt(1, 2));
    }
    {   // Cores empatadas (cinza equidistante de duas regiões idênticas em cor): vence o maior contato.
        const std::map<char, Rgba8> tie{{'k', {0, 0, 0, 255}}, {'w', {0, 0, 0, 255}}, {'g', {40, 40, 40, 255}}};
        Picture p = picture(shape, colors, tie);
        SegmentParams params;
        params.minRegionPixels = 2;
        PixelSegmenter seg(p.image, params, std::move(p.map));
        CHECK(seg.mergeSmallRegions() == 1);
        CHECK(seg.map().regionAt(1, 1) == seg.map().regionAt(0, 1));  // B: contato 3 contra 1
    }
}

// ================================================================ pipeline

CNX_TEST(clean_logo_is_recovered_exactly) {
    SplitMix64 rng{106};
    const Logo logo = makeLogo(64, 48, 0, rng);
    PixelSegmenter seg(logo.image, SegmentParams{});
    cnx::SegmentationResult r = seg.run();
    int regions = 0;
    CHECK(sameAsTruth(r.map.compactLabels(), logo.truth, &regions));
    CHECK(regions == 4);
    CHECK(r.stats.finalRegions == 4);
    CHECK(r.stats.stable);
    // Cores: mediana exata das regiões chapadas.
    CHECK(r.map.regionColor(r.map.compactLabels()(0, 0)) == sla::toRgba(kWhite));
    CHECK(r.map.regionColor(r.map.compactLabels()(10, 10)) == sla::toRgba(kRed));
}

CNX_TEST(noisy_logo_is_recovered_and_the_result_traces_cleanly) {
    SplitMix64 rng{107};
    const Logo logo = makeLogo(96, 64, 3, rng);
    PixelSegmenter seg(logo.image, SegmentParams{});
    cnx::SegmentationResult r = seg.run();
    int regions = 0;
    CHECK(sameAsTruth(r.map.compactLabels(), logo.truth, &regions));
    CHECK(regions == 4);
    // Cor da região ≈ cor verdadeira (ruído de ±3 níveis).
    const sla::Rgba red = r.map.regionColor(r.map.compactLabels()(10, 10));
    CHECK(std::abs(red.r - 1.0f) < 4.0f / 255.0f && red.g < 4.0f / 255.0f && red.b < 4.0f / 255.0f);
    // O mapa resultante alimenta o BoundaryTracer e passa em I1–I4.
    const cnx::TraceResult t = cnx::BoundaryTracer().trace(r.map);
    const auto report = t.image.validate();
    CHECK_CTX(report.ok, report.message);
    CHECK(t.image.faceCount() == static_cast<std::size_t>(r.stats.finalRegions) + 1);
}

CNX_TEST(isolated_outlier_pixels_are_absorbed) {
    SplitMix64 rng{108};
    const int w = 80;
    const int h = 40;
    Array2D<Rgba8> image(w, h);
    Array2D<int> truth(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            truth(x, y) = x < w / 2 ? 0 : 1;
            image(x, y) = truth(x, y) == 0 ? kWhite : kBlack;
        }
    }
    // ~1% de pixels invertidos, isolados entre si e longe da fronteira.
    Array2D<char> flipped(w, h);
    int count = 0;
    for (int tries = 0; tries < 400 && count < 30; ++tries) {
        const int x = 2 + rng.below(w - 4);
        const int y = 2 + rng.below(h - 4);
        if (std::abs(x - w / 2) < 3) continue;
        bool near = false;
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) near = near || flipped(x + dx, y + dy) != 0;
        }
        if (near) continue;
        flipped(x, y) = 1;
        image(x, y) = truth(x, y) == 0 ? kBlack : kWhite;
        ++count;
    }
    CHECK(count >= 20);
    PixelSegmenter seg(image, SegmentParams{});
    cnx::SegmentationResult r = seg.run();
    int regions = 0;
    CHECK(sameAsTruth(r.map.compactLabels(), truth, &regions));
    CHECK(regions == 2);
    CHECK(r.stats.smallMerges >= count);  // cada intruso foi absorvido por enforceMinNumPixels
}

CNX_TEST(pipeline_invariants_on_random_images) {
    SplitMix64 rng{109};
    for (int scene = 0; scene < 5; ++scene) {
        const int w = 30 + 7 * scene;
        const int h = 25 + 3 * scene;
        Array2D<Rgba8> image(w, h);
        // Mosaico de retângulos aleatórios + ruído, com paletas de poucas cores.
        std::vector<Rgba8> palette;
        for (int i = 0; i < 5; ++i) palette.push_back(cnxtest::randomColor(rng, 4));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const std::size_t k = static_cast<std::size_t>(((x / 9) * 3 + (y / 7) * 5 + scene) % 5);
                image(x, y) = rng.below(6) == 0 ? cnxtest::randomColor(rng, 4) : jitter(palette[k], 2, rng);
            }
        }
        SegmentParams params;
        params.minRegionPixels = 3 + scene;
        PixelSegmenter seg(image, params);
        cnx::SegmentationResult r = seg.run();
        const std::string tag = "cena " + std::to_string(scene);
        const auto& labels = r.map.compactLabels();
        CHECK_CTX(allRegionsConnected(w, h, [&](int x, int y) { return labels(x, y); }), tag);
        CHECK_CTX(r.map.isCompact(), tag);
        // Cada região tem ≥ minRegionPixels pixels (a menos que seja a única).
        for (RegionId id = 0; id < r.stats.finalRegions; ++id) {
            CHECK_CTX(r.stats.finalRegions == 1 || r.map.forest().pixelCount(id) >= static_cast<std::uint32_t>(params.minRegionPixels), tag);
        }
        const cnx::TraceResult t = cnx::BoundaryTracer().trace(r.map);
        const auto report = t.image.validate();
        CHECK_CTX(report.ok, tag + ": " + report.message);
        CHECK_CTX(t.image.faceCount() == static_cast<std::size_t>(r.stats.finalRegions) + 1, tag);
    }
}

CNX_TEST(pipeline_is_deterministic) {
    SplitMix64 rngA{110};
    SplitMix64 rngB{110};
    const Logo a = makeLogo(60, 40, 4, rngA);
    const Logo b = makeLogo(60, 40, 4, rngB);
    PixelSegmenter segA(a.image, SegmentParams{});
    PixelSegmenter segB(b.image, SegmentParams{});
    const cnx::SegmentationResult ra = segA.run();
    const cnx::SegmentationResult rb = segB.run();
    bool same = ra.map.compactLabels().width() == rb.map.compactLabels().width();
    for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 60; ++x) same = same && ra.map.compactLabels()(x, y) == rb.map.compactLabels()(x, y);
    }
    CHECK(same);
    CHECK(ra.stats.joins == rb.stats.joins && ra.stats.swaps == rb.stats.swaps);
}

CNX_TEST(pipeline_snaps_to_the_palette_and_honors_cancellation) {
    SplitMix64 rng{111};
    const Logo logo = makeLogo(48, 32, 3, rng);
    cnx::Palette palette;
    palette.colors = {sla::toRgba(kWhite), sla::toRgba(kRed), sla::toRgba(kBlue), sla::toRgba(kGreen)};
    PixelSegmenter seg(logo.image, SegmentParams{});
    cnx::SegmentationResult r = seg.run({}, &palette);
    const auto& labels = r.map.compactLabels();
    // A cor de cada região é exatamente uma entrada da paleta, e o índice confere.
    for (RegionId id = 0; id < r.stats.finalRegions; ++id) {
        const std::int32_t k = r.map.forest().stats(id).paletteIndex;
        CHECK(k >= 0 && k < 4);
        CHECK(r.map.regionColor(id) == palette.colors[static_cast<std::size_t>(k)]);
    }
    CHECK(r.map.regionColor(labels(0, 0)) == sla::toRgba(kWhite));

    cnx::Palette empty;
    CHECK_THROWS(std::invalid_argument, cnx::snapColorsToPalette(r.map, empty));

    std::stop_source source;
    source.request_stop();
    PixelSegmenter cancelled(logo.image, SegmentParams{});
    CHECK_THROWS(cnx::CancelTaskException,
                 (void)cancelled.run(cnx::TaskContext(source.get_token(), nullptr, cnx::ProgressRange{})));
}

CNX_TEST(pipeline_timing_report) {
    SplitMix64 rng{112};
    const Logo logo = makeLogo(640, 480, 2, rng);
    const auto t0 = std::chrono::steady_clock::now();
    PixelSegmenter seg(logo.image, SegmentParams{});
    cnx::SegmentationResult r = seg.run();
    const auto t1 = std::chrono::steady_clock::now();
    int regions = 0;
    CHECK(sameAsTruth(r.map.compactLabels(), logo.truth, &regions));
    std::printf("  640x480: %.0f ms; superpixels %d, juntar %d, trocar %d, diagonal %d, pequenas %d, regiões %d\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(), r.stats.superpixels, r.stats.joins,
                r.stats.swaps, r.stats.diagonalMoves, r.stats.smallMerges, r.stats.finalRegions);

    // Pior caso para a QuadTree: ruído por pixel (todas as folhas são 1×1, ~200 mil regiões iniciais).
#ifdef NDEBUG
    Array2D<Rgba8> noise(512, 384);
#else
    Array2D<Rgba8> noise(160, 120);  // Debug é ~12× mais lento
#endif
    for (int y = 0; y < noise.height(); ++y) {
        for (int x = 0; x < noise.width(); ++x) noise(x, y) = cnxtest::randomColor(rng, 4);
    }
    const auto n0 = std::chrono::steady_clock::now();
    PixelSegmenter noisy(noise, SegmentParams{});
    cnx::SegmentationResult rn = noisy.run();
    const auto n1 = std::chrono::steady_clock::now();
    const auto& noiseLabels = rn.map.compactLabels();
    CHECK(allRegionsConnected(noise.width(), noise.height(), [&](int x, int y) { return noiseLabels(x, y); }));
    const cnx::TraceResult t = cnx::BoundaryTracer().trace(rn.map);
    CHECK(t.image.validate().ok);
    std::printf("  ruído %dx%d: %.0f ms; superpixels %d, juntar %d, trocar %d, diagonal %d, pequenas %d, regiões %d, estável %d\n",
                noise.width(), noise.height(), std::chrono::duration<double, std::milli>(n1 - n0).count(), rn.stats.superpixels, rn.stats.joins,
                rn.stats.swaps, rn.stats.diagonalMoves, rn.stats.smallMerges, rn.stats.finalRegions, rn.stats.stable ? 1 : 0);
}

int main() { return cnxtest::runAll(); }
