// tests/unit/palette_test.cpp — myPrng, histograma ponderado, k-means sobre o coreset e PaletteFinder.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "cnx/palette/ColorHistogram.h"
#include "cnx/palette/KMeans.h"
#include "cnx/palette/PaletteFinder.h"
#include "sla/Prng.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::Color4d;
using cnx::ColorHistogram;
using cnx::KMeansItem;
using cnx::PaletteFinder;
using cnx::PaletteResult;
using cnxtest::SplitMix64;
using sla::Array2D;
using sla::Rgba8;

namespace {

Rgba8 rgb(int r, int g, int b) {
    return {static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b), 255};
}

Color4d asColor(Rgba8 c) { return {c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0}; }

Rgba8 jitter(Rgba8 c, int amplitude, SplitMix64& rng) {
    const auto j = [&](std::uint8_t v) {
        return static_cast<std::uint8_t>(std::clamp(static_cast<int>(v) + rng.below(2 * amplitude + 1) - amplitude, 0, 255));
    };
    return {j(c.r), j(c.g), j(c.b), c.a};
}

/// Imagem w×h em quadrantes de 4 cores com ruído de ±amplitude.
Array2D<Rgba8> quadrants(int w, int h, const std::array<Rgba8, 4>& colors, int noise, SplitMix64& rng) {
    Array2D<Rgba8> img(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) img(x, y) = jitter(colors[static_cast<std::size_t>((y < h / 2 ? 0 : 2) + (x < w / 2 ? 0 : 1))], noise, rng);
    }
    return img;
}

bool hasColorNear(const cnx::Palette& p, Rgba8 truth, double tol) {
    const sla::Rgba t = sla::toRgba(truth);
    return std::any_of(p.colors.begin(), p.colors.end(), [&](const sla::Rgba& c) { return std::sqrt(sla::distance2(c, t)) <= tol; });
}

}  // namespace

// ================================================================ myPrng

CNX_TEST(prng_matches_the_splitmix64_reference_and_is_deterministic) {
    sla::myPrng a(0);
    CHECK(a.next() == 0xE220A8397B1DCDAFull);
    CHECK(a.next() == 0x6E789E6AA1B965F4ull);
    CHECK(a.next() == 0x06C45D188009454Full);
    sla::myPrng b(12345), c(12345), d(12346);
    bool same = true, differ = false;
    for (int i = 0; i < 100; ++i) {
        const auto x = b.next();
        same = same && x == c.next();
        differ = differ || x != d.next();
    }
    CHECK(same && differ);
    sla::myPrng u(7);
    double sum = 0.0;
    for (int i = 0; i < 20000; ++i) {
        const double v = u.uniform();
        CHECK(v >= 0.0 && v < 1.0);
        sum += v;
    }
    CHECK(std::abs(sum / 20000.0 - 0.5) < 0.01);
    sla::myPrng r(9);
    std::array<int, 7> counts{};
    for (int i = 0; i < 70000; ++i) {
        const auto v = r.below(7);
        CHECK(v < 7);
        ++counts[v];
    }
    for (const int n : counts) CHECK(n > 9000 && n < 11000);  // uniforme (±10 %)
    CHECK(sla::myPrng(5).below(1) == 0);
}

// ================================================================ histograma

CNX_TEST(histogram_cells_keep_exact_weighted_means_and_scatter) {
    SplitMix64 rng{901};
    ColorHistogram h(4);  // faixas grandes: muitas cores por célula
    std::vector<std::pair<Rgba8, double>> samples;
    for (int i = 0; i < 600; ++i) {
        const Rgba8 c{static_cast<std::uint8_t>(rng.below(256)), static_cast<std::uint8_t>(rng.below(256)), static_cast<std::uint8_t>(rng.below(256)), static_cast<std::uint8_t>(rng.below(256))};
        const double w = 0.1 + cnxtest::uniform01(rng);
        samples.emplace_back(c, w);
        h.add(c, w);
    }
    CHECK(h.pixelCount() == 600);
    double total = 0.0;
    for (const auto& s : samples) total += s.second;
    CHECK(std::abs(h.totalWeight() - total) < 1e-9);
    const auto& cells = h.cells();
    CHECK(cells.size() == h.cellCount() && cells.size() > 20);
    double sumW = 0.0;
    std::uint64_t sumPixels = 0;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        if (i > 0) CHECK(cells[i - 1].weight > cells[i].weight || (cells[i - 1].weight == cells[i].weight && cells[i - 1].key < cells[i].key));
        sumW += cells[i].weight;
        sumPixels += cells[i].pixels;
        // Média e espalhamento contra força bruta sobre os pixels da célula.
        Color4d mean{};
        double w = 0.0;
        for (const auto& [c, cw] : samples) {
            if (h.keyOf(c) != cells[i].key) continue;
            const Color4d v = asColor(c);
            w += cw;
            for (std::size_t k = 0; k < 4; ++k) mean[k] += cw * v[k];
        }
        for (double& m : mean) m /= w;
        double scatter = 0.0;
        for (const auto& [c, cw] : samples) {
            if (h.keyOf(c) == cells[i].key) scatter += cw * cnx::distance2(asColor(c), mean);
        }
        CHECK(std::abs(w - cells[i].weight) < 1e-9);
        for (std::size_t k = 0; k < 4; ++k) CHECK(std::abs(mean[k] - cells[i].mean[k]) < 1e-9);
        CHECK(std::abs(scatter - cells[i].scatter) < 1e-9);
    }
    CHECK(std::abs(sumW - total) < 1e-9 && sumPixels == 600);
}

CNX_TEST(histogram_groups_by_bin_and_validates_its_inputs) {
    ColorHistogram h(16);  // faixas de 16 níveis
    h.add(rgb(0, 0, 0));
    h.add(rgb(15, 15, 15));    // mesma célula que (0,0,0)
    h.add(rgb(16, 0, 0));      // outra
    CHECK(h.cellCount() == 2);
    CHECK(h.keyOf({10, 10, 10, 0}) != h.keyOf({10, 10, 10, 255}));  // o alfa é um canal da chave
    CHECK(h.keyOf({0, 0, 0, 0}) < 16u * 16u * 16u * 16u && h.keyOf({255, 255, 255, 255}) == 16u * 16u * 16u * 16u - 1u);
    CHECK(h.keyOf(rgb(0, 0, 0)) == h.keyOf(rgb(15, 15, 15)) && h.keyOf(rgb(0, 0, 0)) != h.keyOf(rgb(16, 0, 0)));
    // Célula de uma cor só: média exata e espalhamento zero.
    ColorHistogram one(16);
    one.add(rgb(200, 100, 50), 3.0);
    one.add(rgb(200, 100, 50), 1.0);
    CHECK(one.cells().size() == 1 && one.cells()[0].scatter == 0.0 && std::abs(one.cells()[0].weight - 4.0) < 1e-12);
    for (std::size_t k = 0; k < 3; ++k) CHECK(std::abs(one.cells()[0].mean[k] - asColor(rgb(200, 100, 50))[k]) < 1e-12);
    CHECK_THROWS(std::invalid_argument, (void)ColorHistogram(1));
    CHECK_THROWS(std::invalid_argument, (void)ColorHistogram(257));
    CHECK_THROWS(std::invalid_argument, h.add(rgb(1, 2, 3), -1.0));
    CHECK_THROWS(std::invalid_argument, h.add(rgb(1, 2, 3), std::nan("")));
    h.add(rgb(9, 9, 9), 0.0);  // peso zero é aceito e conta o pixel
    CHECK(h.pixelCount() == 4);
}

// ================================================================ k-means

CNX_TEST(kmeans_error_equals_the_exact_pixel_level_error_of_the_cell_assignment) {
    SplitMix64 rng{902};
    const std::array<Rgba8, 4> truth{rgb(250, 20, 20), rgb(20, 250, 20), rgb(20, 20, 250), rgb(240, 240, 20)};
    const Array2D<Rgba8> img = quadrants(48, 48, truth, 12, rng);
    ColorHistogram hist(8);
    std::vector<double> wts;
    for (int y = 0; y < 48; ++y) {
        for (int x = 0; x < 48; ++x) {
            const double w = 0.5 + cnxtest::uniform01(rng);
            wts.push_back(w);
            hist.add(img(x, y), w);
        }
    }
    std::vector<KMeansItem> items;
    for (const auto& c : hist.cells()) items.push_back({c.mean, c.weight, c.scatter});
    sla::myPrng prng(3);
    const auto seeds = cnx::chooseSeeds(items, 4, 0.5, prng);
    const cnx::KMeansResult r = cnx::kmeans(items, seeds, 10);
    // Erro de pixel com a atribuição da célula: tem de ser idêntico ao erro do coreset.
    std::map<std::uint32_t, int> cellAssign;
    for (std::size_t i = 0; i < hist.cells().size(); ++i) cellAssign[hist.cells()[i].key] = r.assignment[i];
    double err = 0.0, total = 0.0;
    std::size_t i = 0;
    for (int y = 0; y < 48; ++y) {
        for (int x = 0; x < 48; ++x, ++i) {
            err += wts[i] * cnx::distance2(asColor(img(x, y)), r.centers[static_cast<std::size_t>(cellAssign[hist.keyOf(img(x, y))])]);
            total += wts[i];
        }
    }
    CHECK(std::abs(err / total - r.error) < 1e-9);
    CHECK(r.centers.size() == 4);
    // Os centros ficam perto das cores verdadeiras.
    for (const Rgba8 t : truth) {
        double best = 1e9;
        for (const Color4d& c : r.centers) best = std::min(best, std::sqrt(cnx::distance2(c, asColor(t))));
        CHECK(best < 0.03);
    }
    // Ordenados por peso, e os pesos somam o total.
    double sum = 0.0;
    for (std::size_t k = 0; k < r.clusterWeight.size(); ++k) {
        if (k > 0) CHECK(r.clusterWeight[k - 1] >= r.clusterWeight[k]);
        sum += r.clusterWeight[k];
    }
    CHECK(std::abs(sum - hist.totalWeight()) < 1e-9);
}

CNX_TEST(seeds_take_histogram_peaks_with_suppression_and_fill_the_rest_with_kmeans_plus_plus) {
    // Três picos: dois quase coincidentes (0,50 e 0,52) e um distante (0,95).
    const std::vector<KMeansItem> items{{{0.50, 0, 0, 1}, 100.0, 0.0}, {{0.52, 0, 0, 1}, 90.0, 0.0}, {{0.95, 0, 0, 1}, 50.0, 0.0},
                                        {{0.10, 0, 0, 1}, 5.0, 0.0}};
    sla::myPrng rng(1);
    const auto two = cnx::chooseSeeds(items, 2, 0.1, rng);
    CHECK(two.size() == 2 && two[0][0] == 0.50 && two[1][0] == 0.95);  // o vizinho do pico (0,52) é suprimido
    // k = 3: o terceiro pico válido é o de 5 (0,10).
    sla::myPrng rng3(1);
    const auto three = cnx::chooseSeeds(items, 3, 0.1, rng3);
    CHECK(three.size() == 3 && three[2][0] == 0.10);
    // k = 4: sobra o 0,52, que só pode vir do sorteio k-means++ (D² > 0). Determinístico dada a semente.
    sla::myPrng a(42), b(42);
    const auto fa = cnx::chooseSeeds(items, 4, 0.1, a);
    const auto fb = cnx::chooseSeeds(items, 4, 0.1, b);
    CHECK(fa.size() == 4 && fa == fb);
    CHECK(fa[3][0] == 0.52);
    // Mais sementes que itens distintos: devolve só as possíveis.
    sla::myPrng c(5);
    CHECK(cnx::chooseSeeds(items, 9, 0.1, c).size() == 4);
    CHECK(cnx::chooseSeeds(std::span<const KMeansItem>{}, 3, 0.1, c).empty());
}

CNX_TEST(kmeans_moves_each_center_to_the_weighted_mean_of_its_group) {
    // Sementes em 0,0 e 1,0; o item de 0,3 (peso 4) é mais próximo da primeira. O centro do grupo A tem de
    // ir para a média ponderada (0·10 + 0,3·4)/14, não ficar na semente.
    const std::vector<KMeansItem> items{{{0.0, 0, 0, 1}, 10.0, 0.0}, {{0.3, 0, 0, 1}, 4.0, 0.0}, {{1.0, 0, 0, 1}, 8.0, 0.0}};
    const cnx::KMeansResult r = cnx::kmeans(items, {{0.0, 0, 0, 1}, {1.0, 0, 0, 1}}, 10);
    CHECK(r.centers.size() == 2 && r.converged);
    const double meanA = 1.2 / 14.0;
    CHECK(std::abs(r.centers[0][0] - meanA) < 1e-12);  // grupo A tem o maior peso (14 > 8)
    CHECK(std::abs(r.centers[1][0] - 1.0) < 1e-12);
    const double expected = (10.0 * meanA * meanA + 4.0 * (0.3 - meanA) * (0.3 - meanA)) / 22.0;
    CHECK(std::abs(r.error - expected) < 1e-12);
}

CNX_TEST(refinement_on_the_pixels_corrects_the_cell_level_centers) {
    // Rampa vermelha 0..199 (200 colunas × 16 linhas). Com K = 2 o k-means nas células de 16 níveis parte a rampa
    // numa fronteira de célula e os centros saem a ~6 níveis do ótimo dos pixels (49,5 e 149,5); as duas rodadas
    // de refinamento sobre os pixels os aproximam (≈ 1,5 nível).
    Array2D<Rgba8> ramp(200, 16);
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 200; ++x) ramp(x, y) = rgb(x, 0, 0);
    }
    cnx::PaletteParams params;
    params.forcedNumColors = 2;
    const PaletteFinder finder(params, 1);
    const PaletteResult r = finder.find(ramp, Array2D<float>(200, 16, 1.0f));
    CHECK(!r.exact && r.palette.colors.size() == 2);
    std::array<double, 2> reds{r.palette.colors[0].r * 1.0, r.palette.colors[1].r * 1.0};
    std::sort(reds.begin(), reds.end());
    std::printf("  rampa: centros %.2f e %.2f (níveis; ótimo dos pixels 49,50 e 149,50)\n", reds[0] * 255.0, reds[1] * 255.0);
    CHECK(std::abs(reds[0] - 49.5 / 255.0) < 0.004);   // < 1 nível
    CHECK(std::abs(reds[1] - 149.5 / 255.0) < 0.004);
    // Sem rodadas de refinamento os centros ficam onde o k-means das células os deixou.
    params.refineRounds = 0;
    const PaletteResult raw = PaletteFinder(params, 1).find(ramp, Array2D<float>(200, 16, 1.0f));
    std::array<double, 2> rawReds{raw.palette.colors[0].r * 1.0, raw.palette.colors[1].r * 1.0};
    std::sort(rawReds.begin(), rawReds.end());
    std::printf("  rampa sem refinamento: centros %.2f e %.2f\n", rawReds[0] * 255.0, rawReds[1] * 255.0);
    CHECK(std::abs(rawReds[0] - 49.5 / 255.0) > 0.006);  // ≈ 2 níveis
    CHECK(r.error < raw.error);
}
CNX_TEST(kmeans_discards_empty_centers_and_validates_its_inputs) {
    const std::vector<KMeansItem> items{{{0.0, 0, 0, 1}, 1.0, 0.0}, {{1.0, 0, 0, 1}, 2.0, 0.0}};
    // Quatro sementes para dois grupos reais: duas ficam vazias e são descartadas.
    const cnx::KMeansResult r = cnx::kmeans(items, {{0.0, 0, 0, 1}, {1.0, 0, 0, 1}, {5.0, 5, 5, 1}, {-5.0, 5, 5, 1}}, 10);
    CHECK(r.centers.size() == 2 && r.discarded == 2 && r.error == 0.0 && r.converged);
    CHECK(r.centers[0][0] == 1.0 && r.centers[1][0] == 0.0);  // por peso decrescente (2 > 1)
    CHECK(r.assignment[0] == 1 && r.assignment[1] == 0);
    CHECK_THROWS(std::invalid_argument, (void)cnx::kmeans(std::span<const KMeansItem>{}, std::vector<Color4d>{{0, 0, 0, 1}}, 5));
    CHECK_THROWS(std::invalid_argument, (void)cnx::kmeans(items, std::vector<Color4d>{}, 5));
    const std::vector<KMeansItem> zero{{{0.0, 0, 0, 1}, 0.0, 0.0}};
    CHECK_THROWS(std::invalid_argument, (void)cnx::kmeans(zero, std::vector<Color4d>{{0, 0, 0, 1}}, 5));
}

// ================================================================ PaletteFinder

CNX_TEST(edge_magnitude_and_pixel_weights) {
    Array2D<Rgba8> flat(8, 8, rgb(90, 90, 90));
    const Array2D<float> e0 = PaletteFinder::edgeMagnitude(flat);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) CHECK(std::abs(e0(x, y)) < 1e-6f);
    }
    // Degrau preto|branco em x = 4: RGB muda 1 nos 3 canais → magnitude √3 nas duas colunas vizinhas do degrau.
    Array2D<Rgba8> step(8, 8);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) step(x, y) = x < 4 ? rgb(0, 0, 0) : rgb(255, 255, 255);
    }
    const Array2D<float> e = PaletteFinder::edgeMagnitude(step);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            const bool nearEdge = x == 3 || x == 4;
            CHECK_CTX(std::abs(e(x, y) - (nearEdge ? std::sqrt(3.0f) : 0.0f)) < 1e-5f, std::to_string(x) + "," + std::to_string(y));
        }
    }
    cnx::PaletteParams p;
    p.edgeScale = 0.1;
    const PaletteFinder finder(p, 1);
    const Array2D<float> w = finder.pixelWeights(step);
    CHECK(std::abs(w(0, 0) - 1.0f) < 1e-6f);
    CHECK(std::abs(w(3, 2) - 1.0f / (1.0f + std::sqrt(3.0f) / 0.1f)) < 1e-6f);
    CHECK(w(3, 2) < 0.06f);
}

CNX_TEST(four_noisy_colors_give_k_equal_four_with_the_score_minimum_there) {
    SplitMix64 rng{903};
    const std::array<Rgba8, 4> truth{rgb(250, 20, 20), rgb(20, 250, 20), rgb(20, 20, 250), rgb(240, 240, 20)};
    const Array2D<Rgba8> img = quadrants(64, 64, truth, 6, rng);
    cnx::PaletteParams params;
    const PaletteResult r = PaletteFinder(params, 77).find(img);
    CHECK(!r.exact);
    CHECK(r.chosenK == 4 && r.palette.colors.size() == 4);
    for (const Rgba8 t : truth) CHECK(hasColorNear(r.palette, t, 0.02));
    for (const double f : r.fractions) CHECK(std::abs(f - 0.25) < 0.02);
    CHECK(std::abs(std::accumulate(r.fractions.begin(), r.fractions.end(), 0.0) - 1.0) < 1e-9);
    // K = 2..12 testados; a nota usa a fórmula log(erro+1) + (cores−2)·penalidade, e a escolhida é a menor.
    CHECK(r.candidates.size() == 11);
    double best = 1e9;
    int bestK = 0;
    for (const auto& c : r.candidates) {
        CHECK(std::abs(c.score - (std::log(c.error + 1.0) + (c.colors - 2) * params.colorPenalty)) < 1e-12);
        CHECK(c.colors <= c.requested);
        if (c.score < best) {
            best = c.score;
            bestK = c.colors;
        }
    }
    CHECK(bestK == 4 && std::abs(r.palette.score - best) < 1e-12);
    // O erro cai muito de K=3 para K=4 e quase nada depois.
    CHECK(r.candidates[1].error > 10.0 * r.candidates[2].error);
    CHECK(r.error < 0.001);  // ruído de ±6 níveis → erro ≈ 3·(6²/3)/255² ≈ 5e-4
    std::printf("  4 cores com ruído: K=%d, erro %.6f, notas K=2..6: %.3f %.3f %.3f %.3f %.3f\n", r.chosenK, r.error, r.candidates[0].score,
                r.candidates[1].score, r.candidates[2].score, r.candidates[3].score, r.candidates[4].score);
}

CNX_TEST(flat_images_use_their_exact_colors_and_ignore_rare_outliers) {
    SplitMix64 rng{904};
    const std::array<Rgba8, 5> colors{rgb(255, 255, 255), rgb(0, 0, 0), rgb(200, 30, 30), rgb(30, 200, 30), rgb(30, 30, 200)};
    Array2D<Rgba8> img(64, 64);
    std::array<int, 5> counts{};
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            const int k = (x / 16 + 2 * (y / 32)) % 5;
            img(x, y) = colors[static_cast<std::size_t>(k)];
            ++counts[static_cast<std::size_t>(k)];
        }
    }
    const PaletteResult r = PaletteFinder(cnx::PaletteParams{}, 5).find(img);
    CHECK(r.exact && r.chosenK == 5 && r.palette.colors.size() == 5 && r.distinctColors == 5);
    CHECK(r.candidates.empty());
    for (const Rgba8 c : colors) CHECK(hasColorNear(r.palette, c, 1e-6));
    for (std::size_t i = 0; i < r.fractions.size(); ++i) {
        if (i > 0) CHECK(r.fractions[i - 1] >= r.fractions[i]);
    }
    // Três pixels de uma sexta cor (0,07 % < 0,2 %) não contam: continua chapada com 5 cores, 6 distintas.
    img(0, 0) = rgb(123, 45, 67);
    img(1, 0) = rgb(123, 45, 67);
    img(2, 0) = rgb(123, 45, 67);
    const PaletteResult r2 = PaletteFinder(cnx::PaletteParams{}, 5).find(img);
    CHECK(r2.exact && r2.palette.colors.size() == 5 && r2.distinctColors == 6);
    // Uma cor só.
    const PaletteResult one = PaletteFinder(cnx::PaletteParams{}, 5).find(Array2D<Rgba8>(8, 8, rgb(10, 20, 30)));
    CHECK(one.exact && one.palette.colors.size() == 1 && one.fractions[0] == 1.0);
    // Mais de 12 cores significativas: não é chapada.
    Array2D<Rgba8> many(26, 26);
    for (int y = 0; y < 26; ++y) {
        for (int x = 0; x < 26; ++x) many(x, y) = rgb((x + 26 * y) % 40 * 6, (x * 7) % 256, (y * 11) % 256);
    }
    CHECK(!PaletteFinder(cnx::PaletteParams{}, 5).find(many).exact);
}

CNX_TEST(forced_number_of_colors_skips_the_search) {
    SplitMix64 rng{905};
    const std::array<Rgba8, 4> truth{rgb(250, 20, 20), rgb(20, 250, 20), rgb(20, 20, 250), rgb(240, 240, 20)};
    const Array2D<Rgba8> img = quadrants(64, 64, truth, 6, rng);
    cnx::PaletteParams params;
    params.forcedNumColors = 3;
    const PaletteResult r = PaletteFinder(params, 1).find(img);
    CHECK(r.chosenK == 3 && r.palette.colors.size() == 3 && r.candidates.size() == 1);
    // Forçar mais cores que o natural: centros vazios são descartados, nunca passa do pedido.
    params.forcedNumColors = 9;
    const PaletteResult r9 = PaletteFinder(params, 1).find(img);
    CHECK(r9.palette.colors.size() <= 9 && r9.palette.colors.size() >= 4);
    // Forçado ignora a detecção de imagem chapada.
    params.forcedNumColors = 2;
    CHECK(!PaletteFinder(params, 1).find(Array2D<Rgba8>(8, 8, rgb(10, 20, 30))).exact);
}

CNX_TEST(edge_weights_keep_transition_bands_from_becoming_a_false_color) {
    // Esquerda branca, banda de 4 colunas de cinza (10 % dos pixels), direita preta. Sem pesos, a banda
    // custa mais que a penalidade de uma cor e vira a terceira cor; com os pesos de borda, só as 2 colunas
    // internas (sem gradiente) valem peso cheio e a banda fica abaixo da penalidade.
    Array2D<Rgba8> img(40, 40);
    for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 40; ++x) img(x, y) = x < 18 ? rgb(255, 255, 255) : (x < 22 ? rgb(128, 128, 128) : rgb(0, 0, 0));
    }
    cnx::PaletteParams params;
    params.colorPenalty = 0.052;
    params.exactColorMinShare = 0.2;  // a banda (10 %) não conta como cor exata: o teste é do k-means
    const PaletteFinder finder(params, 3);
    const PaletteResult weighted = finder.find(img);
    Array2D<float> ones(40, 40, 1.0f);
    const PaletteResult unweighted = finder.find(img, ones);
    std::printf("  banda de transição: K=%d com pesos de borda (nota K=2 %.4f), K=%d sem (nota K=2 %.4f); penalidade %.3f\n", weighted.chosenK,
                weighted.candidates[0].score, unweighted.chosenK, unweighted.candidates[0].score, params.colorPenalty);
    CHECK(weighted.chosenK == 2);
    CHECK(unweighted.chosenK == 3);
    // O cinza restante (≈ 6 % do peso) puxa o centro do seu grupo em ≈ 0,06: as cores continuam perto do branco e do preto.
    CHECK(hasColorNear(weighted.palette, rgb(255, 255, 255), 0.1) && hasColorNear(weighted.palette, rgb(0, 0, 0), 0.1));
}

CNX_TEST(an_antialiased_disk_gives_a_two_color_palette) {
    const auto scene = cnxtest::aaScene(48, 48, [](double x, double y) { return (x - 24.3) * (x - 24.3) + (y - 23.8) * (y - 23.8) <= 14.0 * 14.0; }, 8);
    Array2D<Rgba8> img(48, 48);
    for (int y = 0; y < 48; ++y) {
        for (int x = 0; x < 48; ++x) img(x, y) = sla::toRgba8(scene.target(x, y));
    }
    const PaletteResult r = PaletteFinder(cnx::PaletteParams{}, 11).find(img);
    CHECK(!r.exact);  // as cores intermediárias do anel (4 % dos pixels) impedem a detecção de imagem chapada
    CHECK(r.chosenK == 2 && r.palette.colors.size() == 2);
    CHECK(hasColorNear(r.palette, rgb(255, 255, 255), 0.02) && hasColorNear(r.palette, rgb(0, 0, 0), 0.02));
    // A fração escura é a área do disco (com o peso reduzido nas bordas): ≈ π·14² / 48².
    double dark = 0.0;
    for (std::size_t i = 0; i < r.palette.colors.size(); ++i) {
        if (r.palette.colors[i].r < 0.5f) dark = r.fractions[i];
    }
    CHECK(std::abs(dark - 3.14159265 * 14.0 * 14.0 / (48.0 * 48.0)) < 0.03);
}

CNX_TEST(the_result_is_deterministic_and_the_seed_only_matters_for_the_random_fill) {
    SplitMix64 rng{906};
    const std::array<Rgba8, 4> truth{rgb(250, 20, 20), rgb(20, 250, 20), rgb(20, 20, 250), rgb(240, 240, 20)};
    const Array2D<Rgba8> img = quadrants(64, 64, truth, 30, rng);
    cnx::PaletteParams params;
    params.forcedNumColors = 10;  // bem mais que os picos: o k-means++ entra
    const PaletteResult a = PaletteFinder(params, 123).find(img);
    const PaletteResult b = PaletteFinder(params, 123).find(img);
    CHECK(a.palette.colors.size() == b.palette.colors.size());
    bool same = true;
    for (std::size_t i = 0; i < a.palette.colors.size(); ++i) same = same && a.palette.colors[i] == b.palette.colors[i];
    CHECK(same && a.error == b.error);
    // Sem sorteio (K = 4 = nº de picos separados), a semente é irrelevante.
    params.forcedNumColors = 4;
    const PaletteResult c = PaletteFinder(params, 1).find(img);
    const PaletteResult d = PaletteFinder(params, 999).find(img);
    bool equal = c.palette.colors.size() == d.palette.colors.size();
    for (std::size_t i = 0; equal && i < c.palette.colors.size(); ++i) equal = c.palette.colors[i] == d.palette.colors[i];
    CHECK(equal);
}

CNX_TEST(argument_validation) {
    const auto bad = [](auto mutate) {
        cnx::PaletteParams p;
        mutate(p);
        return p;
    };
    CHECK_THROWS(std::invalid_argument, (void)PaletteFinder(bad([](cnx::PaletteParams& p) { p.minColors = 0; }), 1));
    CHECK_THROWS(std::invalid_argument, (void)PaletteFinder(bad([](cnx::PaletteParams& p) { p.maxColors = 1; }), 1));
    CHECK_THROWS(std::invalid_argument, (void)PaletteFinder(bad([](cnx::PaletteParams& p) { p.histogramBins = 1; }), 1));
    CHECK_THROWS(std::invalid_argument, (void)PaletteFinder(bad([](cnx::PaletteParams& p) { p.colorPenalty = -1.0; }), 1));
    CHECK_THROWS(std::invalid_argument, (void)PaletteFinder(bad([](cnx::PaletteParams& p) { p.edgeScale = 0.0; }), 1));
    CHECK_THROWS(std::invalid_argument, (void)PaletteFinder(bad([](cnx::PaletteParams& p) { p.forcedNumColors = 0; }), 1));
    const PaletteFinder ok(cnx::PaletteParams{}, 1);
    CHECK_THROWS(std::invalid_argument, (void)ok.find(Array2D<Rgba8>()));
    CHECK_THROWS(std::invalid_argument, (void)ok.find(Array2D<Rgba8>(4, 4, rgb(1, 2, 3)), Array2D<float>(3, 4)));
    SplitMix64 rng{907};
    Array2D<Rgba8> noisy(20, 20);
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) noisy(x, y) = rgb(rng.below(256), rng.below(256), rng.below(256));
    }
    CHECK_THROWS(std::invalid_argument, (void)ok.find(noisy, Array2D<float>(20, 20, 0.0f)));  // pesos todos nulos
}

CNX_TEST(timing_report) {
    SplitMix64 rng{908};
    const std::array<Rgba8, 4> truth{rgb(250, 20, 20), rgb(20, 250, 20), rgb(20, 20, 250), rgb(240, 240, 20)};
    const Array2D<Rgba8> img = quadrants(512, 512, truth, 10, rng);
    const auto t0 = std::chrono::steady_clock::now();
    const PaletteResult r = PaletteFinder(cnx::PaletteParams{}, 1).find(img);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("  512x512: %.0f ms, K=%d, erro %.6f\n", std::chrono::duration<double, std::milli>(t1 - t0).count(), r.chosenK, r.error);
    CHECK(r.chosenK == 4);
}

int main() { return cnxtest::runAll(); }
