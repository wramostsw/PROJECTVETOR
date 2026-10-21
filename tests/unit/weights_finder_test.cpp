// tests/unit/weights_finder_test.cpp — WeightsFinder: mínimos quadrados no simplex contra enumeração
// exaustiva de suportes, propriedades de otimalidade e casos degenerados.
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "cnx/palette/WeightsFinder.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::Palette;
using cnx::WeightsFinder;
using cnxtest::SplitMix64;

namespace {

using Color4 = std::array<double, 4>;

double dot4(const Color4& a, const Color4& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }

Palette randomPalette(std::size_t n, SplitMix64& rng) {
    Palette p;
    for (std::size_t i = 0; i < n; ++i) {
        p.colors.push_back({static_cast<float>(cnxtest::uniform01(rng)), static_cast<float>(cnxtest::uniform01(rng)),
                            static_cast<float>(cnxtest::uniform01(rng)), static_cast<float>(cnxtest::uniform01(rng))});
    }
    return p;
}

Color4 asColor(const sla::Rgba& c) { return {c.r, c.g, c.b, c.a}; }

/// Gauss com pivô parcial; falso se singular.
bool gauss(std::vector<double> a, std::vector<double> b, std::size_t n, std::vector<double>& x) {
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < n; ++r) {
            if (std::abs(a[r * n + c]) > std::abs(a[piv * n + c])) piv = r;
        }
        if (std::abs(a[piv * n + c]) < 1e-10) return false;
        for (std::size_t j = 0; j < n; ++j) std::swap(a[c * n + j], a[piv * n + j]);
        std::swap(b[c], b[piv]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const double f = a[r * n + c] / a[c * n + c];
            for (std::size_t j = c; j < n; ++j) a[r * n + j] -= f * a[c * n + j];
            b[r] -= f * b[c];
        }
    }
    x.assign(n, 0.0);
    for (std::size_t ii = n; ii-- > 0;) {
        double v = b[ii];
        for (std::size_t j = ii + 1; j < n; ++j) v -= a[ii * n + j] * x[j];
        x[ii] = v / a[ii * n + ii];
    }
    return true;
}

/// Referência independente: o ótimo do problema no simplex está num suporte afinamente independente de
/// ≤ 5 cores. Enumera todos os suportes, resolve o problema só com a igualdade Σ z = 1 e fica com o melhor
/// resultado viável (z ≥ 0). Devolve o menor resíduo ‖Σ z_i c_i − p‖².
double bruteForceResidual(const Palette& pal, const Color4& p) {
    const std::size_t n = pal.colors.size();
    double best = 1e300;
    for (unsigned mask = 1; mask < (1u << n); ++mask) {
        std::vector<std::size_t> s;
        for (std::size_t i = 0; i < n; ++i) {
            if (mask & (1u << i)) s.push_back(i);
        }
        if (s.size() > 5) continue;
        const std::size_t m = s.size();
        std::vector<double> a((m + 1) * (m + 1), 0.0), b(m + 1, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) a[i * (m + 1) + j] = dot4(asColor(pal.colors[s[i]]), asColor(pal.colors[s[j]]));
            a[i * (m + 1) + m] = a[m * (m + 1) + i] = 1.0;
            b[i] = dot4(asColor(pal.colors[s[i]]), p);
        }
        b[m] = 1.0;
        std::vector<double> x;
        if (!gauss(a, b, m + 1, x)) continue;
        bool feasible = true;
        Color4 sum{};
        for (std::size_t i = 0; i < m; ++i) {
            feasible = feasible && x[i] >= -1e-12;
            for (std::size_t d = 0; d < 4; ++d) sum[d] += x[i] * asColor(pal.colors[s[i]])[d];
        }
        if (!feasible) continue;
        const Color4 r{sum[0] - p[0], sum[1] - p[1], sum[2] - p[2], sum[3] - p[3]};
        best = std::min(best, dot4(r, r));
    }
    return best;
}

}  // namespace

CNX_TEST(two_colors_are_decomposed_by_projection_onto_the_segment) {
    Palette pal;
    pal.colors = {{1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}};
    const WeightsFinder f(pal);
    // 30 % vermelho + 70 % azul: pesos exatos, resíduo nulo.
    const sla::Rgba mix{0.3f, 0.0f, 0.7f, 1.0f};
    const cnx::PixelWeights w = f.solve(mix);
    CHECK(std::abs(w.w[0] - 0.3) < 1e-6 && std::abs(w.w[1] - 0.7) < 1e-6 && w.residual < 1e-12 && w.support == 2);
    // Fora do segmento: o peso satura em 0 ou 1 (t ∈ [0, 1]).
    const cnx::PixelWeights beyond = f.solve({0.0f, 0.0f, 2.0f, 1.0f});
    CHECK(beyond.w[0] == 0.0 && beyond.w[1] == 1.0 && beyond.support == 1);
    CHECK(std::abs(beyond.residual - 1.0) < 1e-12);  // distância 1 do azul
    // Fora da reta: projeta. Pixel (0,5; 1; 0,5; 1): projeção em t = 0,5, resíduo = 1.
    const cnx::PixelWeights off = f.solve({0.5f, 1.0f, 0.5f, 1.0f});
    CHECK(std::abs(off.w[0] - 0.5) < 1e-6 && std::abs(off.residual - 1.0) < 1e-12);
    // Cores idênticas: sem divisão por zero.
    Palette same;
    same.colors = {{0.2f, 0.2f, 0.2f, 1.0f}, {0.2f, 0.2f, 0.2f, 1.0f}};
    const cnx::PixelWeights sw = WeightsFinder(same).solve({0.5f, 0.5f, 0.5f, 1.0f});
    CHECK(std::isfinite(sw.w[0]) && std::abs(sw.w[0] + sw.w[1] - 1.0) < 1e-12);
}

CNX_TEST(three_or_more_colors_recover_an_exact_mixture) {
    SplitMix64 rng{1001};
    for (const std::size_t n : {3u, 4u, 5u, 7u, 9u}) {
        for (int trial = 0; trial < 20; ++trial) {
            const Palette pal = randomPalette(n, rng);
            // Mistura com suporte de até 5 cores (afinamente independentes com probabilidade 1).
            std::vector<double> truth(n, 0.0);
            const std::size_t support = std::min<std::size_t>(n, 5);
            double sum = 0.0;
            for (std::size_t i = 0; i < support; ++i) {
                truth[i] = 0.1 + cnxtest::uniform01(rng);
                sum += truth[i];
            }
            for (double& v : truth) v /= sum;
            const WeightsFinder f(pal);
            const sla::Rgba pixel = f.blend(truth);
            const cnx::PixelWeights r = f.solve(pixel);
            CHECK_CTX(r.residual < 1e-10, "n=" + std::to_string(n) + " tentativa " + std::to_string(trial));
            // A mistura reproduz o pixel. Com n ≤ 5 as cores são afinamente independentes e os pesos são únicos;
            // com mais cores há várias misturas exatas (dependência afim), e só o resíduo e o blend são exigidos.
            const sla::Rgba again = f.blend(r.w);
            CHECK_CTX(sla::distance2(again, pixel) < 1e-10, "n=" + std::to_string(n));
            if (n <= 5) {
                for (std::size_t i = 0; i < n; ++i) CHECK_CTX(std::abs(r.w[i] - truth[i]) < 1e-4, "n=" + std::to_string(n) + " cor " + std::to_string(i));
            }
        }
    }
}

CNX_TEST(the_solution_matches_exhaustive_enumeration_of_supports) {
    SplitMix64 rng{1002};
    int outside = 0;
    for (const std::size_t n : {3u, 4u, 6u, 8u, 9u}) {
        for (int trial = 0; trial < 25; ++trial) {
            const Palette pal = randomPalette(n, rng);
            // Pixels dentro e fora do fecho convexo (alguns bem fora: o ótimo fica numa face do simplex).
            const double spread = trial % 2 == 0 ? 0.5 : 2.0;
            const sla::Rgba pixel{static_cast<float>(0.5 + cnxtest::symmetric(rng, spread)), static_cast<float>(0.5 + cnxtest::symmetric(rng, spread)),
                                  static_cast<float>(0.5 + cnxtest::symmetric(rng, spread)), static_cast<float>(0.5 + cnxtest::symmetric(rng, spread))};
            const WeightsFinder f(pal, 9);
            const cnx::PixelWeights r = f.solve(pixel);
            const double ref = bruteForceResidual(pal, asColor(pixel));
            const std::string tag = "n=" + std::to_string(n) + " tentativa " + std::to_string(trial);
            CHECK_CTX(std::abs(r.residual - ref) < 1e-9, tag + ": " + std::to_string(r.residual) + " vs " + std::to_string(ref));
            // Viabilidade: simplex.
            double sum = 0.0;
            for (const double v : r.w) {
                CHECK_CTX(v >= 0.0, tag);
                sum += v;
            }
            CHECK_CTX(std::abs(sum - 1.0) < 1e-12, tag);
            // Carathéodory: suporte de no máximo 5 cores em ℝ⁴.
            CHECK_CTX(r.support <= 5, tag);
            outside += r.residual > 1e-6 ? 1 : 0;
        }
    }
    CHECK(outside > 30);  // o teste só vale se muitos pixels ficam fora do fecho convexo
}

CNX_TEST(a_pixel_equal_to_a_palette_color_is_one_hot_and_collinear_palettes_still_work) {
    SplitMix64 rng{1003};
    const Palette pal = randomPalette(6, rng);
    const WeightsFinder f(pal);
    for (std::size_t i = 0; i < 6; ++i) {
        const cnx::PixelWeights r = f.solve(pal.colors[i]);
        CHECK(r.residual < 1e-12);
        CHECK(std::abs(r.w[i] - 1.0) < 1e-9);
    }
    // Três cores colineares (a do meio é a média das pontas): o sistema KKT de 3 cores é singular.
    Palette col;
    col.colors = {{0.0f, 0.0f, 0.0f, 1.0f}, {0.5f, 0.5f, 0.5f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};
    const WeightsFinder cf(col);
    const cnx::PixelWeights r = cf.solve({0.25f, 0.25f, 0.25f, 1.0f});
    CHECK(r.residual < 1e-12);
    double sum = 0.0;
    for (const double v : r.w) {
        CHECK(v >= 0.0);
        sum += v;
    }
    CHECK(std::abs(sum - 1.0) < 1e-12);
    CHECK(std::abs(cf.blend(r.w).r - 0.25f) < 1e-6f);
}

CNX_TEST(many_colors_use_only_the_nearest_candidates_and_subsets_are_respected) {
    SplitMix64 rng{1004};
    const Palette pal = randomPalette(14, rng);
    const WeightsFinder f(pal, 5);
    const sla::Rgba pixel{0.4f, 0.5f, 0.6f, 0.9f};
    const std::vector<std::size_t> near = f.nearest(pixel, 5);
    CHECK(near.size() == 5);
    for (std::size_t k = 1; k < near.size(); ++k) CHECK(sla::distance2(pixel, pal.colors[near[k - 1]]) <= sla::distance2(pixel, pal.colors[near[k]]));
    const cnx::PixelWeights r = f.solve(pixel);
    // Só as 5 mais próximas têm peso.
    for (std::size_t i = 0; i < 14; ++i) {
        if (std::find(near.begin(), near.end(), i) == near.end()) CHECK(r.w[i] == 0.0);
    }
    // Idêntico a resolver o subconjunto diretamente.
    const cnx::PixelWeights s = f.solveSubset(pixel, near);
    CHECK(r.residual == s.residual);
    // Subconjunto menor que a paleta: pesos zero fora dele.
    const std::vector<std::size_t> two{3, 9};
    const cnx::PixelWeights t = f.solveSubset(pixel, two);
    for (std::size_t i = 0; i < 14; ++i) {
        if (i != 3 && i != 9) CHECK(t.w[i] == 0.0);
    }
    CHECK(std::abs(t.w[3] + t.w[9] - 1.0) < 1e-12);
    // Usar mais cores nunca piora: o resíduo com as 5 mais próximas é ≤ o com apenas 2 delas.
    const std::vector<std::size_t> twoNearest{near[0], near[1]};
    CHECK(s.residual <= f.solveSubset(pixel, twoNearest).residual + 1e-12);
}

CNX_TEST(argument_validation) {
    CHECK_THROWS(std::invalid_argument, (void)WeightsFinder(Palette{}));
    SplitMix64 rng{1005};
    const Palette pal = randomPalette(4, rng);
    CHECK_THROWS(std::invalid_argument, (void)WeightsFinder(pal, 0));
    const WeightsFinder f(pal);
    const sla::Rgba px{0.5f, 0.5f, 0.5f, 1.0f};
    CHECK_THROWS(std::invalid_argument, (void)f.solveSubset(px, std::vector<std::size_t>{}));
    CHECK_THROWS(std::invalid_argument, (void)f.solveSubset(px, std::vector<std::size_t>{0, 4}));  // fora da paleta
    CHECK_THROWS(std::invalid_argument, (void)f.solveSubset(px, std::vector<std::size_t>{1, 1}));  // repetido
    CHECK_THROWS(std::invalid_argument, (void)f.blend(std::vector<double>{1.0}));
    CHECK_THROWS(std::invalid_argument, (void)WeightsFinder::simplexLeastSquares(std::span<const std::array<double, 4>>{}, {0, 0, 0, 0}));
    const std::vector<std::array<double, 4>> single{{0.2, 0.3, 0.4, 1.0}};
    CHECK(WeightsFinder::simplexLeastSquares(single, {0, 0, 0, 0}) == std::vector<double>{1.0});
}

int main() { return cnxtest::runAll(); }
