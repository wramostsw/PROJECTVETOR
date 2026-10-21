// src/palette/WeightsFinder.cpp — mínimos quadrados no simplex por conjuntos ativos.
#include "cnx/palette/WeightsFinder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cnx {
namespace {

using Color4 = std::array<double, 4>;

double dot4(const Color4& a, const Color4& b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }

/// Resolve o sistema denso n×n (linha a linha) por eliminação gaussiana com pivô parcial.
/// Falso se algum pivô for pequeno demais (sistema singular).
bool solveDense(std::vector<double> a, std::vector<double> b, std::size_t n, std::vector<double>& x) {
    double scale = 0.0;
    for (const double v : a) scale = std::max(scale, std::abs(v));
    if (!(scale > 0.0)) return false;
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < n; ++r) {
            if (std::abs(a[r * n + c]) > std::abs(a[piv * n + c])) piv = r;
        }
        if (std::abs(a[piv * n + c]) < 1e-11 * scale) return false;
        if (piv != c) {
            for (std::size_t j = 0; j < n; ++j) std::swap(a[c * n + j], a[piv * n + j]);
            std::swap(b[c], b[piv]);
        }
        for (std::size_t r = c + 1; r < n; ++r) {
            const double f = a[r * n + c] / a[c * n + c];
            if (f == 0.0) continue;
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

/// min ‖Σ_{i∈S} z_i c_i − p‖² sujeito a Σ z_i = 1 (sem a restrição z ≥ 0), pelo sistema KKT.
bool equalityConstrained(std::span<const Color4> colors, const std::vector<std::size_t>& s, const Color4& p, std::vector<double>& z) {
    const std::size_t m = s.size();
    const std::size_t n = m + 1;
    std::vector<double> a(n * n, 0.0), b(n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) a[i * n + j] = dot4(colors[s[i]], colors[s[j]]);
        a[i * n + m] = 1.0;
        a[m * n + i] = 1.0;
        b[i] = dot4(colors[s[i]], p);
    }
    b[m] = 1.0;
    std::vector<double> x;
    if (!solveDense(std::move(a), std::move(b), n, x)) return false;
    z.assign(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(m));
    return true;
}

}  // namespace

WeightsFinder::WeightsFinder(Palette palette, int maxCandidates) : palette_(std::move(palette)), maxCandidates_(maxCandidates) {
    if (palette_.colors.empty()) throw std::invalid_argument("WeightsFinder: paleta vazia");
    if (maxCandidates < 1) throw std::invalid_argument("WeightsFinder: maxCandidates < 1");
}

std::vector<double> WeightsFinder::simplexLeastSquares(std::span<const Color4> colors, const Color4& p) {
    const std::size_t n = colors.size();
    if (n == 0) throw std::invalid_argument("simplexLeastSquares: sem cores");
    std::vector<double> w(n, 0.0);
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }
    if (n == 2) {  // projeção na reta, t = fração da segunda cor
        const Color4 d{colors[1][0] - colors[0][0], colors[1][1] - colors[0][1], colors[1][2] - colors[0][2], colors[1][3] - colors[0][3]};
        const Color4 q{p[0] - colors[0][0], p[1] - colors[0][1], p[2] - colors[0][2], p[3] - colors[0][3]};
        const double dd = dot4(d, d);
        const double t = dd > 0.0 ? std::clamp(dot4(q, d) / dd, 0.0, 1.0) : 0.0;
        w[0] = 1.0 - t;
        w[1] = t;
        return w;
    }

    // Início: o vértice (cor única) mais próximo.
    std::size_t first = 0;
    double bestD = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const Color4 r{colors[i][0] - p[0], colors[i][1] - p[1], colors[i][2] - p[2], colors[i][3] - p[3]};
        const double d = dot4(r, r);
        if (d < bestD) {
            bestD = d;
            first = i;
        }
    }
    w[first] = 1.0;
    std::vector<std::size_t> support{first};
    std::vector<char> banned(n, 0);
    constexpr double kTol = 1e-12;
    const auto gradient = [&](std::size_t i) {
        Color4 r{-p[0], -p[1], -p[2], -p[3]};
        for (std::size_t k = 0; k < n; ++k) {
            if (w[k] == 0.0) continue;
            for (std::size_t d = 0; d < 4; ++d) r[d] += w[k] * colors[k][d];
        }
        return dot4(colors[i], r);  // ∂(½‖Cw − p‖²)/∂w_i
    };

    for (std::size_t outer = 0; outer < 20 * n + 50; ++outer) {
        // λ: valor comum do gradiente no suporte (média ponderada).
        double lambda = 0.0;
        for (const std::size_t i : support) lambda += w[i] * gradient(i);
        // Variável fora do suporte que mais reduziria o objetivo.
        std::size_t cand = n;
        double candG = lambda - kTol;
        for (std::size_t i = 0; i < n; ++i) {
            if (banned[i] || std::find(support.begin(), support.end(), i) != support.end()) continue;
            const double g = gradient(i);
            if (g < candG) {
                candG = g;
                cand = i;
            }
        }
        if (cand == n) break;  // KKT satisfeito: ótimo
        support.push_back(cand);

        // Laço interno: resolve no suporte e recua até a primeira restrição w ≥ 0 que seria violada.
        for (std::size_t inner = 0; inner < n + 5; ++inner) {
            std::vector<double> z;
            if (!equalityConstrained(colors, support, p, z)) {  // cores dependentes: a candidata não serve
                support.pop_back();
                banned[cand] = 1;
                break;
            }
            bool allPositive = true;
            for (const double v : z) allPositive = allPositive && v > kTol;
            if (allPositive) {
                std::fill(w.begin(), w.end(), 0.0);
                for (std::size_t k = 0; k < support.size(); ++k) w[support[k]] = z[k];
                break;
            }
            // Passo parcial de w até z, parando quando algum peso chega a zero.
            double alpha = 1.0;
            for (std::size_t k = 0; k < support.size(); ++k) {
                if (z[k] <= kTol) {
                    const double wk = w[support[k]];
                    const double denom = wk - z[k];
                    if (denom > 0.0) alpha = std::min(alpha, wk / denom);
                }
            }
            for (std::size_t k = 0; k < support.size(); ++k) w[support[k]] += alpha * (z[k] - w[support[k]]);
            std::vector<std::size_t> kept;
            for (const std::size_t i : support) {
                if (w[i] > 1e-14) kept.push_back(i);
                else w[i] = 0.0;
            }
            if (kept.empty()) kept.push_back(first);
            const bool dropped = kept.size() != support.size();
            support = std::move(kept);
            if (std::find(support.begin(), support.end(), cand) == support.end()) banned[cand] = 1;  // não entrou de fato
            if (!dropped) break;
        }
    }

    double sum = 0.0;
    for (double& v : w) {
        v = std::max(0.0, v);
        sum += v;
    }
    for (double& v : w) v /= sum;
    return w;
}

std::vector<std::size_t> WeightsFinder::nearest(const sla::Rgba& pixel, std::size_t n) const {
    std::vector<std::size_t> idx(palette_.colors.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        return sla::distance2(pixel, palette_.colors[a]) < sla::distance2(pixel, palette_.colors[b]);
    });
    idx.resize(std::min(n, idx.size()));
    return idx;
}

sla::Rgba WeightsFinder::blend(std::span<const double> w) const {
    if (w.size() != palette_.colors.size()) throw std::invalid_argument("WeightsFinder::blend: um peso por cor da paleta");
    Color4 s{};
    for (std::size_t i = 0; i < w.size(); ++i) {
        const sla::Rgba& c = palette_.colors[i];
        s[0] += w[i] * c.r;
        s[1] += w[i] * c.g;
        s[2] += w[i] * c.b;
        s[3] += w[i] * c.a;
    }
    return {static_cast<float>(s[0]), static_cast<float>(s[1]), static_cast<float>(s[2]), static_cast<float>(s[3])};
}

PixelWeights WeightsFinder::solveSubset(const sla::Rgba& pixel, std::span<const std::size_t> subset) const {
    if (subset.empty()) throw std::invalid_argument("WeightsFinder::solveSubset: subconjunto vazio");
    std::vector<char> seen(palette_.colors.size(), 0);
    std::vector<Color4> colors;
    for (const std::size_t i : subset) {
        if (i >= palette_.colors.size() || seen[i]) throw std::invalid_argument("WeightsFinder::solveSubset: índice inválido ou repetido");
        seen[i] = 1;
        const sla::Rgba& c = palette_.colors[i];
        colors.push_back({c.r, c.g, c.b, c.a});
    }
    const Color4 p{pixel.r, pixel.g, pixel.b, pixel.a};
    const std::vector<double> local = simplexLeastSquares(colors, p);
    PixelWeights out;
    out.w.assign(palette_.colors.size(), 0.0);
    Color4 s{};
    for (std::size_t k = 0; k < subset.size(); ++k) {
        out.w[subset[k]] = local[k];
        out.support += local[k] > 0.0 ? 1 : 0;
        for (std::size_t d = 0; d < 4; ++d) s[d] += local[k] * colors[k][d];
    }
    const Color4 r{s[0] - p[0], s[1] - p[1], s[2] - p[2], s[3] - p[3]};
    out.residual = dot4(r, r);
    return out;
}

PixelWeights WeightsFinder::solve(const sla::Rgba& pixel) const {
    const std::size_t n = palette_.colors.size();
    if (n <= static_cast<std::size_t>(maxCandidates_)) {
        std::vector<std::size_t> all(n);
        std::iota(all.begin(), all.end(), std::size_t{0});
        return solveSubset(pixel, all);
    }
    const std::vector<std::size_t> near = nearest(pixel, static_cast<std::size_t>(maxCandidates_));
    return solveSubset(pixel, near);
}

}  // namespace cnx
