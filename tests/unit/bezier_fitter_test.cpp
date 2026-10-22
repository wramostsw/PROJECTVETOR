// tests/unit/bezier_fitter_test.cpp — BezierFitter: derivadas e passo de Schur contra referência densa,
// recuperação de curvas conhecidas, G¹, fusão com limiar crescente e integração com o VectorImage.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stop_token>
#include <string>
#include <vector>

#include "cnx/fit/BezierFitter.h"
#include "cnx/vector/ContourSmoother.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::BezierFitter;
using cnx::FitDerivatives;
using cnx::FitParams;
using cnx::FragmentEnds;
using cnx::FragmentModel;
using sla::BezierCurve;
using sla::Vec2;
using cnxtest::SplitMix64;

namespace {

constexpr double kPi = std::numbers::pi;

std::vector<Vec2> sampleCurve(const BezierCurve& c, const std::vector<double>& ts) {
    std::vector<Vec2> pts;
    for (const double t : ts) pts.push_back(c.eval(t));
    return pts;
}

std::vector<double> uniformT(std::size_t m) {
    std::vector<double> t(m);
    for (std::size_t i = 0; i < m; ++i) t[i] = static_cast<double>(i) / static_cast<double>(m - 1);
    return t;
}

/// Resolve o sistema denso n×n por eliminação gaussiana com pivô parcial.
std::vector<double> denseSolve(std::vector<double> a, std::vector<double> b, std::size_t n) {
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < n; ++r) {
            if (std::abs(a[r * n + c]) > std::abs(a[piv * n + c])) piv = r;
        }
        for (std::size_t j = 0; j < n; ++j) std::swap(a[c * n + j], a[piv * n + j]);
        std::swap(b[c], b[piv]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const double f = a[r * n + c] / a[c * n + c];
            for (std::size_t j = c; j < n; ++j) a[r * n + j] -= f * a[c * n + j];
            b[r] -= f * b[c];
        }
    }
    std::vector<double> x(n);
    for (std::size_t ii = n; ii-- > 0;) {
        double v = b[ii];
        for (std::size_t j = ii + 1; j < n; ++j) v -= a[ii * n + j] * x[j];
        x[ii] = v / a[ii * n + ii];
    }
    return x;
}

double distanceToChain(const std::vector<sla::CurveSegment>& chain, Vec2 q) {
    double best = 1e300;
    for (const sla::CurveSegment& s : chain) {
        Vec2 prev = s.curve.p[0];
        for (int i = 1; i <= 400; ++i) {
            const Vec2 cur = s.curve.eval(i / 400.0);
            const Vec2 d = cur - prev;
            const double dd = sla::dot(d, d);
            const double t = dd > 0.0 ? std::clamp(sla::dot(q - prev, d) / dd, 0.0, 1.0) : 0.0;
            best = std::min(best, sla::norm(q - (prev + d * t)));
            prev = cur;
        }
    }
    return best;
}

Vec2 unitTangentEnd(const sla::CurveSegment& s) {
    Vec2 v = s.curve.p[3] - s.curve.p[2];
    if (sla::norm(v) < 1e-12) v = s.curve.p[3] - s.curve.p[0];
    return v / sla::norm(v);
}
Vec2 unitTangentStart(const sla::CurveSegment& s) {
    Vec2 v = s.curve.p[1] - s.curve.p[0];
    if (sla::norm(v) < 1e-12) v = s.curve.p[3] - s.curve.p[0];
    return v / sla::norm(v);
}

std::vector<Vec2> circlePolygon(int n, double radius, Vec2 center, bool closeIt) {
    std::vector<Vec2> pts;
    for (int i = 0; i < n; ++i) {
        const double a = 2.0 * kPi * i / n;
        pts.push_back(center + Vec2{radius * std::cos(a), radius * std::sin(a)});
    }
    if (closeIt) pts.push_back(pts.front());
    return pts;
}

std::vector<Vec2> noisyArc(int n, SplitMix64& rng, double noise) {
    std::vector<Vec2> pts;
    for (int i = 0; i < n; ++i) {
        const double a = kPi * 0.8 * i / (n - 1);
        pts.push_back({10.0 * std::cos(a) + cnxtest::symmetric(rng, noise), 10.0 * std::sin(a) + cnxtest::symmetric(rng, noise)});
    }
    return pts;
}

}  // namespace

// ================================================================ derivadas e Schur

CNX_TEST(gradient_and_hessian_match_finite_differences_for_every_end_constraint) {
    SplitMix64 rng{701};
    const std::array<std::pair<bool, bool>, 4> modes{{{false, false}, {true, false}, {false, true}, {true, true}}};
    for (const auto& [g1s, g1e] : modes) {
        std::vector<Vec2> pts;
        for (int i = 0; i < 9; ++i) pts.push_back({static_cast<double>(i) + cnxtest::symmetric(rng, 0.3), cnxtest::symmetric(rng, 1.5)});
        FragmentEnds ends;
        if (g1s) ends.startTangent = Vec2{0.6, 0.8};
        if (g1e) ends.endTangent = Vec2{0.8, -0.6};
        const FragmentModel model(pts, ends);
        const std::size_t k = model.paramCount();
        const std::size_t m = model.interiorCount();
        CHECK(k == (g1s ? 1u : 2u) + (g1e ? 1u : 2u));
        CHECK(m == 7);
        std::vector<double> p(k);
        for (double& v : p) v = 0.5 + cnxtest::symmetric(rng, 2.0);
        std::vector<double> t(m);
        for (std::size_t i = 0; i < m; ++i) t[i] = (static_cast<double>(i) + 1.0) / (static_cast<double>(m) + 1.0) + cnxtest::symmetric(rng, 0.03);
        const std::string tag = std::string(g1s ? "G1" : "livre") + "/" + (g1e ? "G1" : "livre");

        const FitDerivatives d = model.derivatives(p, t);
        CHECK_CTX(std::abs(d.energy - model.energy(p, t)) < 1e-12, tag);
        const double h = 1e-6;
        const auto energyAt = [&](const std::vector<double>& pp, const std::vector<double>& tt) { return model.energy(pp, tt); };
        double worstG = 0.0;
        double worstH = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            auto pp = p, pm = p;
            pp[j] += h;
            pm[j] -= h;
            worstG = std::max(worstG, std::abs((energyAt(pp, t) - energyAt(pm, t)) / (2 * h) - d.gp[j]));
            // Hessiana: ∂g_p/∂p_l e ∂g_p/∂t_i por diferenças do gradiente.
            const FitDerivatives dp = model.derivatives(pp, t), dm = model.derivatives(pm, t);
            for (std::size_t l = 0; l < k; ++l) worstH = std::max(worstH, std::abs((dp.gp[l] - dm.gp[l]) / (2 * h) - d.hpp[j * k + l]));
            for (std::size_t i = 0; i < m; ++i) worstH = std::max(worstH, std::abs((dp.gt[i] - dm.gt[i]) / (2 * h) - d.hpt[i * k + j]));
        }
        for (std::size_t i = 0; i < m; ++i) {
            auto tp = t, tm = t;
            tp[i] += h;
            tm[i] -= h;
            worstG = std::max(worstG, std::abs((energyAt(p, tp) - energyAt(p, tm)) / (2 * h) - d.gt[i]));
            const FitDerivatives dp = model.derivatives(p, tp), dm = model.derivatives(p, tm);
            worstH = std::max(worstH, std::abs((dp.gt[i] - dm.gt[i]) / (2 * h) - d.d[i]));
            for (std::size_t j = 0; j < k; ++j) worstH = std::max(worstH, std::abs((dp.gp[j] - dm.gp[j]) / (2 * h) - d.hpt[i * k + j]));
        }
        CHECK_CTX(worstG < 1e-6, tag + " gradiente " + std::to_string(worstG));
        CHECK_CTX(worstH < 1e-6, tag + " Hessiana " + std::to_string(worstH));
    }
}

CNX_TEST(schur_step_equals_the_dense_newton_step) {
    SplitMix64 rng{702};
    for (int trial = 0; trial < 6; ++trial) {
        std::vector<Vec2> pts;
        for (int i = 0; i < 8; ++i) pts.push_back({static_cast<double>(i), cnxtest::symmetric(rng, 1.0)});
        FragmentEnds ends;
        if (trial % 2 == 1) ends.startTangent = Vec2{1.0, 0.0};
        const FragmentModel model(pts, ends);
        const std::size_t k = model.paramCount();
        const std::size_t m = model.interiorCount();
        std::vector<double> p(k, 1.0);
        for (double& v : p) v += cnxtest::symmetric(rng, 1.0);
        std::vector<double> t(m);
        for (std::size_t i = 0; i < m; ++i) t[i] = (static_cast<double>(i) + 1.0) / (static_cast<double>(m) + 1.0);
        const FitDerivatives d = model.derivatives(p, t);
        // λ pequeno pode deixar o complemento de Schur indefinido (o passo falha e o ajuste aumenta λ);
        // λ grande sempre funciona. Em todo λ que funciona, o passo tem de coincidir com o denso.
        int compared = 0;
        for (const double lambda : {1000.0, 50.0, 5.0, 0.5}) {
            // Sistema denso (k+m): [Hpp+λI, Hptᵀ; Hpt, diag(D+λ)] [δp; δt] = −[gp; gt].
            const std::size_t n = k + m;
            std::vector<double> a(n * n, 0.0), b(n, 0.0);
            for (std::size_t j = 0; j < k; ++j) {
                for (std::size_t l = 0; l < k; ++l) a[j * n + l] = d.hpp[j * k + l] + (j == l ? lambda : 0.0);
                b[j] = -d.gp[j];
            }
            for (std::size_t i = 0; i < m; ++i) {
                const double di = std::max(d.d[i] + lambda, 1e-9);
                a[(k + i) * n + (k + i)] = di;
                b[k + i] = -d.gt[i];
                for (std::size_t j = 0; j < k; ++j) {
                    a[(k + i) * n + j] = d.hpt[i * k + j];
                    a[j * n + (k + i)] = d.hpt[i * k + j];
                }
            }
            const std::vector<double> dense = denseSolve(a, b, n);
            const cnx::SchurStep step = FragmentModel::schurStep(d, lambda);
            if (lambda == 1000.0) CHECK(step.ok);
            if (!step.ok) continue;
            double worst = 0.0;
            for (std::size_t j = 0; j < k; ++j) worst = std::max(worst, std::abs(step.dp[j] - dense[j]));
            for (std::size_t i = 0; i < m; ++i) worst = std::max(worst, std::abs(step.dt[i] - dense[k + i]));
            CHECK_CTX(worst < 1e-9, "tentativa " + std::to_string(trial) + " λ=" + std::to_string(lambda) + ": " + std::to_string(worst));
            ++compared;
        }
        CHECK(compared >= 1);
    }
}

CNX_TEST(least_squares_is_stationary_for_fixed_t) {
    SplitMix64 rng{703};
    std::vector<Vec2> pts;
    for (int i = 0; i < 12; ++i) pts.push_back({static_cast<double>(i), 2.0 * std::sin(0.4 * i) + cnxtest::symmetric(rng, 0.2)});
    FragmentEnds g1;
    g1.startTangent = Vec2{1.0, 0.0};
    g1.endTangent = Vec2{0.8, 0.6};
    for (const FragmentEnds& ends : {FragmentEnds{}, g1}) {
        const FragmentModel model(pts, ends);
        const std::vector<double> t = uniformT(pts.size());
        const std::vector<double> inner(t.begin() + 1, t.end() - 1);
        const std::vector<double> p = model.leastSquares(inner);
        CHECK(p.size() == model.paramCount());
        const FitDerivatives d = model.derivatives(p, inner);
        for (const double g : d.gp) CHECK(std::abs(g) < 1e-9);  // gradiente nulo = mínimo da quadrática
        // Perturbar p só aumenta a energia.
        for (std::size_t j = 0; j < p.size(); ++j) {
            auto q = p;
            q[j] += 0.01;
            CHECK(model.energy(q, inner) > model.energy(p, inner));
        }
    }
}

// ================================================================ ajuste de fragmentos

CNX_TEST(newton_recovers_a_cubic_sampled_at_unknown_parameters) {
    const BezierCurve truth({0, 0}, {3, 8}, {9, 7}, {12, -1});
    // Parâmetros uniformes: o comprimento de corda inicial erra, e o Newton em (p, t) tem de corrigir.
    const std::vector<Vec2> pts = sampleCurve(truth, uniformT(14));
    FitParams params;
    params.newtonMaxIterations = 30;
    const cnx::FragmentFit fit = cnx::fitFragment(pts, FragmentEnds{}, params);
    CHECK(!fit.line && !fit.degenerate);
    CHECK_CTX(fit.energy < 1e-10, std::to_string(fit.energy));
    for (std::size_t j = 1; j <= 2; ++j) CHECK_CTX(sla::norm(fit.curve.p[j] - truth.p[j]) < 1e-3, "P" + std::to_string(j));
    CHECK(fit.curve.p[0] == truth.p[0] && fit.curve.p[3] == truth.p[3]);
    // t recuperados ≈ uniformes.
    for (std::size_t i = 0; i < fit.t.size(); ++i) CHECK(std::abs(fit.t[i] - static_cast<double>(i) / 13.0) < 1e-3);
    // Os t ficam em [0,1] e crescentes.
    for (std::size_t i = 1; i < fit.t.size(); ++i) CHECK(fit.t[i] >= fit.t[i - 1] && fit.t[i] <= 1.0);
}

CNX_TEST(newton_beats_least_squares_on_a_noisy_arc_and_never_increases_the_energy) {
    SplitMix64 rng{704};
    const std::vector<Vec2> pts = noisyArc(26, rng, 0.15);
    FitParams none;
    none.newtonMaxIterations = 0;  // só mínimos quadrados com t por corda
    FitParams full;
    full.newtonMaxIterations = 30;
    const cnx::FragmentFit ls = cnx::fitFragment(pts, FragmentEnds{}, none);
    const cnx::FragmentFit nt = cnx::fitFragment(pts, FragmentEnds{}, full);
    std::printf("  arco ruidoso: E (mínimos quadrados) %.6f -> E (Newton/Schur) %.6f em %d iterações\n", ls.energy, nt.energy,
                nt.newtonIterations);
    CHECK(ls.newtonIterations == 0);
    CHECK(nt.newtonIterations > 0);
    CHECK(nt.energy < 0.9 * ls.energy);
    // Mais iterações nunca pioram.
    double previous = ls.energy;
    for (int iters = 1; iters <= 8; ++iters) {
        FitParams p;
        p.newtonMaxIterations = iters;
        const double e = cnx::fitFragment(pts, FragmentEnds{}, p).energy;
        CHECK(e <= previous + 1e-12);
        previous = e;
    }
}

CNX_TEST(g1_fit_obeys_the_tangents_and_recovers_the_curve) {
    const Vec2 tau0 = Vec2{3, 8} / sla::norm(Vec2{3, 8});
    const Vec2 tau3 = Vec2{3, -8} / sla::norm(Vec2{3, -8});  // direção de percurso no fim: P3 − P2
    const BezierCurve truth({0, 0}, Vec2{0, 0} + tau0 * 5.0, Vec2{12, 0} - tau3 * 4.0, {12, 0});
    const std::vector<Vec2> pts = sampleCurve(truth, uniformT(12));
    FragmentEnds ends;
    ends.startTangent = tau0;
    ends.endTangent = tau3;
    FitParams params;
    params.newtonMaxIterations = 30;
    const cnx::FragmentFit fit = cnx::fitFragment(pts, ends, params);
    CHECK(!fit.degenerate);
    CHECK_CTX(fit.energy < 1e-9, std::to_string(fit.energy));
    const Vec2 a = fit.curve.p[1] - fit.curve.p[0];
    const Vec2 b = fit.curve.p[3] - fit.curve.p[2];
    CHECK(std::abs(sla::cross(a, tau0)) < 1e-9 && sla::dot(a, tau0) > 0.0);  // P₁ − P₀ ∥ τ₀, α > 0
    CHECK(std::abs(sla::cross(b, tau3)) < 1e-9 && sla::dot(b, tau3) > 0.0);
    CHECK(std::abs(sla::dot(a, tau0) - 5.0) < 1e-3 && std::abs(sla::dot(b, tau3) - 4.0) < 1e-3);
}

CNX_TEST(g1_with_a_cusp_is_flagged_and_uses_schneider_only_when_alpha_or_beta_is_not_positive) {
    // Pontos seguem em +x, mas as tangentes impostas apontam para trás nas duas pontas. O mínimo
    // dos mínimos quadrados teria α e β negativos (curva com laço ou cúspide). O resultado deve
    // respeitar o contrato: ou α, β > 0 sem marca, ou a heurística de Schneider com a marca.
    int flagged = 0;
    for (const double sign : {-1.0, +1.0}) {
        std::vector<Vec2> pts;
        // sinal +1 usa um ziguezague de 0,1 (patológico para o Newton em t); sinal -1, quase colinear.
        for (int i = 0; i < 8; ++i) pts.push_back({static_cast<double>(i), (sign > 0 ? 0.1 : 0.01) * (i % 2)});
        FragmentEnds ends;
        ends.startTangent = Vec2{sign, 0.0};
        ends.endTangent = Vec2{sign, 0.0};
        const cnx::FragmentFit fit = cnx::fitFragment(pts, ends, FitParams{});
        const double alpha = sla::dot(fit.curve.p[1] - fit.curve.p[0], *ends.startTangent);
        const double beta = sla::dot(fit.curve.p[3] - fit.curve.p[2], *ends.endTangent);
        const double third = sla::norm(pts.back() - pts.front()) / 3.0;
        std::printf("    sinal %+.0f: α=%.4f β=%.4f E=%.5f degenerado=%d iterações=%d\n", sign, alpha, beta, fit.energy,
                    fit.degenerate ? 1 : 0, fit.newtonIterations);
        // O t devolvido é sempre crescente e dentro de [0, 1], mesmo em dados patológicos.
        for (std::size_t i = 0; i < fit.t.size(); ++i) {
            CHECK(fit.t[i] >= 0.0 && fit.t[i] <= 1.0);
            if (i > 0) CHECK(fit.t[i] >= fit.t[i - 1]);
        }
        if (fit.degenerate) {
            ++flagged;
            CHECK(std::abs(alpha - third) < 1e-12 && std::abs(beta - third) < 1e-12);  // α = β = ‖P₃ − P₀‖/3
            // E* é a energia da curva de Schneider com seus próprios t ótimos (não a de uma parametrização herdada):
            // confere com a projeção densa de cada nó interior sobre a curva.
            double nearest = 0.0;
            for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
                double best = 1e300;
                for (int s = 0; s <= 20000; ++s) best = std::min(best, sla::norm2(fit.curve.eval(s / 20000.0) - pts[i]));
                nearest += 0.5 * best;
            }
            CHECK_CTX(std::abs(fit.energy - nearest) < 1e-6, std::to_string(fit.energy) + " vs " + std::to_string(nearest));
        } else {
            CHECK(alpha > 0.0 && beta > 0.0);
        }
    }
    // Uma só ponta contra o percurso: α quer ser negativo.
    std::vector<Vec2> pts;
    for (int i = 0; i < 8; ++i) pts.push_back({static_cast<double>(i), 0.0});
    FragmentEnds oneSided;
    oneSided.startTangent = Vec2{-1.0, 0.0};
    oneSided.endTangent = Vec2{1.0, 0.0};
    const cnx::FragmentFit fit = cnx::fitFragment(pts, oneSided, FitParams{});
    const double alpha = sla::dot(fit.curve.p[1] - fit.curve.p[0], *oneSided.startTangent);
    std::printf("  cúspide: marcados %d de 2 (+1 caso de uma ponta: degenerado=%d, α=%.4f)\n", flagged, fit.degenerate ? 1 : 0, alpha);
    CHECK(fit.degenerate || alpha > 0.0);
    // O ziguezague com tangentes a favor do percurso leva o Newton a α ≤ 0 e cai em Schneider (determinístico):
    // garante que o caminho degenerado e a marca continuam exercitados.
    CHECK(flagged >= 1);
}

CNX_TEST(zigzag_chain_stays_valid_and_degenerate_merges_are_refused) {
    // Ziguezague regular: cada fusão tem que passar pelo teste G¹; as degeneradas são recusadas.
    std::vector<Vec2> pts;
    for (int i = 0; i < 48; ++i) pts.push_back({0.5 * i, 0.12 * (i % 2)});
    const std::vector<std::uint8_t> corner(pts.size(), 0);
    const cnx::ChainFit fit = BezierFitter(FitParams{}).fitChain(pts, corner, false);
    std::printf("  ziguezague: %zu curvas, %d degeneradas, E %.4f\n", fit.curves.size(), fit.stats.degenerate, fit.stats.energy);
    for (std::size_t k = 0; k + 1 < fit.curves.size(); ++k) CHECK(sla::norm(fit.curves[k].end() - fit.curves[k + 1].start()) < 1e-12);
    // Uma fusão degenerada nunca sobrevive: as marcas restantes vêm só dos fragmentos iniciais.
    CHECK(fit.stats.degenerate <= fit.stats.initialFragments);
    CHECK(fit.stats.degenerate == 0);
}

CNX_TEST(closed_zigzag_never_keeps_a_degenerate_merge_and_t_stays_monotone_when_the_polyline_doubles_back) {
    // Aresta fechada em estrela serrilhada: todas as quebras são lisas (G¹ nas duas pontas), o terreno
    // onde α ≤ 0 aparece. Nenhuma fusão degenerada pode sobreviver.
    std::vector<Vec2> star;
    for (int i = 0; i < 48; ++i) {
        const double a = 2.0 * kPi * i / 48.0;
        const double r = 10.0 + 0.6 * (i % 2);
        star.push_back({20.0 + r * std::cos(a), 20.0 + r * std::sin(a)});
    }
    star.push_back(star.front());
    const std::vector<std::uint8_t> corner(star.size(), 0);
    const cnx::ChainFit fit = BezierFitter(FitParams{}).fitChain(star, corner, true);
    std::printf("  estrela serrilhada: %zu curvas, %d degeneradas, %d fusões\n", fit.curves.size(), fit.stats.degenerate, fit.stats.merges);
    CHECK(fit.stats.degenerate == 0);
    CHECK(fit.curves.size() >= 2);

    // Polilinha que volta sobre si mesma: o melhor t por nó poderia sair de ordem; deve continuar crescente.
    std::vector<Vec2> uturn;
    for (int i = 0; i <= 6; ++i) uturn.push_back({static_cast<double>(i), 0.0});
    for (int j = 1; j <= 6; ++j) uturn.push_back({6.0 - j, 0.4 + 0.05 * j});
    FitParams params;
    params.newtonMaxIterations = 40;
    const cnx::FragmentFit f = cnx::fitFragment(uturn, FragmentEnds{}, params);
    for (std::size_t i = 1; i < f.t.size(); ++i) CHECK_CTX(f.t[i] >= f.t[i - 1], "t[" + std::to_string(i) + "]");
}

CNX_TEST(extra_error_that_is_small_does_not_block_a_merge_even_when_the_fragments_are_noisy) {
    // Ziguezague de amplitude 0,3 em 15 pontos, com fragmentos iniciais de 8 pontos: cada metade já
    // tem E > 0, mas a união custa pouco a mais. A regra é ΔE < τ, não E(união) < τ: com τ_f = 0,1
    // a energia da união passa de τ_f (0,17) e mesmo assim os dois fragmentos se fundem.
    std::vector<Vec2> pts;
    for (int i = 0; i < 15; ++i) pts.push_back({static_cast<double>(i), 0.3 * (i % 2)});
    const std::vector<std::uint8_t> corner(pts.size(), 0);
    FitParams p;
    p.initialFragmentNodes = 8;
    p.statThreshFinal = 0.1;
    const cnx::ChainFit fit = BezierFitter(p).fitChain(pts, corner, false);
    std::printf("  ziguezague 15 pontos: %d fragmentos iniciais -> %zu curvas, E total %.4f (τ_f = %.2f), %d fusões\n",
                fit.stats.initialFragments, fit.curves.size(), fit.stats.energy, p.statThreshFinal, fit.stats.merges);
    CHECK(fit.stats.initialFragments == 2);
    CHECK(fit.stats.energy > p.statThreshFinal);  // E(união) > τ_f: só o delta pode justificar a fusão
    CHECK(fit.curves.size() == 1);
}

CNX_TEST(short_fragments_are_lines) {
    const FitParams params;
    const std::vector<Vec2> two{{0, 0}, {4, 3}};
    const cnx::FragmentFit f2 = cnx::fitFragment(two, FragmentEnds{}, params);
    CHECK(f2.line && f2.energy == 0.0 && f2.curve.p[1] == BezierCurve::fromLine(two[0], two[1]).p[1]);
    const std::vector<Vec2> three{{0, 0}, {2, 2}, {4, 0}};
    const cnx::FragmentFit f3 = cnx::fitFragment(three, FragmentEnds{}, params);
    CHECK(f3.line);
    CHECK(std::abs(f3.energy - 0.5 * 4.0) < 1e-12);  // o ponto do meio está a 2 da corda: ½·2²
    CHECK(std::abs(f3.t[1] - 0.5) < 1e-12);
    CHECK_THROWS(std::invalid_argument, (void)cnx::fitFragment(std::vector<Vec2>{{0, 0}}, FragmentEnds{}, params));
    CHECK_THROWS(std::invalid_argument, (void)FragmentModel(three, FragmentEnds{}));
}

CNX_TEST(warm_start_reaches_the_same_energy) {
    SplitMix64 rng{705};
    const std::vector<Vec2> pts = noisyArc(30, rng, 0.1);
    FitParams params;
    params.newtonMaxIterations = 40;
    const cnx::FragmentFit cold = cnx::fitFragment(pts, FragmentEnds{}, params);
    const cnx::FragmentFit warm = cnx::fitFragment(pts, FragmentEnds{}, params, cold.t);
    CHECK(std::abs(warm.energy - cold.energy) < 1e-6 * (1.0 + cold.energy));
    CHECK(warm.newtonIterations <= cold.newtonIterations);
}

// ================================================================ cadeias

CNX_TEST(threshold_schedule_is_geometric) {
    FitParams p;
    const std::vector<double> t = BezierFitter::thresholdSchedule(p);
    CHECK(t.size() == 20);
    CHECK(t.front() == p.statThreshInitial && t.back() == p.statThreshFinal);
    const double ratio = std::pow(p.statThreshFinal / p.statThreshInitial, 1.0 / 19.0);
    CHECK(std::abs(ratio - 2.42) < 0.01);  // ≈ 2,42 por rodada (§2.5.3)
    for (std::size_t i = 1; i + 1 < t.size(); ++i) CHECK(std::abs(t[i] / t[i - 1] - ratio) < 1e-9);
    p.maxRounds = 1;
    CHECK(BezierFitter::thresholdSchedule(p) == std::vector<double>{p.statThreshFinal});
}

CNX_TEST(collinear_nodes_merge_into_a_single_curve) {
    std::vector<Vec2> pts;
    for (int i = 0; i < 31; ++i) pts.push_back({0.7 * i, 0.3 * i});
    const std::vector<std::uint8_t> corner(pts.size(), 0);
    const cnx::ChainFit fit = BezierFitter(FitParams{}).fitChain(pts, corner, false);
    CHECK(fit.curves.size() == 1);
    CHECK(fit.stats.initialFragments > 5 && fit.stats.finalFragments == 1);
    CHECK(fit.stats.merges == fit.stats.initialFragments - 1);
    CHECK(fit.stats.energy < 1e-12);
    CHECK(fit.curves[0].start() == pts.front() && fit.curves[0].end() == pts.back());
}

CNX_TEST(a_circle_becomes_a_few_g1_cubics_that_stay_close_to_the_nodes) {
    const std::vector<Vec2> pts = circlePolygon(120, 10.0, {20, 20}, true);
    const std::vector<std::uint8_t> corner(pts.size(), 0);
    const cnx::ChainFit fit = BezierFitter(FitParams{}).fitChain(pts, corner, true);
    std::printf("  círculo de 120 nós: %zu curvas, %d fusões, %d deslocamentos, E %.5f\n", fit.curves.size(), fit.stats.merges,
                fit.stats.shifts, fit.stats.energy);
    CHECK(fit.curves.size() >= 2 && fit.curves.size() <= 14);  // poucas cúbicas bastam para um círculo
    double worst = 0.0;
    for (const Vec2& q : pts) worst = std::max(worst, distanceToChain(fit.curves, q));
    std::printf("  círculo: desvio máximo dos nós à cadeia %.5f px\n", worst);
    CHECK_CTX(worst < 0.05, std::to_string(worst));
    // C⁰ e G¹ em todas as quebras, inclusive na âncora (última → primeira).
    for (std::size_t k = 0; k < fit.curves.size(); ++k) {
        const sla::CurveSegment& a = fit.curves[k];
        const sla::CurveSegment& b = fit.curves[(k + 1) % fit.curves.size()];
        CHECK(sla::norm(a.end() - b.start()) < 1e-12);
        const Vec2 ta = unitTangentEnd(a);
        const Vec2 tb = unitTangentStart(b);
        CHECK_CTX(std::abs(sla::cross(ta, tb)) < 1e-9 && sla::dot(ta, tb) > 0.0, "quebra " + std::to_string(k));
    }
    CHECK(fit.stats.degenerate == 0);
    // A tangente na âncora é a diferença central (pts[1] − pts[n−2]), não a lateral.
    const Vec2 central = (pts[1] - pts[pts.size() - 2]) / sla::norm(pts[1] - pts[pts.size() - 2]);
    CHECK(std::abs(sla::cross(unitTangentEnd(fit.curves.back()), central)) < 1e-9);
    CHECK(std::abs(sla::cross(unitTangentStart(fit.curves.front()), central)) < 1e-9);
}

CNX_TEST(corners_split_the_chain_and_keep_their_positions) {
    // Quadrado de lado 10 com 11 nós por lado; os 4 cantos e a âncora são cantos.
    std::vector<Vec2> pts;
    std::vector<std::uint8_t> corner;
    const std::array<Vec2, 5> v{{{0, 0}, {10, 0}, {10, 10}, {0, 10}, {0, 0}}};
    for (std::size_t s = 0; s < 4; ++s) {
        for (int i = 0; i < 10; ++i) {
            pts.push_back(v[s] + (v[s + 1] - v[s]) * (i / 10.0));
            corner.push_back(i == 0 ? 1 : 0);
        }
    }
    pts.push_back(v[4]);
    corner.push_back(1);
    const cnx::ChainFit fit = BezierFitter(FitParams{}).fitChain(pts, corner, true);
    CHECK(fit.curves.size() == 4);
    for (std::size_t s = 0; s < 4; ++s) {
        CHECK(fit.curves[s].start() == v[s] && fit.curves[s].end() == v[s + 1]);
        // Cada lado é uma reta: os pontos de controle ficam sobre ele.
        for (std::size_t j = 1; j <= 2; ++j) {
            const Vec2 d = v[s + 1] - v[s];
            CHECK(std::abs(sla::cross(fit.curves[s].curve.p[j] - v[s], d)) < 1e-6);
        }
    }
    // Nos cantos as tangentes são livres: o giro é de 90°.
    CHECK(std::abs(sla::dot(unitTangentEnd(fit.curves[0]), unitTangentStart(fit.curves[1]))) < 1e-6);
}

CNX_TEST(a_larger_final_threshold_never_produces_more_curves) {
    SplitMix64 rng{706};
    const std::vector<Vec2> pts = noisyArc(80, rng, 0.08);
    const std::vector<std::uint8_t> corner(pts.size(), 0);
    std::size_t previousCount = 1000000;
    double previousEnergy = -1.0;
    for (const double tauF : {1e-6, 1e-3, 1e-1, 3.0, 100.0}) {
        FitParams p;
        p.statThreshFinal = tauF;
        const cnx::ChainFit fit = BezierFitter(p).fitChain(pts, corner, false);
        std::printf("  tau_final %g: %zu curvas, E %.4f\n", tauF, fit.curves.size(), fit.stats.energy);
        CHECK(fit.curves.size() <= previousCount);
        // Menos curvas, mais erro. A fusão é gulosa, então a energia não é estritamente monótona no
        // limiar (caminhos de fusão diferentes chegam a ótimos locais diferentes): folga de 20 %.
        CHECK(fit.stats.energy >= 0.8 * previousEnergy);
        CHECK(fit.stats.finalFragments == fit.stats.initialFragments - fit.stats.merges);
        previousCount = fit.curves.size();
        previousEnergy = fit.stats.energy;
    }
    CHECK(previousCount == 1);  // τ enorme: tudo vira uma curva
}

CNX_TEST(reported_energy_bounds_the_distance_from_nodes_to_the_curves) {
    SplitMix64 rng{707};
    for (int trial = 0; trial < 4; ++trial) {
        const std::vector<Vec2> pts = noisyArc(60, rng, 0.1);
        const std::vector<std::uint8_t> corner(pts.size(), 0);
        const cnx::ChainFit fit = BezierFitter(FitParams{}).fitChain(pts, corner, false);
        // E é medida com t ótimo; a distância ao ponto mais próximo da cadeia só pode ser menor.
        double nearest = 0.0;
        for (const Vec2& q : pts) {
            const double d = distanceToChain(fit.curves, q);
            nearest += 0.5 * d * d;
        }
        CHECK_CTX(nearest <= fit.stats.energy + 1e-4, "série " + std::to_string(trial));
        CHECK(fit.stats.shifts >= 0);
    }
}

CNX_TEST(break_shifting_happens_and_keeps_the_chain_continuous) {
    SplitMix64 rng{708};
    int shifts = 0;
    for (int trial = 0; trial < 6; ++trial) {
        // Curvatura que muda de sinal: o melhor ponto de quebra não coincide com a malha inicial.
        std::vector<Vec2> pts;
        for (int i = 0; i < 70; ++i) {
            const double x = 0.4 * i;
            pts.push_back({x, 3.0 * std::sin(0.35 * x + trial) + cnxtest::symmetric(rng, 0.05)});
        }
        const std::vector<std::uint8_t> corner(pts.size(), 0);
        FitParams p;
        p.statThreshFinal = 0.02;  // exigente: sobram quebras, e o deslocamento entra em ação
        const cnx::ChainFit fit = BezierFitter(p).fitChain(pts, corner, false);
        shifts += fit.stats.shifts;
        for (std::size_t k = 0; k + 1 < fit.curves.size(); ++k) CHECK(sla::norm(fit.curves[k].end() - fit.curves[k + 1].start()) < 1e-12);
        CHECK(fit.curves.front().start() == pts.front() && fit.curves.back().end() == pts.back());
    }
    std::printf("  deslocamentos de quebra aceitos: %d\n", shifts);
    CHECK(shifts > 0);
}

CNX_TEST(fit_chain_validates_its_arguments) {
    FitParams bad;
    bad.statThreshInitial = 0.0;
    CHECK_THROWS(std::invalid_argument, (void)BezierFitter(bad));
    bad = FitParams{};
    bad.statThreshFinal = bad.statThreshInitial / 2;
    CHECK_THROWS(std::invalid_argument, (void)BezierFitter(bad));
    bad = FitParams{};
    bad.maxRounds = 0;
    CHECK_THROWS(std::invalid_argument, (void)BezierFitter(bad));
    bad = FitParams{};
    bad.initialFragmentNodes = 1;
    CHECK_THROWS(std::invalid_argument, (void)BezierFitter(bad));
    const BezierFitter fitter{FitParams{}};
    const std::vector<Vec2> pts{{0, 0}, {1, 0}, {2, 1}, {3, 3}};
    CHECK_THROWS(std::invalid_argument, (void)fitter.fitChain(pts, std::vector<std::uint8_t>(3, 0), false));
    CHECK_THROWS(std::invalid_argument, (void)fitter.fitChain(std::vector<Vec2>{{0, 0}}, std::vector<std::uint8_t>(1, 0), false));
    CHECK_THROWS(std::invalid_argument, (void)fitter.fitChain(pts, std::vector<std::uint8_t>(4, 0), true));  // fechada mas último ≠ primeiro
    // Cadeias minúsculas viram retas.
    const cnx::ChainFit two = fitter.fitChain(std::vector<Vec2>{{0, 0}, {5, 5}}, std::vector<std::uint8_t>(2, 0), false);
    CHECK(two.curves.size() == 1 && two.curves[0].kind == sla::SegmentKind::Line);
}

// ================================================================ integração com o VectorImage

CNX_TEST(fitting_a_smoothed_image_gives_every_edge_a_continuous_chain_with_exact_areas) {
    SplitMix64 rng{709};
    for (int scene = 0; scene < 3; ++scene) {
        cnxtest::VectorScene s = cnxtest::vectorScene(24 + scene, 16, 2, 40, rng, 3);
        cnx::SmoothParams sp;
        sp.maxCgIterations = 40;
        (void)cnx::ContourSmoother(sp).run(s.image);
        const cnx::FitStats stats = BezierFitter(FitParams{}).run(s.image);
        const std::string tag = "cena " + std::to_string(scene);
        CHECK_CTX(stats.finalFragments > 0, tag);
        CHECK_CTX(s.image.hasCurves(), tag);
        int corners = 0;
        for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(s.image.edgeCount()); ++e) {
            const auto curves = s.image.edgeCurves(e);
            const auto nodes = s.image.edgeNodes(e);
            CHECK_CTX(!curves.empty(), tag);
            CHECK_CTX(curves.front().start() == s.image.position(nodes.front()), tag);
            CHECK_CTX(curves.back().end() == s.image.position(nodes.back()), tag);
            for (std::size_t k = 0; k + 1 < curves.size(); ++k) CHECK_CTX(curves[k].end() == curves[k + 1].start(), tag);
            // Cada nó de canto (Corner) é ponta de uma curva.
            for (const cnx::NodeId n : nodes) {
                if ((s.image.flags(n) & cnx::NodeFlag::Corner) == 0) continue;
                bool endpoint = false;
                for (const sla::CurveSegment& c : curves) endpoint = endpoint || c.start() == s.image.position(n) || c.end() == s.image.position(n);
                CHECK_CTX(endpoint, tag + ": canto sem quebra");
                ++corners;
            }
        }
        // Com curvas, as áreas usam a fórmula de Green das cúbicas: continuam somando W·H e a topologia vale.
        s.image.updateFaceAreas();
        const auto report = s.image.validate();
        CHECK_CTX(report.ok, tag + ": " + report.message);
        std::printf("  cena %d: %zu arestas, %d fragmentos finais (%d retas), %d cantos\n", scene, s.image.edgeCount(),
                    stats.finalFragments, stats.lines, corners);
    }
}

CNX_TEST(a_smoothed_disk_becomes_a_few_g1_cubics_with_the_right_area) {
    // Disco de raio 10,6 com antialiasing, suavizado pelo modelo gerador e depois ajustado.
    const Vec2 center{18.3, 17.8};
    const double radius = 10.6;
    cnxtest::AaScene s = cnxtest::aaScene(36, 36, [&](double x, double y) {
        return (x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) <= radius * radius;
    }, 16);
    const cnx::GenerativeModel model(s.target);
    cnx::SmoothParams sp;
    sp.measurementTypes = {cnx::MeasurementType::Generative, cnx::MeasurementType::Generative, cnx::MeasurementType::Generative};
    (void)cnx::ContourSmoother(sp, &model).run(s.image);
    const cnx::FitStats stats = BezierFitter(FitParams{}).run(s.image);

    cnx::EdgeId ring = cnx::kInvalidId;
    for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(s.image.edgeCount()); ++e) {
        const auto nodes = s.image.edgeNodes(e);
        if (nodes.size() > 10 && nodes.front() == nodes.back()) ring = e;
    }
    CHECK(ring != cnx::kInvalidId);
    const auto curves = s.image.edgeCurves(ring);
    std::printf("  disco: %zu nós -> %zu curvas (%d fusões, %d deslocamentos), E %.4f\n", s.image.edgeNodes(ring).size(), curves.size(),
                stats.merges, stats.shifts, stats.energy);
    CHECK(curves.size() >= 2 && curves.size() <= 12);
    // Todos os pontos das curvas ficam a ~radius do centro (o disco foi bem posicionado pelo suavizador).
    // Com τ_final = 0,2 a energia total tolerada é de 0,2 sobre ~50 nós, então ~0,15 px é o esperado.
    const auto radialError = [&](std::span<const sla::CurveSegment> chain) {
        double w = 0.0;
        for (const sla::CurveSegment& c : chain) {
            for (int i = 0; i <= 50; ++i) w = std::max(w, std::abs(sla::norm(c.curve.eval(i / 50.0) - center) - radius));
        }
        return w;
    };
    const double worst = radialError(curves);
    std::printf("  disco: desvio radial máximo das curvas %.4f px\n", worst);
    CHECK(worst < 0.25);
    // G¹ em todas as quebras (a aresta é fechada e a âncora é lisa).
    for (std::size_t k = 0; k < curves.size(); ++k) {
        const Vec2 ta = unitTangentEnd(curves[k]);
        const Vec2 tb = unitTangentStart(curves[(k + 1) % curves.size()]);
        CHECK(std::abs(sla::cross(ta, tb)) < 1e-6 && sla::dot(ta, tb) > 0.0);
    }
    // A área da face do disco, pela fórmula de Green das cúbicas, é a do círculo.
    s.image.updateFaceAreas();
    double diskArea = 0.0;
    for (cnx::FaceId f = 1; f < static_cast<cnx::FaceId>(s.image.faceCount()); ++f) {
        if (s.image.face(f).color.r < 0.5f) diskArea = s.image.face(f).area;
    }
    std::printf("  disco: área %.3f (círculo %.3f)\n", diskArea, std::numbers::pi * radius * radius);
    CHECK(std::abs(diskArea - std::numbers::pi * radius * radius) < 0.02 * std::numbers::pi * radius * radius);
    const auto report = s.image.validate();
    CHECK_CTX(report.ok, report.message);

    // Limiar final menor: mais curvas e mais precisão (o ajuste é monotônico no limiar).
    cnx::VectorImage tight = s.image;
    FitParams strict;
    strict.statThreshFinal = 0.003;
    (void)BezierFitter(strict).run(tight);
    const auto tightCurves = tight.edgeCurves(ring);
    const double tightError = radialError(tightCurves);
    std::printf("  disco com τ_final=0,003: %zu curvas, desvio radial máximo %.4f px\n", tightCurves.size(), tightError);
    CHECK(tightCurves.size() > curves.size());
    CHECK(tightError < worst);  // mais curvas nunca pioram o ajuste (o disco já sai a ~0,03 px com o limiar padrão)
    CHECK(tightError < 0.025);
}

namespace {

/// Distância de q à poligonal (segmentos entre pontos consecutivos).
double distanceToPolyline(const std::vector<Vec2>& poly, bool closed, Vec2 q) {
    double best = 1e300;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i + (closed ? 0 : 1) < n; ++i) {
        const Vec2 a = poly[i];
        const Vec2 b = poly[(i + 1) % n];
        const Vec2 d = b - a;
        const double t = std::clamp(sla::dot(q - a, d) / std::max(1e-300, sla::dot(d, d)), 0.0, 1.0);
        best = std::min(best, sla::norm(q - (a + d * t)));
    }
    return best;
}

}  // namespace

CNX_TEST(long_straight_stretches_do_not_bulge_between_sparse_nodes) {
    // "D": uma reta de 18 px (2 nós só, como deixa o tracer depois de tirar os colineares) fechada por uma semicircunferência
    // de raio 9 em 11 nós. A energia do ajuste só vê os nós: sem amostras na reta, uma cúbica se afasta ~1 px no meio dela.
    std::vector<Vec2> poly;
    poly.push_back({10.0, 2.0});  // topo da reta
    poly.push_back({10.0, 20.0}); // pé da reta
    for (int k = 1; k <= 11; ++k) {
        const double a = kPi * k / 12.0;  // do pé, dando a volta pela esquerda até o topo
        poly.push_back({10.0 - 9.0 * std::sin(a), 11.0 + 9.0 * std::cos(a)});
    }
    cnx::VectorImage v(30.0, 30.0);
    const cnx::FaceId face = v.addFace(sla::Rgba{0.0f, 0.0f, 0.0f, 1.0f}, 0);
    std::vector<cnx::NodeId> ids;
    for (std::size_t i = 0; i < poly.size(); ++i) ids.push_back(v.addNode(poly[i], i == 0 ? cnx::NodeFlag::LoopAnchor : 0));
    ids.push_back(ids.front());
    (void)v.addEdge(ids, face, cnx::kExteriorFace);
    v.linkFaceLoops();
    const std::vector<Vec2> closedPoly = [&] {
        std::vector<Vec2> p = poly;
        return p;
    }();
    (void)BezierFitter(FitParams{}).run(v);
    const auto curves = v.edgeCurves(0);
    double worst = 0.0;
    for (const sla::CurveSegment& c : curves) {
        for (int i = 0; i <= 200; ++i) worst = std::max(worst, distanceToPolyline(closedPoly, true, c.curve.eval(i / 200.0)));
    }
    int lines = 0;
    for (const sla::CurveSegment& c : curves) lines += c.kind == sla::SegmentKind::Line ? 1 : 0;
    std::printf("  D com reta de 18 px: %zu curvas (%d retas), desvio máximo da poligonal %.3f px\n", curves.size(), lines, worst);
    CHECK(worst < 0.25);  // com o limiar padrão o ajuste fica a ~0,17 px da poligonal (contra 0,87 px sem as amostras)
    CHECK(lines >= 1);    // o trecho reto sai como "L"
    // O mesmo ajuste feito diretamente sobre os nós (sem as amostras intermediárias de run()) mostra o defeito.
    std::vector<std::uint8_t> corner(poly.size() + 1, 0);
    std::vector<Vec2> sparse = poly;
    sparse.push_back(poly.front());
    const cnx::ChainFit direct = BezierFitter(FitParams{}).fitChain(sparse, corner, true);
    double directWorst = 0.0;
    for (const sla::CurveSegment& c : direct.curves) {
        for (int i = 0; i <= 200; ++i) directWorst = std::max(directWorst, distanceToPolyline(closedPoly, true, c.curve.eval(i / 200.0)));
    }
    std::printf("  sem as amostras intermediárias: desvio máximo %.3f px\n", directWorst);
    CHECK(directWorst > 3.0 * worst);
    // Extremos da cadeia: a primeira curva começa no primeiro nó e a última termina nele (aresta fechada).
    CHECK(curves.front().start() == poly.front() && curves.back().end() == poly.front());
}

CNX_TEST(fitted_pieces_that_are_straight_become_lines) {
    // Retângulo de 30 × 4 com os 4 cantos marcados: cada lado é uma reta, que sai como "L" e não como cúbica.
    cnx::VectorImage v(40.0, 10.0);
    const cnx::FaceId face = v.addFace(sla::Rgba{0.0f, 0.0f, 0.0f, 1.0f}, 0);
    const cnx::NodeId a = v.addNode({5, 3}, cnx::NodeFlag::LoopAnchor | cnx::NodeFlag::Corner);
    const std::array<cnx::NodeId, 5> loop{a, v.addNode({35, 3}, cnx::NodeFlag::Corner), v.addNode({35, 7}, cnx::NodeFlag::Corner), v.addNode({5, 7}, cnx::NodeFlag::Corner), a};
    (void)v.addEdge(loop, face, cnx::kExteriorFace);
    v.linkFaceLoops();
    const cnx::FitStats stats = BezierFitter(FitParams{}).run(v);
    const auto curves = v.edgeCurves(0);
    CHECK(curves.size() == 4);
    for (const sla::CurveSegment& c : curves) CHECK(c.kind == sla::SegmentKind::Line && c.curve.p == BezierCurve::fromLine(c.curve.p[0], c.curve.p[3]).p);
    CHECK(stats.lines == 4);
    CHECK(curves[0].start() == Vec2(5, 3) && curves[0].end() == Vec2(35, 3) && curves[3].end() == Vec2(5, 3));
    // Um arco de verdade nunca vira reta: um quarto de círculo de raio 10.
    cnx::VectorImage w(30.0, 30.0);
    const cnx::FaceId f2 = w.addFace(sla::Rgba{0.0f, 0.0f, 0.0f, 1.0f}, 0);
    std::vector<cnx::NodeId> ids;
    for (int k = 0; k <= 12; ++k) ids.push_back(w.addNode({5.0 + 10.0 * std::sin(kPi / 2 * k / 12), 25.0 - 10.0 * std::cos(kPi / 2 * k / 12)}, k == 0 || k == 12 ? cnx::NodeFlag::Corner : 0));
    ids.push_back(w.addNode({5.0, 25.0}, cnx::NodeFlag::Corner));
    ids.push_back(ids.front());
    (void)w.addEdge(ids, f2, cnx::kExteriorFace);
    w.linkFaceLoops();
    (void)BezierFitter(FitParams{}).run(w);
    int cubics = 0;
    for (const sla::CurveSegment& c : w.edgeCurves(0)) cubics += c.kind == sla::SegmentKind::Cubic ? 1 : 0;
    CHECK(cubics >= 1);
}

CNX_TEST(only_cubics_with_controls_on_their_chord_in_order_count_as_straight) {
    const auto straight = [](Vec2 p1, Vec2 p2) { return BezierFitter::isStraight(BezierCurve({0, 0}, p1, p2, {15, 0})); };
    CHECK(straight({5, 0}, {10, 0}));            // controles a 1/3 e 2/3 da corda
    CHECK(straight({0, 0}, {15, 0}));            // controles sobre os extremos
    CHECK(straight({5, 0.019}, {10, -0.019}));   // dentro da tolerância de 0,02 px
    CHECK(!straight({5, 0.03}, {10, 0}));        // P1 fora dela
    CHECK(!straight({5, 0}, {10, -0.03}));       // P2 fora dela
    CHECK(!straight({5, 0.3}, {10, 0.3}));       // arco raso (sagita ≈ 0,22 px): nunca vira "L"
    CHECK(!straight({-3, 0}, {10, 0}));          // P1 atrás de P0
    CHECK(!straight({10, 0}, {5, 0}));           // P1 e P2 fora de ordem
    CHECK(!straight({5, 0}, {20, 0}));           // P2 depois de P3
    CHECK(straight({-0.01, 0}, {15.01, 0}));     // dentro da tolerância nas duas pontas
    CHECK(!BezierFitter::isStraight(BezierCurve({3, 3}, {3, 3}, {3, 3}, {3, 3})));  // corda nula
    // A mesma regra em outra direção (a corda não precisa estar nos eixos).
    CHECK(BezierFitter::isStraight(BezierCurve({1, 1}, {4, 5}, {7, 9}, {10, 13})));
    CHECK(!BezierFitter::isStraight(BezierCurve({1, 1}, {4, 5.2}, {7, 9}, {10, 13})));
}

CNX_TEST(fitting_is_deterministic_and_cancellable) {
    SplitMix64 rng{710};
    cnxtest::VectorScene s = cnxtest::vectorScene(22, 14, 2, 30, rng, 3);
    cnx::VectorImage a = s.image;
    cnx::VectorImage b = s.image;
    (void)BezierFitter(FitParams{}).run(a);
    (void)BezierFitter(FitParams{}).run(b);
    bool same = true;
    for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(a.edgeCount()); ++e) {
        const auto ca = a.edgeCurves(e);
        const auto cb = b.edgeCurves(e);
        same = same && ca.size() == cb.size();
        for (std::size_t k = 0; same && k < ca.size(); ++k) same = ca[k].kind == cb[k].kind && ca[k].curve.p == cb[k].curve.p;
    }
    CHECK(same);
    std::stop_source source;
    source.request_stop();
    CHECK_THROWS(cnx::CancelTaskException,
                 (void)BezierFitter(FitParams{}).run(s.image, cnx::TaskContext(source.get_token(), nullptr, cnx::ProgressRange{})));
}

int main() { return cnxtest::runAll(); }
