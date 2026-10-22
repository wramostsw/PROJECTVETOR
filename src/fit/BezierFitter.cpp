// src/fit/BezierFitter.cpp — mínimos quadrados, Newton com complemento de Schur, G¹ e fusão de fragmentos.
#include "cnx/fit/BezierFitter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cnx {
namespace {

using sla::BezierCurve;
using sla::Vec2;

constexpr double kTiny = 1e-12;

/// Resolve A x = b (A simétrica k×k, linha a linha) por Cholesky. Falso se A não for definida positiva.
bool choleskySolve(std::vector<double> a, std::vector<double> b, std::size_t k, std::vector<double>& x) {
    for (std::size_t j = 0; j < k; ++j) {
        double diag = a[j * k + j];
        for (std::size_t s = 0; s < j; ++s) diag -= a[j * k + s] * a[j * k + s];
        if (!(diag > 0.0) || !std::isfinite(diag)) return false;
        const double l = std::sqrt(diag);
        a[j * k + j] = l;
        for (std::size_t i = j + 1; i < k; ++i) {
            double v = a[i * k + j];
            for (std::size_t s = 0; s < j; ++s) v -= a[i * k + s] * a[j * k + s];
            a[i * k + j] = v / l;
        }
    }
    for (std::size_t i = 0; i < k; ++i) {  // L y = b
        double v = b[i];
        for (std::size_t s = 0; s < i; ++s) v -= a[i * k + s] * b[s];
        b[i] = v / a[i * k + i];
    }
    for (std::size_t ii = k; ii-- > 0;) {  // Lᵀ x = y
        double v = b[ii];
        for (std::size_t s = ii + 1; s < k; ++s) v -= a[s * k + ii] * b[s];
        b[ii] = v / a[ii * k + ii];
    }
    x = std::move(b);
    return true;
}

/// t por comprimento de corda: t_i = L_i / L_total (0 se a poligonal tem comprimento nulo).
std::vector<double> chordLengthT(std::span<const Vec2> pts) {
    std::vector<double> t(pts.size(), 0.0);
    for (std::size_t i = 1; i < pts.size(); ++i) t[i] = t[i - 1] + sla::norm(pts[i] - pts[i - 1]);
    const double total = t.back();
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = total > 0.0 ? t[i] / total : static_cast<double>(i) / static_cast<double>(t.size() - 1);
    t.back() = 1.0;
    return t;
}

/// Projeta t em [0, 1] com t_0 ≤ t_1 ≤ … (monotonia).
void projectT(std::vector<double>& t) {
    for (double& v : t) v = std::clamp(v, 0.0, 1.0);
    for (std::size_t i = 1; i < t.size(); ++i) t[i] = std::max(t[i], t[i - 1]);
}

}  // namespace

// ================================================================ FragmentModel

FragmentModel::FragmentModel(std::span<const Vec2> points, const FragmentEnds& ends)
    : points_(points.begin(), points.end()) {
    if (points_.size() < 4) throw std::invalid_argument("FragmentModel: requer ao menos 4 pontos");
    p0_ = points_.front();
    p3_ = points_.back();
    g1Start_ = ends.startTangent.has_value();
    g1End_ = ends.endTangent.has_value();
    if (g1Start_) tau0_ = *ends.startTangent;
    if (g1End_) tau3_ = *ends.endTangent;
    k_ = (g1Start_ ? 1u : 2u) + (g1End_ ? 1u : 2u);
}

FragmentModel::Parts FragmentModel::parts(double t, int derivative) const {
    const std::array<double, 4> b = derivative == 0 ? BezierCurve::basis(t)
                                    : derivative == 1 ? BezierCurve::basisD1(t)
                                                      : BezierCurve::basisD2(t);
    Parts r;
    r.fixed = p0_ * b[0] + p3_ * b[3];
    if (g1Start_) r.fixed += p0_ * b[1];  // P₁ = P₀ + α τ₀
    if (g1End_) r.fixed += p3_ * b[2];    // P₂ = P₃ − β τ₃
    std::size_t j = 0;
    if (g1Start_) {
        r.cols[j++] = tau0_ * b[1];
    } else {
        r.cols[j++] = {b[1], 0.0};
        r.cols[j++] = {0.0, b[1]};
    }
    if (g1End_) {
        r.cols[j++] = tau3_ * (-b[2]);
    } else {
        r.cols[j++] = {b[2], 0.0};
        r.cols[j++] = {0.0, b[2]};
    }
    return r;
}

BezierCurve FragmentModel::curve(std::span<const double> p) const {
    if (p.size() != k_) throw std::invalid_argument("FragmentModel::curve: p com tamanho errado");
    std::size_t j = 0;
    Vec2 p1;
    if (g1Start_) {
        p1 = p0_ + tau0_ * p[j++];
    } else {
        p1 = {p[j], p[j + 1]};
        j += 2;
    }
    Vec2 p2;
    if (g1End_) {
        p2 = p3_ - tau3_ * p[j++];
    } else {
        p2 = {p[j], p[j + 1]};
        j += 2;
    }
    return {p0_, p1, p2, p3_};
}

double FragmentModel::energy(std::span<const double> p, std::span<const double> t) const {
    const std::size_t m = interiorCount();
    if (p.size() != k_ || t.size() != m) throw std::invalid_argument("FragmentModel::energy: tamanhos errados");
    const BezierCurve c = curve(p);
    double e = 0.0;
    for (std::size_t i = 0; i < m; ++i) e += 0.5 * sla::norm2(c.eval(t[i]) - points_[i + 1]);
    return e;
}

FitDerivatives FragmentModel::derivatives(std::span<const double> p, std::span<const double> t) const {
    const std::size_t m = interiorCount();
    if (p.size() != k_ || t.size() != m) throw std::invalid_argument("FragmentModel::derivatives: tamanhos errados");
    FitDerivatives out;
    out.gp.assign(k_, 0.0);
    out.gt.assign(m, 0.0);
    out.hpp.assign(k_ * k_, 0.0);
    out.hpt.assign(m * k_, 0.0);
    out.d.assign(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        const Parts v0 = parts(t[i], 0);
        const Parts v1 = parts(t[i], 1);
        const Parts v2 = parts(t[i], 2);
        Vec2 b0 = v0.fixed;
        Vec2 b1 = v1.fixed;
        Vec2 b2 = v2.fixed;
        for (std::size_t j = 0; j < k_; ++j) {
            b0 += v0.cols[j] * p[j];
            b1 += v1.cols[j] * p[j];
            b2 += v2.cols[j] * p[j];
        }
        const Vec2 r = b0 - points_[i + 1];
        out.energy += 0.5 * sla::norm2(r);
        out.gt[i] = sla::dot(r, b1);
        out.d[i] = sla::norm2(b1) + sla::dot(r, b2);
        for (std::size_t j = 0; j < k_; ++j) {
            out.gp[j] += sla::dot(v0.cols[j], r);
            for (std::size_t l = 0; l < k_; ++l) out.hpp[j * k_ + l] += sla::dot(v0.cols[j], v0.cols[l]);
            // h_j = J_jᵀ B' + J'_jᵀ r
            out.hpt[i * k_ + j] = sla::dot(v0.cols[j], b1) + sla::dot(v1.cols[j], r);
        }
    }
    return out;
}

std::vector<double> FragmentModel::leastSquares(std::span<const double> t) const {
    const std::size_t m = interiorCount();
    if (t.size() != m) throw std::invalid_argument("FragmentModel::leastSquares: t com tamanho errado");
    std::vector<double> a(k_ * k_, 0.0);
    std::vector<double> rhs(k_, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        const Parts v = parts(t[i], 0);
        const Vec2 target = points_[i + 1] - v.fixed;
        for (std::size_t j = 0; j < k_; ++j) {
            rhs[j] += sla::dot(v.cols[j], target);
            for (std::size_t l = 0; l < k_; ++l) a[j * k_ + l] += sla::dot(v.cols[j], v.cols[l]);
        }
    }
    std::vector<double> x;
    if (choleskySolve(a, rhs, k_, x)) return x;
    // Quase singular (t quase coincidentes): uma crista mínima, proporcional ao traço.
    double trace = 0.0;
    for (std::size_t j = 0; j < k_; ++j) trace += a[j * k_ + j];
    if (!(trace > 0.0)) return {};
    for (std::size_t j = 0; j < k_; ++j) a[j * k_ + j] += 1e-10 * trace;
    return choleskySolve(a, rhs, k_, x) ? x : std::vector<double>{};
}

SchurStep FragmentModel::schurStep(const FitDerivatives& d, double lambda) {
    const std::size_t k = d.gp.size();
    const std::size_t m = d.gt.size();
    SchurStep out;
    std::vector<double> s = d.hpp;
    std::vector<double> rhs(k);
    for (std::size_t j = 0; j < k; ++j) {
        s[j * k + j] += lambda;
        rhs[j] = -d.gp[j];
    }
    std::vector<double> denom(m);
    for (std::size_t i = 0; i < m; ++i) {
        denom[i] = std::max(d.d[i] + lambda, 1e-9);  // D_i + λ, mantido positivo
        const double* h = &d.hpt[i * k];
        for (std::size_t j = 0; j < k; ++j) {
            rhs[j] += h[j] * d.gt[i] / denom[i];
            for (std::size_t l = 0; l < k; ++l) s[j * k + l] -= h[j] * h[l] / denom[i];
        }
    }
    if (!choleskySolve(s, rhs, k, out.dp)) return out;
    out.dt.resize(m);
    for (std::size_t i = 0; i < m; ++i) {
        double ht = 0.0;
        for (std::size_t j = 0; j < k; ++j) ht += d.hpt[i * k + j] * out.dp[j];
        out.dt[i] = -(d.gt[i] + ht) / denom[i];
    }
    out.ok = true;
    return out;
}

// ================================================================ ajuste de um fragmento

FragmentFit fitFragment(std::span<const Vec2> points, const FragmentEnds& ends, const FitParams& params,
                        std::span<const double> initialT) {
    const std::size_t m = points.size();
    if (m < 2) throw std::invalid_argument("fitFragment: requer ao menos 2 pontos");
    FragmentFit out;
    out.t.assign(m, 0.0);
    out.t.back() = 1.0;
    const Vec2 p0 = points.front();
    const Vec2 p3 = points.back();

    if (m <= 3) {  // reta (§7): o t de cada nó interior é a projeção sobre a corda
        out.line = true;
        out.curve = BezierCurve::fromLine(p0, p3);
        const Vec2 d = p3 - p0;
        const double dd = sla::dot(d, d);
        for (std::size_t i = 1; i + 1 < m; ++i) {
            const double t = dd > 0.0 ? std::clamp(sla::dot(points[i] - p0, d) / dd, 0.0, 1.0) : 0.0;
            out.t[i] = t;
            out.energy += 0.5 * sla::norm2(p0 + d * t - points[i]);
        }
        return out;
    }

    const FragmentModel model(points, ends);
    const std::size_t k = model.paramCount();
    const std::size_t mi = model.interiorCount();

    std::vector<double> t(mi);
    if (initialT.size() == m) {
        for (std::size_t i = 0; i < mi; ++i) t[i] = initialT[i + 1];
    } else {
        const std::vector<double> chord = chordLengthT(points);
        for (std::size_t i = 0; i < mi; ++i) t[i] = chord[i + 1];
    }
    projectT(t);

    // Heurística de Schneider: P₁ = P₀ + (P₃−P₀)/3 e P₂ = P₃ − (P₃−P₀)/3 (α = β = ‖P₃−P₀‖/3 em G¹).
    const auto schneider = [&]() {
        std::vector<double> p;
        const Vec2 third = (p3 - p0) / 3.0;
        const double len3 = sla::norm(p3 - p0) / 3.0;
        if (model.startIsG1()) {
            p.push_back(len3);
        } else {
            p.push_back(p0.x + third.x);
            p.push_back(p0.y + third.y);
        }
        if (model.endIsG1()) {
            p.push_back(len3);
        } else {
            p.push_back(p3.x - third.x);
            p.push_back(p3.y - third.y);
        }
        return p;
    };
    const auto g1Positive = [&](const std::vector<double>& p) {
        if (model.startIsG1() && !(p[0] > 0.0)) return false;
        if (model.endIsG1() && !(p[k - 1] > 0.0)) return false;
        return true;
    };

    std::vector<double> p = model.leastSquares(t);
    if (p.empty() || !g1Positive(p)) p = schneider();
    double e = model.energy(p, t);

    double lambda = 1e-3;
    for (int it = 0; it < params.newtonMaxIterations; ++it) {
        const FitDerivatives dv = model.derivatives(p, t);
        bool accepted = false;
        std::vector<double> p2;
        std::vector<double> t2;
        double e2 = e;
        for (int tries = 0; tries < 14 && !accepted; ++tries) {
            const SchurStep st = FragmentModel::schurStep(dv, lambda);
            if (st.ok) {
                p2 = p;
                t2 = t;
                for (std::size_t j = 0; j < k; ++j) p2[j] += st.dp[j];
                for (std::size_t i = 0; i < mi; ++i) t2[i] += st.dt[i];
                projectT(t2);
                e2 = model.energy(p2, t2);
                if (std::isfinite(e2) && e2 < e) {
                    accepted = true;
                    break;
                }
            }
            lambda = std::min(lambda * 10.0, 1e12);  // passo rejeitado: λ ← 10 λ
        }
        if (!accepted) break;
        lambda = std::max(lambda / 10.0, 1e-12);     // passo aceito: λ ← λ / 10
        const double drop = e - e2;
        p = std::move(p2);
        t = std::move(t2);
        e = e2;
        ++out.newtonIterations;
        if (drop <= 1e-14 * (1.0 + e)) break;
    }

    if (!g1Positive(p)) {  // laço ou cúspide: Schneider, e o fragmento fica marcado para subdivisão
        p = schneider();
        // Os t vieram de outra curva: reotimiza-os para a de Schneider (Newton 1D por nó, p fixo), para
        // que E* seja a energia dessa curva e não a de uma parametrização herdada.
        const BezierCurve c = model.curve(p);
        for (std::size_t i = 0; i < mi; ++i) {
            double ti = std::clamp(t[i], 0.0, 1.0);
            for (int it = 0; it < 8; ++it) {
                const Vec2 r = c.eval(ti) - points[i + 1];
                const Vec2 d1 = c.d1(ti);
                const double g = sla::dot(r, d1);
                const double hh = std::max(sla::norm2(d1) + sla::dot(r, c.d2(ti)), 1e-9);
                const double next = std::clamp(ti - g / hh, 0.0, 1.0);
                if (std::abs(next - ti) < 1e-14) break;
                ti = next;
            }
            t[i] = ti;
        }
        projectT(t);
        e = model.energy(p, t);
        out.degenerate = true;
    }
    out.curve = model.curve(p);
    out.energy = e;
    for (std::size_t i = 0; i < mi; ++i) out.t[i + 1] = t[i];
    return out;
}

// ================================================================ cadeia de fragmentos

FitStats& FitStats::operator+=(const FitStats& o) noexcept {
    initialFragments += o.initialFragments;
    finalFragments += o.finalFragments;
    merges += o.merges;
    shifts += o.shifts;
    lines += o.lines;
    degenerate += o.degenerate;
    energy += o.energy;
    return *this;
}

BezierFitter::BezierFitter(FitParams params) : params_(params) {
    if (!(params_.statThreshInitial > 0.0)) throw std::invalid_argument("BezierFitter: statThreshInitial <= 0");
    if (!(params_.statThreshFinal >= params_.statThreshInitial)) {
        throw std::invalid_argument("BezierFitter: statThreshFinal < statThreshInitial");
    }
    if (params_.maxRounds < 1) throw std::invalid_argument("BezierFitter: maxRounds < 1");
    if (params_.newtonMaxIterations < 0) throw std::invalid_argument("BezierFitter: newtonMaxIterations < 0");
    if (params_.initialFragmentNodes < 2) throw std::invalid_argument("BezierFitter: initialFragmentNodes < 2");
    if (params_.finalSweeps < 0) throw std::invalid_argument("BezierFitter: finalSweeps < 0");
}

std::vector<double> BezierFitter::thresholdSchedule(const FitParams& p) {
    std::vector<double> tau;
    if (p.maxRounds == 1) return {p.statThreshFinal};
    const double ratio = p.statThreshFinal / p.statThreshInitial;
    for (int r = 0; r < p.maxRounds; ++r) {
        tau.push_back(p.statThreshInitial * std::pow(ratio, static_cast<double>(r) / static_cast<double>(p.maxRounds - 1)));
    }
    tau.back() = p.statThreshFinal;  // sem erro de arredondamento no último
    return tau;
}

namespace {

struct Span {
    std::size_t a = 0;
    std::size_t b = 0;
    std::optional<Vec2> startTangent;  ///< tangente imposta na ponta do trecho (âncora lisa de aresta fechada)
    std::optional<Vec2> endTangent;
};

struct Fragment {
    std::size_t a = 0;
    std::size_t b = 0;
    FragmentFit fit;
    std::size_t size() const noexcept { return b - a + 1; }
};

class ChainFitter {
public:
    ChainFitter(std::span<const Vec2> pts, const FitParams& params) : pts_(pts), params_(params) {}

    /// Tangente (unitária, no sentido do percurso) do nó interior i, por diferença central.
    std::optional<Vec2> tangent(std::size_t i) const {
        if (!params_.enforceG1 || i == 0 || i + 1 >= pts_.size()) return std::nullopt;
        return unit(pts_[i + 1] - pts_[i - 1]);
    }

    static std::optional<Vec2> unit(Vec2 v) {
        const double n = sla::norm(v);
        if (n < kTiny) return std::nullopt;
        return v / n;
    }

    FragmentEnds ends(const Span& s, std::size_t a, std::size_t b) const {
        FragmentEnds e;
        e.startTangent = a == s.a ? s.startTangent : tangent(a);
        e.endTangent = b == s.b ? s.endTangent : tangent(b);
        return e;
    }

    Fragment fit(const Span& s, std::size_t a, std::size_t b, std::span<const double> warm = {}) const {
        Fragment f;
        f.a = a;
        f.b = b;
        f.fit = fitFragment(pts_.subspan(a, b - a + 1), ends(s, a, b), params_, warm);
        return f;
    }

    /// t inicial da união A ∪ B: reescala os t das partes pelo comprimento de corda de cada uma.
    std::vector<double> warmStart(const Fragment& A, const Fragment& B) const {
        const double la = polyLength(A.a, A.b);
        const double lb = polyLength(B.a, B.b);
        if (!(la > 0.0) || !(lb > 0.0)) return {};
        std::vector<double> t;
        for (std::size_t i = 0; i < A.fit.t.size(); ++i) t.push_back(A.fit.t[i] * la / (la + lb));
        for (std::size_t i = 1; i < B.fit.t.size(); ++i) t.push_back((la + B.fit.t[i] * lb) / (la + lb));
        return t;
    }

    double polyLength(std::size_t a, std::size_t b) const {
        double l = 0.0;
        for (std::size_t i = a; i < b; ++i) l += sla::norm(pts_[i + 1] - pts_[i]);
        return l;
    }

    /// Fragmentos iniciais de ~initialFragmentNodes nós: todos com ≥ 4 pontos, salvo um trecho curto.
    std::vector<Fragment> initialFragments(const Span& s) const {
        std::vector<std::size_t> cuts{s.a};
        const std::size_t step = static_cast<std::size_t>(std::max(params_.initialFragmentNodes, 4)) - 1;
        std::size_t pos = s.a;
        while (s.b - pos > step + 2) {
            pos += step;
            cuts.push_back(pos);
        }
        cuts.push_back(s.b);
        std::vector<Fragment> frags;
        for (std::size_t i = 0; i + 1 < cuts.size(); ++i) frags.push_back(fit(s, cuts[i], cuts[i + 1]));
        return frags;
    }

    /// Uma varredura de fusão/deslocamento com limiar τ. Retorna se algo mudou.
    bool sweep(const Span& s, std::vector<Fragment>& frags, double tau, FitStats& stats) const {
        bool changed = false;
        std::size_t i = 0;
        while (i + 1 < frags.size()) {
            const Fragment& A = frags[i];
            const Fragment& B = frags[i + 1];
            const std::vector<double> warm = warmStart(A, B);
            Fragment M = fit(s, A.a, B.b, warm);
            const double before = A.fit.energy + B.fit.energy;
            if (!M.fit.degenerate && M.fit.energy - before < tau) {
                frags[i] = std::move(M);
                frags.erase(frags.begin() + static_cast<std::ptrdiff_t>(i) + 1);
                ++stats.merges;
                changed = true;
                continue;  // tenta fundir o resultado com o próximo vizinho
            }
            // Fusão recusada: desloca a quebra em ±1 nó e fica com a de menor E*(A) + E*(B).
            const std::size_t minA = std::min<std::size_t>(4, A.size());
            const std::size_t minB = std::min<std::size_t>(4, B.size());
            double best = before;
            std::optional<std::pair<Fragment, Fragment>> bestPair;
            for (const int shift : {-1, +1}) {
                const long j = static_cast<long>(A.b) + shift;
                if (j <= static_cast<long>(A.a) || j >= static_cast<long>(B.b)) continue;
                const auto ju = static_cast<std::size_t>(j);
                if (ju - A.a + 1 < minA || B.b - ju + 1 < minB) continue;
                Fragment A2 = fit(s, A.a, ju);
                Fragment B2 = fit(s, ju, B.b);
                const double sum = A2.fit.energy + B2.fit.energy;
                if (sum < best - 1e-12 * (1.0 + best)) {
                    best = sum;
                    bestPair = std::make_pair(std::move(A2), std::move(B2));
                }
            }
            if (bestPair) {
                frags[i] = std::move(bestPair->first);
                frags[i + 1] = std::move(bestPair->second);
                ++stats.shifts;
                changed = true;
            }
            ++i;
        }
        return changed;
    }

private:
    std::span<const Vec2> pts_;
    const FitParams& params_;
};

}  // namespace

ChainFit BezierFitter::fitChain(std::span<const Vec2> pts, std::span<const std::uint8_t> corner, bool closed) const {
    const std::size_t n = pts.size();
    if (n < 2) throw std::invalid_argument("BezierFitter::fitChain: requer ao menos 2 pontos");
    if (corner.size() != n) throw std::invalid_argument("BezierFitter::fitChain: corner.size() != pts.size()");
    if (closed && (n < 4 || pts.front() != pts.back())) {
        throw std::invalid_argument("BezierFitter::fitChain: cadeia fechada exige ≥ 4 pontos com o último igual ao primeiro");
    }

    const ChainFitter cf(pts, params_);
    // Âncora lisa de uma cadeia fechada: tangente compartilhada pelo primeiro e pelo último fragmento.
    std::optional<Vec2> anchor;
    if (closed && corner[0] == 0 && params_.enforceG1) anchor = ChainFitter::unit(pts[1] - pts[n - 2]);

    std::vector<std::size_t> cuts{0};
    for (std::size_t i = 1; i + 1 < n; ++i) {
        if (corner[i] != 0) cuts.push_back(i);
    }
    cuts.push_back(n - 1);

    ChainFit out;
    const std::vector<double> schedule = thresholdSchedule(params_);
    for (std::size_t c = 0; c + 1 < cuts.size(); ++c) {
        Span span;
        span.a = cuts[c];
        span.b = cuts[c + 1];
        if (span.a == 0) span.startTangent = anchor;
        if (span.b == n - 1) span.endTangent = anchor;

        std::vector<Fragment> frags = cf.initialFragments(span);
        out.stats.initialFragments += static_cast<int>(frags.size());
        for (const double tau : schedule) (void)cf.sweep(span, frags, tau, out.stats);
        for (int extra = 0; extra < params_.finalSweeps; ++extra) {
            if (!cf.sweep(span, frags, schedule.back(), out.stats)) break;
        }
        for (const Fragment& f : frags) {
            const Vec2 a = pts[f.a];
            const Vec2 b = pts[f.b];
            if (f.fit.line) {
                out.curves.push_back({sla::SegmentKind::Line, BezierCurve::fromLine(a, b)});
                ++out.stats.lines;
            } else {
                out.curves.push_back({sla::SegmentKind::Cubic, f.fit.curve});
            }
            out.stats.degenerate += f.fit.degenerate ? 1 : 0;
            out.stats.energy += f.fit.energy;
            ++out.stats.finalFragments;
        }
    }
    return out;
}

namespace {

constexpr double kMaxGapPx = 2.0;           ///< maior distância entre amostras consecutivas da poligonal no ajuste
constexpr double kStraightTolerancePx = 0.02;  ///< desvio máximo dos controles à corda para a cúbica contar como reta

}  // namespace

/// P1 e P2 a no máximo kStraightTolerancePx da corda P0–P3 e dentro dela, em ordem.
bool BezierFitter::isStraight(const BezierCurve& c) {
    const Vec2 d = c.p[3] - c.p[0];
    const double len = sla::norm(d);
    if (len < 1e-9) return false;
    double prev = 0.0;
    for (const Vec2 q : {c.p[1], c.p[2]}) {
        const double along = sla::dot(q - c.p[0], d) / len;
        if (std::abs(sla::cross(q - c.p[0], d)) / len > kStraightTolerancePx || along < prev - kStraightTolerancePx || along > len + kStraightTolerancePx) return false;
        prev = along;
    }
    return true;
}

FitStats BezierFitter::run(VectorImage& image, TaskContext ctx) const {
    image.clearCurves();
    FitStats total;
    std::vector<Vec2> pts;
    std::vector<std::uint8_t> corner;
    constexpr std::uint8_t kBreak = NodeFlag::Corner | NodeFlag::Junction | NodeFlag::Pinned;
    for (EdgeId e = 0; e < static_cast<EdgeId>(image.edgeCount()); ++e) {
        ctx.checkpoint(static_cast<double>(e) / static_cast<double>(image.edgeCount()));
        const auto nodes = image.edgeNodes(e);
        pts.clear();
        corner.clear();
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const Vec2 p = image.position(nodes[i]);
            pts.push_back(p);
            corner.push_back((image.flags(nodes[i]) & kBreak) != 0 ? 1 : 0);
            if (i + 1 == nodes.size()) break;
            // A energia do ajuste só vê os pontos: depois que o tracer remove os nós colineares, uma borda reta de 18 px
            // tem 2 nós, e uma cúbica pode se afastar 1 px no meio dela sem custo. Segmentos longos ganham pontos
            // intermediários (não são nós: o ajuste só os usa como amostras da poligonal).
            const Vec2 q = image.position(nodes[i + 1]);
            const double len = sla::norm(q - p);
            const int pieces = static_cast<int>(std::ceil(len / kMaxGapPx));
            for (int k = 1; k < pieces; ++k) {
                pts.push_back(p + (q - p) * (static_cast<double>(k) / pieces));
                corner.push_back(0);
            }
        }
        const bool closed = nodes.front() == nodes.back() && nodes.size() >= 4;
        ChainFit fit = fitChain(pts, corner, closed);
        // Um trecho que ficou reto (controles sobre a corda) é uma reta: sai como "L", não como cúbica degenerada.
        for (sla::CurveSegment& c : fit.curves) {
            if (c.kind == sla::SegmentKind::Cubic && BezierFitter::isStraight(c.curve)) {
                c = {sla::SegmentKind::Line, BezierCurve::fromLine(c.curve.p[0], c.curve.p[3])};
                ++fit.stats.lines;
            }
        }
        image.setEdgeCurves(e, fit.curves);
        total += fit.stats;
    }
    return total;
}

}  // namespace cnx
