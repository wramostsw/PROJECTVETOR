// src/vector/Optimizer.cpp — gradiente conjugado Polak–Ribière+ com busca de Wolfe forte.
#include "cnx/vector/Optimizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cnx {
namespace {

double dotv(std::span<const double> a, std::span<const double> b) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double infNorm(std::span<const double> a) noexcept {
    double m = 0.0;
    for (const double v : a) m = std::max(m, std::abs(v));
    return m;
}

/// Mínimo do interpolante cúbico de φ entre (a0, f0, d0) e (a1, f1, d1) (N&W 3.59), ou NaN.
double cubicMinimizer(double a0, double f0, double d0, double a1, double f1, double d1) noexcept {
    const double h = a0 - a1;
    if (h == 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double q1 = d0 + d1 - 3.0 * (f0 - f1) / h;
    const double disc = q1 * q1 - d0 * d1;
    if (disc < 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double q2 = (a1 > a0 ? 1.0 : -1.0) * std::sqrt(disc);
    const double denom = d1 - d0 + 2.0 * q2;
    if (denom == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return a1 - (a1 - a0) * ((d1 + q2 - q1) / denom);
}

}  // namespace

ConjugateGradient::ConjugateGradient(ObjectiveFunction& objective, CgOptions options)
    : objective_(&objective), options_(options) {
    if (options_.stride < 1) throw std::invalid_argument("ConjugateGradient: stride < 1");
    if (!(options_.c1 > 0.0 && options_.c1 < options_.c2 && options_.c2 < 1.0)) {
        throw std::invalid_argument("ConjugateGradient: requer 0 < c1 < c2 < 1");
    }
}

double ConjugateGradient::maxNodeNorm(std::span<const double> v) const noexcept {
    const std::size_t s = static_cast<std::size_t>(options_.stride);
    double m2 = 0.0;
    for (std::size_t i = 0; i < v.size(); i += s) {
        double n2 = 0.0;
        for (std::size_t c = 0; c < s && i + c < v.size(); ++c) n2 += v[i + c] * v[i + c];
        m2 = std::max(m2, n2);
    }
    return std::sqrt(m2);
}

void ConjugateGradient::reset(std::span<const double> x0) {
    x_.assign(x0.begin(), x0.end());
    g_.assign(x0.size(), 0.0);
    d_.assign(x0.size(), 0.0);
    f_ = objective_->evaluate(x_, g_);
    evaluations_ = 1;
    iterations_ = 0;
    for (std::size_t i = 0; i < g_.size(); ++i) d_[i] = -g_[i];
    const double gn = std::sqrt(dotv(g_, g_));
    alphaPrev_ = gn > 0.0 ? 1.0 / gn : 1.0;  // primeiro passo de comprimento unitário
    lastAlpha_ = 0.0;
    lastBeta_ = 0.0;
    lastCurvatureRatio_ = 0.0;
    started_ = true;
}

void ConjugateGradient::evaluateAt(double alpha, Point& out) {
    out.alpha = alpha;
    out.x.resize(x_.size());
    out.g.resize(x_.size());
    for (std::size_t i = 0; i < x_.size(); ++i) out.x[i] = x_[i] + alpha * d_[i];
    out.f = objective_->evaluate(out.x, out.g);
    out.dphi = dotv(out.g, d_);
    ++evaluations_;
    ++lineSearchEvals_;
}

bool ConjugateGradient::zoom(Point lo, Point hi, Point& accepted) {
    Point t;
    while (lineSearchEvals_ < options_.maxLineSearchEvals) {
        const double width = hi.alpha - lo.alpha;
        if (std::abs(width) <= 1e-14 * std::max(1.0, std::abs(lo.alpha))) break;
        double a = cubicMinimizer(lo.alpha, lo.f, lo.dphi, hi.alpha, hi.f, hi.dphi);
        // Protege: dentro do intervalo, longe das pontas (senão bisseção).
        const double lower = std::min(lo.alpha, hi.alpha) + 0.1 * std::abs(width);
        const double upper = std::max(lo.alpha, hi.alpha) - 0.1 * std::abs(width);
        if (!std::isfinite(a) || !std::isfinite(hi.f) || a < lower || a > upper) a = 0.5 * (lo.alpha + hi.alpha);
        evaluateAt(a, t);
        if (!std::isfinite(t.f) || t.f > f0_ + options_.c1 * a * dphi0_ || t.f >= lo.f) {
            hi = t;
        } else {
            if (std::abs(t.dphi) <= -options_.c2 * dphi0_) {
                accepted = t;
                return true;
            }
            if (t.dphi * (hi.alpha - lo.alpha) >= 0.0) hi = lo;
            lo = t;
        }
    }
    if (lo.alpha > 0.0) {  // todo `lo` com alpha > 0 satisfaz Armijo e reduz f
        accepted = lo;
        return true;
    }
    return false;
}

bool ConjugateGradient::lineSearch(double alphaInit, double alphaMax, Point& accepted) {
    lineSearchEvals_ = 0;
    Point prev;
    prev.alpha = 0.0;
    prev.f = f0_;
    prev.dphi = dphi0_;
    prev.x = x_;
    prev.g = g_;
    Point cur;
    double a = std::min(alphaInit, alphaMax);
    for (int i = 1; lineSearchEvals_ < options_.maxLineSearchEvals; ++i) {
        evaluateAt(a, cur);
        if (!std::isfinite(cur.f) || cur.f > f0_ + options_.c1 * a * dphi0_ || (i > 1 && cur.f >= prev.f)) {
            return zoom(prev, cur, accepted);
        }
        if (std::abs(cur.dphi) <= -options_.c2 * dphi0_) {
            accepted = cur;
            return true;
        }
        if (cur.dphi >= 0.0) return zoom(cur, prev, accepted);
        if (a >= alphaMax) {  // teto da região de confiança: Armijo basta
            accepted = cur;
            return true;
        }
        prev = cur;
        a = std::min(2.0 * a, alphaMax);
    }
    accepted = cur;  // orçamento esgotado com Armijo satisfeito até aqui
    return std::isfinite(cur.f) && cur.f <= f0_ + options_.c1 * cur.alpha * dphi0_;
}

CgStatus ConjugateGradient::iterate() {
    if (!started_) throw std::logic_error("ConjugateGradient::iterate: chame reset() antes");
    if (infNorm(g_) <= options_.gradientTol) return CgStatus::ConvergedGradient;
    if (iterations_ >= options_.maxIterations) return CgStatus::MaxIterations;

    dphi0_ = dotv(g_, d_);
    if (!(dphi0_ < 0.0)) {  // não é direção de descida: reinicia no gradiente
        for (std::size_t i = 0; i < g_.size(); ++i) d_[i] = -g_[i];
        dphi0_ = -dotv(g_, g_);
        lastBeta_ = 0.0;
    }
    f0_ = f_;

    const double dn = maxNodeNorm(d_);
    const double alphaMax = dn > 0.0 ? options_.maxStep / dn : std::numeric_limits<double>::infinity();
    const double alphaInit = std::min(alphaPrev_, alphaMax);

    Point accepted;
    if (!lineSearch(alphaInit, alphaMax, accepted) && dotv(g_, g_) > 0.0) {
        // Última tentativa: gradiente puro (descarta a memória do CG).
        for (std::size_t i = 0; i < g_.size(); ++i) d_[i] = -g_[i];
        dphi0_ = -dotv(g_, g_);
        const double dn2 = maxNodeNorm(d_);
        const double aMax2 = dn2 > 0.0 ? options_.maxStep / dn2 : std::numeric_limits<double>::infinity();
        if (!lineSearch(std::min(1.0 / std::sqrt(-dphi0_), aMax2), aMax2, accepted)) return CgStatus::LineSearchFailed;
        lastBeta_ = 0.0;
    }

    const double gOldSq = dotv(g_, g_);
    const double gOldDotNew = dotv(g_, accepted.g);
    const double gNewSq = dotv(accepted.g, accepted.g);
    const double fOld = f_;
    const double dphiOld = dphi0_;

    lastAlpha_ = accepted.alpha;
    lastCurvatureRatio_ = dphiOld != 0.0 ? accepted.dphi / dphiOld : 0.0;
    x_ = accepted.x;
    g_ = accepted.g;
    f_ = accepted.f;
    ++iterations_;

    // Próxima direção: PR+ com reinício de Powell.
    const double beta = gOldSq > 0.0 ? std::max(0.0, (gNewSq - gOldDotNew) / gOldSq) : 0.0;
    const bool powellRestart = std::abs(gOldDotNew) >= 0.2 * gNewSq;
    if (powellRestart) {
        for (std::size_t i = 0; i < g_.size(); ++i) d_[i] = -g_[i];
        lastBeta_ = 0.0;
    } else {
        for (std::size_t i = 0; i < g_.size(); ++i) d_[i] = -g_[i] + beta * d_[i];
        lastBeta_ = beta;
    }
    // Estimativa do próximo passo inicial (N&W 3.60).
    const double dphiNew = dotv(g_, d_);
    alphaPrev_ = (dphiNew < 0.0 && dphiOld < 0.0) ? lastAlpha_ * dphiOld / dphiNew : lastAlpha_ * 2.0;
    if (!(alphaPrev_ > 1e-12) || !std::isfinite(alphaPrev_)) alphaPrev_ = lastAlpha_ > 0.0 ? lastAlpha_ : 1.0;

    if (infNorm(g_) <= options_.gradientTol) return CgStatus::ConvergedGradient;
    if (fOld - f_ <= options_.relEnergyTol * std::max(1.0, std::abs(f_))) return CgStatus::ConvergedEnergy;
    return CgStatus::Running;
}

CgStatus ConjugateGradient::run() {
    CgStatus s = CgStatus::Running;
    while (s == CgStatus::Running) s = iterate();
    return s;
}

}  // namespace cnx
