// src/sla/BezierCurve.cpp — limites, discretização e comprimento de arco da Bézier cúbica.
#include "sla/BezierCurve.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sla {
namespace {

/// Gauss–Legendre de 5 pontos em [-1, 1].
constexpr std::array<double, 5> kGlNodes{0.0, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640};
constexpr std::array<double, 5> kGlWeights{0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891};

/// ∫_a^b ‖B'(t)‖ dt por GL5.
double glLength(const BezierCurve& c, double a, double b) {
    const double half = 0.5 * (b - a);
    const double mid = 0.5 * (a + b);
    double s = 0.0;
    for (std::size_t i = 0; i < kGlNodes.size(); ++i) s += kGlWeights[i] * norm(c.d1(mid + half * kGlNodes[i]));
    return s * half;
}

double adaptiveLength(const BezierCurve& c, double a, double b, double whole, double tol, int depth) {
    const double m = 0.5 * (a + b);
    const double left = glLength(c, a, m);
    const double right = glLength(c, m, b);
    // Critério relativo ao valor local, com piso absoluto para trechos de comprimento nulo.
    if (depth <= 0 || std::abs(left + right - whole) <= tol * std::max(left + right, 1e-300)) return left + right;
    return adaptiveLength(c, a, m, left, tol, depth - 1) + adaptiveLength(c, m, b, right, tol, depth - 1);
}

/// Raízes de a t² + b t + c = 0 (no máximo 2), anexadas a `out` a partir de `count`.
void collectRoots(double a, double b, double c, std::array<double, 4>& out, int& count) noexcept {
    constexpr double eps = 1e-14;
    if (std::abs(a) < eps * (std::abs(b) + std::abs(c) + 1.0)) {
        if (std::abs(b) > eps) out[static_cast<std::size_t>(count++)] = -c / b;
    } else {
        const double disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
            const double s = std::sqrt(disc);
            // Forma estável: evita o cancelamento de -b ± s.
            const double q = -0.5 * (b + (b >= 0.0 ? s : -s));
            out[static_cast<std::size_t>(count++)] = q / a;
            if (q != 0.0) out[static_cast<std::size_t>(count++)] = c / q;
        }
    }
}

}  // namespace

double BezierCurve::secondDifferenceBound() const noexcept {
    return std::max(norm(p[0] - p[1] * 2.0 + p[2]), norm(p[1] - p[2] * 2.0 + p[3]));
}

int BezierCurve::flatteningSegments(double tol) const {
    if (!(tol > 0.0)) throw std::invalid_argument("BezierCurve::flatteningSegments: tol <= 0");
    const double m = secondDifferenceBound();
    const double n = std::ceil(std::sqrt(3.0 * m / (4.0 * tol)));
    return n < 1.0 ? 1 : (n > 1.0e6 ? 1000000 : static_cast<int>(n));
}

void BezierCurve::flatten(double tol, std::vector<Vec2>& out, bool includeStart) const {
    const int n = flatteningSegments(tol);
    if (includeStart) out.push_back(p[0]);
    for (int i = 1; i < n; ++i) out.push_back(eval(static_cast<double>(i) / n));
    out.push_back(p[3]);  // ponta exata, sem erro de arredondamento de eval(1)
}

Box2 BezierCurve::bounds() const noexcept {
    Box2 box;
    box.expand(p[0]);
    box.expand(p[3]);
    // B'(t)/3 = (d0 − 2d1 + d2) t² + 2(d1 − d0) t + d0, com d_k = P_{k+1} − P_k.
    const Vec2 d0 = p[1] - p[0];
    const Vec2 d1 = p[2] - p[1];
    const Vec2 d2 = p[3] - p[2];
    const Vec2 a = d0 - d1 * 2.0 + d2;
    const Vec2 b = (d1 - d0) * 2.0;
    std::array<double, 4> roots{};
    int count = 0;
    collectRoots(a.x, b.x, d0.x, roots, count);
    collectRoots(a.y, b.y, d0.y, roots, count);
    for (int i = 0; i < count; ++i) {
        const double t = roots[static_cast<std::size_t>(i)];
        if (t > 0.0 && t < 1.0) box.expand(eval(t));
    }
    return box;
}

double BezierCurve::length(double tol) const {
    if (!(tol > 0.0)) throw std::invalid_argument("BezierCurve::length: tol <= 0");
    return adaptiveLength(*this, 0.0, 1.0, glLength(*this, 0.0, 1.0), tol, 30);
}

}  // namespace sla
