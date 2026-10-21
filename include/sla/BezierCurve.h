// sla/BezierCurve.h — Bézier cúbica e segmentos de cadeia de curvas.
//
//   B(t) = Σ_{k=0}^{3} b_k(t) P_k,   b_k(t) = C(3,k) (1−t)^{3−k} t^k,   t ∈ [0, 1]
//
// O ajuste (mínimos quadrados + Newton/Schur) fica em cnx/fit/BezierFitter; aqui está só a
// geometria, usada pelo ajuste, pela exportação e pela renderização da prévia.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "sla/Vec.h"

namespace sla {

/// Caixa alinhada aos eixos; vazia por padrão (min = +∞, max = −∞).
struct Box2 {
    Vec2 min{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    Vec2 max{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};

    constexpr bool empty() const noexcept { return min.x > max.x || min.y > max.y; }
    constexpr void expand(Vec2 p) noexcept {
        if (p.x < min.x) min.x = p.x;
        if (p.y < min.y) min.y = p.y;
        if (p.x > max.x) max.x = p.x;
        if (p.y > max.y) max.y = p.y;
    }
};

class BezierCurve {
public:
    std::array<Vec2, 4> p{};

    constexpr BezierCurve() noexcept = default;
    constexpr BezierCurve(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3) noexcept : p{p0, p1, p2, p3} {}

    /// Reta como cúbica com elevação de grau: P1 = P0 + Δ/3, P2 = P0 + 2Δ/3.
    /// A parametrização fica uniforme (B'(t) = Δ constante).
    static constexpr BezierCurve fromLine(Vec2 a, Vec2 b) noexcept {
        const Vec2 d = b - a;
        return {a, a + d / 3.0, a + d * (2.0 / 3.0), b};
    }

    /// Bases de Bernstein e derivadas (u = 1 − t):
    ///   b   = (u³, 3u²t, 3ut², t³)
    ///   b'  = (−3u², 3u² − 6ut, 6ut − 3t², 3t²)
    ///   b'' = (6u, −12u + 6t, 6u − 12t, 6t)
    static constexpr std::array<double, 4> basis(double t) noexcept {
        const double u = 1.0 - t;
        return {u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t};
    }
    static constexpr std::array<double, 4> basisD1(double t) noexcept {
        const double u = 1.0 - t;
        return {-3.0 * u * u, 3.0 * u * u - 6.0 * u * t, 6.0 * u * t - 3.0 * t * t, 3.0 * t * t};
    }
    static constexpr std::array<double, 4> basisD2(double t) noexcept {
        const double u = 1.0 - t;
        return {6.0 * u, -12.0 * u + 6.0 * t, 6.0 * u - 12.0 * t, 6.0 * t};
    }

    constexpr Vec2 eval(double t) const noexcept { return combine(basis(t)); }
    constexpr Vec2 d1(double t) const noexcept { return combine(basisD1(t)); }
    constexpr Vec2 d2(double t) const noexcept { return combine(basisD2(t)); }

    /// Mesma curva percorrida de trás para frente: B_rev(t) = B(1 − t).
    /// É o que a face vizinha escreve na exportação com fronteira compartilhada.
    constexpr BezierCurve reversed() const noexcept { return {p[3], p[2], p[1], p[0]}; }

    /// Subdivisão exata de de Casteljau em t ∈ (0, 1).
    constexpr std::pair<BezierCurve, BezierCurve> split(double t) const noexcept {
        const Vec2 a = lerp(p[0], p[1], t);
        const Vec2 b = lerp(p[1], p[2], t);
        const Vec2 c = lerp(p[2], p[3], t);
        const Vec2 d = lerp(a, b, t);
        const Vec2 e = lerp(b, c, t);
        const Vec2 f = lerp(d, e, t);
        return {BezierCurve{p[0], a, d, f}, BezierCurve{f, e, c, p[3]}};
    }

    /// M = max(‖P0 − 2P1 + P2‖, ‖P1 − 2P2 + P3‖); limita a curvatura: ‖B''(t)‖ ≤ 6M.
    double secondDifferenceBound() const noexcept;

    /// Número de segmentos uniformes com desvio ≤ tol. A interpolação linear com passo 1/n erra
    /// no máximo (1/8)·max‖B''‖/n² ≤ (6M/8)/n², logo n = ⌈√(3M / (4·tol))⌉ (mínimo 1).
    int flatteningSegments(double tol) const;

    /// Discretização para CAD/CNC (dxf_mode 1/2). Anexa a `out`; omite P0 se includeStart = false,
    /// para encadear curvas sem duplicar vértices.
    void flatten(double tol, std::vector<Vec2>& out, bool includeStart = true) const;

    /// Caixa justa: extremos em t ∈ {0, 1} ∪ raízes de B'_x(t) = 0 e B'_y(t) = 0 (quadráticas).
    Box2 bounds() const noexcept;

    /// Comprimento de arco por Gauss–Legendre adaptativo até a tolerância relativa `tol`.
    double length(double tol = 1e-9) const;

private:
    constexpr Vec2 combine(const std::array<double, 4>& b) const noexcept {
        return p[0] * b[0] + p[1] * b[1] + p[2] * b[2] + p[3] * b[3];
    }
};

enum class SegmentKind : std::uint8_t { Line, Cubic };

/// Elemento de uma cadeia de curvas exportável. Uma reta mantém kind = Line (escrita como "L"/lineto),
/// mas a geometria fica sempre como cúbica (fromLine), unificando avaliação, inversão e discretização.
struct CurveSegment {
    SegmentKind kind = SegmentKind::Cubic;
    BezierCurve curve;

    constexpr Vec2 start() const noexcept { return curve.p[0]; }
    constexpr Vec2 end() const noexcept { return curve.p[3]; }
    constexpr CurveSegment reversed() const noexcept { return {kind, curve.reversed()}; }
};

}  // namespace sla
