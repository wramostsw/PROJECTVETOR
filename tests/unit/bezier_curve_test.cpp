// tests/unit/bezier_curve_test.cpp — geometria da Bézier cúbica: limites, discretização, comprimento.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "sla/BezierCurve.h"
#include "test_harness.h"
#include "test_support.h"

using sla::BezierCurve;
using sla::Vec2;

namespace {

BezierCurve randomCurve(cnxtest::SplitMix64& rng, double span = 10.0) {
    const auto pt = [&] { return Vec2{cnxtest::symmetric(rng, span), cnxtest::symmetric(rng, span)}; };
    return {pt(), pt(), pt(), pt()};
}

/// Distância de q ao segmento [a, b].
double distToSegment(Vec2 q, Vec2 a, Vec2 b) {
    const Vec2 d = b - a;
    const double dd = sla::dot(d, d);
    const double t = dd > 0.0 ? std::clamp(sla::dot(q - a, d) / dd, 0.0, 1.0) : 0.0;
    return sla::norm(q - (a + d * t));
}

}  // namespace

CNX_TEST(basis_functions_form_a_partition_of_unity_with_consistent_derivatives) {
    for (int i = 0; i <= 20; ++i) {
        const double t = i / 20.0;
        const auto b = BezierCurve::basis(t);
        CHECK(std::abs(b[0] + b[1] + b[2] + b[3] - 1.0) < 1e-14);
        const auto d1 = BezierCurve::basisD1(t);
        CHECK(std::abs(d1[0] + d1[1] + d1[2] + d1[3]) < 1e-13);
        // d/dt por diferença central.
        const double h = 1e-6;
        const auto bp = BezierCurve::basis(t + h);
        const auto bm = BezierCurve::basis(t - h);
        for (std::size_t k = 0; k < 4; ++k) CHECK(std::abs((bp[k] - bm[k]) / (2 * h) - d1[k]) < 1e-7);
        const auto dp = BezierCurve::basisD1(t + h);
        const auto dm = BezierCurve::basisD1(t - h);
        const auto d2 = BezierCurve::basisD2(t);
        for (std::size_t k = 0; k < 4; ++k) CHECK(std::abs((dp[k] - dm[k]) / (2 * h) - d2[k]) < 1e-6);
    }
}

CNX_TEST(split_and_reverse_preserve_the_curve) {
    cnxtest::SplitMix64 rng{601};
    for (int trial = 0; trial < 20; ++trial) {
        const BezierCurve c = randomCurve(rng);
        const double u = 0.1 + 0.8 * cnxtest::uniform01(rng);
        const auto [left, right] = c.split(u);
        for (int i = 0; i <= 10; ++i) {
            const double s = i / 10.0;
            CHECK(sla::norm(left.eval(s) - c.eval(u * s)) < 1e-12);
            CHECK(sla::norm(right.eval(s) - c.eval(u + (1.0 - u) * s)) < 1e-12);
            CHECK(sla::norm(c.reversed().eval(s) - c.eval(1.0 - s)) < 1e-12);
        }
        CHECK(left.p[3] == right.p[0]);
    }
}

CNX_TEST(bounds_contain_the_curve_and_touch_it) {
    cnxtest::SplitMix64 rng{602};
    for (int trial = 0; trial < 50; ++trial) {
        const BezierCurve c = randomCurve(rng);
        const sla::Box2 box = c.bounds();
        sla::Box2 sampled;
        for (int i = 0; i <= 20000; ++i) sampled.expand(c.eval(i / 20000.0));
        // Contém todas as amostras (com folga de arredondamento) e é justa: difere das amostras só pelo passo.
        CHECK(box.min.x <= sampled.min.x + 1e-12 && box.min.y <= sampled.min.y + 1e-12);
        CHECK(box.max.x >= sampled.max.x - 1e-12 && box.max.y >= sampled.max.y - 1e-12);
        CHECK_CTX(sla::norm(box.min - sampled.min) < 1e-5 && sla::norm(box.max - sampled.max) < 1e-5, "curva " + std::to_string(trial));
        // Mais justa que a caixa dos pontos de controle.
        sla::Box2 control;
        for (const Vec2& q : c.p) control.expand(q);
        CHECK(box.min.x >= control.min.x - 1e-12 && box.max.x <= control.max.x + 1e-12);
    }
    // Casos degenerados: reta e curva com derivada nula em um eixo.
    const BezierCurve line = BezierCurve::fromLine({0, 0}, {4, 2});
    const sla::Box2 lb = line.bounds();
    CHECK(lb.min == Vec2(0, 0) && lb.max == Vec2(4, 2));
    const BezierCurve flatX({0, 0}, {1, 3}, {2, -3}, {3, 0});  // x monótono, y com extremos interiores
    const sla::Box2 fb = flatX.bounds();
    CHECK(fb.min.x == 0.0 && fb.max.x == 3.0 && fb.max.y > 0.0 && fb.min.y < 0.0);
}

CNX_TEST(length_matches_closed_forms_and_fine_polylines) {
    const BezierCurve line = BezierCurve::fromLine({1, 2}, {4, 6});
    CHECK(std::abs(line.length() - 5.0) < 1e-12);
    cnxtest::SplitMix64 rng{603};
    for (int trial = 0; trial < 10; ++trial) {
        const BezierCurve c = randomCurve(rng);
        double ref = 0.0;
        Vec2 prev = c.p[0];
        constexpr int n = 200000;
        for (int i = 1; i <= n; ++i) {
            const Vec2 q = c.eval(static_cast<double>(i) / n);
            ref += sla::norm(q - prev);
            prev = q;
        }
        CHECK_CTX(std::abs(c.length(1e-10) - ref) < 1e-6 * ref, "curva " + std::to_string(trial));
    }
    // Quarto de círculo unitário pela aproximação clássica (κ = 0,5522847498): comprimento ≈ π/2 dentro de 3e-4.
    const double k = 0.5522847498;
    const BezierCurve quarter({1, 0}, {1, k}, {k, 1}, {0, 1});
    CHECK(std::abs(quarter.length() - std::numbers::pi / 2) < 3e-4);
    CHECK_THROWS(std::invalid_argument, (void)quarter.length(0.0));
}

CNX_TEST(second_difference_bound_limits_the_curvature) {
    cnxtest::SplitMix64 rng{604};
    for (int trial = 0; trial < 20; ++trial) {
        const BezierCurve c = randomCurve(rng);
        const double m = c.secondDifferenceBound();
        for (int i = 0; i <= 100; ++i) CHECK(sla::norm(c.d2(i / 100.0)) <= 6.0 * m + 1e-9);
    }
    CHECK(BezierCurve::fromLine({0, 0}, {5, 5}).secondDifferenceBound() < 1e-14);
}

CNX_TEST(flattening_respects_the_tolerance) {
    cnxtest::SplitMix64 rng{605};
    for (int trial = 0; trial < 15; ++trial) {
        const BezierCurve c = randomCurve(rng);
        for (const double tol : {0.5, 0.05, 0.005}) {
            std::vector<Vec2> poly;
            c.flatten(tol, poly);
            CHECK(static_cast<int>(poly.size()) == c.flatteningSegments(tol) + 1);
            CHECK(poly.front() == c.p[0] && poly.back() == c.p[3]);
            double worst = 0.0;
            for (int i = 0; i <= 400; ++i) {
                const Vec2 q = c.eval(i / 400.0);
                double d = 1e300;
                for (std::size_t s = 0; s + 1 < poly.size(); ++s) d = std::min(d, distToSegment(q, poly[s], poly[s + 1]));
                worst = std::max(worst, d);
            }
            CHECK_CTX(worst <= tol, "tol " + std::to_string(tol) + " desvio " + std::to_string(worst));
        }
        // Menos tolerância nunca usa menos segmentos.
        CHECK(c.flatteningSegments(0.001) >= c.flatteningSegments(0.01));
    }
    const BezierCurve line = BezierCurve::fromLine({0, 0}, {10, 0});
    CHECK(line.flatteningSegments(1e-6) == 1);
    std::vector<Vec2> out{{99, 99}};
    line.flatten(0.1, out, false);  // sem o ponto inicial
    CHECK(out.size() == 2 && out.back() == Vec2(10, 0));
    CHECK_THROWS(std::invalid_argument, (void)line.flatteningSegments(0.0));
}

int main() { return cnxtest::runAll(); }
