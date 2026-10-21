// tests/unit/optimizer_test.cpp — gradiente conjugado Polak–Ribière+ com Wolfe forte e região de confiança.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnx/vector/Optimizer.h"
#include "test_harness.h"

using cnx::CgOptions;
using cnx::CgStatus;
using cnx::ConjugateGradient;

namespace {

/// Objetivo definido por lambda; conta avaliações.
class FnObjective final : public cnx::ObjectiveFunction {
public:
    using Fn = std::function<double(std::span<const double>, std::span<double>)>;
    explicit FnObjective(Fn fn) : fn_(std::move(fn)) {}
    double evaluate(std::span<const double> x, std::span<double> grad) override {
        ++calls;
        return fn_(x, grad);
    }
    int calls = 0;

private:
    Fn fn_;
};

double dotv(std::span<const double> a, std::span<const double> b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

/// f = ½ Σ λ_i x_i² − Σ b_i x_i, com λ de 1 a 100.
FnObjective quadratic(std::size_t n) {
    return FnObjective([n](std::span<const double> x, std::span<double> g) {
        double f = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double lambda = 1.0 + 99.0 * static_cast<double>(i) / static_cast<double>(n - 1);
            const double b = 1.0 + 0.1 * static_cast<double>(i);
            f += 0.5 * lambda * x[i] * x[i] - b * x[i];
            g[i] = lambda * x[i] - b;
        }
        return f;
    });
}

FnObjective rosenbrock() {
    return FnObjective([](std::span<const double> x, std::span<double> g) {
        const double a = 1.0 - x[0];
        const double b = x[1] - x[0] * x[0];
        g[0] = -2.0 * a - 400.0 * x[0] * b;
        g[1] = 200.0 * b;
        return a * a + 100.0 * b * b;
    });
}

}  // namespace

CNX_TEST(minimizes_an_ill_conditioned_quadratic) {
    constexpr std::size_t n = 20;
    FnObjective f = quadratic(n);
    CgOptions o;
    o.stride = 1;
    o.gradientTol = 1e-9;
    o.relEnergyTol = 0.0;  // só o gradiente decide: a queda de energia fica abaixo do arredondamento muito antes
    o.maxIterations = 200;
    ConjugateGradient cg(f, o);
    cg.reset(std::vector<double>(n, 0.0));
    CHECK(cg.run() == CgStatus::ConvergedGradient);
    for (std::size_t i = 0; i < n; ++i) {
        const double lambda = 1.0 + 99.0 * static_cast<double>(i) / static_cast<double>(n - 1);
        const double b = 1.0 + 0.1 * static_cast<double>(i);
        CHECK_CTX(std::abs(cg.x()[i] - b / lambda) < 1e-7, std::to_string(i));
    }
    CHECK(cg.iterations() < 100);
}

CNX_TEST(minimizes_the_rosenbrock_function) {
    FnObjective f = rosenbrock();
    CgOptions o;
    o.stride = 1;
    o.gradientTol = 1e-6;
    o.maxIterations = 500;
    ConjugateGradient cg(f, o);
    cg.reset(std::vector<double>{-1.2, 1.0});
    const CgStatus s = cg.run();
    CHECK(s == CgStatus::ConvergedGradient || s == CgStatus::ConvergedEnergy);
    CHECK(std::abs(cg.x()[0] - 1.0) < 1e-4 && std::abs(cg.x()[1] - 1.0) < 1e-4);
    CHECK(cg.value() < 1e-8);
    std::printf("  Rosenbrock: %d iterações, %d avaliações\n", cg.iterations(), cg.evaluations());
}

CNX_TEST(every_accepted_step_satisfies_the_strong_wolfe_conditions) {
    FnObjective f = rosenbrock();
    CgOptions o;
    o.stride = 1;
    o.gradientTol = 1e-8;
    o.maxIterations = 300;
    ConjugateGradient cg(f, o);
    cg.reset(std::vector<double>{-1.2, 1.0});
    int checked = 0;
    for (int it = 0; it < 300; ++it) {
        const std::vector<double> xOld(cg.x().begin(), cg.x().end());
        const std::vector<double> gOld(cg.gradient().begin(), cg.gradient().end());
        const double fOld = cg.value();
        const CgStatus s = cg.iterate();
        if (cg.iterations() == it) break;  // nada aceito
        const double alpha = cg.lastStep();
        CHECK(alpha > 0.0);
        std::vector<double> d(xOld.size());
        for (std::size_t i = 0; i < d.size(); ++i) d[i] = (cg.x()[i] - xOld[i]) / alpha;
        const double dphi0 = dotv(gOld, d);
        const double dphi = dotv(cg.gradient(), d);
        CHECK_CTX(dphi0 < 0.0, "direção de descida, iteração " + std::to_string(it));
        CHECK_CTX(cg.value() <= fOld + o.c1 * alpha * dphi0 + 1e-14, "Armijo, iteração " + std::to_string(it));
        CHECK_CTX(std::abs(dphi) <= o.c2 * std::abs(dphi0) * (1.0 + 1e-9) + 1e-12, "curvatura, iteração " + std::to_string(it));
        CHECK(std::abs(cg.lastCurvatureRatio()) <= o.c2 * (1.0 + 1e-9) + 1e-12);
        CHECK(cg.lastBeta() >= 0.0);  // PR+
        ++checked;
        if (s != CgStatus::Running) break;
    }
    CHECK(checked > 10);
}

CNX_TEST(energy_never_increases) {
    FnObjective f = quadratic(12);
    CgOptions o;
    o.stride = 1;
    o.gradientTol = 1e-10;
    ConjugateGradient cg(f, o);
    cg.reset(std::vector<double>(12, 5.0));
    double previous = cg.value();
    for (int it = 0; it < 60; ++it) {
        const CgStatus s = cg.iterate();
        CHECK(cg.value() <= previous + 1e-12);
        previous = cg.value();
        if (s != CgStatus::Running) break;
    }
}

CNX_TEST(trust_region_caps_the_displacement_of_each_node) {
    // Quadrática distante do mínimo: sem o teto o primeiro passo seria enorme.
    FnObjective f([](std::span<const double> x, std::span<double> g) {
        double v = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double d = x[i] - 100.0;
            v += d * d;
            g[i] = 2.0 * d;
        }
        return v;
    });
    CgOptions o;
    o.stride = 2;
    o.maxStep = 0.5;
    o.gradientTol = 1e-9;
    ConjugateGradient cg(f, o);
    cg.reset(std::vector<double>{0.0, 0.0, 3.0, 4.0, -2.0, 1.0});  // 3 nós
    for (int it = 0; it < 50; ++it) {
        const std::vector<double> before(cg.x().begin(), cg.x().end());
        const CgStatus s = cg.iterate();
        for (std::size_t node = 0; node < 3; ++node) {
            const double dx = cg.x()[2 * node] - before[2 * node];
            const double dy = cg.x()[2 * node + 1] - before[2 * node + 1];
            CHECK_CTX(std::sqrt(dx * dx + dy * dy) <= 0.5 + 1e-12, "iteração " + std::to_string(it));
        }
        if (s != CgStatus::Running) break;
    }
    CHECK(cg.x()[0] > 10.0);  // andou: passos de 0,5 por iteração, 50 iterações
}

CNX_TEST(converges_when_the_function_is_undefined_outside_a_domain) {
    // f = x − ln x (x > 0), mínimo em 1; fora do domínio a função é NaN.
    FnObjective f([](std::span<const double> x, std::span<double> g) {
        g[0] = 1.0 - 1.0 / x[0];
        return x[0] > 0.0 ? x[0] - std::log(x[0]) : std::numeric_limits<double>::quiet_NaN();
    });
    CgOptions o;
    o.stride = 1;
    o.gradientTol = 1e-7;  // abaixo disso a energia (≈ 1) já não distingue os pontos em ponto flutuante
    ConjugateGradient cg(f, o);
    cg.reset(std::vector<double>{0.05});
    const CgStatus s = cg.run();
    std::printf("  x - ln x: status %d, x = %.10f, %d iterações\n", static_cast<int>(s), cg.x()[0], cg.iterations());
    CHECK(s == CgStatus::ConvergedGradient || s == CgStatus::ConvergedEnergy);
    CHECK(std::abs(cg.x()[0] - 1.0) < 1e-6);
}

CNX_TEST(reports_line_search_failure_and_other_statuses) {
    // Gradiente com o sinal errado: a direção "de descida" sobe a função de verdade.
    FnObjective bad([](std::span<const double> x, std::span<double> g) {
        g[0] = -2.0 * x[0];
        return x[0] * x[0];
    });
    CgOptions o;
    o.stride = 1;
    ConjugateGradient cg(bad, o);
    cg.reset(std::vector<double>{1.0});
    CHECK(cg.run() == CgStatus::LineSearchFailed);
    CHECK(cg.x()[0] == 1.0);  // nenhum passo aceito

    FnObjective f = rosenbrock();
    CgOptions few;
    few.stride = 1;
    few.maxIterations = 2;
    ConjugateGradient limited(f, few);
    limited.reset(std::vector<double>{-1.2, 1.0});
    CHECK(limited.run() == CgStatus::MaxIterations);
    CHECK(limited.iterations() == 2);

    // Já no mínimo: converge sem iterar.
    FnObjective q = quadratic(3);
    CgOptions tight;
    tight.stride = 1;
    ConjugateGradient atMin(q, tight);
    atMin.reset(std::vector<double>{1.0, 1.1 / 50.5, 1.2 / 100.0});
    CHECK(atMin.run() == CgStatus::ConvergedGradient);
    CHECK(atMin.iterations() == 0);

    // Dimensão zero.
    FnObjective empty([](std::span<const double>, std::span<double>) { return 0.0; });
    ConjugateGradient none(empty, CgOptions{});
    none.reset(std::vector<double>{});
    CHECK(none.run() == CgStatus::ConvergedGradient);
}

CNX_TEST(argument_validation) {
    FnObjective f = rosenbrock();
    CgOptions bad;
    bad.stride = 0;
    CHECK_THROWS(std::invalid_argument, ConjugateGradient(f, bad));
    CgOptions c2;
    c2.c1 = 0.5;
    c2.c2 = 0.4;
    CHECK_THROWS(std::invalid_argument, ConjugateGradient(f, c2));
    ConjugateGradient fresh(f, CgOptions{});
    CHECK_THROWS(std::logic_error, (void)fresh.iterate());
}

CNX_TEST(reset_can_change_the_dimension_and_is_deterministic) {
    FnObjective f = rosenbrock();
    CgOptions o;
    o.stride = 1;
    o.gradientTol = 1e-7;
    o.maxIterations = 500;
    ConjugateGradient a(f, o);
    ConjugateGradient b(f, o);
    a.reset(std::vector<double>{-1.2, 1.0});
    b.reset(std::vector<double>{-1.2, 1.0});
    (void)a.run();
    (void)b.run();
    CHECK(a.x()[0] == b.x()[0] && a.x()[1] == b.x()[1] && a.iterations() == b.iterations());
    // Objetivo isotrópico, válido em qualquer dimensão: reset() troca de 5 para 3 variáveis.
    FnObjective iso([](std::span<const double> x, std::span<double> g) {
        double v = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            v += (x[i] - 2.0) * (x[i] - 2.0);
            g[i] = 2.0 * (x[i] - 2.0);
        }
        return v;
    });
    ConjugateGradient c(iso, o);
    c.reset(std::vector<double>(5, 0.0));
    CHECK(c.run() == CgStatus::ConvergedGradient);
    CHECK(c.x().size() == 5);
    c.reset(std::vector<double>(3, 10.0));
    CHECK(c.iterations() == 0);
    CHECK(c.run() == CgStatus::ConvergedGradient);
    CHECK(c.x().size() == 3 && std::abs(c.x()[2] - 2.0) < 1e-6);
}

int main() { return cnxtest::runAll(); }
