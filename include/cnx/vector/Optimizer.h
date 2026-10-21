// cnx/vector/Optimizer.h — gradiente conjugado não linear Polak–Ribière+ com busca em linha de
// Wolfe forte e região de confiança por nó (docs/ARQUITETURA.md §2.3, §1.13).
//
//   d₀ = −g₀;   x_{k+1} = x_k + α_k d_k;   β_{k+1} = max(0, g_{k+1}·(g_{k+1} − g_k) / ‖g_k‖²)
//   d_{k+1} = −g_{k+1} + β_{k+1} d_k    (reinício em d = −g se d não for de descida ou se
//                                          |g_{k+1}·g_k| ≥ 0,2‖g_{k+1}‖², critério de Powell)
//   Wolfe forte:  f(x+αd) ≤ f(x) + c₁ α g·d   e   |g(x+αd)·d| ≤ c₂ |g·d|     (c₁ = 1e-4, c₂ = 0,1)
//   Região de confiança: max_nó ‖α d_nó‖ ≤ maxStep (o fiscal anti-cruzamento precisa de passos curtos).
//   No teto da região de confiança basta a condição de Armijo.
//
// A busca é a de Nocedal & Wright (alg. 3.5 e 3.6), com interpolação cúbica protegida.
// Determinístico: sem aleatoriedade.
#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace cnx {

/// Função objetivo. `grad` tem o mesmo tamanho de `x`.
class ObjectiveFunction {
public:
    virtual ~ObjectiveFunction() = default;
    virtual double evaluate(std::span<const double> x, std::span<double> grad) = 0;
};

struct CgOptions {
    int maxIterations = 200;
    double gradientTol = 1e-6;   ///< ‖g‖∞ ≤ tol → convergiu
    double relEnergyTol = 1e-10; ///< f_k − f_{k+1} ≤ tol·max(1, |f_{k+1}|) → convergiu
    double c1 = 1e-4;
    double c2 = 0.1;
    /// Deslocamento máximo por nó em uma iteração; os nós são grupos de `stride` componentes.
    double maxStep = std::numeric_limits<double>::infinity();
    int stride = 2;
    int maxLineSearchEvals = 25;
};

enum class CgStatus {
    Running,
    ConvergedGradient,
    ConvergedEnergy,
    LineSearchFailed,
    MaxIterations,
};

class ConjugateGradient {
public:
    /// `objective` deve sobreviver ao otimizador.
    ConjugateGradient(ObjectiveFunction& objective, CgOptions options);

    /// Recomeça em x0 (a dimensão pode mudar): avalia f e g, d = −g. Zera o contador de iterações.
    void reset(std::span<const double> x0);
    /// Uma iteração: busca em linha e atualização de d. Pré-condição: reset() chamado.
    CgStatus iterate();
    /// iterate() até deixar de ser Running.
    CgStatus run();

    std::span<const double> x() const noexcept { return x_; }
    std::span<const double> gradient() const noexcept { return g_; }
    double value() const noexcept { return f_; }
    int iterations() const noexcept { return iterations_; }
    int evaluations() const noexcept { return evaluations_; }
    double lastStep() const noexcept { return lastAlpha_; }
    /// β da última atualização de direção (0 após um reinício).
    double lastBeta() const noexcept { return lastBeta_; }
    /// φ'(α)/φ'(0) no passo aceito — |·| ≤ c₂ quando a condição de curvatura vale.
    double lastCurvatureRatio() const noexcept { return lastCurvatureRatio_; }

private:
    struct Point {
        double alpha = 0.0;
        double f = 0.0;
        double dphi = 0.0;
        std::vector<double> x;
        std::vector<double> g;
    };

    void evaluateAt(double alpha, Point& out);
    bool lineSearch(double alphaInit, double alphaMax, Point& accepted);
    bool zoom(Point lo, Point hi, Point& accepted);
    double maxNodeNorm(std::span<const double> v) const noexcept;

    ObjectiveFunction* objective_;
    CgOptions options_;
    std::vector<double> x_, g_, d_;
    double f_ = 0.0;
    double dphi0_ = 0.0;
    double f0_ = 0.0;
    int iterations_ = 0;
    int evaluations_ = 0;
    int lineSearchEvals_ = 0;
    double alphaPrev_ = 0.0;
    double lastAlpha_ = 0.0;
    double lastBeta_ = 0.0;
    double lastCurvatureRatio_ = 0.0;
    bool started_ = false;
};

}  // namespace cnx
