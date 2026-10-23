// cnx/fit/BezierFitter.h — ajuste de Bézier cúbica por mínimos quadrados + Newton/Schur (§2.5, docs §1.14).
//
// Cada aresta do VectorImage suavizado é cortada nos nós fixos (cantos perfurados e junções) e
// cada trecho vira uma sequência de cúbicas:
//   1. fragmentos iniciais de `initialFragmentNodes` nós;
//   2. por fragmento, P₀ e P₃ ficam nos nós das pontas e (P₁, P₂) saem de mínimos quadrados com
//      t por comprimento de corda, refinados por Newton exato em (p, t) com complemento de Schur
//      e amortecimento de Levenberg–Marquardt (custo O(m) por iteração);
//   3. fusão iterativa de vizinhos: funde se ΔE = E*(A∪B) − E*(A) − E*(B) < τ_r, com τ_r crescendo
//      de statThreshInitial a statThreshFinal (o teto, atingido exatamente na última rodada; as duas últimas
//      rodadas repetem a varredura até o ponto fixo) em maxRounds rodadas; quando recusa, desloca a quebra
//      em {−1, 0, +1} nó e fica com o menor E*(A) + E*(B);
//   4. nas quebras lisas a tangente é compartilhada (G¹): P₁ = P₀ + α·τ₀, P₂ = P₃ − β·τ₃.
// Fragmentos de ≤ 3 pontos viram retas (SegmentKind::Line).
//
// A energia de um fragmento é E = ½ Σ_{nós interiores} ‖B(t_k) − Q_k‖² com t_k ótimo.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/vector/VectorImage.h"
#include "sla/BezierCurve.h"
#include "sla/Vec.h"

namespace cnx {

/// Restrições de tangente nas pontas de um fragmento (vetores unitários no sentido do percurso).
/// Presente: G¹ — P₁ = P₀ + α·start e P₂ = P₃ − β·end. Ausente: ponta livre (canto ou junção).
struct FragmentEnds {
    std::optional<sla::Vec2> startTangent;
    std::optional<sla::Vec2> endTangent;
};

/// Valor, gradiente e Hessiana em seta de E(p, t), com k = paramCount() e m = interiorCount().
struct FitDerivatives {
    double energy = 0.0;
    std::vector<double> gp;   ///< k
    std::vector<double> gt;   ///< m
    std::vector<double> hpp;  ///< k × k, linha a linha
    std::vector<double> hpt;  ///< m × k: linha i = h_i = ∂²E/∂p∂t_i
    std::vector<double> d;    ///< m: ∂²E/∂t_i²
};

struct SchurStep {
    bool ok = false;
    std::vector<double> dp;
    std::vector<double> dt;
};

/// Modelo de um fragmento de m ≥ 4 pontos (≥ 2 interiores): B(t; p) = B_fixo(t) + J(t)·p, linear em p.
/// p = [P₁ (2) | α (1)] seguido de [P₂ (2) | β (1)], conforme a ponta seja livre ou G¹.
class FragmentModel {
public:
    /// Lança std::invalid_argument se houver menos de 4 pontos.
    FragmentModel(std::span<const sla::Vec2> points, const FragmentEnds& ends);

    std::size_t paramCount() const noexcept { return k_; }
    std::size_t interiorCount() const noexcept { return points_.size() - 2; }
    bool startIsG1() const noexcept { return g1Start_; }
    bool endIsG1() const noexcept { return g1End_; }

    sla::BezierCurve curve(std::span<const double> p) const;
    /// t: parâmetros dos nós interiores (m valores).
    double energy(std::span<const double> p, std::span<const double> t) const;
    FitDerivatives derivatives(std::span<const double> p, std::span<const double> t) const;
    /// Mínimos quadrados em p com t fixo (equações normais k×k por Cholesky). Vazio se singular.
    std::vector<double> leastSquares(std::span<const double> t) const;
    /// Passo de Newton com complemento de Schur: S δp = −g_p + Σ h g_t/(D+λ), δt = −(g_t + hᵀδp)/(D+λ).
    static SchurStep schurStep(const FitDerivatives& d, double lambda);

private:
    struct Parts {
        sla::Vec2 fixed;
        std::array<sla::Vec2, 4> cols{};
    };
    Parts parts(double t, int derivative) const;

    std::vector<sla::Vec2> points_;
    sla::Vec2 p0_, p3_, tau0_, tau3_;
    bool g1Start_ = false;
    bool g1End_ = false;
    std::size_t k_ = 0;
};

struct FragmentFit {
    sla::BezierCurve curve;
    double energy = 0.0;
    std::vector<double> t;  ///< parâmetro de cada ponto (m valores; t[0] = 0 e t.back() = 1)
    bool line = false;
    /// G¹ com α ≤ 0 ou β ≤ 0 (laço ou cúspide): trocado pela heurística de Schneider α = β = ‖P₃ − P₀‖/3.
    bool degenerate = false;
    int newtonIterations = 0;
};

/// Ajusta um fragmento. `initialT` (m valores) é o ponto de partida opcional dos t (warm start);
/// por padrão, comprimento de corda. Lança std::invalid_argument se houver menos de 2 pontos.
FragmentFit fitFragment(std::span<const sla::Vec2> points, const FragmentEnds& ends, const FitParams& params,
                        std::span<const double> initialT = {});

struct FitStats {
    int initialFragments = 0;
    int finalFragments = 0;
    int merges = 0;
    int shifts = 0;     ///< deslocamentos de quebra aceitos
    int lines = 0;      ///< fragmentos finais que são retas
    int degenerate = 0; ///< fragmentos finais marcados como degenerados (G¹ com α ≤ 0 ou β ≤ 0)
    double energy = 0.0;  ///< Σ E* dos fragmentos finais

    FitStats& operator+=(const FitStats& o) noexcept;
};

struct ChainFit {
    std::vector<sla::CurveSegment> curves;
    FitStats stats;
};

class BezierFitter {
public:
    /// Lança std::invalid_argument se os parâmetros forem inválidos.
    explicit BezierFitter(FitParams params);

    /// τ_r = τ₀·(τ_f/τ₀)^{r/(R−1)}, r = 0..R−1 (R = 1: só τ_f).
    static std::vector<double> thresholdSchedule(const FitParams& params);

    /// Uma cúbica é reta quando P1 e P2 ficam a no máximo 0,02 px da corda P0–P3, dentro dela e em ordem (P0, P1, P2, P3).
    /// `run` escreve esses trechos como `Line`. Corda de comprimento ~0 não é reta.
    static bool isStraight(const sla::BezierCurve& c);

    /// Ajusta uma poligonal. `corner[i]` marca quebra obrigatória com tangente livre (canto ou junção).
    /// Aberta: os extremos são sempre livres. Fechada (pts.front() == pts.back(), ≥ 4 pontos):
    /// corner[0] diz se o nó âncora é canto; se não for, a tangente é compartilhada por ele.
    ChainFit fitChain(std::span<const sla::Vec2> points, std::span<const std::uint8_t> corner, bool closed) const;

    /// Ajusta todas as arestas do VectorImage (substitui as curvas existentes). Nós com as flags
    /// Corner, Junction ou Pinned são quebras de tangente livre.
    FitStats run(VectorImage& image, TaskContext ctx = {}) const;

private:
    FitParams params_;
};

}  // namespace cnx
