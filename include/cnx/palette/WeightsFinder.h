// cnx/palette/WeightsFinder.h — decomposição de um pixel em mistura linear das cores da paleta (§3).
//
// "Pixel de borda = mistura linear de tintas": dado o pixel p e cores c_1..c_n, acha pesos
//     w = argmin ‖Σ w_i c_i − p‖²   sujeito a   w_i ≥ 0,  Σ w_i = 1        (NNLS no simplex)
// Duas cores: projeção na reta com t ∈ [0, 1] (forma fechada). Três ou mais: método de conjuntos
// ativos (como Lawson–Hanson, com a restrição Σ w = 1 resolvida pelo sistema KKT de cada suporte).
//
// Por que não o Cholesky de CᵀC da especificação: com 4 canais (RGBA), CᵀC tem posto ≤ 4 e é singular
// para n ≥ 5. O sistema KKT [CᵀC 1; 1ᵀ 0] continua não singular enquanto as cores do suporte são
// afinamente independentes, e o ótimo sempre tem suporte de no máximo 5 cores (Carathéodory em ℝ⁴).
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "cnx/segment/PixelMap.h"
#include "sla/Vec.h"

namespace cnx {

struct PixelWeights {
    std::vector<double> w;   ///< um peso por cor da paleta (zero fora das candidatas); soma 1
    double residual = 0.0;   ///< ‖Σ w_i c_i − p‖²
    int support = 0;         ///< nº de pesos > 0
};

class WeightsFinder {
public:
    /// `maxCandidates` (3..9 na especificação): com mais cores que isso, só as mais próximas do pixel
    /// entram na decomposição. Lança std::invalid_argument se a paleta for vazia ou maxCandidates < 1.
    explicit WeightsFinder(Palette palette, int maxCandidates = 9);

    const Palette& palette() const noexcept { return palette_; }

    /// Decompõe o pixel usando a paleta inteira (ou as `maxCandidates` cores mais próximas).
    PixelWeights solve(const sla::Rgba& pixel) const;
    /// Decompõe usando só as cores de `subset` (índices distintos e válidos, senão std::invalid_argument).
    PixelWeights solveSubset(const sla::Rgba& pixel, std::span<const std::size_t> subset) const;
    /// Índices das n cores mais próximas do pixel (distância em RGBA; empate → menor índice).
    std::vector<std::size_t> nearest(const sla::Rgba& pixel, std::size_t n) const;
    /// Σ w_i c_i.
    sla::Rgba blend(std::span<const double> w) const;

    /// O problema em si: min ‖Σ w_i c_i − target‖² no simplex. Devolve n pesos. Exposto para testes.
    static std::vector<double> simplexLeastSquares(std::span<const std::array<double, 4>> colors, const std::array<double, 4>& target);

private:
    Palette palette_;
    int maxCandidates_;
};

}  // namespace cnx
