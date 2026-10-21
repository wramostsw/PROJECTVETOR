// cnx/palette/ColorHistogram.h — histograma de cores ponderado e esparso (§3, docs §1.16).
//
// O espaço RGBA é dividido em `bins` faixas por canal (16 → 65 536 células; 8 no modo rápido). Só as
// células ocupadas existem. Cada célula guarda o peso total, a MÉDIA ponderada das cores que caíram
// nela (não o centro da faixa) e o espalhamento interno Σ w‖c − média‖², de modo que o erro de um
// agrupamento calculado sobre as células é EXATO, igual ao calculado sobre os pixels.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "sla/Vec.h"

namespace cnx {

using Color4d = std::array<double, 4>;

struct HistogramCell {
    std::uint32_t key = 0;      ///< índice da célula: ((r·B + g)·B + b)·B + a
    double weight = 0.0;        ///< Σ w
    std::uint64_t pixels = 0;   ///< nº de pixels (independe do peso)
    Color4d mean{};             ///< média ponderada, canais em [0, 1]
    double scatter = 0.0;       ///< Σ w ‖c − média‖² (≥ 0)
};

class ColorHistogram {
public:
    /// `bins` ∈ [2, 256]. Lança std::invalid_argument fora disso.
    explicit ColorHistogram(int bins = 16);

    int bins() const noexcept { return bins_; }
    /// Pesos negativos ou não finitos lançam std::invalid_argument; peso 0 é aceito e conta o pixel.
    void add(sla::Rgba8 color, double weight = 1.0);

    std::uint64_t pixelCount() const noexcept { return pixels_; }
    double totalWeight() const noexcept { return totalWeight_; }
    std::size_t cellCount() const;

    /// Células ocupadas por peso decrescente (empate → chave crescente): ordem determinística.
    const std::vector<HistogramCell>& cells() const;

    /// Índice da célula de uma cor.
    std::uint32_t keyOf(sla::Rgba8 color) const noexcept;

private:
    struct Accum {
        double weight = 0.0;
        std::uint64_t pixels = 0;
        std::array<double, 4> sum{};
        double sumSq = 0.0;
    };

    int bins_;
    std::uint64_t pixels_ = 0;
    double totalWeight_ = 0.0;
    std::unordered_map<std::uint32_t, Accum> accum_;
    mutable std::vector<HistogramCell> cells_;
    mutable bool dirty_ = true;
};

}  // namespace cnx
