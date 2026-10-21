// cnx/palette/KMeans.h — k-means ponderado sobre o coreset do histograma (§3, docs §1.16).
//
// Cada item é uma célula (posição = média, peso, espalhamento interno). O erro devolvido é EXATO para
// os pixels: Σ_itens [w·‖média − centro‖² + espalhamento] / Σ w. Distância em RGBA, canais em [0, 1].
// Sementes: picos do histograma com supressão de vizinhos; as que faltarem saem de k-means++
// (amostragem D² com myPrng). Lloyd com no máximo `maxRounds` rodadas; centros vazios são descartados.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "cnx/palette/ColorHistogram.h"
#include "sla/Prng.h"

namespace cnx {

struct KMeansItem {
    Color4d pos{};
    double weight = 0.0;
    double scatter = 0.0;  ///< Σ w‖c − pos‖² dentro do item
};

struct KMeansResult {
    std::vector<Color4d> centers;       ///< só os centros não vazios, por peso decrescente
    std::vector<double> clusterWeight;  ///< peso de cada centro (mesma ordem)
    std::vector<int> assignment;        ///< centro de cada item (índices de `centers`)
    double error = 0.0;                 ///< erro quadrático médio por unidade de peso
    int rounds = 0;
    bool converged = false;
    int discarded = 0;                  ///< centros que ficaram vazios
};

inline double distance2(const Color4d& a, const Color4d& b) noexcept {
    const double d0 = a[0] - b[0], d1 = a[1] - b[1], d2 = a[2] - b[2], d3 = a[3] - b[3];
    return d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
}

/// Sementes para k centros. `items` deve vir ordenado por peso decrescente (como as células do
/// histograma). Primeiro os picos, pulando os que ficam a < `minSeparation` de uma semente já
/// escolhida; o restante por k-means++ (probabilidade ∝ peso · D²). Se houver menos de k itens
/// distintos, devolve menos sementes. Determinístico dado `rng`.
std::vector<Color4d> chooseSeeds(std::span<const KMeansItem> items, std::size_t k, double minSeparation, sla::myPrng& rng);

/// Lloyd ponderado a partir de `seeds`. Lança std::invalid_argument se não houver itens ou sementes,
/// ou se o peso total for nulo.
KMeansResult kmeans(std::span<const KMeansItem> items, std::vector<Color4d> seeds, int maxRounds);

}  // namespace cnx
