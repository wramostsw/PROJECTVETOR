// cnx/palette/PaletteFinder.h — extração da paleta oficial da imagem (§3, docs §1.16).
//
//   1. peso do pixel = 1 / (1 + borda/edgeScale): pixels de transição valem menos, para o
//      antialiasing não criar cores falsas (EdgeImage = gradiente de Sobel nos 4 canais);
//   2. histograma esparso 16⁴ ponderado (média e espalhamento por célula);
//   3. para K = minColors..maxColors: sementes (picos + k-means++ com myPrng) e k-means sobre as células;
//   4. nota(K) = log(erro + 1) + (K − 2)·penalidade, com o erro quadrático médio por unidade de peso
//      (RGBA em [0, 1]); fica a menor nota (empate → menos cores);
//   5. imagem chapada (poucas cores exatas, cobrindo ≥ 99 % dos pixels): a paleta são as cores exatas;
//   6. refinamento de `refineRounds` rodadas de k-means sobre os PIXELS, só para o K escolhido.
//
// Não implementado: a segunda árvore de decisão de 62 medidas (2–9 cores ou "livre") da especificação,
// que não está definida; a escolha de K é só a menor nota (docs §1.16).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/segment/PixelMap.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

struct PaletteCandidate {
    int requested = 0;  ///< K pedido ao k-means
    int colors = 0;     ///< cores que sobraram (centros vazios são descartados)
    double error = 0.0; ///< erro quadrático médio por unidade de peso, nas células
    double score = 0.0;
};

struct PaletteResult {
    Palette palette;                  ///< por fração de peso decrescente; palette.score = nota da escolhida (0 se chapada)
    std::vector<double> fractions;    ///< fração do peso (ou dos pixels, se chapada) de cada cor; soma 1
    std::vector<PaletteCandidate> candidates;  ///< uma por K tentado (vazio se chapada ou K forçado sem busca)
    bool exact = false;               ///< imagem chapada: paleta = cores exatas, sem k-means
    int chosenK = 0;
    std::size_t distinctColors = 0;   ///< cores exatas distintas na imagem
    double error = 0.0;               ///< erro da paleta final medido nos pixels (0 se chapada)
};

class PaletteFinder {
public:
    /// `seed`: semente do myPrng (EngineParameters::seed). Lança std::invalid_argument se
    /// minColors < 1, maxColors < minColors, histogramBins ∉ [2, 256], penalidade < 0, edgeScale ≤ 0 ou
    /// forcedNumColors < 1.
    PaletteFinder(PaletteParams params, std::uint64_t seed);

    /// Magnitude da borda: ‖∇‖ de Sobel nos 4 canais (canais em [0, 1]), dividida por 4, de modo que um
    /// degrau de amplitude a em um canal dá a. Bordas replicadas.
    static sla::Array2D<float> edgeMagnitude(const sla::Array2D<sla::Rgba8>& image);
    /// Pesos 1 / (1 + borda/edgeScale).
    sla::Array2D<float> pixelWeights(const sla::Array2D<sla::Rgba8>& image) const;

    /// nota = log(erro + 1) + (cores − 2)·penalidade.
    static double score(double error, int colors, double penalty);

    PaletteResult find(const sla::Array2D<sla::Rgba8>& image) const;
    /// Com pesos próprios (≥ 0, mesma dimensão da imagem).
    PaletteResult find(const sla::Array2D<sla::Rgba8>& image, const sla::Array2D<float>& weights) const;

private:
    PaletteParams params_;
    std::uint64_t seed_;
};

}  // namespace cnx
