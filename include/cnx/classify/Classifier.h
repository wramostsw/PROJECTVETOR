// cnx/classify/Classifier.h — medidas da imagem e escolha do perfil (§2 classifyImageType, docs §1.19).
//
// A especificação descreve uma árvore de decisão TREINADA sobre 20 medidas (energia de sinal, perfil de
// nitidez das bordas, erro nas bordas × no interior, contagem de cores reais, segmentos preliminares, fração
// do histograma fora dos picos e "entre" dois picos, extensão do arquivo), com limiares aprendidos num
// conjunto de referência que não temos. Aqui as medidas são reais e calculadas de fato (20, nessas mesmas
// famílias), e a árvore é ESCRITA À MÃO, curta e legível, com limiares derivados do que cada medida vale em
// famílias sintéticas conhecidas (logo chapado, logo com AA, logo desfocado, logo com ruído, foto). Os
// limiares ficam em ClassificationRules, para que uma árvore treinada os substitua sem mexer nas medidas.
//
// O classificador escolhe a FAMÍLIA (foto, logo, logo com AA); o nível (HIG/MED/LOW) não é medido e fica
// MED — os outros níveis continuam disponíveis por ClassifierParams::forcedProfile.
#pragma once

#include <cstddef>

#include "cnx/engine/CoreEngine.h"
#include "cnx/engine/Parameters.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/preprocess/AADetector.h"
#include "cnx/preprocess/Preprocessor.h"

namespace cnx {

/// As 20 medidas. Frações em [0, 1]; cores RGBA em [0, 1].
struct ImageMeasures {
    // ---- cores (histograma exato e de 16 faixas por canal)
    int significantColors = 0;      ///< cores exatas com ≥ exactColorMinShare dos pixels
    double flatShare = 0.0;         ///< fração dos pixels nessas cores exatas
    double distinctColorRatio = 0.0;///< cores exatas distintas / pixels
    int mainPeaks = 0;              ///< picos do histograma 16⁴: células com ≥ peakMinShare dos pixels, a > peakRadius uma da outra
    double peakShare = 0.0;         ///< fração dos pixels a ≤ peakRadius de uma tinta (pico, ou cor exata significativa longe dos picos)
    double betweenShare = 0.0;      ///< fração dos pixels fora das tintas mas sobre o segmento entre dois picos (mistura: AA)
    // ---- bordas (EdgeImage)
    double edgeFraction = 0.0;      ///< pixels com magnitude > edgeMagnitude
    double strongEdgeFraction = 0.0;///< cristas fortes por pixel
    double meanEdgeMagnitude = 0.0;
    double interiorNoise = 0.0;     ///< RMS da magnitude nos pixels abaixo do limiar de crista forte ("energia de sinal")
    // ---- antialiasing (AADetector)
    std::size_t edgeProfiles = 0;   ///< perfis válidos (degraus + rampas)
    double stepShare = 0.0;         ///< degraus / válidos
    double rampShare = 0.0;         ///< rampas / válidos
    double otherShare = 0.0;        ///< outros / cristas (textura, terceira cor, ringing, perfil cortado)
    double meanTransitionWidth = 0.0;///< amostras intermediárias por perfil válido (px)
    // ---- erro de uma paleta de tintas (os picos principais + as cores exatas significativas)
    double quantizationRmse = 0.0;  ///< √(média de ‖pixel − tinta mais próxima‖²)
    double edgeErrorShare = 0.0;    ///< parte desse erro nos pixels de borda (0 se o erro é nulo)
    // ---- segmentos preliminares
    double quadTreeLeavesPerBlock = 0.0;  ///< folhas da QuadTree (blocos 16×16, variância 1e-3) por bloco inicial
    // ---- arquivo
    bool lossyFormat = false;       ///< extensão .jpg/.jpeg
    bool hasAlpha = false;
};

/// Constantes das medidas (não são parâmetros do usuário: não entram em impressão digital nem em .apf).
struct MeasureOptions {
    double peakMinShare = 0.01;        ///< célula do histograma que pode ser "pico"
    double peakRadius = 0.06;          ///< células a até essa distância de um pico pertencem a ele (≈ 15 níveis num canal)
    int maxPeaks = 64;                 ///< parada da busca de picos (bem acima de qualquer maxColors)
    int maxPalettePeaks = 12;          ///< picos usados como cores (erro e "entre picos")
    double betweenMaxDistance = 0.04;  ///< distância máxima ao segmento entre dois picos
    double betweenEndMargin = 0.1;     ///< parâmetro t do ponto mais próximo ∈ [margem, 1 − margem]
    double minPeakSeparation = 0.2;    ///< pares de picos mais próximos que isso não contam (são a mesma cor)
    float edgeMagnitude = 0.05f;       ///< pixel "de borda" (edgeFraction, edgeErrorShare)
    AAParams aa;                       ///< inclui o limiar de crista forte
};

/// Limiares da árvore (escritos à mão — a calibrar com o conjunto de referência; docs §1.19).
struct ClassificationRules {
    double minLogoCoverage = 0.90;       ///< peakShare + betweenShare: quase tudo é tinta ou mistura de duas tintas
    double minLogoCoverageLossy = 0.95;  ///< o mesmo para JPEG (o ruído de compressão pede mais evidência)
    double maxFlatRmse = 0.05;           ///< erro que dispensa explicação (ruído leve)
    double minEdgeErrorLift = 1.5;       ///< edgeErrorShare / edgeFraction: o erro se concentra nas bordas
    double minBetweenShare = 0.01;       ///< misturas entre tintas → AA
    std::size_t minAAProfiles = 16;      ///< perfis válidos para confiar em rampShare
    double minRampShare = 0.25;          ///< rampas entre os perfis válidos → AA (ou desfoque)
};

struct ImageClassification {
    ImageProfile profile = ImageProfile::Custom;
    ImageMeasures measures;
    bool automatic = true;  ///< false: o perfil veio de ClassifierParams::forcedProfile
};

/// Calcula as 20 medidas sobre `pre.image` (sem desfoque). Lança std::invalid_argument se a imagem for vazia
/// e CancelTaskException se `ctx` for cancelado.
ImageMeasures measureImage(const PreprocessedImage& pre, const SourceImage& source, const PaletteParams& palette, TaskContext& ctx,
                           const MeasureOptions& options = {});

/// A árvore:
///   1. mainPeaks > maxColors                                         → foto
///   2. peakShare + betweenShare < minLogoCoverage (JPEG: …Lossy)     → foto
///   3. quantizationRmse > maxFlatRmse e edgeErrorShare/edgeFraction < minEdgeErrorLift  → foto
///      (sobra erro, e ele não está nas bordas: textura ou gradiente)
///   4. betweenShare ≥ minBetweenShare, ou rampShare ≥ minRampShare com ≥ minAAProfiles perfis → logo com AA
///   5. senão                                                         → logo sem AA
ImageProfile decideProfile(const ImageMeasures& m, int maxColors, const ClassificationRules& rules = {});

/// Mede e decide. Com autoClassify = false devolve o perfil forçado, sem medir.
ImageClassification classify(const PreprocessedImage& pre, const SourceImage& source, const ClassifierParams& params,
                             const PaletteParams& palette, TaskContext& ctx);
ImageClassification classify(const PreprocessedImage& pre, const SourceImage& source, const ClassifierParams& params,
                             const PaletteParams& palette);

}  // namespace cnx
