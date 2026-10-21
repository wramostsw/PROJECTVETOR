// cnx/classify/Classifier.h — escolha do perfil da imagem (§2, docs §1.17).
//
// PLACEHOLDER HONESTO: a especificação fala de uma árvore de decisão sobre 20 medidas e 10 perfis, mas
// não define a árvore nem as medidas. Aqui as medidas são poucas e legíveis, e a regra são três
// degraus; devem ser substituídas por uma árvore treinada no conjunto de referência (SegmentationTrainer).
#pragma once

#include "cnx/engine/CoreEngine.h"
#include "cnx/engine/Parameters.h"
#include "cnx/preprocess/Preprocessor.h"

namespace cnx {

struct ImageMeasures {
    int significantColors = 0;    ///< cores exatas com ≥ exactColorMinShare dos pixels
    double flatShare = 0.0;       ///< fração dos pixels nessas cores (o resto é antialiasing, ruído ou gradiente)
    double edgeFraction = 0.0;    ///< fração dos pixels com borda (Sobel) > 0,05
    bool lossyFormat = false;     ///< JPEG: ruído de compressão
    bool hasAlpha = false;
};

struct ImageClassification {
    ImageProfile profile = ImageProfile::Custom;
    ImageMeasures measures;
    bool automatic = true;  ///< false: o perfil veio de ClassifierParams::forcedProfile
};

/// Regra (provisória):
///   formato com perda, ou mais de maxColors cores exatas, ou flatShare < 0,80  → foto (PhotoMed);
///   ≤ maxColors cores exatas cobrindo ≥ 99 %                                   → logo sem AA (LogoMed);
///   ≤ maxColors cores exatas cobrindo 80–99 %                                  → logo com AA (LogoMedAA).
/// Com autoClassify = false devolve o perfil forçado, sem medir.
ImageClassification classify(const PreprocessedImage& pre, const SourceImage& source, const ClassifierParams& params,
                             const PaletteParams& palette);

}  // namespace cnx
