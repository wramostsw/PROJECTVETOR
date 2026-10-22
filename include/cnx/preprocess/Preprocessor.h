// cnx/preprocess/Preprocessor.h — achatamento do alfa, reamostragem para a área-alvo e desfoque (§1.3, §2).
//
//   1. alfa constante (qualquer valor) → vira opaco: o canal não carrega informação (ensureFlattened);
//   2. área > targetAreaPx → reduz por média de área EXATA (cada pixel de saída é a média ponderada pela
//      cobertura dos pixels de origem), s = √(A_alvo / (W·H)). Imagens menores só são ampliadas (bilinear)
//      com allowUpscale: ampliar um logo chapado o borra;
//   3. blurSigmaPx > 0 → cópia desfocada (gaussiana separável, bordas replicadas) usada pela paleta e pela
//      segmentação. O sinal NÃO desfocado continua sendo o alvo do modelo gerador (docs §0).
// A reamostragem e o desfoque são feitos no sinal pré-multiplicado (SignalImage, docs §1.22): o RGB escondido
// sob alfa 0 não vaza para as bordas, e todo pixel invisível vira (0, 0, 0, 0) nas imagens de 8 bits.
#pragma once

#include "cnx/engine/CoreEngine.h"
#include "cnx/engine/Parameters.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/preprocess/SignalImage.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

struct PreprocessedImage {
    SignalImage signal;               ///< achatado e reamostrado, sem desfoque, pré-multiplicado em float: alvo do GenerativeModel
    sla::Array2D<sla::Rgba8> image;   ///< = signal.toStraight8(): o classificador e quem mais lê 8 bits sem desfoque
    sla::Array2D<sla::Rgba8> smooth;  ///< = image se blurSigmaPx == 0; senão o sinal desfocado em 8 bits (paleta, segmentação)
    double scale = 1.0;               ///< fator linear aplicado (saída / entrada)
    bool alphaFlattened = false;
};

/// Lança std::invalid_argument se a imagem for vazia, targetAreaPx ≤ 0 ou blurSigmaPx < 0 ou não finito, e
/// CancelTaskException se `ctx` for cancelado.
PreprocessedImage preprocess(const SourceImage& source, const PreprocessParams& params, TaskContext& ctx);

/// Média de área exata para o novo tamanho (w × h), no sinal pré-multiplicado; exposta para testes.
sla::Array2D<sla::Rgba8> resampleArea(const sla::Array2D<sla::Rgba8>& in, int w, int h);

/// Gaussiana separável de desvio sigma (px), bordas replicadas, no sinal pré-multiplicado. sigma ≤ 0 → cópia.
sla::Array2D<sla::Rgba8> gaussianBlur(const sla::Array2D<sla::Rgba8>& in, double sigma);

}  // namespace cnx
