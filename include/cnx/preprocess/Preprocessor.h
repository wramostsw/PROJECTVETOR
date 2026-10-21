// cnx/preprocess/Preprocessor.h — achatamento do alfa, reamostragem para a área-alvo e desfoque (§1.3, §2).
//
//   1. alfa constante (qualquer valor) → vira opaco: o canal não carrega informação (ensureFlattened);
//   2. área > targetAreaPx → reduz por média de área EXATA (cada pixel de saída é a média ponderada pela
//      cobertura dos pixels de origem), s = √(A_alvo / (W·H)). Imagens menores só são ampliadas (bilinear)
//      com allowUpscale: ampliar um logo chapado o borra;
//   3. blurSigmaPx > 0 → cópia desfocada (gaussiana separável, bordas replicadas) usada pela paleta e pela
//      segmentação. A imagem NÃO desfocada continua sendo o alvo do modelo gerador (docs §0).
#pragma once

#include "cnx/engine/CoreEngine.h"
#include "cnx/engine/Parameters.h"
#include "cnx/engine/TaskContext.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

struct PreprocessedImage {
    sla::Array2D<sla::Rgba8> image;   ///< achatada e reamostrada, sem desfoque: alvo do GenerativeModel
    sla::Array2D<sla::Rgba8> smooth;  ///< = image se blurSigmaPx == 0; senão a cópia desfocada (paleta, segmentação)
    double scale = 1.0;               ///< fator linear aplicado (saída / entrada)
    bool alphaFlattened = false;
};

/// Lança std::invalid_argument se a imagem for vazia, targetAreaPx ≤ 0 ou blurSigmaPx < 0, e
/// CancelTaskException se `ctx` for cancelado.
PreprocessedImage preprocess(const SourceImage& source, const PreprocessParams& params, TaskContext& ctx);

/// Média de área exata para o novo tamanho (w × h); exposta para testes.
sla::Array2D<sla::Rgba8> resampleArea(const sla::Array2D<sla::Rgba8>& in, int w, int h);

/// Gaussiana separável de desvio sigma (px), bordas replicadas.
sla::Array2D<sla::Rgba8> gaussianBlur(const sla::Array2D<sla::Rgba8>& in, double sigma);

}  // namespace cnx
