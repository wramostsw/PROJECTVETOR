// cnx/segment/QuadTree.h — pré-corte em blocos (§4, passo 1 do PixelSegmenter).
#pragma once

#include <vector>

#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

/// Retângulo de pixels [x, x+w) × [y, y+h).
struct QuadBlock {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;

    friend constexpr bool operator==(const QuadBlock&, const QuadBlock&) noexcept = default;
};

/// Divide a imagem em blocos `maxBlock × maxBlock` (os da borda direita/inferior são truncados).
/// Um bloco cuja variância de cor (Σ dos 4 canais da variância por canal, canais em [0, 1]) é
/// ≤ `variance` — ou que tem 1 pixel — vira uma folha; os demais são divididos ao meio em cada
/// eixo (metades ⌊n/2⌋ e n − ⌊n/2⌋; um eixo de tamanho 1 não é dividido) e examinados de novo.
///
/// Retorna as folhas, que particionam a imagem exatamente, em ordem determinística: blocos
/// iniciais em varredura por linhas, e dentro de cada um a ordem Z (NO, NE, SO, SE).
/// Custo: O(N · profundidade) = O(N log₂ maxBlock); sem imagens integrais, memória O(folhas).
/// Lança std::invalid_argument se maxBlock < 1 ou a imagem for vazia.
std::vector<QuadBlock> quadTreeSplit(const sla::Array2D<sla::Rgba8>& image, int maxBlock, double variance);

/// Variância de cor do bloco, como definida acima (exposta para testes).
double blockColorVariance(const sla::Array2D<sla::Rgba8>& image, const QuadBlock& block);

}  // namespace cnx
