// cnx/vector/BoundaryTracer.h — traçado inicial das fronteiras na grade de quinas dos pixels (§5).
//
// Entrada: um rótulo por pixel (0..R−1) e a cor de cada rótulo. Saída: VectorImage em que
//   * cada face é uma componente 4-conexa de um rótulo (logo, um único laço externo por face);
//   * os nós ficam nas quinas dos pixels: junções (quinas de grau ≥ 3, inclusive selas 2×2 em
//     xadrez), cantos da moldura (Pinned | Corner) e as dobras das escadinhas; quinas colineares
//     são omitidas;
//   * cada aresta é uma cadeia de fendas entre dois vértices. Um laço sem junção recebe como
//     vértice sintético (LoopAnchor) sua quina mais alta à esquerda.
// Tempo e memória O(W·H). Determinístico: ids seguem a ordem de varredura.
//
// Selas: numa quina 2×2 em xadrez as quatro fendas se encontram num único vértice de grau 4.
// Se uma face toca a si mesma ali (diagonal), seu contorno é um único passeio "beliscado" nesse
// vértice. Ver docs/ARQUITETURA.md §1.8.
#pragma once

#include <cstdint>
#include <span>
#include <stop_token>
#include <vector>

#include "cnx/segment/PixelMap.h"
#include "cnx/vector/VectorImage.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

struct TraceResult {
    VectorImage image;
    sla::Array2D<FaceId> pixelFaces;   ///< face de cada pixel (≥ 1)
    std::vector<RegionId> faceRegion;  ///< rótulo de origem de cada face (kNoRegion no exterior)
};

class BoundaryTracer {
public:
    struct Options {
        bool removeCollinearNodes = true;  ///< false: um nó em cada quina de fronteira (depuração/pesquisa)
    };

    BoundaryTracer() = default;
    explicit BoundaryTracer(Options options) : options_(options) {}

    /// labels: W×H com W, H ≥ 1 e valores em [0, regionColors.size()).
    /// regionPaletteIndex: vazio ou com o tamanho de regionColors.
    /// Lança std::invalid_argument em entrada inválida e CancelTaskException se `stop` for acionado.
    TraceResult trace(const sla::Array2D<RegionId>& labels, std::span<const sla::Rgba> regionColors,
                      std::span<const std::int32_t> regionPaletteIndex = {}, const std::stop_token& stop = {}) const;

    /// Traça a segmentação final. Pré-condição: map.isCompact() (senão std::logic_error).
    /// A cor e o índice de paleta de cada face vêm de RegionStats::color e ::paletteIndex da
    /// região; TraceResult::faceRegion usa os ids compactos do PixelMap.
    TraceResult trace(const PixelMap& map, const std::stop_token& stop = {}) const;

private:
    Options options_{};
};

}  // namespace cnx
