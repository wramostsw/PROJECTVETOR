// cnx/segment/SubPixelSegmenter.h — especialista em antialiasing (§4, docs §1.15).
//
// Roda depois do PixelSegmenter, sobre o mesmo PixelMap e a mesma imagem. Em ordem:
//   1. micro-regiões: limpa regiões de 1 px, depois de 2 px, … (cada uma vai para a vizinha de cor
//      mais próxima; a ordem crescente de tamanho importa);
//   2. praias (splitAndMergeBeaches): faixas finas de pixels de transição entre duas cores dominantes
//      são repartidas entre as duas, pixel a pixel, para a fronteira cair no meio da transição;
//   3. sombras (flagShadowSegments + rebuild): regiões de gradiente suave (sombra ou desfoque) são
//      marcadas e reconstruídas em faixas pela cor da imagem DESFOCADA mais próxima da paleta;
//   4. paleta: cada região recebe a cor da paleta mais próxima de sua média (snapColorsToPalette),
//      vizinhas de mesma cor da paleta são fundidas, enforceMinNumPixels varre os fragmentos
//      residuais e compact() (buildRegions) renumera.
//
// Invariantes: toda região continua 4-conexa e com ≥ 1 pixel; contatos exatos; determinístico.
// A imagem passada deve ser a usada para montar o PixelMap (as cores dos pixels alimentam as medianas).
#pragma once

#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/segment/PixelMap.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

struct SubPixelStats {
    int microMerges = 0;       ///< regiões absorvidas pela limpeza progressiva
    int beaches = 0;           ///< praias encontradas
    int beachPixelsMoved = 0;  ///< pixels de praia entregues a uma das duas dominantes
    int beachesMerged = 0;     ///< praias cujo resto foi fundido inteiro a uma dominante
    int shadowFlagged = 0;     ///< regiões marcadas como sombra/desfoque
    int shadowNewRegions = 0;  ///< faixas novas criadas na reconstrução
    int splitRegions = 0;      ///< regiões novas criadas por splitDisconnected (esperado: 0)
    int paletteMerges = 0;     ///< fusões de vizinhas com a mesma cor da paleta
    int smallMerges = 0;       ///< fusões por enforceMinNumPixels
    int finalRegions = 0;
};

struct SubPixelResult {
    PixelMap map;  ///< compactado; cada região tem cor e índice de paleta
    SubPixelStats stats;
};

class SubPixelSegmenter {
public:
    struct BeachResult {
        int beaches = 0;
        int moved = 0;
        int merged = 0;
    };

    /// `image` deve sobreviver ao objeto. Lança std::invalid_argument se a paleta for vazia.
    SubPixelSegmenter(const sla::Array2D<sla::Rgba8>& image, SegmentParams params, Palette palette);

    /// Pipeline completo. Lança CancelTaskException se `ctx` for cancelado e std::invalid_argument se
    /// o mapa não tiver as dimensões da imagem ou tiver pixel sem região.
    SubPixelResult run(PixelMap map, TaskContext ctx = {}) const;

    // ---- estágios individuais (também usados pelos testes); todos modificam `map` ----

    /// Limpeza progressiva de micro-regiões. Retorna o nº de fusões.
    int mergeMicroRegions(PixelMap& map) const;
    /// Divide ou funde as praias entre suas duas cores dominantes.
    BeachResult splitAndMergeBeaches(PixelMap& map) const;
    /// Marca com RegionFlag::Shadow as regiões de gradiente suave. Retorna quantas.
    int flagShadowSegments(PixelMap& map) const;
    /// Reconstrói as regiões marcadas em faixas pela cor da paleta mais próxima da imagem desfocada.
    /// Retorna o nº de regiões novas.
    int rebuildShadowSegments(PixelMap& map) const;
    /// Funde vizinhas 4-adjacentes com o mesmo paletteIndex (≥ 0). Retorna o nº de fusões.
    int mergeSamePaletteNeighbors(PixelMap& map) const;
    /// enforceMinNumPixels: funde regiões com menos de SegmentParams::minRegionPixels pixels à vizinha
    /// de cor (RegionStats::color) mais próxima, que prevalece. Retorna o nº de fusões.
    int enforceMinNumPixels(PixelMap& map) const;

    /// Divide cada região em suas componentes 4-conexas: a maior mantém o id, as demais viram regiões
    /// novas. Retorna quantas regiões novas. Pré-condição: nenhum pixel sem região.
    static int splitDisconnected(PixelMap& map, const sla::Array2D<sla::Rgba8>& image);

private:
    const sla::Array2D<sla::Rgba>& blurred() const;
    std::size_t nearestPalette(const sla::Rgba& c) const;

    const sla::Array2D<sla::Rgba8>& image_;
    SegmentParams params_;
    Palette palette_;
    mutable sla::Array2D<sla::Rgba> blurred_;  ///< cache preguiçoso do desfoque de caixa
};

}  // namespace cnx
