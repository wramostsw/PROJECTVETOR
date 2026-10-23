// cnx/segment/PixelSegmenter.h — segmentação por QuadTree + juntar/trocar + passada diagonal (§4).
//
// Fluxo (docs/ARQUITETURA.md §1.11):
//   1. QuadTree: blocos 16×16 homogêneos viram uma região; os demais são subdivididos.
//   2. Para cada limiar t da escala multiplicativa t₀ · g^k (até o teto):
//        juntar — fundir regiões vizinhas cujas cores medianas estão a dist² ≤ t e que passam na
//                 simulação "e se" (aborto antecipado);
//        trocar — mover pixels de fronteira para a vizinha de cor mediana mais próxima, só em
//                 pontos simples (PixelMap::tryMovePixel), então nenhuma região se parte.
//   3. Repetição no teto até estabilizar; ligação de traços finos diagonais ao longo do FlowField
//      (linkStrokes, docs §1.20); fusão das regiões menores que minRegionPixels;
//      passada diagonal (uma região que toca a si mesma só pela quina de uma janela 2×2 — "sela" —
//      é desfeita movendo um pixel da janela); renumeração.
//
// Invariantes mantidos em todos os estágios: toda região é 4-conexa; toda região tem ≥ 1 pixel;
// contatos exatos (PixelMap). Determinístico: sem aleatoriedade, ordem de varredura fixa.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/preprocess/FlowField.h"
#include "cnx/segment/PixelMap.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

struct SegmentationStats {
    int superpixels = 0;      ///< folhas da QuadTree (ou regiões do mapa inicial)
    int rounds = 0;           ///< limiares percorridos na escala multiplicativa
    int joins = 0;            ///< fusões aceitas
    int swaps = 0;            ///< pixels trocados
    int diagonalMoves = 0;    ///< pixels movidos para desfazer selas
    int strokeLinks = 0;      ///< pares diagonais de traço fino ligados
    int smallMerges = 0;      ///< fusões por minRegionPixels
    int finalRegions = 0;
    bool stable = true;       ///< falso se finalSweepCap acabou antes de juntar/trocar/diagonal estabilizarem
};

struct SegmentationResult {
    PixelMap map;  ///< compactado, com cores (mediana ou paleta) atribuídas
    SegmentationStats stats;
};

/// Atribui a cada raiz com pixels a cor da paleta mais próxima de sua média (uma rodada de
/// k-means, só o passo de atribuição, §4). Empate → menor índice. Se a distância² da média a essa cor passa de
/// `maxDistance2`, a raiz não é encaixada: mantém a cor que tinha e paletteIndex = -1 (faixas de degradê, §1.28; o
/// padrão 4,0 é o máximo possível em RGBA, ou seja, sempre encaixa). Lança std::invalid_argument se a paleta for
/// vazia ou se `maxDistance2` for negativo ou NaN. Pode ser chamada antes ou depois de compact().
void snapColorsToPalette(PixelMap& map, const Palette& palette, double maxDistance2 = 4.0);

/// Modo paleta exata (docs §1.21): cada raiz recebe a cor da paleta que a maioria dos seus pixels tem (cada
/// pixel vota na cor da paleta mais próxima da sua, a mesma regra que montou o mapa inicial). A média não
/// serve aqui: um fragmento que junta preto e branco tem média cinza, e a cor da paleta mais próxima do cinza
/// pode ser uma terceira cor que nenhum pixel dele tem. Pixels fora das estatísticas de cor (pontes de traço)
/// não votam, a menos que a região não tenha outros. Empate → a empatada mais próxima da média, depois o menor
/// índice. A cor escolhida é sempre a de pelo menos um pixel da região. Lança std::invalid_argument se a
/// paleta for vazia ou `image` não tiver as dimensões do mapa.
void snapColorsToExactPalette(PixelMap& map, const sla::Array2D<sla::Rgba8>& image, const Palette& palette);

class PixelSegmenter {
public:
    /// Mapa inicial pela QuadTree (chame initializeFromQuadTree()). `image` deve sobreviver ao objeto.
    PixelSegmenter(const sla::Array2D<sla::Rgba8>& image, SegmentParams params);
    /// Mapa inicial fornecido (testes, modo paleta exata). Todos os pixels devem ter região.
    PixelSegmenter(const sla::Array2D<sla::Rgba8>& image, SegmentParams params, PixelMap initial);

    /// Limiares t₀, t₀g, t₀g², … < teto, mais o teto. Lança std::invalid_argument se
    /// mergeThreshStart ≤ 0, growth ≤ 1 ou teto < início.
    static std::vector<double> thresholdSchedule(const SegmentParams& params);

    /// Pipeline completo. Lança CancelTaskException se `ctx` for cancelado.
    SegmentationResult run(TaskContext ctx = {}, const Palette* palette = nullptr);

    // ---- estágios individuais (também usados pelos testes) ----

    /// Passo 1. Pré-condição: o mapa ainda está vazio (construtor sem mapa inicial).
    void initializeFromQuadTree();
    /// Uma varredura de "juntar" no limiar t. Retorna o número de fusões.
    int joinSweep(double threshold);
    /// Uma varredura de "trocar" com as cores das regiões congeladas no início da varredura
    /// (a soma dos erros pixel↔cor decresce estritamente a cada troca). Retorna o nº de trocas.
    int swapSweep();
    /// Uma varredura da passada diagonal. Retorna o nº de pixels movidos.
    int diagonalSweep();
    /// Ligação de traços finos (docs §1.20). Para cada par de pixels diagonais p, q = p + (±1, 1) de regiões
    /// diferentes, cujas duas "pontes" (os 4-vizinhos comuns) são de outras regiões, liga p a q se:
    ///   cores    dist²(p, q) ≤ mergeThreshMax e dist² de p e de q a cada ponte > mergeThreshMax;
    ///   não é AA o traço não é uma mistura das duas pontes (o mesmo teste de reta das praias:
    ///            beachMinContrast, beachLineTolerance) — uma escadinha de AA entre duas cores não é traço;
    ///   campo    no campo do par (a soma dos tensores de p e de q): coerência ≥ strokeMinCoherence e
    ///            alinhamento com a diagonal ≥ cos(2·strokeMaxAngle).
    /// A ligação move uma ponte para a região de p (só se for ponto simples da região dela) e une p e q. Ponte
    /// preferida: a que não encosta em outra região além da própria (não cria contatos novos), depois a de cor
    /// mais próxima do traço, depois a horizontal. Retorna o nº de ligações; 0 se linkStrokes = false.
    int linkStrokeSweep();
    /// enforceMinNumPixels. Retorna o nº de fusões.
    int mergeSmallRegions();
    /// Número de selas (ver passada diagonal) no mapa atual.
    int countSaddles();
    /// Autoverificação: toda cor em cache marcada como atualizada coincide com a recalculada a partir
    /// das amostras da região. Deve valer entre estágios; usado pelos testes.
    bool colorCacheConsistent() const;

    /// Atribui cores (mediana, ou paleta: snapColorsToExactPalette se palette->exact, senão
    /// snapColorsToPalette), compacta e devolve o mapa. Consome o mapa interno: depois disso o segmentador
    /// não deve mais ser usado.
    PixelMap finish(const Palette* palette = nullptr);

    PixelMap& map() noexcept { return map_; }
    const SegmentationStats& stats() const noexcept { return stats_; }

private:
    sla::Rgba regionColor(RegionId root) const;
    const sla::Rgba& colorOf(RegionId root);
    void markDirty(RegionId root);
    void refreshAll();
    void ensureCache();
    bool whatIfAccepts(RegionId a, RegionId b, double threshold);
    bool saddleAt(int x, int y);
    bool isBlend(const sla::Rgba& c, const sla::Rgba& a, const sla::Rgba& b) const;

    const sla::Array2D<sla::Rgba8>& image_;
    SegmentParams params_;
    PixelMap map_;
    std::optional<FlowField> flow_;      ///< calculado na primeira ligação de traços
    SegmentationStats stats_;
    std::vector<sla::Rgba> color_;      ///< cor efetiva por raiz (índice = id)
    std::vector<std::uint8_t> dirty_;   ///< 1 = cor desatualizada
    bool initialized_ = false;
};

}  // namespace cnx
