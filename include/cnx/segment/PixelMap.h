// cnx/segment/PixelMap.h — mapa de rótulos + Union-Find de regiões (§4).
//
// Fusões não reescrevem pixels: labels_ guarda o id de uma região (inicialmente o superpixel) e a
// região atual é find(label). Ao fim da segmentação, compact() (= buildRegions) resolve tudo e
// renumera. Ver docs/ARQUITETURA.md §1.4 (fluxo), §1.5 (teste topológico) e §1.10 (contatos).
#pragma once

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

using RegionId = std::int32_t;
inline constexpr RegionId kNoRegion = -1;

/// Paleta oficial (§3).
struct Palette {
    std::vector<sla::Rgba> colors;
    double score = 0.0;  ///< log(erro + 1) + (K − 2)·penalidade
    bool exact = false;  ///< imagem chapada: as cores são as cores exatas da imagem (modo "paleta exata" da segmentação)
};

namespace RegionFlag {
inline constexpr std::uint8_t Shadow = 1u << 0;      ///< flagShadowSegments
inline constexpr std::uint8_t Beach = 1u << 1;       ///< faixa de transição (splitAndMergeBeaches)
inline constexpr std::uint8_t Background = 1u << 2;  ///< marcada pelo usuário como fundo
}  // namespace RegionFlag

/// Mediana marginal por canal com fusão barata.
///
/// Modo pequeno: amostras brutas (até kSwitchToHistogram = 1024, ou seja 4 KiB). Modo grande:
/// histograma 4 × 256 de uint32 (também 4 KiB). A troca de modo é transparente e só ocorre ao
/// crescer. Custos: add O(1); remove O(1) no histograma e O(n ≤ 1024) no modo pequeno;
/// merge O(1024) ou O(n_menor); median O(256 + n).
class MedianAccumulator {
public:
    static constexpr std::size_t kSwitchToHistogram = 1024;

    void add(sla::Rgba8 c);
    /// Pré-condição: c foi adicionado antes (operação "trocar"). Lança std::logic_error se a cor
    /// não estiver registrada (detecção completa no modo pequeno, por canal no histograma).
    void remove(sla::Rgba8 c);
    /// Absorve `other`, que fica vazio.
    void merge(MedianAccumulator&& other);

    std::uint32_t count() const noexcept { return count_; }
    bool usesHistogram() const noexcept { return !histogram_.empty(); }

    /// Mediana inferior de cada canal: o valor de posição ⌊(n−1)/2⌋ na ordem crescente
    /// (determinística em empates). Lança std::logic_error se vazio.
    sla::Rgba8 median() const;
    /// Média aritmética (fallback para regiões com pouquíssimos pixels); {} se vazio.
    sla::Rgba mean() const noexcept;

    /// Mediana inferior da união dos multiconjuntos de a e b, sem modificá-los (simulação "e se"
    /// de uma fusão, §4). O(256 + nₐ + n_b). Lança std::logic_error se ambos estiverem vazios.
    static sla::Rgba8 unionMedian(const MedianAccumulator& a, const MedianAccumulator& b);

    /// Σ_amostras Σ_canais (v/255 − center_c)²: o erro quadrático da região em torno de `center`.
    /// Aborto antecipado: se a soma parcial passar de `abortAbove`, retorna na hora um valor
    /// > abortAbove (não o total). Custo O(n) no modo pequeno e O(1024) no histograma.
    double squaredDeviation(const sla::Rgba& center, double abortAbove = std::numeric_limits<double>::infinity()) const;
    /// Mesmo erro para a união de a e b (aditivo: SSE(A ∪ B) = SSE(A) + SSE(B) para o mesmo centro).
    static double unionSquaredDeviation(const MedianAccumulator& a, const MedianAccumulator& b, const sla::Rgba& center,
                                        double abortAbove = std::numeric_limits<double>::infinity());

private:
    void promoteToHistogram();
    /// Soma as contagens deste acumulador em bins[4·256].
    void addToBins(std::uint32_t* bins) const;

    std::uint32_t count_ = 0;
    std::array<std::uint64_t, 4> sum_{};    // somas por canal (média)
    std::vector<sla::Rgba8> samples_;       // modo pequeno
    std::vector<std::uint32_t> histogram_;  // modo grande: [canal·256 + valor]; vazio no modo pequeno
};

struct RegionStats {
    MedianAccumulator samples;  ///< pixels da região (mediana por canal = cor verdadeira)
    /// Caixa inclusiva. Exata após compact(); entre trocas de pixel é um superconjunto (só cresce).
    int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;
    std::uint8_t flags = 0;          ///< RegionFlag
    std::int32_t paletteIndex = -1;  ///< após snapColorsToPalette
    sla::Rgba color{};               ///< cor efetiva: mediana (assignMedianColors) ou cor da paleta
};

/// Union-Find sobre ids de regiões, com estatísticas e contatos vivendo na raiz.
///
/// find: compressão por halving; unite: união por tamanho em pixels (empate → menor id), o que
/// dá O(α(n)) amortizado. Contatos: para cada par de raízes, o número exato de pares de pixels
/// 4-adjacentes entre elas (o comprimento da fronteira compartilhada, em lados de pixel). Ficam num
/// mapa de hash com chaves sempre formadas por raízes atuais; unite re-indexa as entradas da raiz
/// absorvida em O(d) e PixelMap mantém as contagens a cada setLabel/tryMovePixel.
class RegionForest {
public:
    RegionForest() = default;
    explicit RegionForest(std::size_t reserve);

    /// Novo singleton com 0 pixels; os pixels entram por PixelMap::setLabel.
    RegionId create();

    std::size_t capacity() const noexcept { return parent_.size(); }
    /// Raízes, incluindo regiões criadas e ainda sem pixels.
    std::size_t rootCount() const noexcept { return roots_; }

    /// Pré-condição (não verificada, laço quente): 0 ≤ r < capacity().
    RegionId find(RegionId r) noexcept {
        while (parent_[ix(r)] != r) {
            parent_[ix(r)] = parent_[ix(parent_[ix(r)])];  // halving: aponta para o avô
            r = parent_[ix(r)];
        }
        return r;
    }
    RegionId findNoCompress(RegionId r) const noexcept {
        while (parent_[ix(r)] != r) r = parent_[ix(r)];
        return r;
    }
    bool isRoot(RegionId r) const noexcept { return parent_[ix(r)] == r; }

    /// Une as árvores de a e b (qualquer membro) e retorna a nova raiz. Funde estatísticas
    /// (MedianAccumulator::merge, caixa, flags; cor e paleta ficam as da raiz) e contatos; os
    /// contatos entre a e b viram fronteira interna e desaparecem.
    RegionId unite(RegionId a, RegionId b);

    std::uint32_t pixelCount(RegionId root) const noexcept { return size_[ix(root)]; }
    RegionStats& stats(RegionId root) noexcept { return stats_[ix(root)]; }
    const RegionStats& stats(RegionId root) const noexcept { return stats_[ix(root)]; }

    /// Registra/remove n pares de pixels 4-adjacentes entre as regiões de a e b (ignorado se forem
    /// a mesma). Usados por PixelMap; removeContacts lança std::logic_error se a contagem não bastar.
    void addContacts(RegionId a, RegionId b, std::uint32_t n = 1);
    void removeContacts(RegionId a, RegionId b, std::uint32_t n = 1);
    /// Comprimento da fronteira entre as regiões de a e b (0 se não são vizinhas ou são a mesma).
    std::uint32_t contactCount(RegionId a, RegionId b) const;

    /// Vizinhos da região de `r`: as raízes com contato > 0, únicas, em ordem crescente.
    /// Canonização preguiçosa, O(d log d); o span vale até a próxima mutação da floresta.
    std::span<const RegionId> neighbors(RegionId r);

private:
    friend class PixelMap;

    static constexpr std::size_t ix(RegionId r) noexcept { return static_cast<std::size_t>(r); }
    static constexpr std::uint64_t contactKey(RegionId a, RegionId b) noexcept {
        const auto lo = static_cast<std::uint64_t>(a < b ? a : b);
        const auto hi = static_cast<std::uint64_t>(a < b ? b : a);
        return (lo << 32) | hi;
    }
    void checkId(RegionId r, const char* where) const;

    std::vector<RegionId> parent_;
    std::vector<std::uint32_t> size_;
    std::vector<RegionStats> stats_;
    std::vector<std::vector<RegionId>> neighbors_;  // superconjunto preguiçoso (ids possivelmente obsoletos)
    std::unordered_map<std::uint64_t, std::uint32_t> contacts_;
    std::size_t roots_ = 0;
};

class PixelMap {
public:
    PixelMap() = default;
    /// Todos os pixels começam sem região (kNoRegion).
    PixelMap(int width, int height);

    int width() const noexcept { return labels_.width(); }
    int height() const noexcept { return labels_.height(); }

    /// Rótulo bruto (sem seguir a cadeia de fusões).
    RegionId rawLabel(int x, int y) const noexcept { return labels_(x, y); }
    /// Região atual do pixel (kNoRegion se ainda não atribuído), comprimindo o caminho.
    RegionId regionAt(int x, int y) noexcept {
        const RegionId l = labels_(x, y);
        return l == kNoRegion ? kNoRegion : forest_.find(l);
    }
    /// Idem, sem mutação (consumidores de um snapshot const).
    RegionId regionAtConst(int x, int y) const noexcept {
        const RegionId l = labels_(x, y);
        return l == kNoRegion ? kNoRegion : forest_.findNoCompress(l);
    }

    /// Atalho para forest().create().
    RegionId createRegion() { return forest_.create(); }

    /// Inicialização (QuadTree / modo paleta exata): atribui o pixel, atualiza estatísticas e
    /// contatos com os 4-vizinhos já atribuídos. Lança std::logic_error se o pixel já tem região.
    void setLabel(int x, int y, RegionId region, sla::Rgba8 color);

    /// "Trocar" (§4): move o pixel para a região de `to` e atualiza estatísticas e contatos.
    /// Retorna false, sem efeito, se o pixel não for um ponto simples da região de origem
    /// (isSimplePoint). `color` deve ser a mesma cor usada no setLabel desse pixel.
    /// Lança std::invalid_argument se `to` já é a região do pixel ou não é 4-vizinha dele.
    bool tryMovePixel(int x, int y, RegionId to, sla::Rgba8 color);

    /// Move o pixel para a região de `to` SEM teste topológico (reconstrução de regiões, como as de
    /// sombra). Atualiza estatísticas e contatos como tryMovePixel. O chamador responde pela
    /// 4-conexidade das regiões afetadas. Retorna false (sem efeito) se o pixel já é da região;
    /// não exige que `to` seja vizinha, e a região de origem pode ficar vazia.
    bool reassignPixel(int x, int y, RegionId to, sla::Rgba8 color);

    /// Tira o pixel das estatísticas de cor (MedianAccumulator) da sua região: ele continua nela para a
    /// topologia (contagem, contatos, ponto simples), e ao ser movido depois nenhuma amostra o acompanha.
    /// É a ponte de um traço fino diagonal (PixelSegmenter::linkStrokeSweep): um pixel de fundo que só existe
    /// para dar 4-conexidade ao traço e não deve puxar a cor dele. `color` deve ser a cor usada no setLabel.
    /// Retorna false, sem efeito, se o pixel já estava excluído. Lança std::logic_error se não tem região.
    bool excludeFromColorStats(int x, int y, sla::Rgba8 color);
    bool isExcludedFromColorStats(int x, int y) const noexcept { return !noStats_.empty() && noStats_(x, y) != 0; }

    /// Teste de ponto simples, O(1) (Rosenfeld 1970; Yokoi et al. 1975).
    /// Com R = região de p em 4-conectividade e R̄ em 8-conectividade (fora da imagem ∈ R̄):
    ///   p é simples  ⇔  T4(p, R) = 1 ∧ T8(p, R̄) = 1.
    /// Tabela de 256 entradas indexada pela máscara de R ∩ N8*(p), gerada em tempo de compilação.
    /// Falso para pixel sem região.
    bool isSimplePoint(int x, int y) const noexcept;

    RegionForest& forest() noexcept { return forest_; }
    const RegionForest& forest() const noexcept { return forest_; }

    /// Cor efetiva de cada raiz: mediana por canal, ou a média se a região tem menos de
    /// `meanFallbackBelow` pixels (§4: a média só é fallback para regiões minúsculas).
    void assignMedianColors(std::uint32_t meanFallbackBelow = 3);

    /// buildRegions: rótulos ← find(rótulos); raízes com pixels renumeradas 0..F−1 em ordem de
    /// varredura (determinístico); regiões vazias descartadas; a floresta vira identidade; caixas
    /// e contatos são recalculados exatamente. Lança std::logic_error se houver pixel sem região.
    /// Retorna F.
    RegionId compact();
    /// Verdadeiro após compact() e enquanto nenhuma fusão ou troca ocorrer.
    bool isCompact() const noexcept { return compact_ && forest_.rootCount() == forest_.capacity(); }

    /// Rótulos densos 0..F−1. Pré-condição: isCompact() (senão std::logic_error).
    const sla::Array2D<RegionId>& compactLabels() const;
    /// Cor final da região. Pré-condição: isCompact().
    const sla::Rgba& regionColor(RegionId region) const;

private:
    void moveUnchecked(int x, int y, RegionId from, RegionId target, sla::Rgba8 color, const std::array<RegionId, 4>& around);

    sla::Array2D<RegionId> labels_;
    RegionForest forest_;
    sla::Array2D<std::uint8_t> noStats_;  ///< 1 = pixel fora das estatísticas de cor; vazio até a primeira exclusão
    bool compact_ = false;
};

}  // namespace cnx
