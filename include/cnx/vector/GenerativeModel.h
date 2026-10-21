// cnx/vector/GenerativeModel.h — modelo direto de análise por síntese (docs/ARQUITETURA.md §2.2–2.3).
//
// Dado um VectorImage (poligonais entre os nós), a cor sintetizada do pixel p é
//     ĉ_p = Σ_f a_p(f) · c_f,      a_p(f) = |f ∩ p| ∈ [0, 1]      (filtro-caixa exato; exterior = cor 0)
// e a energia de fidelidade é E(P) = Σ_p w_p ‖ĉ_p − I_p‖² nos 4 canais RGBA.
//
//   * Cobertura exata por área: acumulador de área assinada por linha (teorema de Green), sem
//     recortar polígonos. Uma única passada pelas arestas, com o vetor (c_esq − c_dir) por aresta.
//   * Energia esparsa: entre duas colunas cruzadas consecutivas a cor é constante, então o erro de
//     um trecho sai de somas prefixadas em O(1). Custo por avaliação O(P log P) em P = pedaços
//     de aresta por pixel, independente de W·H (a construção do modelo é O(W·H)).
//   * Gradiente analítico por derivada de forma (Reynolds): ∂a_p/∂P_k = n_k·∫(1−t) + n_{k−1}·∫t, na
//     mesma passada. Exato exceto quando um segmento é colinear a uma linha da grade (medida nula).
//
// As cores das faces são fixas; as variáveis são as posições dos nós. A cor de uma face é
// Face::color (não pré-multiplicada); a mistura é linear nos 4 canais, no espaço em que a imagem
// alvo e as cores das faces foram dadas.
//
// Thread-safety: todos os métodos são const e sem estado mutável; podem rodar em paralelo.
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "cnx/vector/VectorImage.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

/// Cor RGBA em precisão dupla (os acumuladores do modelo).
using Color4 = std::array<double, 4>;

namespace detail {
struct Group;  // incremento do acumulador de uma linha/coluna (GenerativeModel.cpp)
}

/// Contadores da última avaliação (diagnóstico e testes de complexidade).
struct GenerativeStats {
    std::size_t segments = 0;       ///< segmentos de poligonal com contribuição
    std::size_t pieces = 0;         ///< migalhas de pão (segmento ∩ pixel) dentro da imagem
    std::size_t events = 0;         ///< incrementos do acumulador, antes da fusão
    std::size_t crossedPixels = 0;  ///< pares (linha, coluna) distintos com incremento
};

class GenerativeModel {
public:
    /// `target`: imagem original (sem desfoque), RGBA em [0, 1]. Pesos 1 em todos os pixels.
    /// Memória: ≈ 48 bytes por pixel para as somas prefixadas. Lança std::invalid_argument se vazia.
    explicit GenerativeModel(sla::Array2D<sla::Rgba> target);
    /// `weights` ≥ 0 e finitos, mesma dimensão de `target` (w_p = 0 ignora o pixel).
    GenerativeModel(sla::Array2D<sla::Rgba> target, const sla::Array2D<float>& weights);

    /// Conversão da entrada decodificada (8 bits) para o alvo em ponto flutuante.
    static sla::Array2D<sla::Rgba> targetFrom(const sla::Array2D<sla::Rgba8>& image);

    int width() const noexcept { return target_.width(); }
    int height() const noexcept { return target_.height(); }

    /// E(P). Lança std::invalid_argument se as dimensões de `image` forem diferentes das do alvo.
    double energy(const VectorImage& image, GenerativeStats* stats = nullptr) const;

    /// E(P) e ∂E/∂P_n para todos os nós (zerado e depois acumulado). O chamador mascara os nós
    /// fixos (Pinned/quarentena). Junções somam a contribuição de cada aresta incidente.
    /// Lança std::invalid_argument se gradient.size() != image.nodeCount().
    double energyAndGradient(const VectorImage& image, std::span<sla::Vec2> gradient,
                             GenerativeStats* stats = nullptr) const;

    /// Imagem sintetizada densa (W × H), O(W·H + P log P). Diagnóstico e testes.
    void render(const VectorImage& image, sla::Array2D<Color4>& out) const;

    /// Cobertura a_p(f) de uma face, densa (soma assinada dos laços: furos subtraem). A face exterior
    /// (0) é o complemento da imagem e dá cobertura nula dentro dela. Lança std::out_of_range se `face`
    /// não existir.
    void faceCoverage(const VectorImage& image, FaceId face, sla::Array2D<double>& out) const;

    /// E(P) pela definição, a partir de render(): O(W·H). Referência para os testes.
    double denseEnergy(const VectorImage& image) const;

private:
    void checkDimensions(const VectorImage& image) const;
    /// Varredura esparsa por linhas sobre os incrementos fundidos; devolve E e (opcional) a cor ĉ de cada grupo.
    double sweep(const std::vector<detail::Group>& groups, std::vector<Color4>* groupColors) const;
    double runEnergy(int y, int x0, int x1, const Color4& c) const;
    double pixelWeight(int x, int y) const noexcept { return weights_.empty() ? 1.0 : weights_(x, y); }

    sla::Array2D<sla::Rgba> target_;
    sla::Array2D<float> weights_;     ///< vazio = todos 1
    std::vector<double> prefW_;       ///< (W+1)·H somas de w (só se houver pesos)
    std::vector<double> prefS1_;      ///< (W+1)·H·4 somas de w·I
    std::vector<double> prefS2_;      ///< (W+1)·H somas de w·‖I‖²
};

}  // namespace cnx
