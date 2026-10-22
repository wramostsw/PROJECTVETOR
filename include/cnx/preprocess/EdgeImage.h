// cnx/preprocess/EdgeImage.h — mapa de nitidez de bordas (§2 "Mapas de suporte", docs §1.19).
//
// Por pixel, o gradiente de Sobel 3×3 em cada um dos 4 canais (RGBA em [0, 1], bordas replicadas):
//   magnitude  m = √(Σ_c gx_c² + gy_c²) / 4      (um degrau de amplitude a em um canal dá m = a)
//   normal     n = (cos θ, sin θ),  θ = ½·atan2(2·Jxy, Jxx − Jyy)
// com o tensor de estrutura J = Σ_c ∇_c ∇_cᵀ: a orientação dominante dos canais juntos, sem que gradientes
// de sinais opostos em canais diferentes se cancelem. n é um eixo (o sinal é arbitrário); (0, 0) onde m = 0.
//
// Usos: os pesos 1/(1 + m/edgeScale) da paleta (pixels de transição valem menos) e as cristas de borda
// que o detector de antialiasing (AADetector) atravessa perpendicularmente.
#pragma once

#include <vector>

#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

/// Pixel de crista: máximo local da magnitude ao longo da própria normal.
struct EdgePoint {
    int x = 0;
    int y = 0;
    float magnitude = 0.0f;
    sla::Vec2 normal;  ///< unitária
};

class EdgeImage {
public:
    EdgeImage() = default;

    /// Imagem vazia → EdgeImage vazio.
    static EdgeImage compute(const sla::Array2D<sla::Rgba8>& image);

    int width() const noexcept { return magnitude_.width(); }
    int height() const noexcept { return magnitude_.height(); }

    float magnitude(int x, int y) const noexcept { return magnitude_(x, y); }
    const sla::Array2D<float>& magnitudes() const noexcept { return magnitude_; }
    /// Eixo unitário da direção de maior variação; (0, 0) onde a magnitude é nula.
    sla::Vec2 normal(int x, int y) const noexcept { return {normalX_(x, y), normalY_(x, y)}; }

    /// Pesos 1 / (1 + m/edgeScale) ∈ (0, 1]. Lança std::invalid_argument se edgeScale ≤ 0 ou não finito.
    sla::Array2D<float> sharpnessWeights(double edgeScale) const;

    /// Cristas com m ≥ minMagnitude (e m > 0): m é estritamente maior que no vizinho p − n e ≥ que em p + n
    /// (vizinhos pelo arredondamento de p ± n; fora da imagem conta 0). A assimetria escolhe exatamente um dos
    /// dois pixels de um degrau, que têm a mesma magnitude. Ordem: varredura por linhas.
    std::vector<EdgePoint> ridges(float minMagnitude) const;

private:
    sla::Array2D<float> magnitude_;
    sla::Array2D<float> normalX_;
    sla::Array2D<float> normalY_;
};

}  // namespace cnx
