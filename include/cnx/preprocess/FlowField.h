// cnx/preprocess/FlowField.h — campo de orientação (§2 "Mapas de suporte", docs §1.20).
//
// O tensor de estrutura de Sobel (EdgeImage.h, sobelTensors) integrado por uma gaussiana de desvio σ:
//     J_σ = G_σ * Σ_c ∇I_c ∇I_cᵀ        (bordas replicadas, unidades de m²: tr J_0 = m²)
// Autovalores λ₁ ≥ λ₂ ≥ 0. Por pixel:
//     energia   λ₁ + λ₂ = Jxx + Jyy
//     coerência (λ₁ − λ₂)/(λ₁ + λ₂) = √((Jxx − Jyy)² + 4·Jxy²) / (Jxx + Jyy) ∈ [0, 1]
//     tangente  autovetor de λ₂: a direção AO LONGO da borda ou do traço (um eixo; o sinal é arbitrário)
//
// Por que integrar: no centro de um traço de 1 px o gradiente é nulo (os dois lados se cancelam no Sobel),
// e EdgeImage não dá direção ali. O tensor soma ∇∇ᵀ, que não depende do sinal, então os dois flancos do
// traço somam a mesma orientação e o centro herda a direção do traço. Um xadrez de 1 px tem gradiente de
// Sobel exatamente nulo e cantos e texturas têm coerência baixa: nenhum deles parece traço.
//
// Uso: a ligação de traços finos diagonais na segmentação (PixelSegmenter::linkStrokeSweep).
#pragma once

#include "cnx/preprocess/EdgeImage.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

class FlowField {
public:
    FlowField() = default;

    /// sigma = desvio (px) da gaussiana de integração (raio ⌈3σ⌉, pesos exp(−k²/2σ²) normalizados);
    /// 0 = tensor bruto. Lança std::invalid_argument se sigma < 0 ou não finito. Imagem vazia → campo vazio.
    static FlowField compute(const sla::Array2D<sla::Rgba8>& image, double sigma);

    int width() const noexcept { return tensor_.width(); }
    int height() const noexcept { return tensor_.height(); }
    double sigma() const noexcept { return sigma_; }

    /// Tensor integrado, em unidades de m².
    const StructureTensor& tensor(int x, int y) const noexcept { return tensor_(x, y); }
    /// λ₁ + λ₂.
    double energy(int x, int y) const noexcept;
    /// (λ₁ − λ₂)/(λ₁ + λ₂); 0 onde a energia é nula.
    double coherence(int x, int y) const noexcept;
    /// Eixo unitário ao longo da borda; (0, 0) onde o tensor é isotrópico (λ₁ = λ₂, inclusive energia nula).
    sla::Vec2 tangent(int x, int y) const noexcept;
    /// Eixo unitário através da borda (a tangente girada de 90°); (0, 0) onde ela é (0, 0).
    sla::Vec2 normal(int x, int y) const noexcept;
    /// cos 2α, com α o ângulo entre a tangente e `direction` (não precisa ser unitária): 1 ao longo do traço,
    /// −1 através dele, 0 a 45°. 0 onde o tensor é isotrópico ou `direction` é nula. Sem trigonometria.
    double alignment(int x, int y, sla::Vec2 direction) const noexcept;

    /// As mesmas medidas para um tensor qualquer — por exemplo a soma dos tensores de dois pixels, que é o
    /// campo do par (a integração sobre a união das duas vizinhanças).
    static double coherenceOf(const StructureTensor& t) noexcept;
    static double alignmentOf(const StructureTensor& t, sla::Vec2 direction) noexcept;

private:
    sla::Array2D<StructureTensor> tensor_;
    double sigma_ = 0.0;
};

}  // namespace cnx
