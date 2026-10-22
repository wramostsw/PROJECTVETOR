// cnx/preprocess/SignalImage.h — o sinal da imagem: RGBA em float, PRÉ-MULTIPLICADO (§2, docs §1.22).
//
// Cada pixel guarda (r·a, g·a, b·a, a) em [0, 1]. É a grandeza que se soma fisicamente: a cobertura de uma
// borda com AA, a média de área de uma redução e um desfoque são combinações lineares desses 4 números.
// Em RGBA não pré-multiplicado ("straight") a média mistura o RGB escondido sob alfa 0, que não é visível:
// vermelho opaco ao lado de transparente com RGB branco por baixo vira rosa ao reduzir, e com RGB preto
// vira vermelho-escuro (a franja). Pré-multiplicado, qualquer pixel invisível é (0, 0, 0, 0).
//
// Papéis no motor:
//   - fonte única da reamostragem (média de área exata, bilinear) e do desfoque do Preprocessor;
//   - alvo do GenerativeModel, em float, sem a quantização de 8 bits (docs §0, item 2);
//   - origem da imagem de 8 bits (toStraight8) que a paleta, a segmentação e o classificador usam, com os
//     pixels invisíveis canônicos (0, 0, 0, 0).
// Para imagens opacas, pré-multiplicado e straight coincidem: nada muda além da precisão.
#pragma once

#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

class SignalImage {
public:
    SignalImage() = default;

    /// Pré-multiplica uma imagem de 8 bits straight. Imagem vazia → sinal vazio.
    static SignalImage fromStraight(const sla::Array2D<sla::Rgba8>& image);

    int width() const noexcept { return signal_.width(); }
    int height() const noexcept { return signal_.height(); }

    /// O sinal pré-multiplicado (r·a, g·a, b·a, a).
    const sla::Array2D<sla::Rgba>& premultiplied() const noexcept { return signal_; }
    /// Cor straight do pixel: (r, g, b) = sinal/a, limitado a [0, 1]; (0, 0, 0, 0) onde a = 0.
    sla::Rgba straight(int x, int y) const noexcept;
    /// Imagem straight de 8 bits (arredondamento ao mais próximo). Alfa que arredonda para 0 → (0, 0, 0, 0);
    /// para uma imagem de 8 bits sem reamostragem, devolve os mesmos pixels, exceto o RGB escondido sob a = 0.
    sla::Array2D<sla::Rgba8> toStraight8() const;

    /// Redução por média de área exata: o pixel de saída i cobre [i·r, (i+1)·r) da entrada (r = entrada/saída)
    /// e pondera cada pixel pela fração coberta. Separável, em double. Conserva Σ sinal · área. Lança
    /// std::invalid_argument se w ou h < 1 ou maiores que a imagem (só reduz).
    SignalImage resampledArea(int w, int h) const;
    /// Ampliação (ou redução) bilinear com centros de pixel, bordas replicadas. Lança std::invalid_argument se
    /// w ou h < 1 ou o sinal estiver vazio.
    SignalImage resampledBilinear(int w, int h) const;
    /// Gaussiana separável de desvio sigma (px): raio ⌈3σ⌉, pesos exp(−k²/2σ²) normalizados, bordas
    /// replicadas. sigma ≤ 0 (ou NaN) → cópia; std::invalid_argument se sigma for infinito.
    SignalImage blurred(double sigma) const;

private:
    sla::Array2D<sla::Rgba> signal_;
};

}  // namespace cnx
