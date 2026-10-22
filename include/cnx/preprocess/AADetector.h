// cnx/preprocess/AADetector.h — detector de antialiasing por perfis perpendiculares às bordas (§2, docs §1.19).
//
// Em cada crista forte do EdgeImage, lê 6 pixels consecutivos ao longo da normal (offsets −3..+2, vizinho
// mais próximo: interpolar criaria rampas falsas em bordas duras oblíquas). Com a = v₀ e b = v₅ (as pontas),
// cada amostra interna é projetada no eixo a→b:
//     t_k = ⟨v_k − a, b − a⟩ / ‖b − a‖²,     r_k = ‖v_k − a − t_k (b − a)‖ / ‖b − a‖.
//   - Other: contraste ‖b − a‖ < minContrast, alguma amostra fora do eixo (r_k > lineTolerance — uma terceira
//     cor, não a mistura das duas tintas), fora de [−tol, 1 + tol] (ringing) ou não monótona (retrocesso > tol);
//   - Step:  todas as amostras internas num patamar (t ≤ stepTolerance ou t ≥ 1 − stepTolerance): borda dura;
//   - Ramp:  pelo menos uma amostra intermediária: borda com AA (1–2 intermediárias) ou desfocada (3–4).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "cnx/preprocess/EdgeImage.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

struct AAParams {
    float minEdgeMagnitude = 0.2f;   ///< crista forte (unidades do EdgeImage: degrau de a num canal = a)
    double minContrast = 0.15;       ///< ‖b − a‖ mínimo entre as pontas (RGBA em [0, 1])
    double stepTolerance = 0.12;     ///< t a menos disso de 0 ou de 1 conta como patamar
    double lineTolerance = 0.15;     ///< distância máxima ao eixo a→b, em fração do contraste
    double monotoneTolerance = 0.15; ///< retrocesso (e transbordo de [0, 1]) tolerado, em t
};

enum class EdgeProfileKind : std::uint8_t { Step, Ramp, Other };

struct EdgeProfile {
    EdgeProfileKind kind = EdgeProfileKind::Other;
    int intermediate = 0;  ///< amostras internas (1..4) estritamente entre os patamares
};

/// Classifica um perfil de 6 amostras (v₀ … v₅) ao longo da normal. Função pura.
EdgeProfile analyzeEdgeProfile(const std::array<sla::Rgba, 6>& samples, const AAParams& params = {});

struct AAStats {
    std::size_t ridges = 0;  ///< cristas fortes encontradas
    std::size_t steps = 0;
    std::size_t ramps = 0;
    std::size_t others = 0;  ///< inclui as que não cabem na imagem (perfil cortado)
    std::array<std::size_t, 5> rampWidths{};  ///< [k] = perfis com k amostras intermediárias (0 = Step)

    std::size_t valid() const noexcept { return steps + ramps; }
    /// Fração de rampas entre os perfis válidos (0 sem perfis válidos).
    double rampShare() const noexcept { return valid() == 0 ? 0.0 : static_cast<double>(ramps) / static_cast<double>(valid()); }
    /// Média de amostras intermediárias nos perfis válidos (largura da transição em px; 0 = todas duras).
    double meanTransitionWidth() const noexcept;
};

/// Percorre as cristas de `edges` (calculado sobre `image`) e acumula os perfis.
/// Lança std::invalid_argument se as dimensões diferirem.
AAStats detectAntialiasing(const sla::Array2D<sla::Rgba8>& image, const EdgeImage& edges, const AAParams& params = {});

}  // namespace cnx
