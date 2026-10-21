// cnx/engine/Parameters.h — parâmetros por estágio, perfis embutidos, controles de usuário e .apf.
//
// Valores marcados "a calibrar" são provisórios: devem sair do treinamento no conjunto de dados de
// referência (SegmentationTrainer + busca de hiperparâmetros), nunca de constantes copiadas.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnx/engine/Fingerprint.h"

namespace cnx {

/// Perfis embutidos (§2). A especificação fala em 10 perfis e enumera 9; `Custom` (parâmetros de
/// um .apf ou da interface) completa a lista até a confirmação.
enum class ImageProfile : std::uint8_t {
    LogoHighAA,  // EP_LOGO_HIG_AA
    LogoMedAA,   // EP_LOGO_MED_AA
    LogoLowAA,   // EP_LOGO_LOW_AA
    LogoHigh,    // EP_LOGO_HIG
    LogoMed,     // EP_LOGO_MED
    LogoLow,     // EP_LOGO_LOW
    PhotoHigh,   // EP_PHOTO_HIG
    PhotoMed,    // EP_PHOTO_MED
    PhotoLow,    // EP_PHOTO_LOW
    Custom,
};

constexpr bool isAntialiasedLogo(ImageProfile p) noexcept { return p <= ImageProfile::LogoLowAA; }
constexpr bool isPhoto(ImageProfile p) noexcept {
    return p >= ImageProfile::PhotoHigh && p <= ImageProfile::PhotoLow;
}

/// Termo de fidelidade de cada fase da suavização (`measurement_types`).
enum class MeasurementType : std::uint8_t {
    Spring = 0,      ///< mola até a escadinha original (com zona livre opcional)
    Generative = 1,  ///< análise por síntese (§6) — logos com AA, nas 3 fases
    None = 2,        ///< sem termo de fidelidade — fotos, fases 2–3 (interpretação da spec, a confirmar)
};

/// Escala multiplicativa dos controles de usuário:
///   v ∈ [vMin, vMax]  ↦  p = pMin · (pMax / pMin)^((v − vMin) / (vMax − vMin)),   pMin, pMax > 0.
/// Passos iguais do controle multiplicam o parâmetro pelo mesmo fator (efeito suave no início,
/// forte no fim, quando pMax ≫ pMin). v fora da faixa é saturado.
inline double multiplicativeScale(double v, double vMin, double vMax, double pMin, double pMax) {
    if (!(vMax > vMin) || !(pMin > 0.0) || !(pMax > 0.0)) {
        throw std::invalid_argument("multiplicativeScale: requer vMax > vMin e pMin, pMax > 0");
    }
    const double s = std::clamp((v - vMin) / (vMax - vMin), 0.0, 1.0);
    return pMin * std::pow(pMax / pMin, s);
}

struct PreprocessParams {
    double targetAreaPx = 1.0e6;  ///< área-alvo do redimensionamento (px²) — a calibrar
    double blurSigmaPx = 0.0;     ///< σ do desfoque gaussiano (paleta/segmentação); 0 desativa
    bool flattenAlpha = true;     ///< ensureFlattened quando o alfa é constante

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct ClassifierParams {
    bool autoClassify = true;
    ImageProfile forcedProfile = ImageProfile::Custom;  ///< usado quando autoClassify = false

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct PaletteParams {
    int minColors = 2;
    int maxColors = 12;
    int kmeansMaxRounds = 10;
    int histogramBins = 16;              ///< por canal; 8 no modo rápido
    double colorPenalty = 0.05;          ///< nota = log(erro + 1) + (K − 2)·penalidade — a calibrar
    std::optional<int> forcedNumColors;  ///< ignora a seleção automática

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct SegmentParams {
    int quadTreeBlock = 16;
    double quadTreeVariance = 1.0e-3;  ///< — a calibrar
    double mergeThreshStart = 1.0e-4;  ///< limiar inicial de "juntar" (distância² RGBA) — a calibrar
    double mergeThreshGrowth = 1.5;    ///< fator multiplicativo por rodada — a calibrar
    double mergeThreshMax = 2.5e-2;    ///< teto — a calibrar
    int minRegionPixels = 4;           ///< enforceMinNumPixels — a calibrar
    /// Orçamento da simulação "e se" de uma fusão: o erro quadrático da união pode superar a soma
    /// dos erros das partes em no máximo `whatIfFactor · limiar · min(nₐ, n_b)`. Para regiões
    /// chapadas isso equivale a dist²(medianas) ≤ limiar (fator 1); para regiões heterogêneas é mais severo.
    double whatIfFactor = 1.0;
    int finalSweepCap = 8;       ///< varreduras extras de juntar/trocar/diagonal no limiar máximo (convergência)

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct SmoothParams {
    static constexpr int kPhases = 3;

    std::array<MeasurementType, kPhases> measurementTypes{MeasurementType::Spring, MeasurementType::Spring,
                                                          MeasurementType::Spring};
    std::array<double, kPhases> spacingWeight{5.0, 0.5, 0.5};
    double fidelityWeight = 1.0;          ///< λ_fid: peso da mola (ou do modelo gerador) frente aos priors — a calibrar
    int priorType = 3;                    ///< 3 = suavidade angular (único implementado; o "raiz quadrada" fica desativado)
    double priorStrength = 10.0;          ///< prior_strengths
    double airPressureWeight = 5.0;       ///< air_pressure_weight
    int airPressureMaxNodes = 8;          ///< aplica-se a contornos com menos de 8 nós
    double borderAnchorWeight = 10.0;     ///< peso da mola de margem — a calibrar
    /// Além da mola, trava exatamente a coordenada perpendicular dos nós da moldura: eles deslizam
    /// ao longo dela e nunca saem. Sem isso a moldura deriva e a soma das áreas deixa de ser W·H (I4).
    bool borderHardConstraint = true;
    double fidelityFreeRadiusPx = 0.0;    ///< raio da zona livre da mola; 0 = mola simples
    int quarantineRounds = 2;
    /// Segmentos mais curtos que isto são tratados como colapso (nós coincidentes): a iteração é
    /// desfeita e os nós vão para a quarentena, como num cruzamento.
    double minSegmentPx = 0.05;
    /// Canto perfurado: um nó é canto se o giro entre os braços de comprimento `punctureArmPx` for
    /// ≥ `punctureCornerThresh` (rad) e os dois braços forem retos, isto é, nenhum nó intermediário
    /// se afasta da corda em mais de `punctureFlatnessThresh` (px). Limiar ≤ 0 desativa. A regra
    /// substitui o vetor de 21 medidas da especificação, que não está definido — a calibrar.
    double punctureCornerThresh = 0.8;    ///< puncture_corner_thresh (rad)
    double punctureFlatnessThresh = 0.3;  ///< puncture_flatness_thresh (px)
    double punctureArmPx = 4.0;
    int punctureOnPhase = 1;              ///< índice 0-based: só na fase do meio
    int maxCgIterations = 200;            ///< por fase
    double maxNodeStepPx = 0.5;           ///< região de confiança da busca em linha (§2.3)
    double gradientTolerance = 1.0e-5;    ///< norma do infinito do gradiente nos nós livres
    double energyTolerance = 1.0e-10;     ///< decréscimo relativo de energia por iteração

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct FitParams {
    double statThreshInitial = 1.0e-8;
    double statThreshFinal = 0.2;  ///< stat_thresh_final (0.2 alta fidelidade/AA; 0.5…10.0 compactos)
    int maxRounds = 20;
    int initialFragmentNodes = 4;  ///< — a calibrar
    int newtonMaxIterations = 10;
    int finalSweeps = 8;           ///< varreduras extras no limiar final, até nenhuma fusão/deslocamento restar
    bool enforceG1 = true;         ///< tangente compartilhada nas quebras que não são cantos/junções

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct EngineParameters {
    std::uint64_t seed = 0x9E3779B97F4A7C15ull;  ///< semente do myPrng; entra em todas as impressões digitais

    PreprocessParams preprocess;
    ClassifierParams classifier;
    PaletteParams palette;
    SegmentParams segment;
    SmoothParams smooth;
    FitParams fit;

    /// Sobrescreve os campos que o perfil define (measurement_types, stat_thresh_final, etc.).
    void applyProfile(ImageProfile profile);
};

/// Os ajustes de usuário. A especificação fala em 11 e nomeia 5; os demais entram quando definidos.
enum class UserControl : std::uint8_t { Sensitivity, Complexity, Smoothing, Curves, Antialiasing };

/// Aplica um controle (valor inteiro da interface) via multiplicativeScale sobre o(s) parâmetro(s) associado(s).
void applyUserControl(EngineParameters& params, UserControl control, int value);

/// Erro de leitura de .apf com o número da linha (1-based; 0 = erro de E/S).
class ParameterFileError : public std::runtime_error {
public:
    ParameterFileError(const std::string& message, int line) : std::runtime_error(message), line_(line) {}
    int line() const noexcept { return line_; }

private:
    int line_;
};

/// Perfis .apf em texto plano, uma entrada por linha: `Classe::parametro valor`
/// (ex.: `ContourSmoother::prior_strengths 10`). Linhas vazias e iniciadas por '#' são ignoradas.
class ParameterFiler {
public:
    enum class Mode : std::uint8_t {
        Strict,   ///< chave desconhecida → ParameterFileError
        Lenient,  ///< chave desconhecida → devolvida na lista de avisos (compatibilidade entre versões)
    };

    /// Retorna as chaves ignoradas (sempre vazio no modo Strict).
    static std::vector<std::string> readAPF(std::istream& in, EngineParameters& params, Mode mode = Mode::Strict);
    static std::vector<std::string> readAPF(const std::filesystem::path& file, EngineParameters& params,
                                            Mode mode = Mode::Strict);
    static void writeAPF(std::ostream& out, const EngineParameters& params);
    static void writeAPF(const std::filesystem::path& file, const EngineParameters& params);
};

}  // namespace cnx
