// src/engine/Parameters.cpp — impressões digitais por estágio, perfis embutidos e controles de usuário.
#include "cnx/engine/Parameters.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cnx {
namespace {

template <class E>
std::uint64_t code(E e) noexcept {
    return static_cast<std::uint64_t>(static_cast<std::uint8_t>(e));
}

}  // namespace

// Cada fingerprint() lista TODOS os campos do struct, na ordem de declaração: um campo esquecido faria o
// recálculo inteligente reaproveitar um cache indevido (parameters_test confere campo a campo).

Fingerprint PreprocessParams::fingerprint(Fingerprint upstream) const {
    Fingerprint f;
    f.add(std::string_view("preprocess")).add(targetAreaPx).add(blurSigmaPx).add(static_cast<int>(flattenAlpha)).add(static_cast<int>(allowUpscale));
    return f.add(upstream);
}

Fingerprint ClassifierParams::fingerprint(Fingerprint upstream) const {
    Fingerprint f;
    f.add(std::string_view("classifier")).add(static_cast<int>(autoClassify)).add(code(forcedProfile));
    return f.add(upstream);
}

Fingerprint PaletteParams::fingerprint(Fingerprint upstream) const {
    Fingerprint f;
    f.add(std::string_view("palette")).add(minColors).add(maxColors).add(kmeansMaxRounds).add(histogramBins).add(colorPenalty);
    f.add(static_cast<int>(forcedNumColors.has_value())).add(forcedNumColors.value_or(0));
    f.add(edgeScale).add(exactColorMinShare).add(refineRounds);
    return f.add(upstream);
}

Fingerprint SegmentParams::fingerprint(Fingerprint upstream) const {
    Fingerprint f;
    f.add(std::string_view("segment")).add(quadTreeBlock).add(quadTreeVariance).add(mergeThreshStart).add(mergeThreshGrowth).add(mergeThreshMax);
    f.add(paletteSnapMax).add(minRegionPixels).add(whatIfFactor).add(finalSweepCap);
    f.add(static_cast<int>(linkStrokes)).add(flowSigmaPx).add(strokeMinCoherence).add(strokeMaxAngle);
    f.add(microRegionMaxPixels).add(beachLineTolerance).add(beachMinContrast).add(beachNeighborShare);
    f.add(shadowVariance).add(shadowSmoothRatio).add(shadowMinPixels).add(shadowBlurRadius);
    return f.add(upstream);
}

Fingerprint SmoothParams::fingerprint(Fingerprint upstream) const {
    Fingerprint f;
    f.add(std::string_view("smooth"));
    for (const MeasurementType m : measurementTypes) f.add(code(m));
    for (const double w : spacingWeight) f.add(w);
    f.add(fidelityWeight).add(priorType).add(priorStrength).add(airPressureWeight).add(airPressureMaxNodes);
    f.add(borderAnchorWeight).add(static_cast<int>(borderHardConstraint)).add(fidelityFreeRadiusPx).add(quarantineRounds).add(minSegmentPx);
    f.add(punctureCornerThresh).add(punctureFlatnessThresh).add(punctureArmPx).add(punctureOnPhase);
    f.add(punctureMinContrast).add(contrastFullLevels).add(contrastFloor);
    f.add(maxCgIterations).add(maxNodeStepPx).add(gradientTolerance).add(energyTolerance);
    return f.add(upstream);
}

Fingerprint FitParams::fingerprint(Fingerprint upstream) const {
    Fingerprint f;
    f.add(std::string_view("fit")).add(statThreshInitial).add(statThreshFinal).add(maxRounds).add(initialFragmentNodes);
    f.add(newtonMaxIterations).add(finalSweeps).add(static_cast<int>(enforceG1));
    return f.add(upstream);
}

void EngineParameters::applyProfile(ImageProfile profile) {
    if (profile == ImageProfile::Custom) return;
    // Medição do suavizador (§6): logos com AA usam o modelo gerador nas 3 fases; logos sem AA, a mola
    // simples; fotos, a mola só na 1ª fase (as demais sem termo de fidelidade — interpretação, ver Parameters.h).
    if (isAntialiasedLogo(profile)) {
        smooth.measurementTypes = {MeasurementType::Generative, MeasurementType::Generative, MeasurementType::Generative};
    } else if (isPhoto(profile)) {
        smooth.measurementTypes = {MeasurementType::Spring, MeasurementType::None, MeasurementType::None};
    } else {
        smooth.measurementTypes = {MeasurementType::Spring, MeasurementType::Spring, MeasurementType::Spring};
    }
    // stat_thresh_final: 0,2 nos perfis de alta fidelidade/AA; valores maiores nos compactos (0,5 … 10,0).
    // A especificação dá a faixa, não a tabela: estes valores são provisórios (a calibrar).
    switch (profile) {
        case ImageProfile::LogoHighAA:
        case ImageProfile::LogoHigh: fit.statThreshFinal = 0.2; break;
        case ImageProfile::LogoMedAA:
        case ImageProfile::LogoMed: fit.statThreshFinal = 0.7; break;
        case ImageProfile::LogoLowAA:
        case ImageProfile::LogoLow: fit.statThreshFinal = 3.0; break;
        case ImageProfile::PhotoHigh: fit.statThreshFinal = 0.5; break;
        case ImageProfile::PhotoMed: fit.statThreshFinal = 3.0; break;
        case ImageProfile::PhotoLow: fit.statThreshFinal = 10.0; break;
        case ImageProfile::Custom: break;
    }
    fit.statThreshFinal = std::max(fit.statThreshFinal, fit.statThreshInitial);

    // Passo das faixas de degradê (§1.28): o teto de fusão é o do usuário (12 níveis de cor por padrão) vezes o fator
    // do detalhe do perfil, (8/12)² para alto e (24/12)² para baixo. Multiplicar (e não atribuir) preserva o controle
    // Sensitivity, que escala o mesmo campo. O início nunca passa do teto.
    double detail = 1.0;
    bool snapOnlyNearInks = false;  // fotos e degradês: a paleta não explica a imagem; só encaixa quem está perto de uma tinta
    switch (profile) {
        case ImageProfile::PhotoHigh: detail = (8.0 / 12.0) * (8.0 / 12.0); snapOnlyNearInks = true; break;
        case ImageProfile::PhotoMed: snapOnlyNearInks = true; break;
        case ImageProfile::PhotoLow: detail = (24.0 / 12.0) * (24.0 / 12.0); snapOnlyNearInks = true; break;
        case ImageProfile::LogoHighAA:
        case ImageProfile::LogoHigh: detail = (8.0 / 12.0) * (8.0 / 12.0); break;
        case ImageProfile::LogoLowAA:
        case ImageProfile::LogoLow: detail = (24.0 / 12.0) * (24.0 / 12.0); break;
        case ImageProfile::LogoMedAA:
        case ImageProfile::LogoMed:
        case ImageProfile::Custom: break;
    }
    segment.mergeThreshMax *= detail;
    segment.mergeThreshStart = std::min(segment.mergeThreshStart, segment.mergeThreshMax);
    // A tolerância do encaixe é a do ruído, um terço do passo da faixa (4 níveis no perfil médio): mais do que isso
    // puxaria a faixa de um degradê até a tinta, que é só o centro de um grupo do k-means, e pioraria o erro.
    if (snapOnlyNearInks) segment.paletteSnapMax = segment.mergeThreshMax / 9.0;
}

void applyUserControl(EngineParameters& params, UserControl control, int value) {
    // Escala multiplicativa: 0..100 ↦ fator em [¼, 4], com 50 = identidade (§0.1: a especificação nomeia
    // 5 dos 11 controles, e os alvos abaixo são provisórios). Passos iguais multiplicam o parâmetro
    // pelo mesmo fator.
    const double f = multiplicativeScale(static_cast<double>(value), 0.0, 100.0, 0.25, 4.0);
    switch (control) {
        case UserControl::Sensitivity:  // mais sensível = mais detalhe = menos fusão de regiões
            params.segment.mergeThreshStart /= f;
            params.segment.mergeThreshMax /= f;
            break;
        case UserControl::Complexity:  // mais complexidade = mais curvas = limiar de fusão menor
            params.fit.statThreshFinal = std::max(params.fit.statThreshFinal / f, params.fit.statThreshInitial);
            break;
        case UserControl::Smoothing:
            params.smooth.priorStrength *= f;
            break;
        case UserControl::Curves:  // fragmentos iniciais mais longos = curvas mais longas e suaves
            params.fit.initialFragmentNodes = std::clamp(static_cast<int>(std::lround(params.fit.initialFragmentNodes * f)), 2, 64);
            break;
        case UserControl::Antialiasing:  // mais tolerância na detecção de praias
            params.segment.beachLineTolerance = std::min(1.0, params.segment.beachLineTolerance * f);
            break;
    }
}

}  // namespace cnx
