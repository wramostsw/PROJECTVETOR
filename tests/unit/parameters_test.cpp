// tests/unit/parameters_test.cpp — impressões digitais por estágio (campo a campo), perfis embutidos,
// controles de usuário e BackgroundSelection.
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <string>
#include <utility>
#include <vector>

#include "cnx/engine/Fingerprint.h"
#include "cnx/engine/Parameters.h"
#include "cnx/export/Exporter.h"
#include "test_harness.h"

using cnx::Fingerprint;
using cnx::MeasurementType;

namespace {

template <class P>
using Mutations = std::vector<std::pair<const char*, std::function<void(P&)>>>;

/// Toda mudança de campo tem de mudar a impressão digital — senão o recálculo inteligente reaproveitaria
/// um cache indevido. Também confere estabilidade e o encadeamento com o estágio anterior.
template <class P>
void checkEveryFieldIsHashed(const char* name, const P& base, const Mutations<P>& mutations) {
    const Fingerprint baseFp = base.fingerprint(Fingerprint{});
    CHECK_CTX(baseFp == base.fingerprint(Fingerprint{}), name);  // estável
    CHECK_CTX(baseFp != Fingerprint{}, name);
    CHECK_CTX(base.fingerprint(Fingerprint{1}) != baseFp, std::string(name) + ": não encadeia o estágio anterior");
    for (const auto& [field, mutate] : mutations) {
        P p = base;
        mutate(p);
        CHECK_CTX(p.fingerprint(Fingerprint{}) != baseFp, std::string(name) + "::" + field);
    }
}

}  // namespace

CNX_TEST(every_field_of_every_stage_parameter_set_enters_its_fingerprint) {
    checkEveryFieldIsHashed<cnx::PreprocessParams>("PreprocessParams", {}, {
        {"targetAreaPx", [](auto& p) { p.targetAreaPx *= 2; }},
        {"blurSigmaPx", [](auto& p) { p.blurSigmaPx = 1.5; }},
        {"flattenAlpha", [](auto& p) { p.flattenAlpha = !p.flattenAlpha; }},
        {"allowUpscale", [](auto& p) { p.allowUpscale = !p.allowUpscale; }},
    });
    checkEveryFieldIsHashed<cnx::ClassifierParams>("ClassifierParams", {}, {
        {"autoClassify", [](auto& p) { p.autoClassify = !p.autoClassify; }},
        {"forcedProfile", [](auto& p) { p.forcedProfile = cnx::ImageProfile::PhotoLow; }},
    });
    checkEveryFieldIsHashed<cnx::PaletteParams>("PaletteParams", {}, {
        {"minColors", [](auto& p) { p.minColors = 3; }},
        {"maxColors", [](auto& p) { p.maxColors = 9; }},
        {"kmeansMaxRounds", [](auto& p) { p.kmeansMaxRounds = 5; }},
        {"histogramBins", [](auto& p) { p.histogramBins = 8; }},
        {"colorPenalty", [](auto& p) { p.colorPenalty = 0.2; }},
        {"forcedNumColors", [](auto& p) { p.forcedNumColors = 4; }},
        {"edgeScale", [](auto& p) { p.edgeScale = 0.5; }},
        {"exactColorMinShare", [](auto& p) { p.exactColorMinShare = 0.01; }},
        {"refineRounds", [](auto& p) { p.refineRounds = 0; }},
    });
    {   // forçar 0 não é o mesmo que não forçar (has_value entra na impressão)
        cnx::PaletteParams a;
        cnx::PaletteParams b;
        b.forcedNumColors = 0;
        CHECK(a.fingerprint({}) != b.fingerprint({}));
    }
    checkEveryFieldIsHashed<cnx::SegmentParams>("SegmentParams", {}, {
        {"quadTreeBlock", [](auto& p) { p.quadTreeBlock = 8; }},
        {"quadTreeVariance", [](auto& p) { p.quadTreeVariance *= 2; }},
        {"mergeThreshStart", [](auto& p) { p.mergeThreshStart *= 2; }},
        {"mergeThreshGrowth", [](auto& p) { p.mergeThreshGrowth = 2.0; }},
        {"mergeThreshMax", [](auto& p) { p.mergeThreshMax *= 2; }},
        {"minRegionPixels", [](auto& p) { p.minRegionPixels = 9; }},
        {"whatIfFactor", [](auto& p) { p.whatIfFactor = 2.0; }},
        {"finalSweepCap", [](auto& p) { p.finalSweepCap = 3; }},
        {"microRegionMaxPixels", [](auto& p) { p.microRegionMaxPixels = 5; }},
        {"beachLineTolerance", [](auto& p) { p.beachLineTolerance = 0.5; }},
        {"beachMinContrast", [](auto& p) { p.beachMinContrast = 0.1; }},
        {"beachNeighborShare", [](auto& p) { p.beachNeighborShare = 0.5; }},
        {"shadowVariance", [](auto& p) { p.shadowVariance = 0.01; }},
        {"shadowSmoothRatio", [](auto& p) { p.shadowSmoothRatio = 0.9; }},
        {"shadowMinPixels", [](auto& p) { p.shadowMinPixels = 99; }},
        {"shadowBlurRadius", [](auto& p) { p.shadowBlurRadius = 4; }},
    });
    checkEveryFieldIsHashed<cnx::SmoothParams>("SmoothParams", {}, {
        {"measurementTypes[0]", [](auto& p) { p.measurementTypes[0] = MeasurementType::Generative; }},
        {"measurementTypes[1]", [](auto& p) { p.measurementTypes[1] = MeasurementType::None; }},
        {"measurementTypes[2]", [](auto& p) { p.measurementTypes[2] = MeasurementType::Generative; }},
        {"spacingWeight[0]", [](auto& p) { p.spacingWeight[0] = 9.0; }},
        {"spacingWeight[1]", [](auto& p) { p.spacingWeight[1] = 9.0; }},
        {"spacingWeight[2]", [](auto& p) { p.spacingWeight[2] = 9.0; }},
        {"fidelityWeight", [](auto& p) { p.fidelityWeight = 7.0; }},
        {"priorType", [](auto& p) { p.priorType = 2; }},
        {"priorStrength", [](auto& p) { p.priorStrength = 3.0; }},
        {"airPressureWeight", [](auto& p) { p.airPressureWeight = 1.0; }},
        {"airPressureMaxNodes", [](auto& p) { p.airPressureMaxNodes = 12; }},
        {"borderAnchorWeight", [](auto& p) { p.borderAnchorWeight = 1.0; }},
        {"borderHardConstraint", [](auto& p) { p.borderHardConstraint = !p.borderHardConstraint; }},
        {"fidelityFreeRadiusPx", [](auto& p) { p.fidelityFreeRadiusPx = 0.5; }},
        {"quarantineRounds", [](auto& p) { p.quarantineRounds = 5; }},
        {"minSegmentPx", [](auto& p) { p.minSegmentPx = 0.2; }},
        {"punctureCornerThresh", [](auto& p) { p.punctureCornerThresh = 1.2; }},
        {"punctureFlatnessThresh", [](auto& p) { p.punctureFlatnessThresh = 0.9; }},
        {"punctureArmPx", [](auto& p) { p.punctureArmPx = 8.0; }},
        {"punctureOnPhase", [](auto& p) { p.punctureOnPhase = 2; }},
        {"maxCgIterations", [](auto& p) { p.maxCgIterations = 17; }},
        {"maxNodeStepPx", [](auto& p) { p.maxNodeStepPx = 0.25; }},
        {"gradientTolerance", [](auto& p) { p.gradientTolerance = 1e-3; }},
        {"energyTolerance", [](auto& p) { p.energyTolerance = 1e-6; }},
    });
    checkEveryFieldIsHashed<cnx::FitParams>("FitParams", {}, {
        {"statThreshInitial", [](auto& p) { p.statThreshInitial = 1e-6; }},
        {"statThreshFinal", [](auto& p) { p.statThreshFinal = 0.7; }},
        {"maxRounds", [](auto& p) { p.maxRounds = 10; }},
        {"initialFragmentNodes", [](auto& p) { p.initialFragmentNodes = 8; }},
        {"newtonMaxIterations", [](auto& p) { p.newtonMaxIterations = 3; }},
        {"finalSweeps", [](auto& p) { p.finalSweeps = 1; }},
        {"enforceG1", [](auto& p) { p.enforceG1 = !p.enforceG1; }},
    });
    checkEveryFieldIsHashed<cnx::ExporterSettings>("ExporterSettings", {}, {
        {"cakeStackMode", [](auto& p) { p.cakeStackMode = cnx::CakeStackMode::Stacked; }},
        {"shapeStroking", [](auto& p) { p.shapeStroking = !p.shapeStroking; }},
        {"strokeWidthPt", [](auto& p) { p.strokeWidthPt = 0.5; }},
        {"epsCompatibilityMode", [](auto& p) { p.epsCompatibilityMode = !p.epsCompatibilityMode; }},
        {"dxfMode", [](auto& p) { p.dxfMode = cnx::DxfMode::Polylines; }},
        {"flattenTolerancePx", [](auto& p) { p.flattenTolerancePx = 0.5; }},
        {"coordinateDecimals", [](auto& p) { p.coordinateDecimals = 4; }},
        {"pointsPerPixel", [](auto& p) { p.pointsPerPixel = 0.75; }},
        {"background", [](auto& p) { p.background.setHidden(3, true); }},
    });
}

CNX_TEST(fingerprints_ignore_negative_zero_and_canonicalize_nan) {
    cnx::FitParams a;
    cnx::FitParams b;
    a.statThreshInitial = 0.0;
    b.statThreshInitial = -0.0;
    CHECK(a.fingerprint({}) == b.fingerprint({}));
    Fingerprint x;
    Fingerprint y;
    x.add(std::nan("1"));
    y.add(std::nan("2"));
    CHECK(x == y);
    // Separa "ab"+"c" de "a"+"bc".
    Fingerprint p;
    Fingerprint q;
    p.add(std::string_view("ab")).add(std::string_view("c"));
    q.add(std::string_view("a")).add(std::string_view("bc"));
    CHECK(p != q);
}

CNX_TEST(built_in_profiles_set_the_measurement_types_and_the_final_fit_threshold) {
    using cnx::ImageProfile;
    struct Row {
        ImageProfile profile;
        std::array<MeasurementType, 3> types;
        double threshold;
    };
    const MeasurementType G = MeasurementType::Generative, S = MeasurementType::Spring, N = MeasurementType::None;
    const std::vector<Row> table{
        {ImageProfile::LogoHighAA, {G, G, G}, 0.2}, {ImageProfile::LogoMedAA, {G, G, G}, 0.7}, {ImageProfile::LogoLowAA, {G, G, G}, 3.0},
        {ImageProfile::LogoHigh, {S, S, S}, 0.2},   {ImageProfile::LogoMed, {S, S, S}, 0.7},   {ImageProfile::LogoLow, {S, S, S}, 3.0},
        {ImageProfile::PhotoHigh, {S, N, N}, 0.5},  {ImageProfile::PhotoMed, {S, N, N}, 3.0},  {ImageProfile::PhotoLow, {S, N, N}, 10.0},
    };
    for (const Row& r : table) {
        cnx::EngineParameters p;
        p.applyProfile(r.profile);
        CHECK_CTX(p.smooth.measurementTypes == r.types, std::to_string(static_cast<int>(r.profile)));
        CHECK_CTX(p.fit.statThreshFinal == r.threshold, std::to_string(static_cast<int>(r.profile)));
        CHECK(p.fit.statThreshFinal >= p.fit.statThreshInitial);
    }
    // Custom não mexe em nada.
    cnx::EngineParameters custom;
    custom.smooth.measurementTypes = {N, G, S};
    custom.fit.statThreshFinal = 1.23;
    const cnx::EngineParameters before = custom;
    custom.applyProfile(ImageProfile::Custom);
    CHECK(custom.smooth.measurementTypes == before.smooth.measurementTypes && custom.fit.statThreshFinal == 1.23);
    // O perfil muda a impressão digital dos estágios que ele toca, e só deles.
    cnx::EngineParameters a;
    cnx::EngineParameters b;
    b.applyProfile(ImageProfile::LogoMedAA);
    CHECK(a.smooth.fingerprint({}) != b.smooth.fingerprint({}));
    CHECK(a.fit.fingerprint({}) != b.fit.fingerprint({}));
    CHECK(a.palette.fingerprint({}) == b.palette.fingerprint({}));
    // O limiar final nunca fica abaixo do inicial, mesmo que a tabela do perfil peça menos.
    cnx::EngineParameters floorTest;
    floorTest.fit.statThreshInitial = 5.0;
    floorTest.applyProfile(ImageProfile::LogoHigh);
    CHECK(floorTest.fit.statThreshFinal == 5.0);
    // Aplicar duas vezes é idempotente.
    cnx::EngineParameters c = b;
    c.applyProfile(ImageProfile::LogoMedAA);
    CHECK(c.smooth.fingerprint({}) == b.smooth.fingerprint({}) && c.fit.fingerprint({}) == b.fit.fingerprint({}));
}

CNX_TEST(user_controls_scale_their_parameters_multiplicatively_around_the_midpoint) {
    const cnx::EngineParameters base;
    const auto apply = [&](cnx::UserControl c, int v) {
        cnx::EngineParameters p = base;
        cnx::applyUserControl(p, c, v);
        return p;
    };
    // 50 é a identidade para todos os controles.
    for (const cnx::UserControl c : {cnx::UserControl::Sensitivity, cnx::UserControl::Complexity, cnx::UserControl::Smoothing,
                                     cnx::UserControl::Curves, cnx::UserControl::Antialiasing}) {
        const cnx::EngineParameters p = apply(c, 50);
        CHECK(p.segment.fingerprint({}) == base.segment.fingerprint({}) && p.fit.fingerprint({}) == base.fit.fingerprint({}) &&
              p.smooth.fingerprint({}) == base.smooth.fingerprint({}));
    }
    // Suavização: passos iguais do controle multiplicam o parâmetro pelo mesmo fator.
    const double s25 = apply(cnx::UserControl::Smoothing, 25).smooth.priorStrength;
    const double s50 = apply(cnx::UserControl::Smoothing, 50).smooth.priorStrength;
    const double s75 = apply(cnx::UserControl::Smoothing, 75).smooth.priorStrength;
    CHECK(std::abs(s75 / s50 - s50 / s25) < 1e-12 && s75 > s50 && s50 > s25);
    CHECK(std::abs(apply(cnx::UserControl::Smoothing, 100).smooth.priorStrength / base.smooth.priorStrength - 4.0) < 1e-12);
    CHECK(std::abs(apply(cnx::UserControl::Smoothing, 0).smooth.priorStrength / base.smooth.priorStrength - 0.25) < 1e-12);
    // Sensibilidade: mais sensível = menos fusão (limiares menores).
    CHECK(apply(cnx::UserControl::Sensitivity, 80).segment.mergeThreshMax < base.segment.mergeThreshMax);
    CHECK(apply(cnx::UserControl::Sensitivity, 20).segment.mergeThreshStart > base.segment.mergeThreshStart);
    // Complexidade: mais complexo = limiar final menor (mais curvas), nunca abaixo do inicial.
    CHECK(apply(cnx::UserControl::Complexity, 90).fit.statThreshFinal < base.fit.statThreshFinal);
    CHECK(apply(cnx::UserControl::Complexity, 100).fit.statThreshFinal >= base.fit.statThreshInitial);
    // Curvas: fragmentos iniciais maiores ou menores, dentro de [2, 64].
    CHECK(apply(cnx::UserControl::Curves, 100).fit.initialFragmentNodes == 16 && apply(cnx::UserControl::Curves, 0).fit.initialFragmentNodes == 2);
    CHECK(apply(cnx::UserControl::Antialiasing, 100).segment.beachLineTolerance == 1.0);  // satura em 1
    // Valores fora da faixa saturam.
    CHECK(apply(cnx::UserControl::Smoothing, 500).smooth.priorStrength == apply(cnx::UserControl::Smoothing, 100).smooth.priorStrength);
    CHECK(std::abs(cnx::multiplicativeScale(5.0, 0.0, 10.0, 1.0, 100.0) - 10.0) < 1e-12);
    CHECK_THROWS(std::invalid_argument, (void)cnx::multiplicativeScale(1.0, 5.0, 5.0, 1.0, 2.0));
    CHECK_THROWS(std::invalid_argument, (void)cnx::multiplicativeScale(1.0, 0.0, 5.0, 0.0, 2.0));
}

CNX_TEST(background_selection_supports_undo_redo_and_branch_discard) {
    cnx::BackgroundSelection b;
    CHECK(b.isShapeVisible(1) && !b.canUndo() && !b.canRedo());
    b.setHidden(4, true);
    b.setHidden(2, true);
    b.setHidden(2, true);  // já oculta: nenhum passo novo
    CHECK(!b.isShapeVisible(2) && !b.isShapeVisible(4) && b.isShapeVisible(3));
    CHECK((b.hiddenFaces() == std::vector<cnx::FaceId>{2, 4}));  // ordenada
    CHECK(b.undo() && b.isShapeVisible(2) && !b.isShapeVisible(4));
    CHECK(b.canRedo());
    CHECK(b.redo() && !b.isShapeVisible(2) && !b.redo());
    // Novo passo depois de desfazer descarta o ramo de refazer.
    CHECK(b.undo());
    b.setHidden(9, true);
    CHECK(!b.canRedo() && !b.isShapeVisible(9) && b.isShapeVisible(2));
    b.setHidden(4, false);
    CHECK(b.isShapeVisible(4));
    b.setHidden(77, false);  // nada a desfazer: sem passo
    const bool canUndoBeforeClear = b.canUndo();
    b.clear();
    CHECK(canUndoBeforeClear && b.hiddenFaces().empty() && b.undo() && !b.isShapeVisible(9));
    // Mesmo conjunto, históricos diferentes → mesma impressão digital.
    cnx::BackgroundSelection x;
    cnx::BackgroundSelection y;
    x.setHidden(1, true);
    x.setHidden(2, true);
    y.setHidden(2, true);
    y.setHidden(1, true);
    y.setHidden(5, true);
    y.setHidden(5, false);
    CHECK(x.fingerprint({}) == y.fingerprint({}));
    y.setHidden(3, true);
    CHECK(x.fingerprint({}) != y.fingerprint({}));
    // Conjuntos diferentes de mesmo tamanho também têm impressões diferentes.
    cnx::BackgroundSelection z;
    z.setHidden(1, true);
    z.setHidden(3, true);
    CHECK(x.fingerprint({}) != z.fingerprint({}));
    // Limpar o que já está vazio não registra passo.
    cnx::BackgroundSelection fresh;
    fresh.clear();
    CHECK(!fresh.canUndo());
}

int main() { return cnxtest::runAll(); }
