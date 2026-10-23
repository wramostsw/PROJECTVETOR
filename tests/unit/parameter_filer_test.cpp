// tests/unit/parameter_filer_test.cpp — leitura/escrita de .apf: ida e volta exata, sintaxe, erros com número
// de linha, transacionalidade, modos estrito/tolerante, regras de validação e E/S de arquivo.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/fit/BezierFitter.h"
#include "cnx/palette/PaletteFinder.h"
#include "cnx/segment/PixelSegmenter.h"
#include "cnx/vector/ContourSmoother.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::EngineParameters;
using cnx::ImageProfile;
using cnx::MeasurementType;
using cnx::ParameterFileError;
using cnx::ParameterFiler;
using Mode = cnx::ParameterFiler::Mode;

namespace {

/// Igualdade campo a campo, pelas impressões digitais (que cobrem todos os campos: parameters_test).
bool same(const EngineParameters& a, const EngineParameters& b) {
    return a.seed == b.seed && a.preprocess.fingerprint({}) == b.preprocess.fingerprint({}) && a.classifier.fingerprint({}) == b.classifier.fingerprint({}) &&
           a.palette.fingerprint({}) == b.palette.fingerprint({}) && a.segment.fingerprint({}) == b.segment.fingerprint({}) &&
           a.smooth.fingerprint({}) == b.smooth.fingerprint({}) && a.fit.fingerprint({}) == b.fit.fingerprint({});
}

std::string toText(const EngineParameters& p) {
    std::ostringstream out;
    ParameterFiler::writeAPF(out, p);
    return out.str();
}

struct Outcome {
    bool threw = false;
    int line = -1;
    std::string message;
    std::vector<std::string> ignored;
};

Outcome tryRead(const std::string& text, EngineParameters& p, Mode mode = Mode::Strict) {
    Outcome o;
    std::istringstream in(text);
    try {
        o.ignored = ParameterFiler::readAPF(in, p, mode);
    } catch (const ParameterFileError& e) {
        o.threw = true;
        o.line = e.line();
        o.message = e.what();
    }
    return o;
}

/// Erro esperado: a leitura falha na `line`, a mensagem cita `needle`, e `params` fica intacto.
void expectError(const std::string& text, int line, const std::string& needle, Mode mode = Mode::Strict) {
    EngineParameters p;
    p.seed = 99;  // diferente do padrão, para provar que nada foi tocado
    const EngineParameters before = p;
    const Outcome o = tryRead(text, p, mode);
    CHECK_CTX(o.threw, "não falhou: " + text);
    CHECK_CTX(o.line == line, "linha " + std::to_string(o.line) + " em vez de " + std::to_string(line) + ": " + o.message + " <- " + text);
    CHECK_CTX(o.message.find(needle) != std::string::npos, "'" + needle + "' não aparece em: " + o.message);
    CHECK_CTX(same(p, before), "params mudou apesar do erro: " + text);
}

double wide(cnxtest::SplitMix64& rng) { return std::pow(10.0, -9.0 + 12.0 * cnxtest::uniform01(rng)); }  // 1e-9 … 1e3, mantissa cheia

/// Parâmetros válidos aleatórios, com mantissas cheias para exercitar a menor representação que reconstrói o double.
EngineParameters randomParams(cnxtest::SplitMix64& rng) {
    EngineParameters p;
    p.seed = rng.next();
    p.preprocess.targetAreaPx = 1.0 + wide(rng) * 1.0e5;
    p.preprocess.blurSigmaPx = rng.below(3) == 0 ? 0.0 : wide(rng);
    p.preprocess.flattenAlpha = rng.below(2) != 0;
    p.preprocess.allowUpscale = rng.below(2) != 0;
    p.classifier.autoClassify = rng.below(2) != 0;
    p.classifier.forcedProfile = static_cast<ImageProfile>(rng.below(10));
    p.palette.minColors = 1 + rng.below(5);
    p.palette.maxColors = p.palette.minColors + rng.below(20);
    p.palette.kmeansMaxRounds = rng.below(30);
    p.palette.histogramBins = 2 + rng.below(255);
    p.palette.colorPenalty = wide(rng);
    if (rng.below(2) != 0) p.palette.forcedNumColors = 1 + rng.below(20);
    else p.palette.forcedNumColors.reset();
    p.palette.edgeScale = wide(rng);
    p.palette.exactColorMinShare = cnxtest::uniform01(rng);
    p.palette.refineRounds = rng.below(5);
    p.segment.quadTreeBlock = 1 + rng.below(64);
    p.segment.quadTreeVariance = wide(rng);
    p.segment.mergeThreshStart = wide(rng);
    p.segment.mergeThreshMax = p.segment.mergeThreshStart * (1.0 + wide(rng));
    p.segment.mergeThreshGrowth = 1.0 + wide(rng) + 1e-6;
    p.segment.paletteSnapMax = wide(rng);
    p.segment.minRegionPixels = rng.below(50);
    p.segment.whatIfFactor = wide(rng);
    p.segment.finalSweepCap = rng.below(20);
    p.segment.linkStrokes = rng.below(2) != 0;
    p.segment.flowSigmaPx = wide(rng);
    p.segment.strokeMinCoherence = cnxtest::uniform01(rng);
    p.segment.strokeMaxAngle = 0.78 * cnxtest::uniform01(rng) + 1e-6;
    p.segment.microRegionMaxPixels = rng.below(8);
    p.segment.beachLineTolerance = wide(rng);
    p.segment.beachMinContrast = wide(rng);
    p.segment.beachNeighborShare = cnxtest::uniform01(rng);
    p.segment.shadowVariance = wide(rng);
    p.segment.shadowSmoothRatio = wide(rng);
    p.segment.shadowMinPixels = rng.below(100);
    p.segment.shadowBlurRadius = rng.below(6);
    for (auto& m : p.smooth.measurementTypes) m = static_cast<MeasurementType>(rng.below(3));
    for (auto& w : p.smooth.spacingWeight) w = wide(rng);
    p.smooth.fidelityWeight = wide(rng);
    p.smooth.priorType = 3;
    p.smooth.priorStrength = wide(rng);
    p.smooth.airPressureWeight = wide(rng);
    p.smooth.airPressureMaxNodes = rng.below(20);
    p.smooth.borderAnchorWeight = wide(rng);
    p.smooth.borderHardConstraint = rng.below(2) != 0;
    p.smooth.fidelityFreeRadiusPx = rng.below(2) == 0 ? 0.0 : wide(rng);
    p.smooth.quarantineRounds = rng.below(5);
    p.smooth.minSegmentPx = wide(rng);
    p.smooth.punctureCornerThresh = cnxtest::symmetric(rng, 3.0);
    p.smooth.punctureFlatnessThresh = wide(rng);
    p.smooth.punctureArmPx = wide(rng) + 1e-3;
    p.smooth.punctureOnPhase = rng.below(3);
    p.smooth.maxCgIterations = rng.below(500);
    p.smooth.maxNodeStepPx = wide(rng) + 1e-6;
    p.smooth.gradientTolerance = wide(rng);
    p.smooth.energyTolerance = wide(rng);
    p.fit.statThreshInitial = wide(rng) * 1.0e-6 + 1e-12;
    p.fit.statThreshFinal = p.fit.statThreshInitial * (1.0 + wide(rng));
    p.fit.maxRounds = 1 + rng.below(30);
    p.fit.initialFragmentNodes = 2 + rng.below(10);
    p.fit.newtonMaxIterations = rng.below(20);
    p.fit.finalSweeps = rng.below(20);
    p.fit.enforceG1 = rng.below(2) != 0;
    return p;
}

std::vector<std::string> linesOf(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);) out.push_back(l);
    return out;
}

struct FailingBuf final : std::streambuf {
    int_type underflow() override { throw std::runtime_error("leitura falhou"); }
};

}  // namespace

CNX_TEST(defaults_round_trip_and_the_spec_example_line_is_written_as_is) {
    const EngineParameters defaults;
    const std::string text = toText(defaults);
    // A linha de exemplo da especificação sai exatamente assim.
    CHECK(text.find("\nContourSmoother::prior_strengths 10\n") != std::string::npos);
    CHECK(text.find("\nCoreEngine::seed 11400714819323198485\n") != std::string::npos);  // 0x9E3779B97F4A7C15
    CHECK(text.find("\nPaletteFinder::forced_num_colors none\n") != std::string::npos);
    CHECK(text.find("\nContourSmoother::measurement_types spring spring spring\n") != std::string::npos);
    CHECK(text.find("\nContourSmoother::spacing_weight 5 0.5 0.5\n") != std::string::npos);
    CHECK(text.find("\nParameterClassifier::forced_profile EP_CUSTOM\n") != std::string::npos);
    CHECK(text.find("\nPreprocessor::flatten_alpha true\n") != std::string::npos);
    CHECK(text.rfind("# cnx .apf v1\n", 0) == 0);
    for (const char* section : {"CoreEngine", "Preprocessor", "ParameterClassifier", "PaletteFinder", "PixelSegmenter", "SubPixelSegmenter", "ContourSmoother", "BezierFitter"}) {
        CHECK_CTX(text.find(std::string("\n# ") + section + "\n" + section + "::") != std::string::npos, section);  // cabeçalho logo antes da 1ª chave da classe
    }
    CHECK(text.back() == '\n' && text.find('\r') == std::string::npos);

    cnxtest::SplitMix64 rng{7};
    EngineParameters p = randomParams(rng);
    CHECK(!same(p, defaults));
    const Outcome o = tryRead(text, p);
    CHECK(!o.threw && o.ignored.empty());
    CHECK(same(p, defaults));
}

CNX_TEST(random_valid_parameters_survive_write_read_write_bit_for_bit) {
    cnxtest::SplitMix64 rng{20260101};
    const sla::Array2D<sla::Rgba8> tiny(4, 4, sla::Rgba8{10, 20, 30, 255});
    for (int i = 0; i < 300; ++i) {
        const EngineParameters original = randomParams(rng);
        CHECK_CTX(!ParameterFiler::validate(original), "gerador inválido na rodada " + std::to_string(i));
        const std::string text = toText(original);
        EngineParameters back = randomParams(rng);  // começa de outros valores: tudo tem de ser sobrescrito
        const Outcome o = tryRead(text, back);
        CHECK_CTX(!o.threw && o.ignored.empty(), o.message);
        CHECK_CTX(same(back, original), "rodada " + std::to_string(i));
        CHECK_CTX(toText(back) == text, "segunda escrita difere na rodada " + std::to_string(i));
        // O que o leitor aceita os módulos aceitam (as regras espelham os construtores).
        (void)cnx::BezierFitter(original.fit);
        (void)cnx::ContourSmoother(original.smooth, nullptr);
        (void)cnx::PaletteFinder(original.palette, original.seed);
        (void)cnx::PixelSegmenter(tiny, original.segment);
    }
}

CNX_TEST(extreme_doubles_and_integers_round_trip_exactly) {
    EngineParameters p;
    p.seed = std::numeric_limits<std::uint64_t>::max();
    p.preprocess.targetAreaPx = std::numeric_limits<double>::max();
    p.preprocess.blurSigmaPx = std::numeric_limits<double>::denorm_min();
    p.segment.quadTreeVariance = -0.0;  // >= 0 vale para o zero negativo; precisa sobreviver com o sinal
    p.segment.mergeThreshStart = std::numeric_limits<double>::min();
    p.segment.mergeThreshMax = 1.0e300;
    p.palette.minColors = 1;
    p.palette.maxColors = std::numeric_limits<int>::max();
    p.fit.maxRounds = std::numeric_limits<int>::max();
    p.fit.statThreshInitial = 5e-324;
    p.fit.statThreshFinal = 1.7976931348623157e308;
    p.smooth.punctureCornerThresh = -std::numeric_limits<double>::max();
    p.smooth.punctureArmPx = 0.1 + 0.2;  // 0.30000000000000004: só a menor representação exata sobrevive
    const std::string text = toText(p);
    EngineParameters back;
    CHECK(!tryRead(text, back).threw);
    CHECK(same(back, p));
    CHECK(back.seed == std::numeric_limits<std::uint64_t>::max());
    CHECK(std::signbit(back.segment.quadTreeVariance) && back.segment.quadTreeVariance == 0.0);
    CHECK(back.smooth.punctureArmPx == 0.1 + 0.2 && back.smooth.punctureArmPx != 0.3);
    CHECK(back.preprocess.blurSigmaPx == std::numeric_limits<double>::denorm_min());
}

CNX_TEST(every_field_has_exactly_one_line_and_changing_it_changes_only_that_line) {
    using Mut = std::pair<const char*, std::function<void(EngineParameters&)>>;
    const std::vector<Mut> muts{
        {"CoreEngine::seed", [](auto& p) { p.seed ^= 1; }},
        {"Preprocessor::target_area_px", [](auto& p) { p.preprocess.targetAreaPx *= 2; }},
        {"Preprocessor::blur_sigma_px", [](auto& p) { p.preprocess.blurSigmaPx = 1.25; }},
        {"Preprocessor::flatten_alpha", [](auto& p) { p.preprocess.flattenAlpha = !p.preprocess.flattenAlpha; }},
        {"Preprocessor::allow_upscale", [](auto& p) { p.preprocess.allowUpscale = !p.preprocess.allowUpscale; }},
        {"ParameterClassifier::auto_classify", [](auto& p) { p.classifier.autoClassify = !p.classifier.autoClassify; }},
        {"ParameterClassifier::forced_profile", [](auto& p) { p.classifier.forcedProfile = ImageProfile::PhotoLow; }},
        {"PaletteFinder::min_colors", [](auto& p) { p.palette.minColors = 3; }},
        {"PaletteFinder::max_colors", [](auto& p) { p.palette.maxColors = 9; }},
        {"PaletteFinder::kmeans_max_rounds", [](auto& p) { p.palette.kmeansMaxRounds = 5; }},
        {"PaletteFinder::histogram_bins", [](auto& p) { p.palette.histogramBins = 8; }},
        {"PaletteFinder::color_penalty", [](auto& p) { p.palette.colorPenalty = 0.2; }},
        {"PaletteFinder::forced_num_colors", [](auto& p) { p.palette.forcedNumColors = 4; }},
        {"PaletteFinder::edge_scale", [](auto& p) { p.palette.edgeScale = 0.5; }},
        {"PaletteFinder::exact_color_min_share", [](auto& p) { p.palette.exactColorMinShare = 0.01; }},
        {"PaletteFinder::refine_rounds", [](auto& p) { p.palette.refineRounds = 0; }},
        {"PixelSegmenter::quad_tree_block", [](auto& p) { p.segment.quadTreeBlock = 8; }},
        {"PixelSegmenter::quad_tree_variance", [](auto& p) { p.segment.quadTreeVariance = 0.5; }},
        {"PixelSegmenter::merge_thresh_start", [](auto& p) { p.segment.mergeThreshStart = 2e-4; }},
        {"PixelSegmenter::merge_thresh_growth", [](auto& p) { p.segment.mergeThreshGrowth = 2.0; }},
        {"PixelSegmenter::merge_thresh_max", [](auto& p) { p.segment.mergeThreshMax = 0.1; }},
        {"PixelSegmenter::palette_snap_max", [](auto& p) { p.segment.paletteSnapMax = 0.01; }},
        {"PixelSegmenter::min_region_pixels", [](auto& p) { p.segment.minRegionPixels = 9; }},
        {"PixelSegmenter::what_if_factor", [](auto& p) { p.segment.whatIfFactor = 2.0; }},
        {"PixelSegmenter::final_sweep_cap", [](auto& p) { p.segment.finalSweepCap = 3; }},
        {"PixelSegmenter::link_strokes", [](auto& p) { p.segment.linkStrokes = !p.segment.linkStrokes; }},
        {"PixelSegmenter::flow_sigma_px", [](auto& p) { p.segment.flowSigmaPx = 2.0; }},
        {"PixelSegmenter::stroke_min_coherence", [](auto& p) { p.segment.strokeMinCoherence = 0.75; }},
        {"PixelSegmenter::stroke_max_angle", [](auto& p) { p.segment.strokeMaxAngle = 0.5; }},
        {"SubPixelSegmenter::micro_region_max_pixels", [](auto& p) { p.segment.microRegionMaxPixels = 5; }},
        {"SubPixelSegmenter::beach_line_tolerance", [](auto& p) { p.segment.beachLineTolerance = 0.5; }},
        {"SubPixelSegmenter::beach_min_contrast", [](auto& p) { p.segment.beachMinContrast = 0.1; }},
        {"SubPixelSegmenter::beach_neighbor_share", [](auto& p) { p.segment.beachNeighborShare = 0.6; }},
        {"SubPixelSegmenter::shadow_variance", [](auto& p) { p.segment.shadowVariance = 0.01; }},
        {"SubPixelSegmenter::shadow_smooth_ratio", [](auto& p) { p.segment.shadowSmoothRatio = 0.25; }},
        {"SubPixelSegmenter::shadow_min_pixels", [](auto& p) { p.segment.shadowMinPixels = 32; }},
        {"SubPixelSegmenter::shadow_blur_radius", [](auto& p) { p.segment.shadowBlurRadius = 3; }},
        {"ContourSmoother::measurement_types", [](auto& p) { p.smooth.measurementTypes[2] = MeasurementType::Generative; }},
        {"ContourSmoother::spacing_weight", [](auto& p) { p.smooth.spacingWeight[1] = 0.75; }},
        {"ContourSmoother::fidelity_weight", [](auto& p) { p.smooth.fidelityWeight = 2.0; }},
        {"ContourSmoother::prior_strengths", [](auto& p) { p.smooth.priorStrength = 12.5; }},
        {"ContourSmoother::air_pressure_weight", [](auto& p) { p.smooth.airPressureWeight = 6.0; }},
        {"ContourSmoother::air_pressure_max_nodes", [](auto& p) { p.smooth.airPressureMaxNodes = 6; }},
        {"ContourSmoother::border_anchor_weight", [](auto& p) { p.smooth.borderAnchorWeight = 3.0; }},
        {"ContourSmoother::border_hard_constraint", [](auto& p) { p.smooth.borderHardConstraint = !p.smooth.borderHardConstraint; }},
        {"ContourSmoother::fidelity_free_radius_px", [](auto& p) { p.smooth.fidelityFreeRadiusPx = 0.5; }},
        {"ContourSmoother::quarantine_rounds", [](auto& p) { p.smooth.quarantineRounds = 3; }},
        {"ContourSmoother::min_segment_px", [](auto& p) { p.smooth.minSegmentPx = 0.1; }},
        {"ContourSmoother::puncture_corner_thresh", [](auto& p) { p.smooth.punctureCornerThresh = 1.0; }},
        {"ContourSmoother::puncture_flatness_thresh", [](auto& p) { p.smooth.punctureFlatnessThresh = 0.4; }},
        {"ContourSmoother::puncture_arm_px", [](auto& p) { p.smooth.punctureArmPx = 5.0; }},
        {"ContourSmoother::puncture_on_phase", [](auto& p) { p.smooth.punctureOnPhase = 2; }},
        {"ContourSmoother::max_cg_iterations", [](auto& p) { p.smooth.maxCgIterations = 77; }},
        {"ContourSmoother::max_node_step_px", [](auto& p) { p.smooth.maxNodeStepPx = 0.25; }},
        {"ContourSmoother::gradient_tolerance", [](auto& p) { p.smooth.gradientTolerance = 1e-4; }},
        {"ContourSmoother::energy_tolerance", [](auto& p) { p.smooth.energyTolerance = 1e-8; }},
        {"BezierFitter::stat_thresh_initial", [](auto& p) { p.fit.statThreshInitial = 1e-6; }},
        {"BezierFitter::stat_thresh_final", [](auto& p) { p.fit.statThreshFinal = 0.7; }},
        {"BezierFitter::max_rounds", [](auto& p) { p.fit.maxRounds = 7; }},
        {"BezierFitter::initial_fragment_nodes", [](auto& p) { p.fit.initialFragmentNodes = 6; }},
        {"BezierFitter::newton_max_iterations", [](auto& p) { p.fit.newtonMaxIterations = 4; }},
        {"BezierFitter::final_sweeps", [](auto& p) { p.fit.finalSweeps = 2; }},
        {"BezierFitter::enforce_g1", [](auto& p) { p.fit.enforceG1 = !p.fit.enforceG1; }},
    };
    const EngineParameters base;
    const std::vector<std::string> baseLines = linesOf(toText(base));
    std::vector<std::string> entries;
    for (const std::string& l : baseLines) {
        if (!l.empty() && l[0] != '#') entries.push_back(l);
    }
    // 59 campos no total; ContourSmoother::prior_type só tem o valor 3 válido e é conferido à parte.
    CHECK(entries.size() == muts.size() + 1);
    CHECK(std::count(entries.begin(), entries.end(), std::string("ContourSmoother::prior_type 3")) == 1);

    for (const auto& [key, mutate] : muts) {
        EngineParameters p = base;
        mutate(p);
        const std::vector<std::string> lines = linesOf(toText(p));
        CHECK_CTX(lines.size() == baseLines.size(), key);
        int differing = 0;
        for (std::size_t i = 0; i < std::min(lines.size(), baseLines.size()); ++i) {
            if (lines[i] == baseLines[i]) continue;
            ++differing;
            CHECK_CTX(lines[i].rfind(std::string(key) + " ", 0) == 0, std::string(key) + " mudou a linha: " + lines[i]);
        }
        CHECK_CTX(differing == 1, std::string(key) + ": linhas diferentes = " + std::to_string(differing));
        EngineParameters back;
        CHECK_CTX(!tryRead(toText(p), back).threw && same(back, p), key);
    }
    // Todas as chaves são distintas.
    std::vector<std::string> keys;
    for (const std::string& e : entries) keys.push_back(e.substr(0, e.find(' ')));
    std::sort(keys.begin(), keys.end());
    CHECK(std::adjacent_find(keys.begin(), keys.end()) == keys.end());
}

CNX_TEST(syntax_accepts_comments_blank_lines_crlf_bom_tabs_and_partial_files) {
    cnxtest::SplitMix64 rng{11};
    const EngineParameters base = randomParams(rng);
    EngineParameters p = base;
    const std::string text =
        "\xEF\xBB\xBF# comentário com acento: ção\r\n"
        "\r\n"
        "   \t\r\n"
        "  PaletteFinder::max_colors\t 30  \r\n"
        "   # outro comentário\r\n"
        "ContourSmoother::spacing_weight  1   2\t3\r\n"
        "ParameterClassifier::forced_profile EP_PHOTO_HIG\r\n"
        "BezierFitter::enforce_g1 0";  // sem quebra de linha final
    const Outcome o = tryRead(text, p);
    CHECK_CTX(!o.threw, o.message);
    CHECK(p.palette.maxColors == 30);
    CHECK((p.smooth.spacingWeight == std::array<double, 3>{1.0, 2.0, 3.0}));
    CHECK(p.classifier.forcedProfile == ImageProfile::PhotoHigh);
    CHECK(!p.fit.enforceG1);
    // Um .apf parcial só muda o que cita.
    EngineParameters expected = base;
    expected.palette.maxColors = 30;
    expected.smooth.spacingWeight = {1.0, 2.0, 3.0};
    expected.classifier.forcedProfile = ImageProfile::PhotoHigh;
    expected.fit.enforceG1 = false;
    CHECK(same(p, expected));
    // Arquivo vazio e só de comentários: nada muda.
    EngineParameters q = base;
    CHECK(!tryRead("", q).threw && same(q, base));
    CHECK(!tryRead("# nada\n\n   \n", q).threw && same(q, base));
    // Ordem das linhas é livre.
    EngineParameters a;
    EngineParameters b;
    CHECK(!tryRead("BezierFitter::max_rounds 3\nPaletteFinder::max_colors 5\n", a).threw);
    CHECK(!tryRead("PaletteFinder::max_colors 5\nBezierFitter::max_rounds 3\n", b).threw);
    CHECK(same(a, b) && a.fit.maxRounds == 3 && a.palette.maxColors == 5);
}

CNX_TEST(value_forms_booleans_optionals_enums_lists_and_unsigned) {
    EngineParameters p;
    CHECK(!tryRead("Preprocessor::flatten_alpha false\nPreprocessor::allow_upscale 1\nParameterClassifier::auto_classify 0\nContourSmoother::border_hard_constraint true\n", p).threw);
    CHECK(!p.preprocess.flattenAlpha && p.preprocess.allowUpscale && !p.classifier.autoClassify && p.smooth.borderHardConstraint);
    // forced_num_colors: número ou none.
    CHECK(!tryRead("PaletteFinder::forced_num_colors 5\n", p).threw && p.palette.forcedNumColors == 5);
    CHECK(!tryRead("PaletteFinder::forced_num_colors none\n", p).threw && !p.palette.forcedNumColors.has_value());
    // Todas as enumerações por nome.
    const std::array<std::pair<const char*, ImageProfile>, 10> profiles{{{"EP_LOGO_HIG_AA", ImageProfile::LogoHighAA},
                                                                         {"EP_LOGO_MED_AA", ImageProfile::LogoMedAA},
                                                                         {"EP_LOGO_LOW_AA", ImageProfile::LogoLowAA},
                                                                         {"EP_LOGO_HIG", ImageProfile::LogoHigh},
                                                                         {"EP_LOGO_MED", ImageProfile::LogoMed},
                                                                         {"EP_LOGO_LOW", ImageProfile::LogoLow},
                                                                         {"EP_PHOTO_HIG", ImageProfile::PhotoHigh},
                                                                         {"EP_PHOTO_MED", ImageProfile::PhotoMed},
                                                                         {"EP_PHOTO_LOW", ImageProfile::PhotoLow},
                                                                         {"EP_CUSTOM", ImageProfile::Custom}}};
    for (const auto& [name, profile] : profiles) {
        EngineParameters e;
        CHECK_CTX(!tryRead(std::string("ParameterClassifier::forced_profile ") + name + "\n", e).threw && e.classifier.forcedProfile == profile, name);
        EngineParameters w;
        w.classifier.forcedProfile = profile;
        CHECK_CTX(toText(w).find(std::string("forced_profile ") + name + "\n") != std::string::npos, name);
    }
    CHECK(!tryRead("ContourSmoother::measurement_types generative none spring\n", p).threw);
    CHECK((p.smooth.measurementTypes == std::array<MeasurementType, 3>{MeasurementType::Generative, MeasurementType::None, MeasurementType::Spring}));
    // Semente de 64 bits sem sinal.
    CHECK(!tryRead("CoreEngine::seed 18446744073709551615\n", p).threw && p.seed == std::numeric_limits<std::uint64_t>::max());
    CHECK(!tryRead("CoreEngine::seed 0\n", p).threw && p.seed == 0);
    // Real em notação científica e com ponto inicial.
    CHECK(!tryRead("PaletteFinder::color_penalty 1.5e-2\nPaletteFinder::edge_scale .25\n", p).threw);
    CHECK(p.palette.colorPenalty == 0.015 && p.palette.edgeScale == 0.25);
    // Inteiro negativo onde a regra aceita (puncture_corner_thresh é real; aqui, só o sinal do real).
    CHECK(!tryRead("ContourSmoother::puncture_corner_thresh -1\n", p).threw && p.smooth.punctureCornerThresh == -1.0);
}

CNX_TEST(malformed_lines_report_the_line_number_and_leave_the_parameters_untouched) {
    expectError("# c\n\nsem_dois_pontos 3\n", 3, "chave malformada");
    expectError("Foo:: 3\n", 1, "chave malformada");
    expectError("::x 3\n", 1, "chave malformada");
    expectError("Foo::1x 3\n", 1, "chave malformada");
    expectError("A::b::c 1\n", 1, "chave malformada");
    expectError("Foo::bar-baz 1\n", 1, "chave malformada");
    expectError("BezierFitter::max_rounds\n", 1, "falta o valor");
    expectError("BezierFitter::max_rounds   \t \n", 1, "falta o valor");
    expectError("BezierFitter::max_rounds 3 4\n", 1, "esperado 1 valor");
    expectError("ContourSmoother::measurement_types spring spring\n", 1, "esperados 3 valores");
    expectError("ContourSmoother::measurement_types spring spring spring spring\n", 1, "esperados 3 valores");
    expectError("ContourSmoother::spacing_weight 1 2\n", 1, "esperados 3 valores");
    expectError("BezierFitter::max_rounds 3\n\nBezierFitter::max_rounds 4\n", 3, "linha 1");
    expectError("BezierFitter::max_rounds 3\nBezierFitter::max_rounds 3\n", 2, "repetida");
    expectError("BezierFitter::max_rounds 3\x01\n", 1, "controle");
    expectError("# ok\nBezierFitter::max_rounds 3\rPaletteFinder::max_colors 5\n", 2, "controle");
    expectError(std::string("BezierFitter::max_rounds 3") + std::string(1, '\0') + "\n", 1, "controle");
    expectError("BezierFitter::max_rounds " + std::string(5000, '1') + "\n", 1, "longa demais");
}

CNX_TEST(bad_scalar_values_are_rejected_with_the_offending_text) {
    for (const char* bad : {"1.5", "abc", "3x", "+3", "0x10", "99999999999", "-99999999999", "1e3", "--1", "3 3"}) {
        expectError(std::string("PaletteFinder::max_colors ") + bad + "\n", 1, "PaletteFinder::max_colors");
    }
    expectError("PaletteFinder::max_colors 99999999999\n", 1, "fora do intervalo");
    expectError("PaletteFinder::max_colors 1.5\n", 1, "inteiro inválido");
    expectError("CoreEngine::seed -1\n", 1, "inteiro inválido");
    expectError("CoreEngine::seed 18446744073709551616\n", 1, "fora do intervalo");
    for (const char* bad : {"nan", "inf", "-inf", "NaN", "1e999", "1,5", "abc", "1.5.2", "--1", "+1", "0x1p3", "1e"}) {
        expectError(std::string("PaletteFinder::edge_scale ") + bad + "\n", 1, "PaletteFinder::edge_scale");
    }
    expectError("PaletteFinder::edge_scale nan\n", 1, "não finito");
    expectError("PaletteFinder::edge_scale 1e999\n", 1, "fora do intervalo");
    for (const char* bad : {"yes", "TRUE", "2", "", "truee"}) {
        const std::string value = bad;
        if (value.empty()) continue;
        expectError("BezierFitter::enforce_g1 " + value + "\n", 1, "booleano inválido");
    }
    expectError("ParameterClassifier::forced_profile EP_LOGO\n", 1, "nome desconhecido");
    expectError("ParameterClassifier::forced_profile ep_logo_med\n", 1, "EP_LOGO_MED");  // lista os válidos
    expectError("ContourSmoother::measurement_types Spring spring spring\n", 1, "nome desconhecido");
    expectError("PaletteFinder::forced_num_colors None\n", 1, "inteiro inválido");
    expectError("PaletteFinder::forced_num_colors 1.5\n", 1, "inteiro inválido");
    // O erro cita a linha certa mesmo depois de linhas boas.
    expectError("# 1\nBezierFitter::max_rounds 3\n\nPaletteFinder::max_colors oito\nBezierFitter::final_sweeps 1\n", 4, "oito");
}

CNX_TEST(range_and_cross_field_rules_blame_the_last_line_that_defines_a_field_involved) {
    expectError("PaletteFinder::min_colors 5\nPaletteFinder::max_colors 3\n", 2, "max_colors");
    expectError("PaletteFinder::max_colors 3\nPaletteFinder::min_colors 5\n", 2, "max_colors");
    expectError("PaletteFinder::max_colors 1\n", 1, "max_colors");  // 1 < min_colors (2) padrão
    expectError("PaletteFinder::histogram_bins 1\n", 1, "histogram_bins");
    expectError("PaletteFinder::histogram_bins 257\n", 1, "histogram_bins");
    expectError("PaletteFinder::min_colors 0\n", 1, "min_colors");
    expectError("PaletteFinder::forced_num_colors 0\n", 1, "forced_num_colors");
    expectError("PaletteFinder::edge_scale 0\n", 1, "edge_scale");
    expectError("PaletteFinder::color_penalty -0.5\n", 1, "color_penalty");
    expectError("PaletteFinder::exact_color_min_share 1.5\n", 1, "exact_color_min_share");
    expectError("PaletteFinder::kmeans_max_rounds -1\n", 1, "kmeans_max_rounds");
    expectError("Preprocessor::target_area_px 0\n", 1, "target_area_px");
    expectError("Preprocessor::blur_sigma_px -1\n", 1, "blur_sigma_px");
    expectError("PixelSegmenter::quad_tree_block 0\n", 1, "quad_tree_block");
    expectError("PixelSegmenter::merge_thresh_growth 1\n", 1, "merge_thresh_growth");
    expectError("PixelSegmenter::merge_thresh_max 1e-5\n", 1, "merge_thresh_max");  // < start (1e-4)
    expectError("PixelSegmenter::merge_thresh_start 0\n", 1, "merge_thresh_start");
    expectError("PixelSegmenter::min_region_pixels -1\n", 1, "min_region_pixels");
    expectError("PixelSegmenter::flow_sigma_px -0.5\n", 1, "flow_sigma_px");
    expectError("PixelSegmenter::stroke_min_coherence 1.01\n", 1, "stroke_min_coherence");
    expectError("PixelSegmenter::stroke_min_coherence -0.01\n", 1, "stroke_min_coherence");
    expectError("PixelSegmenter::stroke_max_angle 0\n", 1, "stroke_max_angle");
    expectError("PixelSegmenter::stroke_max_angle 0.7853981633974483\n", 1, "stroke_max_angle");  // π/4: as duas diagonais empatam
    expectError("SubPixelSegmenter::beach_neighbor_share 2\n", 1, "beach_neighbor_share");
    expectError("SubPixelSegmenter::shadow_blur_radius -1\n", 1, "shadow_blur_radius");
    expectError("ContourSmoother::prior_type 2\n", 1, "prior_type");
    expectError("ContourSmoother::puncture_on_phase 3\n", 1, "puncture_on_phase");
    expectError("ContourSmoother::puncture_on_phase -1\n", 1, "puncture_on_phase");
    expectError("ContourSmoother::max_node_step_px 0\n", 1, "max_node_step_px");
    expectError("ContourSmoother::spacing_weight 1 -1 1\n", 1, "spacing_weight");
    expectError("ContourSmoother::max_cg_iterations -1\n", 1, "max_cg_iterations");
    expectError("BezierFitter::stat_thresh_final 1e-9\n", 1, "stat_thresh_final");  // < inicial (1e-8)
    expectError("BezierFitter::stat_thresh_initial 1\n", 1, "stat_thresh_final");    // inicial > final (0.2)
    expectError("BezierFitter::stat_thresh_initial 0\n", 1, "stat_thresh_initial");
    expectError("BezierFitter::max_rounds 0\n", 1, "max_rounds");
    expectError("BezierFitter::initial_fragment_nodes 1\n", 1, "initial_fragment_nodes");
    expectError("BezierFitter::newton_max_iterations -1\n", 1, "newton_max_iterations");
    expectError("BezierFitter::final_sweeps -1\n", 1, "final_sweeps");
    // Valores-limite aceitos.
    EngineParameters p;
    CHECK(!tryRead("PaletteFinder::min_colors 4\nPaletteFinder::max_colors 4\nPaletteFinder::histogram_bins 2\nPaletteFinder::exact_color_min_share 1\n"
                   "PixelSegmenter::merge_thresh_max 1e-4\nBezierFitter::stat_thresh_final 1e-8\nBezierFitter::initial_fragment_nodes 2\nBezierFitter::max_rounds 1\n"
                   "ContourSmoother::puncture_on_phase 2\nContourSmoother::spacing_weight 0 0 0\n"
                   "PixelSegmenter::flow_sigma_px 0\nPixelSegmenter::stroke_min_coherence 1\nPixelSegmenter::stroke_max_angle 0.785\n",
                   p)
                .threw);
    // Incoerência que vem dos valores anteriores ao arquivo: linha 0.
    EngineParameters bad;
    bad.palette.minColors = 20;  // > max_colors (12)
    const Outcome o = tryRead("BezierFitter::max_rounds 3\n", bad);
    CHECK(o.threw && o.line == 0 && o.message.find("anteriores") != std::string::npos);
    CHECK(bad.fit.maxRounds == EngineParameters{}.fit.maxRounds);
    // ...mas o arquivo pode consertá-la.
    EngineParameters fixable = bad;
    CHECK(!tryRead("PaletteFinder::max_colors 25\n", fixable).threw && fixable.palette.maxColors == 25);
}

CNX_TEST(a_failed_read_never_changes_the_parameters) {
    cnxtest::SplitMix64 rng{33};
    EngineParameters p = randomParams(rng);
    const EngineParameters before = p;
    // Várias linhas boas e uma ruim no fim: nada pode ter vazado para `p`.
    std::string text = "BezierFitter::max_rounds 2\nPaletteFinder::max_colors 9\nContourSmoother::prior_strengths 77\nCoreEngine::seed 5\nPaletteFinder::edge_scale zero\n";
    const Outcome o = tryRead(text, p);
    CHECK(o.threw && o.line == 5);
    CHECK(same(p, before));
    // E uma regra violada só no fim também desfaz tudo.
    text = "BezierFitter::max_rounds 2\nCoreEngine::seed 5\nBezierFitter::final_sweeps -3\n";
    CHECK(tryRead(text, p).threw && same(p, before));
    // Stream que falha no meio (badbit) também não mexe.
    FailingBuf buf;
    std::istream broken(&buf);
    CHECK_THROWS(ParameterFileError, (void)ParameterFiler::readAPF(broken, p));
    CHECK(same(p, before));
    try {
        std::istream again(&buf);
        (void)ParameterFiler::readAPF(again, p);
    } catch (const ParameterFileError& e) {
        CHECK(e.line() == 0);
    }
}

CNX_TEST(strict_rejects_unknown_keys_and_lenient_collects_them_once_in_order) {
    const std::string text =
        "BezierFitter::max_rounds 4\n"
        "Futuro::chave_nova 1 2 3\n"
        "PaletteFinder::max_colors 6\n"
        "BezierFitter::nao_existe sim\n"
        "Futuro::chave_nova 9\n"
        "sem valor::\n";
    expectError("BezierFitter::max_rounds 4\nBezierFitter::nao_existe 1\n", 2, "chave desconhecida");
    expectError("Futuro::chave_nova 1\n", 1, "Futuro::chave_nova");
    // Estrito é o padrão.
    EngineParameters strict;
    CHECK(tryRead("Futuro::x 1\n", strict).threw);
    // Tolerante: aplica o que conhece, devolve o resto (sem repetir). Linha malformada continua erro.
    EngineParameters p;
    const Outcome o = tryRead(text.substr(0, text.find("sem valor")), p, Mode::Lenient);
    CHECK_CTX(!o.threw, o.message);
    CHECK((o.ignored == std::vector<std::string>{"Futuro::chave_nova", "BezierFitter::nao_existe"}));
    CHECK(p.fit.maxRounds == 4 && p.palette.maxColors == 6);
    expectError(text, 6, "chave malformada", Mode::Lenient);
    // Chave conhecida com valor ruim continua erro no modo tolerante, e repetição também.
    expectError("PaletteFinder::max_colors x\n", 1, "inteiro inválido", Mode::Lenient);
    expectError("PaletteFinder::max_colors 3\nPaletteFinder::max_colors 4\n", 2, "repetida", Mode::Lenient);
    // Chave desconhecida sem valor é só ignorada no modo tolerante.
    EngineParameters q;
    const Outcome o2 = tryRead("Futuro::sozinha\n", q, Mode::Lenient);
    CHECK(!o2.threw && o2.ignored.size() == 1);
    // Um erro de regra no modo tolerante ainda desfaz tudo.
    expectError("Futuro::a 1\nPaletteFinder::histogram_bins 1\n", 2, "histogram_bins", Mode::Lenient);
}

CNX_TEST(validate_reports_the_first_violation_and_the_writer_refuses_invalid_parameters) {
    CHECK(!ParameterFiler::validate(EngineParameters{}));
    const auto problem = [](const std::function<void(EngineParameters&)>& mutate) {
        EngineParameters p;
        mutate(p);
        return ParameterFiler::validate(p);
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(problem([&](auto& p) { p.preprocess.targetAreaPx = nan; }).value_or("").find("target_area_px") != std::string::npos);
    CHECK(problem([&](auto& p) { p.preprocess.blurSigmaPx = inf; }).value_or("").find("blur_sigma_px") != std::string::npos);
    CHECK(problem([&](auto& p) { p.smooth.spacingWeight[2] = nan; }).value_or("").find("spacing_weight") != std::string::npos);
    CHECK(problem([&](auto& p) { p.smooth.punctureCornerThresh = -inf; }).value_or("").find("puncture_corner_thresh") != std::string::npos);
    CHECK(problem([&](auto& p) { p.fit.statThreshFinal = nan; }).value_or("").find("stat_thresh_final") != std::string::npos);
    CHECK(problem([&](auto& p) { p.palette.maxColors = 1; }).value_or("").find("max_colors") != std::string::npos);
    // Enumerações fora da tabela (conversão forçada) também são recusadas.
    CHECK(problem([&](auto& p) { p.classifier.forcedProfile = static_cast<ImageProfile>(77); }).value_or("").find("forced_profile") != std::string::npos);
    CHECK(problem([&](auto& p) { p.smooth.measurementTypes[1] = static_cast<MeasurementType>(9); }).value_or("").find("measurement_types") != std::string::npos);
    // O escritor não produz nada quando recusa.
    EngineParameters bad;
    bad.fit.maxRounds = 0;
    std::ostringstream out;
    CHECK_THROWS(std::invalid_argument, ParameterFiler::writeAPF(out, bad));
    CHECK(out.str().empty());
    // E o que o escritor aceita, o leitor reabre (já coberto no aleatório); o que ele recusa, o leitor também.
    EngineParameters viaText;
    CHECK(tryRead("BezierFitter::max_rounds 0\n", viaText).threw);
}

CNX_TEST(file_round_trip_replaces_atomically_and_reports_io_errors_with_line_zero) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "cnx_apf_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path file = dir / "perfil.apf";
    const auto slurp = [](const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };

    cnxtest::SplitMix64 rng{55};
    const EngineParameters a = randomParams(rng);
    ParameterFiler::writeAPF(file, a);
    CHECK(fs::exists(file) && !fs::exists(dir / "perfil.apf.tmp"));
    EngineParameters back;
    CHECK(ParameterFiler::readAPF(file, back).empty());
    CHECK(same(back, a));
    CHECK(slurp(file) == toText(a));

    // Sobrescrever troca o conteúdo inteiro.
    const EngineParameters b = randomParams(rng);
    ParameterFiler::writeAPF(file, b);
    CHECK(slurp(file) == toText(b) && !fs::exists(dir / "perfil.apf.tmp"));

    // Parâmetros inválidos: lança e o arquivo anterior fica como estava, sem temporário.
    EngineParameters invalid = b;
    invalid.fit.maxRounds = 0;
    CHECK_THROWS(std::invalid_argument, ParameterFiler::writeAPF(file, invalid));
    CHECK(slurp(file) == toText(b) && !fs::exists(dir / "perfil.apf.tmp"));

    // Leitura de arquivo com erro: a mensagem de leitura traz a linha, params não muda.
    {
        std::ofstream broken(dir / "ruim.apf", std::ios::binary);
        broken << "BezierFitter::max_rounds 3\nPaletteFinder::max_colors nove\n";
    }
    EngineParameters keep = b;
    try {
        (void)ParameterFiler::readAPF(dir / "ruim.apf", keep);
        CHECK(false);
    } catch (const ParameterFileError& e) {
        CHECK(e.line() == 2);
    }
    CHECK(same(keep, b));

    // Arquivo inexistente e diretório inexistente: erro de E/S (linha 0).
    try {
        (void)ParameterFiler::readAPF(dir / "nao_existe.apf", keep);
        CHECK(false);
    } catch (const ParameterFileError& e) {
        CHECK(e.line() == 0);
    }
    try {
        ParameterFiler::writeAPF(dir / "sub" / "x.apf", a);
        CHECK(false);
    } catch (const ParameterFileError& e) {
        CHECK(e.line() == 0);
    }
    // Destino que é um diretório: a troca falha e o temporário não fica.
    fs::create_directories(dir / "pasta.apf");
    CHECK_THROWS(ParameterFileError, ParameterFiler::writeAPF(dir / "pasta.apf", a));
    CHECK(!fs::exists(dir / "pasta.apf.tmp") && fs::is_directory(dir / "pasta.apf"));

    // Stream de saída em falha.
    std::ostringstream failing;
    failing.setstate(std::ios::badbit);
    try {
        ParameterFiler::writeAPF(failing, a);
        CHECK(false);
    } catch (const ParameterFileError& e) {
        CHECK(e.line() == 0);
    }
    fs::remove_all(dir);
}

CNX_TEST(a_profile_file_drives_the_fingerprints_like_the_ui_would) {
    // Um .apf que só muda o ajuste suja só o ajuste (mesma lógica do recálculo inteligente).
    const EngineParameters base;
    EngineParameters p;
    CHECK(!tryRead("BezierFitter::stat_thresh_final 0.9\n", p).threw);
    CHECK(base.palette.fingerprint({}) == p.palette.fingerprint({}) && base.segment.fingerprint({}) == p.segment.fingerprint({}) &&
          base.smooth.fingerprint({}) == p.smooth.fingerprint({}) && base.fit.fingerprint({}) != p.fit.fingerprint({}));
    // Mesmo arquivo lido duas vezes: mesma impressão digital.
    EngineParameters q;
    CHECK(!tryRead("BezierFitter::stat_thresh_final 0.9\n", q).threw);
    CHECK(q.fit.fingerprint({}) == p.fit.fingerprint({}));
}

int main() { return cnxtest::runAll(); }
