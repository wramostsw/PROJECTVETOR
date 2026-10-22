// src/engine/ParameterFiler.cpp — leitura e escrita de perfis .apf (`Classe::parametro valor`).
//
// Uma única tabela de campos (chave, leitor, escritor) serve às duas direções; as regras de validação
// (faixas e relações entre campos) também são únicas: o leitor as aplica depois de ler, o escritor antes de
// escrever. Assim, tudo o que o escritor produz o leitor reabre, e vice-versa (parameter_filer_test).
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <istream>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#include "cnx/engine/Parameters.h"

namespace cnx {
namespace {

using Tokens = std::vector<std::string_view>;

/// Valor de campo ilegível (tipo, faixa de inteiro, nome de enumeração…); o leitor acrescenta a linha.
class FieldError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

constexpr std::size_t kMaxLineChars = 4096;
constexpr std::string_view kBom = "\xEF\xBB\xBF";

// ---------------------------------------------------------------- valores escalares

std::string quoted(std::string_view s) { return "'" + std::string(s) + "'"; }

template <class T>
T parseInteger(std::string_view s) {
    T v{};
    const char* end = s.data() + s.size();
    const auto [ptr, ec] = std::from_chars(s.data(), end, v);
    if (ec == std::errc::result_out_of_range) throw FieldError("inteiro fora do intervalo: " + quoted(s));
    if (ec != std::errc{} || ptr != end) throw FieldError("inteiro inválido: " + quoted(s));
    return v;
}

double parseDouble(std::string_view s) {
    double v = 0.0;
    const char* end = s.data() + s.size();
    const auto [ptr, ec] = std::from_chars(s.data(), end, v, std::chars_format::general);
    if (ec == std::errc::result_out_of_range) throw FieldError("real fora do intervalo: " + quoted(s));
    if (ec != std::errc{} || ptr != end) throw FieldError("real inválido: " + quoted(s));
    if (!std::isfinite(v)) throw FieldError("real não finito: " + quoted(s));
    return v;
}

bool parseBool(std::string_view s) {
    if (s == "true" || s == "1") return true;
    if (s == "false" || s == "0") return false;
    throw FieldError("booleano inválido (use true/false): " + quoted(s));
}

/// to_chars sem formato: a menor representação que reconstrói exatamente o mesmo double.
std::string formatDouble(double v) {
    std::array<char, 40> buf{};
    const auto r = std::to_chars(buf.data(), buf.data() + buf.size(), v);
    return std::string(buf.data(), r.ptr);
}

std::string_view single(const Tokens& t) {
    if (t.size() != 1) throw FieldError("esperado 1 valor, encontrados " + std::to_string(t.size()));
    return t[0];
}

template <class E, std::size_t N>
using EnumTable = std::array<std::pair<std::string_view, E>, N>;

template <class E, std::size_t N>
E parseEnum(std::string_view s, const EnumTable<E, N>& table) {
    for (const auto& [name, value] : table) {
        if (name == s) return value;
    }
    std::string known;
    for (const auto& entry : table) known += (known.empty() ? "" : ", ") + std::string(entry.first);
    throw FieldError("nome desconhecido " + quoted(s) + " (válidos: " + known + ")");
}

template <class E, std::size_t N>
std::string_view enumName(E value, const EnumTable<E, N>& table) {
    for (const auto& [name, v] : table) {
        if (v == value) return name;
    }
    throw FieldError("valor de enumeração fora da tabela");
}

const EnumTable<ImageProfile, 10> kProfiles{{
    {"EP_LOGO_HIG_AA", ImageProfile::LogoHighAA},
    {"EP_LOGO_MED_AA", ImageProfile::LogoMedAA},
    {"EP_LOGO_LOW_AA", ImageProfile::LogoLowAA},
    {"EP_LOGO_HIG", ImageProfile::LogoHigh},
    {"EP_LOGO_MED", ImageProfile::LogoMed},
    {"EP_LOGO_LOW", ImageProfile::LogoLow},
    {"EP_PHOTO_HIG", ImageProfile::PhotoHigh},
    {"EP_PHOTO_MED", ImageProfile::PhotoMed},
    {"EP_PHOTO_LOW", ImageProfile::PhotoLow},
    {"EP_CUSTOM", ImageProfile::Custom},
}};

const EnumTable<MeasurementType, 3> kMeasurements{{
    {"spring", MeasurementType::Spring},
    {"generative", MeasurementType::Generative},
    {"none", MeasurementType::None},
}};

// ---------------------------------------------------------------- tabela de campos

struct Field {
    std::string key;
    std::function<void(EngineParameters&, const Tokens&)> read;  ///< lança FieldError
    std::function<std::string(const EngineParameters&)> write;   ///< lança FieldError (enumeração inválida)
};

// `acc` é um lambda genérico `[](auto& p) -> auto& { return p.palette.maxColors; }`: serve a EngineParameters
// com e sem const.

template <class T, class Acc>
Field integerField(const char* key, Acc acc) {
    return {key, [acc](EngineParameters& p, const Tokens& t) { acc(p) = parseInteger<T>(single(t)); },
            [acc](const EngineParameters& p) { return std::to_string(acc(p)); }};
}

template <class Acc>
Field doubleField(const char* key, Acc acc) {
    return {key, [acc](EngineParameters& p, const Tokens& t) { acc(p) = parseDouble(single(t)); },
            [acc](const EngineParameters& p) { return formatDouble(acc(p)); }};
}

template <class Acc>
Field boolField(const char* key, Acc acc) {
    return {key, [acc](EngineParameters& p, const Tokens& t) { acc(p) = parseBool(single(t)); },
            [acc](const EngineParameters& p) { return std::string(acc(p) ? "true" : "false"); }};
}

template <class Acc>
Field optionalIntField(const char* key, Acc acc) {
    return {key,
            [acc](EngineParameters& p, const Tokens& t) {
                const std::string_view s = single(t);
                if (s == "none") {
                    acc(p).reset();
                } else {
                    acc(p) = parseInteger<int>(s);
                }
            },
            [acc](const EngineParameters& p) { return acc(p) ? std::to_string(*acc(p)) : std::string("none"); }};
}

template <class E, std::size_t N, class Acc>
Field enumField(const char* key, Acc acc, const EnumTable<E, N>& table) {
    return {key, [acc, &table](EngineParameters& p, const Tokens& t) { acc(p) = parseEnum(single(t), table); },
            [acc, &table](const EngineParameters& p) { return std::string(enumName(acc(p), table)); }};
}

template <std::size_t K, class Acc>
Field doubleListField(const char* key, Acc acc) {
    return {key,
            [acc](EngineParameters& p, const Tokens& t) {
                if (t.size() != K) throw FieldError("esperados " + std::to_string(K) + " valores, encontrados " + std::to_string(t.size()));
                std::array<double, K> values{};
                for (std::size_t i = 0; i < K; ++i) values[i] = parseDouble(t[i]);
                acc(p) = values;
            },
            [acc](const EngineParameters& p) {
                std::string out;
                for (const double v : acc(p)) out += (out.empty() ? "" : " ") + formatDouble(v);
                return out;
            }};
}

template <class E, std::size_t N, std::size_t K, class Acc>
Field enumListField(const char* key, Acc acc, const EnumTable<E, N>& table) {
    return {key,
            [acc, &table](EngineParameters& p, const Tokens& t) {
                if (t.size() != K) throw FieldError("esperados " + std::to_string(K) + " valores, encontrados " + std::to_string(t.size()));
                std::array<E, K> values{};
                for (std::size_t i = 0; i < K; ++i) values[i] = parseEnum(t[i], table);
                acc(p) = values;
            },
            [acc, &table](const EngineParameters& p) {
                std::string out;
                for (const E v : acc(p)) out += (out.empty() ? "" : " ") + std::string(enumName(v, table));
                return out;
            }};
}

// Classes das chaves: as da especificação (§1–§6). PixelSegmenter e SubPixelSegmenter dividem o SegmentParams.
std::vector<Field> buildFields() {
    std::vector<Field> f;
    f.push_back(integerField<std::uint64_t>("CoreEngine::seed", [](auto& p) -> auto& { return p.seed; }));

    f.push_back(doubleField("Preprocessor::target_area_px", [](auto& p) -> auto& { return p.preprocess.targetAreaPx; }));
    f.push_back(doubleField("Preprocessor::blur_sigma_px", [](auto& p) -> auto& { return p.preprocess.blurSigmaPx; }));
    f.push_back(boolField("Preprocessor::flatten_alpha", [](auto& p) -> auto& { return p.preprocess.flattenAlpha; }));
    f.push_back(boolField("Preprocessor::allow_upscale", [](auto& p) -> auto& { return p.preprocess.allowUpscale; }));

    f.push_back(boolField("ParameterClassifier::auto_classify", [](auto& p) -> auto& { return p.classifier.autoClassify; }));
    f.push_back(enumField("ParameterClassifier::forced_profile", [](auto& p) -> auto& { return p.classifier.forcedProfile; }, kProfiles));

    f.push_back(integerField<int>("PaletteFinder::min_colors", [](auto& p) -> auto& { return p.palette.minColors; }));
    f.push_back(integerField<int>("PaletteFinder::max_colors", [](auto& p) -> auto& { return p.palette.maxColors; }));
    f.push_back(integerField<int>("PaletteFinder::kmeans_max_rounds", [](auto& p) -> auto& { return p.palette.kmeansMaxRounds; }));
    f.push_back(integerField<int>("PaletteFinder::histogram_bins", [](auto& p) -> auto& { return p.palette.histogramBins; }));
    f.push_back(doubleField("PaletteFinder::color_penalty", [](auto& p) -> auto& { return p.palette.colorPenalty; }));
    f.push_back(optionalIntField("PaletteFinder::forced_num_colors", [](auto& p) -> auto& { return p.palette.forcedNumColors; }));
    f.push_back(doubleField("PaletteFinder::edge_scale", [](auto& p) -> auto& { return p.palette.edgeScale; }));
    f.push_back(doubleField("PaletteFinder::exact_color_min_share", [](auto& p) -> auto& { return p.palette.exactColorMinShare; }));
    f.push_back(integerField<int>("PaletteFinder::refine_rounds", [](auto& p) -> auto& { return p.palette.refineRounds; }));

    f.push_back(integerField<int>("PixelSegmenter::quad_tree_block", [](auto& p) -> auto& { return p.segment.quadTreeBlock; }));
    f.push_back(doubleField("PixelSegmenter::quad_tree_variance", [](auto& p) -> auto& { return p.segment.quadTreeVariance; }));
    f.push_back(doubleField("PixelSegmenter::merge_thresh_start", [](auto& p) -> auto& { return p.segment.mergeThreshStart; }));
    f.push_back(doubleField("PixelSegmenter::merge_thresh_growth", [](auto& p) -> auto& { return p.segment.mergeThreshGrowth; }));
    f.push_back(doubleField("PixelSegmenter::merge_thresh_max", [](auto& p) -> auto& { return p.segment.mergeThreshMax; }));
    f.push_back(integerField<int>("PixelSegmenter::min_region_pixels", [](auto& p) -> auto& { return p.segment.minRegionPixels; }));
    f.push_back(doubleField("PixelSegmenter::what_if_factor", [](auto& p) -> auto& { return p.segment.whatIfFactor; }));
    f.push_back(integerField<int>("PixelSegmenter::final_sweep_cap", [](auto& p) -> auto& { return p.segment.finalSweepCap; }));

    f.push_back(integerField<int>("SubPixelSegmenter::micro_region_max_pixels", [](auto& p) -> auto& { return p.segment.microRegionMaxPixels; }));
    f.push_back(doubleField("SubPixelSegmenter::beach_line_tolerance", [](auto& p) -> auto& { return p.segment.beachLineTolerance; }));
    f.push_back(doubleField("SubPixelSegmenter::beach_min_contrast", [](auto& p) -> auto& { return p.segment.beachMinContrast; }));
    f.push_back(doubleField("SubPixelSegmenter::beach_neighbor_share", [](auto& p) -> auto& { return p.segment.beachNeighborShare; }));
    f.push_back(doubleField("SubPixelSegmenter::shadow_variance", [](auto& p) -> auto& { return p.segment.shadowVariance; }));
    f.push_back(doubleField("SubPixelSegmenter::shadow_smooth_ratio", [](auto& p) -> auto& { return p.segment.shadowSmoothRatio; }));
    f.push_back(integerField<int>("SubPixelSegmenter::shadow_min_pixels", [](auto& p) -> auto& { return p.segment.shadowMinPixels; }));
    f.push_back(integerField<int>("SubPixelSegmenter::shadow_blur_radius", [](auto& p) -> auto& { return p.segment.shadowBlurRadius; }));

    f.push_back((enumListField<MeasurementType, 3, SmoothParams::kPhases>("ContourSmoother::measurement_types",
                                                                         [](auto& p) -> auto& { return p.smooth.measurementTypes; }, kMeasurements)));
    f.push_back(doubleListField<SmoothParams::kPhases>("ContourSmoother::spacing_weight", [](auto& p) -> auto& { return p.smooth.spacingWeight; }));
    f.push_back(doubleField("ContourSmoother::fidelity_weight", [](auto& p) -> auto& { return p.smooth.fidelityWeight; }));
    f.push_back(integerField<int>("ContourSmoother::prior_type", [](auto& p) -> auto& { return p.smooth.priorType; }));
    f.push_back(doubleField("ContourSmoother::prior_strengths", [](auto& p) -> auto& { return p.smooth.priorStrength; }));
    f.push_back(doubleField("ContourSmoother::air_pressure_weight", [](auto& p) -> auto& { return p.smooth.airPressureWeight; }));
    f.push_back(integerField<int>("ContourSmoother::air_pressure_max_nodes", [](auto& p) -> auto& { return p.smooth.airPressureMaxNodes; }));
    f.push_back(doubleField("ContourSmoother::border_anchor_weight", [](auto& p) -> auto& { return p.smooth.borderAnchorWeight; }));
    f.push_back(boolField("ContourSmoother::border_hard_constraint", [](auto& p) -> auto& { return p.smooth.borderHardConstraint; }));
    f.push_back(doubleField("ContourSmoother::fidelity_free_radius_px", [](auto& p) -> auto& { return p.smooth.fidelityFreeRadiusPx; }));
    f.push_back(integerField<int>("ContourSmoother::quarantine_rounds", [](auto& p) -> auto& { return p.smooth.quarantineRounds; }));
    f.push_back(doubleField("ContourSmoother::min_segment_px", [](auto& p) -> auto& { return p.smooth.minSegmentPx; }));
    f.push_back(doubleField("ContourSmoother::puncture_corner_thresh", [](auto& p) -> auto& { return p.smooth.punctureCornerThresh; }));
    f.push_back(doubleField("ContourSmoother::puncture_flatness_thresh", [](auto& p) -> auto& { return p.smooth.punctureFlatnessThresh; }));
    f.push_back(doubleField("ContourSmoother::puncture_arm_px", [](auto& p) -> auto& { return p.smooth.punctureArmPx; }));
    f.push_back(integerField<int>("ContourSmoother::puncture_on_phase", [](auto& p) -> auto& { return p.smooth.punctureOnPhase; }));
    f.push_back(integerField<int>("ContourSmoother::max_cg_iterations", [](auto& p) -> auto& { return p.smooth.maxCgIterations; }));
    f.push_back(doubleField("ContourSmoother::max_node_step_px", [](auto& p) -> auto& { return p.smooth.maxNodeStepPx; }));
    f.push_back(doubleField("ContourSmoother::gradient_tolerance", [](auto& p) -> auto& { return p.smooth.gradientTolerance; }));
    f.push_back(doubleField("ContourSmoother::energy_tolerance", [](auto& p) -> auto& { return p.smooth.energyTolerance; }));

    f.push_back(doubleField("BezierFitter::stat_thresh_initial", [](auto& p) -> auto& { return p.fit.statThreshInitial; }));
    f.push_back(doubleField("BezierFitter::stat_thresh_final", [](auto& p) -> auto& { return p.fit.statThreshFinal; }));
    f.push_back(integerField<int>("BezierFitter::max_rounds", [](auto& p) -> auto& { return p.fit.maxRounds; }));
    f.push_back(integerField<int>("BezierFitter::initial_fragment_nodes", [](auto& p) -> auto& { return p.fit.initialFragmentNodes; }));
    f.push_back(integerField<int>("BezierFitter::newton_max_iterations", [](auto& p) -> auto& { return p.fit.newtonMaxIterations; }));
    f.push_back(integerField<int>("BezierFitter::final_sweeps", [](auto& p) -> auto& { return p.fit.finalSweeps; }));
    f.push_back(boolField("BezierFitter::enforce_g1", [](auto& p) -> auto& { return p.fit.enforceG1; }));
    return f;
}

const std::vector<Field>& fields() {
    static const std::vector<Field> table = buildFields();
    return table;
}

const std::map<std::string_view, const Field*>& fieldIndex() {
    static const std::map<std::string_view, const Field*> index = [] {
        std::map<std::string_view, const Field*> m;
        for (const Field& f : fields()) m.emplace(f.key, &f);
        return m;
    }();
    return index;
}

// ---------------------------------------------------------------- regras de validação

struct Rule {
    std::vector<std::string_view> keys;  ///< campos envolvidos (a linha do erro é a do último que o arquivo definiu)
    std::string message;
    std::function<bool(const EngineParameters&)> ok;
};

bool atLeast(double x, double lo) { return std::isfinite(x) && x >= lo; }
bool above(double x, double lo) { return std::isfinite(x) && x > lo; }
bool within(double x, double lo, double hi) { return std::isfinite(x) && x >= lo && x <= hi; }

// Espelha o que cada módulo exige ao construir-se (PaletteFinder, PixelSegmenter, ContourSmoother, BezierFitter,
// preprocess…) e acrescenta o que é absurdo (contagens negativas, reais não finitos). Cada campo real tem regra.
const std::vector<Rule>& rules() {
    static const std::vector<Rule> table = [] {
        std::vector<Rule> r;
        const auto add = [&r](std::vector<std::string_view> keys, std::string message, std::function<bool(const EngineParameters&)> ok) {
            r.push_back({std::move(keys), std::move(message), std::move(ok)});
        };
        add({"Preprocessor::target_area_px"}, "Preprocessor::target_area_px deve ser finito e > 0", [](const EngineParameters& p) { return above(p.preprocess.targetAreaPx, 0.0); });
        add({"Preprocessor::blur_sigma_px"}, "Preprocessor::blur_sigma_px deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.preprocess.blurSigmaPx, 0.0); });

        add({"PaletteFinder::min_colors"}, "PaletteFinder::min_colors deve ser >= 1", [](const EngineParameters& p) { return p.palette.minColors >= 1; });
        add({"PaletteFinder::max_colors", "PaletteFinder::min_colors"}, "PaletteFinder::max_colors deve ser >= min_colors", [](const EngineParameters& p) { return p.palette.maxColors >= p.palette.minColors; });
        add({"PaletteFinder::kmeans_max_rounds"}, "PaletteFinder::kmeans_max_rounds deve ser >= 0", [](const EngineParameters& p) { return p.palette.kmeansMaxRounds >= 0; });
        add({"PaletteFinder::histogram_bins"}, "PaletteFinder::histogram_bins deve estar em [2, 256]", [](const EngineParameters& p) { return p.palette.histogramBins >= 2 && p.palette.histogramBins <= 256; });
        add({"PaletteFinder::color_penalty"}, "PaletteFinder::color_penalty deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.palette.colorPenalty, 0.0); });
        add({"PaletteFinder::forced_num_colors"}, "PaletteFinder::forced_num_colors deve ser >= 1 (ou none)", [](const EngineParameters& p) { return !p.palette.forcedNumColors || *p.palette.forcedNumColors >= 1; });
        add({"PaletteFinder::edge_scale"}, "PaletteFinder::edge_scale deve ser finito e > 0", [](const EngineParameters& p) { return above(p.palette.edgeScale, 0.0); });
        add({"PaletteFinder::exact_color_min_share"}, "PaletteFinder::exact_color_min_share deve estar em [0, 1]", [](const EngineParameters& p) { return within(p.palette.exactColorMinShare, 0.0, 1.0); });
        add({"PaletteFinder::refine_rounds"}, "PaletteFinder::refine_rounds deve ser >= 0", [](const EngineParameters& p) { return p.palette.refineRounds >= 0; });

        add({"PixelSegmenter::quad_tree_block"}, "PixelSegmenter::quad_tree_block deve ser >= 1", [](const EngineParameters& p) { return p.segment.quadTreeBlock >= 1; });
        add({"PixelSegmenter::quad_tree_variance"}, "PixelSegmenter::quad_tree_variance deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.segment.quadTreeVariance, 0.0); });
        add({"PixelSegmenter::merge_thresh_start"}, "PixelSegmenter::merge_thresh_start deve ser finito e > 0", [](const EngineParameters& p) { return above(p.segment.mergeThreshStart, 0.0); });
        add({"PixelSegmenter::merge_thresh_max", "PixelSegmenter::merge_thresh_start"}, "PixelSegmenter::merge_thresh_max deve ser finito e >= merge_thresh_start", [](const EngineParameters& p) { return std::isfinite(p.segment.mergeThreshMax) && p.segment.mergeThreshMax >= p.segment.mergeThreshStart; });
        add({"PixelSegmenter::merge_thresh_growth"}, "PixelSegmenter::merge_thresh_growth deve ser finito e > 1", [](const EngineParameters& p) { return above(p.segment.mergeThreshGrowth, 1.0); });
        add({"PixelSegmenter::min_region_pixels"}, "PixelSegmenter::min_region_pixels deve ser >= 0", [](const EngineParameters& p) { return p.segment.minRegionPixels >= 0; });
        add({"PixelSegmenter::what_if_factor"}, "PixelSegmenter::what_if_factor deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.segment.whatIfFactor, 0.0); });
        add({"PixelSegmenter::final_sweep_cap"}, "PixelSegmenter::final_sweep_cap deve ser >= 0", [](const EngineParameters& p) { return p.segment.finalSweepCap >= 0; });

        add({"SubPixelSegmenter::micro_region_max_pixels"}, "SubPixelSegmenter::micro_region_max_pixels deve ser >= 0", [](const EngineParameters& p) { return p.segment.microRegionMaxPixels >= 0; });
        add({"SubPixelSegmenter::beach_line_tolerance"}, "SubPixelSegmenter::beach_line_tolerance deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.segment.beachLineTolerance, 0.0); });
        add({"SubPixelSegmenter::beach_min_contrast"}, "SubPixelSegmenter::beach_min_contrast deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.segment.beachMinContrast, 0.0); });
        add({"SubPixelSegmenter::beach_neighbor_share"}, "SubPixelSegmenter::beach_neighbor_share deve estar em [0, 1]", [](const EngineParameters& p) { return within(p.segment.beachNeighborShare, 0.0, 1.0); });
        add({"SubPixelSegmenter::shadow_variance"}, "SubPixelSegmenter::shadow_variance deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.segment.shadowVariance, 0.0); });
        add({"SubPixelSegmenter::shadow_smooth_ratio"}, "SubPixelSegmenter::shadow_smooth_ratio deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.segment.shadowSmoothRatio, 0.0); });
        add({"SubPixelSegmenter::shadow_min_pixels"}, "SubPixelSegmenter::shadow_min_pixels deve ser >= 0", [](const EngineParameters& p) { return p.segment.shadowMinPixels >= 0; });
        add({"SubPixelSegmenter::shadow_blur_radius"}, "SubPixelSegmenter::shadow_blur_radius deve ser >= 0", [](const EngineParameters& p) { return p.segment.shadowBlurRadius >= 0; });

        add({"ContourSmoother::spacing_weight"}, "ContourSmoother::spacing_weight deve ter valores finitos e >= 0", [](const EngineParameters& p) {
            return std::all_of(p.smooth.spacingWeight.begin(), p.smooth.spacingWeight.end(), [](double w) { return atLeast(w, 0.0); });
        });
        add({"ContourSmoother::fidelity_weight"}, "ContourSmoother::fidelity_weight deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.smooth.fidelityWeight, 0.0); });
        add({"ContourSmoother::prior_type"}, "ContourSmoother::prior_type: só 3 (suavidade angular) está implementado", [](const EngineParameters& p) { return p.smooth.priorType == 3; });
        add({"ContourSmoother::prior_strengths"}, "ContourSmoother::prior_strengths deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.smooth.priorStrength, 0.0); });
        add({"ContourSmoother::air_pressure_weight"}, "ContourSmoother::air_pressure_weight deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.smooth.airPressureWeight, 0.0); });
        add({"ContourSmoother::air_pressure_max_nodes"}, "ContourSmoother::air_pressure_max_nodes deve ser >= 0", [](const EngineParameters& p) { return p.smooth.airPressureMaxNodes >= 0; });
        add({"ContourSmoother::border_anchor_weight"}, "ContourSmoother::border_anchor_weight deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.smooth.borderAnchorWeight, 0.0); });
        add({"ContourSmoother::fidelity_free_radius_px"}, "ContourSmoother::fidelity_free_radius_px deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.smooth.fidelityFreeRadiusPx, 0.0); });
        add({"ContourSmoother::quarantine_rounds"}, "ContourSmoother::quarantine_rounds deve ser >= 0", [](const EngineParameters& p) { return p.smooth.quarantineRounds >= 0; });
        add({"ContourSmoother::min_segment_px"}, "ContourSmoother::min_segment_px deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.smooth.minSegmentPx, 0.0); });
        add({"ContourSmoother::puncture_corner_thresh"}, "ContourSmoother::puncture_corner_thresh deve ser finito (<= 0 desativa)", [](const EngineParameters& p) { return std::isfinite(p.smooth.punctureCornerThresh); });
        add({"ContourSmoother::puncture_flatness_thresh"}, "ContourSmoother::puncture_flatness_thresh deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.smooth.punctureFlatnessThresh, 0.0); });
        add({"ContourSmoother::puncture_arm_px"}, "ContourSmoother::puncture_arm_px deve ser finito e > 0", [](const EngineParameters& p) { return above(p.smooth.punctureArmPx, 0.0); });
        add({"ContourSmoother::puncture_on_phase"}, "ContourSmoother::puncture_on_phase deve estar em [0, 2]", [](const EngineParameters& p) { return p.smooth.punctureOnPhase >= 0 && p.smooth.punctureOnPhase < SmoothParams::kPhases; });
        add({"ContourSmoother::max_cg_iterations"}, "ContourSmoother::max_cg_iterations deve ser >= 0", [](const EngineParameters& p) { return p.smooth.maxCgIterations >= 0; });
        add({"ContourSmoother::max_node_step_px"}, "ContourSmoother::max_node_step_px deve ser finito e > 0", [](const EngineParameters& p) { return above(p.smooth.maxNodeStepPx, 0.0); });
        add({"ContourSmoother::gradient_tolerance"}, "ContourSmoother::gradient_tolerance deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.smooth.gradientTolerance, 0.0); });
        add({"ContourSmoother::energy_tolerance"}, "ContourSmoother::energy_tolerance deve ser finito e >= 0", [](const EngineParameters& p) { return atLeast(p.smooth.energyTolerance, 0.0); });

        add({"BezierFitter::stat_thresh_initial"}, "BezierFitter::stat_thresh_initial deve ser finito e > 0", [](const EngineParameters& p) { return above(p.fit.statThreshInitial, 0.0); });
        add({"BezierFitter::stat_thresh_final", "BezierFitter::stat_thresh_initial"}, "BezierFitter::stat_thresh_final deve ser finito e >= stat_thresh_initial", [](const EngineParameters& p) { return std::isfinite(p.fit.statThreshFinal) && p.fit.statThreshFinal >= p.fit.statThreshInitial; });
        add({"BezierFitter::max_rounds"}, "BezierFitter::max_rounds deve ser >= 1", [](const EngineParameters& p) { return p.fit.maxRounds >= 1; });
        add({"BezierFitter::initial_fragment_nodes"}, "BezierFitter::initial_fragment_nodes deve ser >= 2", [](const EngineParameters& p) { return p.fit.initialFragmentNodes >= 2; });
        add({"BezierFitter::newton_max_iterations"}, "BezierFitter::newton_max_iterations deve ser >= 0", [](const EngineParameters& p) { return p.fit.newtonMaxIterations >= 0; });
        add({"BezierFitter::final_sweeps"}, "BezierFitter::final_sweeps deve ser >= 0", [](const EngineParameters& p) { return p.fit.finalSweeps >= 0; });
        return r;
    }();
    return table;
}

// ---------------------------------------------------------------- análise de linhas

constexpr std::string_view kBlanks = " \t";

bool isNameChar(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

bool isName(std::string_view s) { return !s.empty() && !(s.front() >= '0' && s.front() <= '9') && std::all_of(s.begin(), s.end(), isNameChar); }

/// `Classe::nome`, ambos identificadores.
bool wellFormedKey(std::string_view key) {
    const std::size_t sep = key.find("::");
    return sep != std::string_view::npos && isName(key.substr(0, sep)) && isName(key.substr(sep + 2));
}

Tokens split(std::string_view s) {
    Tokens out;
    std::size_t i = 0;
    while (i < s.size()) {
        i = s.find_first_not_of(kBlanks, i);
        if (i == std::string_view::npos) break;
        const std::size_t end = s.find_first_of(kBlanks, i);
        out.push_back(s.substr(i, end == std::string_view::npos ? std::string_view::npos : end - i));
        if (end == std::string_view::npos) break;
        i = end;
    }
    return out;
}

std::string describe(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

std::optional<std::string> findProblem(const EngineParameters& params) {
    for (const Rule& rule : rules()) {
        if (!rule.ok(params)) return rule.message;
    }
    for (const Field& f : fields()) {
        try {
            (void)f.write(params);
        } catch (const FieldError& e) {
            return f.key + ": " + e.what();
        }
    }
    return std::nullopt;
}

}  // namespace

// ---------------------------------------------------------------- API

std::optional<std::string> ParameterFiler::validate(const EngineParameters& params) { return findProblem(params); }

std::vector<std::string> ParameterFiler::readAPF(std::istream& in, EngineParameters& params, Mode mode) {
    EngineParameters work = params;  // transacional: só vira `params` se o arquivo inteiro for aceito
    std::map<std::string, int, std::less<>> definedAt;
    std::vector<std::string> ignored;

    std::string raw;
    int lineNo = 0;
    while (std::getline(in, raw)) {
        ++lineNo;
        std::string_view line = raw;
        if (lineNo == 1 && line.substr(0, kBom.size()) == kBom) line.remove_prefix(kBom.size());
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.size() > kMaxLineChars) throw ParameterFileError("linha longa demais (máximo " + std::to_string(kMaxLineChars) + " caracteres)", lineNo);

        const std::size_t first = line.find_first_not_of(kBlanks);
        if (first == std::string_view::npos || line[first] == '#') continue;
        if (std::any_of(line.begin(), line.end(), [](char c) { return static_cast<unsigned char>(c) < 0x20 && c != '\t'; })) {
            throw ParameterFileError("caractere de controle na linha", lineNo);
        }
        line = line.substr(first);

        const std::size_t keyEnd = line.find_first_of(kBlanks);
        const std::string_view key = line.substr(0, keyEnd);
        if (!wellFormedKey(key)) throw ParameterFileError("chave malformada " + quoted(key) + " (esperado Classe::parametro)", lineNo);

        const auto it = fieldIndex().find(key);
        if (it == fieldIndex().end()) {
            if (mode == Mode::Strict) throw ParameterFileError("chave desconhecida " + quoted(key), lineNo);
            if (std::find(ignored.begin(), ignored.end(), key) == ignored.end()) ignored.emplace_back(key);
            continue;
        }
        if (const auto dup = definedAt.find(key); dup != definedAt.end()) {
            throw ParameterFileError("chave repetida " + quoted(key) + " (já definida na linha " + std::to_string(dup->second) + ")", lineNo);
        }
        const Tokens values = keyEnd == std::string_view::npos ? Tokens{} : split(line.substr(keyEnd));
        if (values.empty()) throw ParameterFileError("falta o valor de " + quoted(key), lineNo);
        try {
            it->second->read(work, values);
        } catch (const FieldError& e) {
            throw ParameterFileError(std::string(key) + ": " + e.what(), lineNo);
        }
        definedAt.emplace(std::string(key), lineNo);
    }
    if (in.bad()) throw ParameterFileError("falha de leitura", 0);

    for (const Rule& rule : rules()) {
        if (rule.ok(work)) continue;
        int where = 0;  // a linha do último campo envolvido que o arquivo definiu; 0 se nenhum
        for (const std::string_view k : rule.keys) {
            if (const auto d = definedAt.find(k); d != definedAt.end()) where = std::max(where, d->second);
        }
        throw ParameterFileError(rule.message + (where == 0 ? " (valores anteriores ao arquivo)" : ""), where);
    }
    if (const auto problem = findProblem(work)) throw ParameterFileError(*problem, 0);

    params = std::move(work);
    return ignored;
}

std::vector<std::string> ParameterFiler::readAPF(const std::filesystem::path& file, EngineParameters& params, Mode mode) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw ParameterFileError("não foi possível abrir " + describe(file), 0);
    return readAPF(in, params, mode);
}

void ParameterFiler::writeAPF(std::ostream& out, const EngineParameters& params) {
    if (const auto problem = findProblem(params)) throw std::invalid_argument("ParameterFiler::writeAPF: " + *problem);
    std::string text = "# cnx .apf v1\n# Classe::parametro valor (linhas vazias e iniciadas por '#' sao ignoradas)\n";
    std::string_view lastClass;
    for (const Field& f : fields()) {
        const std::string_view cls = std::string_view(f.key).substr(0, f.key.find("::"));
        if (cls != lastClass) {
            text += "\n# ";
            text += cls;
            text += '\n';
            lastClass = cls;
        }
        text += f.key;
        text += ' ';
        text += f.write(params);
        text += '\n';
    }
    out << text;
    out.flush();
    if (!out) throw ParameterFileError("falha ao escrever o .apf", 0);
}

void ParameterFiler::writeAPF(const std::filesystem::path& file, const EngineParameters& params) {
    std::ostringstream buffer;
    writeAPF(buffer, params);  // valida antes de tocar no disco

    std::filesystem::path temp = file;
    temp += ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) throw ParameterFileError("não foi possível criar " + describe(temp), 0);
        out << buffer.str();
        out.flush();
        if (!out) {
            out.close();
            std::error_code ignore;
            std::filesystem::remove(temp, ignore);
            throw ParameterFileError("falha ao escrever " + describe(temp), 0);
        }
    }
    std::error_code ec;
    std::filesystem::rename(temp, file, ec);  // substitui o destino de uma vez
    if (ec) {
        std::error_code ignore;
        std::filesystem::remove(temp, ignore);
        throw ParameterFileError("não foi possível gravar " + describe(file) + ": " + ec.message(), 0);
    }
}

}  // namespace cnx
