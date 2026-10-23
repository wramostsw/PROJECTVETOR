// src/engine/CoreEngine.cpp — orquestrador do pipeline: impressões digitais, recálculo inteligente,
// cancelamento e progresso (docs/ARQUITETURA.md §1.3, §1.6, §1.17).
#include "cnx/engine/CoreEngine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "cnx/classify/Classifier.h"
#include "cnx/fit/BezierFitter.h"
#include "cnx/palette/PaletteFinder.h"
#include "cnx/preprocess/Preprocessor.h"
#include "cnx/segment/PixelMap.h"
#include "cnx/segment/PixelSegmenter.h"
#include "cnx/segment/SubPixelSegmenter.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/vector/ContourSmoother.h"
#include "cnx/vector/EdgeAnchors.h"
#include "cnx/vector/GenerativeModel.h"
#include "cnx/vector/VectorImage.h"

namespace cnx {

std::string_view toString(Stage s) noexcept {
    switch (s) {
        case Stage::LoadImage: return "LoadImage";
        case Stage::Classify: return "Classify";
        case Stage::Palette: return "Palette";
        case Stage::Segment: return "Segment";
        case Stage::SmoothContours: return "SmoothContours";
        case Stage::FitBezier: return "FitBezier";
        case Stage::Output: return "Output";
    }
    return "?";
}

namespace {

constexpr std::array<Stage, kStageCount> kStages{Stage::LoadImage, Stage::Classify,       Stage::Palette,  Stage::Segment,
                                                 Stage::SmoothContours, Stage::FitBezier, Stage::Output};

/// Edição manual da segmentação (§4): vale enquanto a impressão digital da paleta for a do momento da edição.
struct SegmentationOverride {
    std::shared_ptr<const PixelMap> map;
    Fingerprint editsHash;
    Fingerprint upstream;
};

/// Cópia da configuração tomada no início de uma execução: nada mais é lido sob lock durante o trabalho.
struct RunConfig {
    std::shared_ptr<IImageLoader> loader;
    std::shared_ptr<IEngineObserver> observer;
    std::shared_ptr<IOutputStage> output;
    std::filesystem::path input;
    std::uint64_t revision = 0;
    EngineParameters params;
    std::vector<ExportRequest> requests;
    std::shared_ptr<const PipelineSnapshot> previous;
    std::optional<SegmentationOverride> segOverride;
};

struct RunOutcome {
    RunStatus status = RunStatus::Completed;
    std::string message;
    std::shared_ptr<const PipelineSnapshot> snapshot;
};

// ---------------------------------------------------------------- impressões digitais

Fingerprint inputFingerprint(const RunConfig& cfg) {
    Fingerprint f;
    f.add(std::string_view("load")).add(std::string_view(cfg.input.generic_string()));
    std::error_code ec;
    const auto size = std::filesystem::file_size(cfg.input, ec);
    f.add(static_cast<std::uint64_t>(ec ? 0 : size));
    ec.clear();
    const auto mtime = std::filesystem::last_write_time(cfg.input, ec);
    f.add(static_cast<std::int64_t>(ec ? 0 : mtime.time_since_epoch().count()));
    return f.add(cfg.revision);
}

EngineParameters effectiveParameters(const EngineParameters& params, ImageProfile profile) {
    EngineParameters eff = params;
    eff.applyProfile(profile);
    return eff;
}

/// Impressão digital do estágio s, dada a do estágio anterior (`up`) e os parâmetros efetivos (com o perfil
/// já aplicado; para LoadImage e Classify valem os parâmetros brutos).
Fingerprint stageFingerprint(Stage s, const RunConfig& cfg, const EngineParameters& eff, ImageProfile profile, Fingerprint up) {
    switch (s) {
        case Stage::LoadImage: return inputFingerprint(cfg);
        case Stage::Classify: {
            Fingerprint f = cfg.params.preprocess.fingerprint(up);
            f = cfg.params.classifier.fingerprint(f);
            return f.add(cfg.params.seed);
        }
        case Stage::Palette: return eff.palette.fingerprint(up).add(eff.seed);
        case Stage::Segment: {
            Fingerprint f = eff.segment.fingerprint(up);
            // O perfil decide o modo (paleta exata, só PixelSegmenter, ou + SubPixelSegmenter).
            return f.add(static_cast<std::uint64_t>(static_cast<std::uint8_t>(profile))).add(eff.seed);
        }
        case Stage::SmoothContours: return eff.smooth.fingerprint(up).add(eff.seed);
        case Stage::FitBezier: return eff.fit.fingerprint(up).add(eff.seed);
        case Stage::Output: {
            Fingerprint f;
            f.add(std::string_view("output"));
            for (const ExportRequest& r : cfg.requests) {
                f.add(std::string_view(r.file.generic_string())).add(static_cast<std::uint64_t>(static_cast<std::uint8_t>(r.format)));
                f = r.settings.fingerprint(f);
            }
            return f.add(up);
        }
    }
    return {};
}

Fingerprint segmentEditFingerprint(const SegmentationOverride& o) {
    Fingerprint f;
    f.add(std::string_view("segment-edit")).add(o.editsHash);
    return f.add(o.upstream);
}

// ---------------------------------------------------------------- estágios

/// Modo "paleta exata": cada pixel vai para a cor da paleta mais próxima; as regiões são as componentes
/// 4-conexas. Depois, as quinas só-diagonais são resolvidas (performDiagonalExtrusions) sem agrupar mais.
PixelMap exactPaletteMap(const sla::Array2D<sla::Rgba8>& image, const Palette& palette) {
    const int w = image.width();
    const int h = image.height();
    std::vector<int> index(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const sla::Rgba c = sla::toRgba(image(x, y));
            std::size_t best = 0;
            double bestD = sla::distance2(c, palette.colors[0]);
            for (std::size_t k = 1; k < palette.colors.size(); ++k) {
                const double d = sla::distance2(c, palette.colors[k]);
                if (d < bestD) {
                    bestD = d;
                    best = k;
                }
            }
            index[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] = static_cast<int>(best);
        }
    }
    PixelMap map(w, h);
    std::vector<std::pair<int, int>> stack;
    constexpr std::array<int, 4> dx{1, 0, -1, 0};
    constexpr std::array<int, 4> dy{0, 1, 0, -1};
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (map.regionAtConst(x, y) != kNoRegion) continue;
            const int label = index[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)];
            const RegionId id = map.createRegion();
            map.setLabel(x, y, id, image(x, y));
            stack.assign(1, {x, y});
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + dx[d];
                    const int ny = cy + dy[d];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h || map.regionAtConst(nx, ny) != kNoRegion) continue;
                    if (index[static_cast<std::size_t>(ny) * static_cast<std::size_t>(w) + static_cast<std::size_t>(nx)] != label) continue;
                    map.setLabel(nx, ny, id, image(nx, ny));
                    stack.emplace_back(nx, ny);
                }
            }
        }
    }
    return map;
}

PixelMap segmentStage(const PreprocessedImage& pre, const Palette& palette, const EngineParameters& eff, ImageProfile profile, TaskContext& ctx) {
    if (palette.exact) {
        PixelSegmenter seg(pre.smooth, eff.segment, exactPaletteMap(pre.smooth, palette));
        for (int k = 0; k < eff.segment.finalSweepCap; ++k) {
            ctx.checkpoint(static_cast<double>(k) / std::max(1, eff.segment.finalSweepCap));
            // Traços diagonais de 1 px primeiro: mergeSmallRegions apagaria os pixels deles um a um.
            if (seg.linkStrokeSweep() == 0 && seg.diagonalSweep() == 0 && seg.mergeSmallRegions() == 0) break;
        }
        return seg.finish(&palette);
    }
    const bool antialiased = isAntialiasedLogo(profile);
    PixelSegmenter seg(pre.smooth, eff.segment);
    SegmentationResult r = seg.run(ctx.sub(0.0, antialiased ? 0.7 : 1.0), &palette);
    if (!antialiased) return std::move(r.map);
    TaskContext sub = ctx.sub(0.7, 1.0);
    return SubPixelSegmenter(pre.smooth, eff.segment, palette).run(std::move(r.map), sub).map;
}

bool anyGenerative(const SmoothParams& p) {
    return std::any_of(p.measurementTypes.begin(), p.measurementTypes.end(), [](MeasurementType m) { return m == MeasurementType::Generative; });
}

// ---------------------------------------------------------------- execução

RunOutcome runPipeline(const RunConfig& cfg, Stage target, const std::stop_token& stop) {
    using Clock = std::chrono::steady_clock;
    RunOutcome outcome;
    const auto notify = [&](auto&& fn) {
        if (!cfg.observer) return;
        try {
            fn(*cfg.observer);
        } catch (...) {  // o contrato proíbe exceções; um observador defeituoso não derruba o motor
        }
    };

    auto next = cfg.previous ? std::make_shared<PipelineSnapshot>(*cfg.previous) : std::make_shared<PipelineSnapshot>();
    const PipelineSnapshot* prev = cfg.previous.get();
    bool dirty = false;
    EngineParameters eff = cfg.params;
    ImageProfile profile = ImageProfile::Custom;
    Fingerprint up;

    try {
        for (std::size_t i = 0; i <= stageIndex(target); ++i) {
            const Stage stage = kStages[i];
            if (stop.stop_requested()) throw CancelTaskException{};

            Fingerprint fp = stageFingerprint(stage, cfg, eff, profile, up);
            bool overridden = false;
            if (stage == Stage::Segment && cfg.segOverride && cfg.segOverride->upstream == up) {
                fp = segmentEditFingerprint(*cfg.segOverride);
                overridden = true;
            }
            const bool present = prev != nullptr && [&] {
                switch (stage) {
                    case Stage::LoadImage: return prev->source != nullptr;
                    case Stage::Classify: return prev->preprocessed != nullptr && prev->classification != nullptr;
                    case Stage::Palette: return prev->palette != nullptr;
                    case Stage::Segment: return prev->segmentation != nullptr;
                    case Stage::SmoothContours: return prev->smoothed != nullptr;
                    case Stage::FitBezier: return prev->fitted != nullptr;
                    case Stage::Output: return true;
                }
                return false;
            }();
            const bool cached = !dirty && present && prev->fingerprints[i] == fp;

            const ProgressRange range = kStageProgress[i];
            if (cached) {
                notify([&](IEngineObserver& o) { o.onProgress(stage, range.end); });
                notify([&](IEngineObserver& o) { o.onStageFinished(stage, 0.0, true); });
            } else {
                dirty = true;
                const auto progress = std::make_shared<const TaskContext::ProgressFn>(
                    [obs = cfg.observer, stage](double g) {
                        if (!obs) return;
                        try {
                            obs->onProgress(stage, g);
                        } catch (...) {
                        }
                    });
                TaskContext ctx(stop, progress, range);
                const auto t0 = Clock::now();
                switch (stage) {
                    case Stage::LoadImage: {
                        SourceImage src = cfg.loader->load(cfg.input, stop);
                        if (src.pixels.width() <= 0 || src.pixels.height() <= 0) throw std::runtime_error("imagem de entrada vazia");
                        next->source = std::make_shared<const SourceImage>(std::move(src));
                        break;
                    }
                    case Stage::Classify: {
                        TaskContext preCtx = ctx.sub(0.0, 0.4);
                        auto pre = std::make_shared<PreprocessedImage>(preprocess(*next->source, cfg.params.preprocess, preCtx));
                        TaskContext clsCtx = ctx.sub(0.4, 1.0);
                        auto cls = std::make_shared<ImageClassification>(
                            classify(*pre, *next->source, cfg.params.classifier, cfg.params.palette, clsCtx));
                        next->preprocessed = std::move(pre);
                        next->classification = std::move(cls);
                        break;
                    }
                    case Stage::Palette: {
                        const PaletteFinder finder(eff.palette, eff.seed);
                        ctx.checkpoint(0.0);
                        next->palette = std::make_shared<const Palette>(finder.find(next->preprocessed->smooth).palette);
                        break;
                    }
                    case Stage::Segment: {
                        if (overridden) {
                            next->segmentation = cfg.segOverride->map;
                        } else {
                            next->segmentation = std::make_shared<const PixelMap>(
                                segmentStage(*next->preprocessed, *next->palette, eff, profile, ctx));
                        }
                        break;
                    }
                    case Stage::SmoothContours: {
                        TraceResult traced = BoundaryTracer().trace(*next->segmentation, stop);
                        if (eff.smooth.subPixelAnchors) refineEdgeAnchors(traced.image, next->preprocessed->image, eff.smooth);
                        std::optional<GenerativeModel> model;
                        if (anyGenerative(eff.smooth)) model.emplace(next->preprocessed->signal.premultiplied());  // float, sem quantizar
                        ContourSmoother(eff.smooth, model ? &*model : nullptr).run(traced.image, ctx);
                        next->smoothed = std::make_shared<const VectorImage>(std::move(traced.image));
                        break;
                    }
                    case Stage::FitBezier: {
                        auto fitted = std::make_shared<VectorImage>(*next->smoothed);
                        BezierFitter(eff.fit).run(*fitted, ctx);
                        fitted->updateFaceAreas();  // áreas pela fórmula de Green das cúbicas
                        next->fitted = std::move(fitted);
                        break;
                    }
                    case Stage::Output: {
                        if (cfg.output && !cfg.requests.empty()) cfg.output->write(*next, cfg.requests, ctx);
                        break;
                    }
                }
                const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
                notify([&](IEngineObserver& o) { o.onProgress(stage, range.end); });
                notify([&](IEngineObserver& o) { o.onStageFinished(stage, elapsed, false); });
            }
            next->fingerprints[i] = fp;

            if (stage == Stage::Classify) {
                profile = next->classification->profile;
                eff = effectiveParameters(cfg.params, profile);
            }
            up = fp;
        }
        // Estágios depois do alvo: valem só se nada acima foi recalculado.
        if (dirty) {
            for (std::size_t i = stageIndex(target) + 1; i < kStageCount; ++i) {
                next->fingerprints[i] = Fingerprint{};
                switch (kStages[i]) {
                    case Stage::LoadImage: break;
                    case Stage::Classify: next->preprocessed.reset(); next->classification.reset(); break;
                    case Stage::Palette: next->palette.reset(); break;
                    case Stage::Segment: next->segmentation.reset(); break;
                    case Stage::SmoothContours: next->smoothed.reset(); break;
                    case Stage::FitBezier: next->fitted.reset(); break;
                    case Stage::Output: break;
                }
            }
        }
        outcome.status = RunStatus::Completed;
        outcome.snapshot = std::move(next);
    } catch (const CancelTaskException&) {
        outcome.status = RunStatus::Cancelled;
        outcome.message = "cancelado";
    } catch (const std::exception& e) {
        outcome.status = RunStatus::Failed;
        outcome.message = e.what();
    }
    return outcome;
}

}  // namespace

// ---------------------------------------------------------------- Impl

struct CoreEngine::Impl {
    std::shared_ptr<IImageLoader> loader;
    unsigned workers = 0;

    mutable std::mutex mu;  // configuração, snapshot, status e a condição de término
    std::condition_variable cv;
    std::shared_ptr<IEngineObserver> observer;
    std::shared_ptr<IOutputStage> output;
    std::filesystem::path input;
    std::uint64_t revision = 0;
    EngineParameters params;
    std::vector<ExportRequest> requests;
    std::shared_ptr<const PipelineSnapshot> snap;
    std::optional<SegmentationOverride> segOverride;
    RunStatus lastStatus = RunStatus::Completed;
    std::string lastMessage;
    bool running = false;

    std::mutex ctlMu;  // worker e stop_source
    std::jthread worker;
    std::stop_source stop;
    std::atomic<bool> runningFlag{false};

    RunConfig makeConfig() const {
        std::lock_guard<std::mutex> l(mu);
        RunConfig c;
        c.loader = loader;
        c.observer = observer;
        c.output = output;
        c.input = input;
        c.revision = revision;
        c.params = params;
        c.requests = requests;
        c.previous = snap;
        c.segOverride = segOverride;
        return c;
    }

    void joinWorker(bool requestStop) {
        std::jthread w;
        {
            std::lock_guard<std::mutex> l(ctlMu);
            if (requestStop) stop.request_stop();
            w = std::move(worker);
        }
        if (w.joinable()) w.join();
    }

    void execute(RunConfig cfg, Stage target, std::stop_token token) {
        RunOutcome outcome;
        try {
            if (!cfg.loader) throw std::logic_error("CoreEngine: sem IImageLoader");
            if (cfg.input.empty()) throw std::logic_error("CoreEngine: sem arquivo de entrada (setInput)");
            outcome = runPipeline(cfg, target, token);
        } catch (const std::exception& e) {
            outcome.status = RunStatus::Failed;
            outcome.message = e.what();
        }
        {
            std::lock_guard<std::mutex> l(mu);
            if (outcome.status == RunStatus::Completed && outcome.snapshot) snap = outcome.snapshot;
            lastStatus = outcome.status;
            lastMessage = outcome.message;
        }
        if (cfg.observer) {
            try {
                cfg.observer->onRunFinished(outcome.status, outcome.message);
            } catch (...) {
            }
        }
        {
            std::lock_guard<std::mutex> l(mu);
            running = false;
            runningFlag = false;
        }
        cv.notify_all();
    }
};

// ---------------------------------------------------------------- CoreEngine

CoreEngine::CoreEngine(std::shared_ptr<IImageLoader> loader, unsigned workerThreads) : impl_(std::make_unique<Impl>()) {
    impl_->loader = std::move(loader);
    impl_->workers = workerThreads != 0 ? workerThreads : std::max(1u, std::thread::hardware_concurrency());
}

CoreEngine::~CoreEngine() { impl_->joinWorker(true); }

void CoreEngine::setObserver(std::shared_ptr<IEngineObserver> observer) {
    std::lock_guard<std::mutex> l(impl_->mu);
    impl_->observer = std::move(observer);
}

void CoreEngine::setOutputStage(std::shared_ptr<IOutputStage> output) {
    std::lock_guard<std::mutex> l(impl_->mu);
    impl_->output = std::move(output);
}

void CoreEngine::setInput(const std::filesystem::path& file) {
    std::lock_guard<std::mutex> l(impl_->mu);
    impl_->input = file;
    impl_->segOverride.reset();  // as edições eram de outra imagem
}

void CoreEngine::invalidateInput() {
    std::lock_guard<std::mutex> l(impl_->mu);
    ++impl_->revision;
}

void CoreEngine::setParameters(const EngineParameters& params) {
    std::lock_guard<std::mutex> l(impl_->mu);
    impl_->params = params;
}

EngineParameters CoreEngine::parameters() const {
    std::lock_guard<std::mutex> l(impl_->mu);
    return impl_->params;
}

void CoreEngine::setExportRequests(std::vector<ExportRequest> requests) {
    std::lock_guard<std::mutex> l(impl_->mu);
    impl_->requests = std::move(requests);
}

std::optional<Stage> CoreEngine::firstDirtyStage() const {
    const RunConfig cfg = impl_->makeConfig();
    const PipelineSnapshot* prev = cfg.previous.get();
    if (prev == nullptr) return Stage::LoadImage;

    EngineParameters eff = cfg.params;
    ImageProfile profile = ImageProfile::Custom;
    Fingerprint up;
    for (std::size_t i = 0; i < kStageCount; ++i) {
        const Stage stage = kStages[i];
        Fingerprint fp = stageFingerprint(stage, cfg, eff, profile, up);
        if (stage == Stage::Segment && cfg.segOverride && cfg.segOverride->upstream == up) fp = segmentEditFingerprint(*cfg.segOverride);
        if (prev->fingerprints[i] != fp) return stage;
        if (stage == Stage::Classify) {
            if (!prev->classification) return stage;
            profile = prev->classification->profile;
            eff = effectiveParameters(cfg.params, profile);
        }
        up = fp;
    }
    return std::nullopt;
}

bool CoreEngine::needsToSegment() const {
    const auto s = firstDirtyStage();
    return s.has_value() && stageIndex(*s) <= stageIndex(Stage::Segment);
}

bool CoreEngine::needsToBezierFit() const {
    const auto s = firstDirtyStage();
    return s.has_value() && stageIndex(*s) <= stageIndex(Stage::FitBezier);
}

void CoreEngine::start(Stage target) {
    impl_->joinWorker(true);  // cancela e aguarda a execução anterior
    RunConfig cfg = impl_->makeConfig();
    std::stop_token token;
    {
        std::lock_guard<std::mutex> l(impl_->mu);
        impl_->running = true;
        impl_->runningFlag = true;
    }
    {
        std::lock_guard<std::mutex> l(impl_->ctlMu);
        impl_->stop = std::stop_source{};
        token = impl_->stop.get_token();
        impl_->worker = std::jthread([impl = impl_.get(), cfg = std::move(cfg), target, token]() mutable {
            impl->execute(std::move(cfg), target, token);
        });
    }
}

void CoreEngine::cancel() noexcept {
    try {
        std::lock_guard<std::mutex> l(impl_->ctlMu);
        impl_->stop.request_stop();
    } catch (...) {
    }
}

RunStatus CoreEngine::wait() {
    std::unique_lock<std::mutex> l(impl_->mu);
    impl_->cv.wait(l, [&] { return !impl_->running; });
    return impl_->lastStatus;
}

bool CoreEngine::isRunning() const noexcept { return impl_->runningFlag.load(); }

std::string CoreEngine::lastMessage() const {
    std::lock_guard<std::mutex> l(impl_->mu);
    return impl_->lastMessage;
}

std::shared_ptr<const PipelineSnapshot> CoreEngine::snapshot() const {
    std::lock_guard<std::mutex> l(impl_->mu);
    return impl_->snap;
}

void CoreEngine::replaceSegmentation(std::shared_ptr<const PixelMap> edited, Fingerprint editsHash) {
    if (!edited) throw std::invalid_argument("CoreEngine::replaceSegmentation: segmentação nula");
    if (!edited->isCompact()) throw std::invalid_argument("CoreEngine::replaceSegmentation: o mapa deve estar compactado");
    std::lock_guard<std::mutex> l(impl_->mu);
    if (!impl_->snap || !impl_->snap->palette) throw std::logic_error("CoreEngine::replaceSegmentation: execute até a paleta antes de editar");
    impl_->segOverride = SegmentationOverride{std::move(edited), editsHash, impl_->snap->fingerprints[stageIndex(Stage::Palette)]};
}

}  // namespace cnx
