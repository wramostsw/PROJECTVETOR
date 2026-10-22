// tests/unit/core_engine_test.cpp — CoreEngine: pipeline completo, recálculo inteligente por impressão
// digital, cancelamento, progresso, edição manual da segmentação e o estágio de saída plugável.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

#include "cnx/classify/Classifier.h"
#include "cnx/engine/CoreEngine.h"
#include "cnx/preprocess/Preprocessor.h"
#include "cnx/segment/PixelMap.h"
#include "cnx/vector/VectorImage.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::CoreEngine;
using cnx::RunStatus;
using cnx::Stage;
using sla::Array2D;
using sla::Rgba8;

namespace {

Rgba8 rgb(int r, int g, int b) { return {static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b), 255}; }

/// Logo chapado: fundo branco, retângulo vermelho e retângulo azul.
Array2D<Rgba8> flatLogo(int w = 48, int h = 36) {
    Array2D<Rgba8> img(w, h, rgb(255, 255, 255));
    for (int y = 4; y < 18; ++y) {
        for (int x = 5; x < 24; ++x) img(x, y) = rgb(220, 30, 30);
    }
    for (int y = 20; y < 32; ++y) {
        for (int x = 26; x < 43; ++x) img(x, y) = rgb(30, 30, 220);
    }
    return img;
}

/// Disco escuro com antialiasing (supersampling) sobre fundo branco.
Array2D<Rgba8> aaDisk(int w = 48, int h = 48) {
    const auto scene = cnxtest::aaScene(w, h, [&](double x, double y) { return (x - w * 0.5 - 0.3) * (x - w * 0.5 - 0.3) + (y - h * 0.5 + 0.2) * (y - h * 0.5 + 0.2) <= 14.0 * 14.0; }, 8);
    Array2D<Rgba8> img(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) img(x, y) = sla::toRgba8(scene.target(x, y));
    }
    return img;
}

/// Gradiente com ruído: "foto".
Array2D<Rgba8> photoLike(int w = 32, int h = 32) {
    cnxtest::SplitMix64 rng{4242};
    Array2D<Rgba8> img(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            img(x, y) = rgb(std::clamp(8 * x + rng.below(24), 0, 255), std::clamp(8 * y + rng.below(24), 0, 255), rng.below(256));
        }
    }
    return img;
}

/// Carregador falso: devolve a imagem pedida, conta as chamadas e pode atrasar, falhar ou esperar o cancelamento.
class FakeLoader final : public cnx::IImageLoader {
public:
    cnx::SourceImage load(const std::filesystem::path&, const std::stop_token& stop) override {
        ++calls;
        if (blockUntilStopped) {
            while (!stop.stop_requested()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            throw cnx::CancelTaskException{};
        }
        if (failWith) throw std::runtime_error(*failWith);
        cnx::SourceImage s;
        s.pixels = image;
        s.formatHint = hint;
        return s;
    }
    Array2D<Rgba8> image = flatLogo();
    std::string hint = ".png";
    std::atomic<int> calls{0};
    std::atomic<bool> blockUntilStopped{false};
    std::optional<std::string> failWith;
};

struct Event {
    enum Kind { Progress, Finished, Run } kind;
    Stage stage = Stage::LoadImage;
    double value = 0.0;
    bool reused = false;
    RunStatus status = RunStatus::Completed;
    std::string message;
};

class Recorder final : public cnx::IEngineObserver {
public:
    void onProgress(Stage s, double g) override {
        std::lock_guard<std::mutex> l(mu);
        events.push_back({Event::Progress, s, g, false, RunStatus::Completed, {}});
        if (onProgressHook) onProgressHook(s, g);
    }
    void onStageFinished(Stage s, double secs, bool reused) override {
        std::lock_guard<std::mutex> l(mu);
        events.push_back({Event::Finished, s, secs, reused, RunStatus::Completed, {}});
        if (onFinishedHook) onFinishedHook(s);
    }
    void onRunFinished(RunStatus st, const std::string& msg) override {
        std::lock_guard<std::mutex> l(mu);
        events.push_back({Event::Run, Stage::Output, 0.0, false, st, msg});
    }

    std::vector<Event> take() {
        std::lock_guard<std::mutex> l(mu);
        std::vector<Event> out;
        out.swap(events);
        return out;
    }
    std::function<void(Stage, double)> onProgressHook;
    std::function<void(Stage)> onFinishedHook;

private:
    std::mutex mu;
    std::vector<Event> events;
};

/// Estágios concluídos, na ordem, com o flag "reaproveitado do cache".
std::vector<std::pair<Stage, bool>> finishedStages(const std::vector<Event>& events) {
    std::vector<std::pair<Stage, bool>> out;
    for (const Event& e : events) {
        if (e.kind == Event::Finished) out.emplace_back(e.stage, e.reused);
    }
    return out;
}

struct Rig {
    std::shared_ptr<FakeLoader> loader = std::make_shared<FakeLoader>();
    std::shared_ptr<Recorder> rec = std::make_shared<Recorder>();
    CoreEngine engine{loader};
    Rig() {
        engine.setObserver(rec);
        engine.setInput("logo.png");
        cnx::EngineParameters p = engine.parameters();
        p.smooth.maxCgIterations = 40;  // testes rápidos
        engine.setParameters(p);
    }
    RunStatus run(Stage target = Stage::Output) {
        engine.start(target);
        return engine.wait();
    }
    template <class Fn>
    void edit(Fn&& fn) {
        cnx::EngineParameters p = engine.parameters();
        fn(p);
        engine.setParameters(p);
    }
};

class FakeOutput final : public cnx::IOutputStage {
public:
    void write(const cnx::PipelineSnapshot& snap, const std::vector<cnx::ExportRequest>& requests, cnx::TaskContext&) override {
        ++writes;
        lastRequests = requests.size();
        sawFitted = snap.fitted != nullptr && snap.fitted->hasCurves();
    }
    std::atomic<int> writes{0};
    std::size_t lastRequests = 0;
    bool sawFitted = false;
};

}  // namespace

CNX_TEST(a_full_run_builds_every_artifact_and_reports_progress_in_order) {
    Rig rig;
    CHECK(rig.engine.firstDirtyStage() == Stage::LoadImage);  // nada em cache
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(rig.engine.lastMessage().empty() && !rig.engine.isRunning());
    const auto snap = rig.engine.snapshot();
    CHECK(snap && snap->source && snap->preprocessed && snap->classification && snap->palette && snap->segmentation && snap->smoothed && snap->fitted);
    CHECK(snap->classification->profile == cnx::ImageProfile::LogoMed);
    CHECK(snap->palette->exact && snap->palette->colors.size() == 3);
    CHECK(snap->segmentation->forest().rootCount() == 3);  // fundo, vermelho, azul
    CHECK(snap->smoothed->validate().ok);
    CHECK(snap->fitted->hasCurves() && snap->fitted->validate().ok);
    CHECK(!snap->smoothed->hasCurves());  // a poligonal fica intacta; as curvas vão numa cópia
    CHECK(rig.engine.firstDirtyStage() == std::nullopt);
    CHECK(!rig.engine.needsToSegment() && !rig.engine.needsToBezierFit());

    const std::vector<Event> events = rig.rec->take();
    const auto stages = finishedStages(events);
    CHECK(stages.size() == 7);
    for (std::size_t i = 0; i < stages.size(); ++i) {
        CHECK(stages[i].first == static_cast<Stage>(i) && !stages[i].second);
    }
    // Progresso global monótono, dentro da faixa de cada estágio e terminando em 100 %.
    double last = 0.0;
    for (const Event& e : events) {
        if (e.kind != Event::Progress) continue;
        CHECK_CTX(e.value >= last - 1e-12, "retrocesso em " + std::string(cnx::toString(e.stage)));
        const cnx::ProgressRange r = cnx::kStageProgress[cnx::stageIndex(e.stage)];
        CHECK_CTX(e.value >= r.begin - 1e-12 && e.value <= r.end + 1e-12, std::string(cnx::toString(e.stage)));
        last = std::max(last, e.value);
    }
    CHECK(std::abs(last - 1.0) < 1e-12);
    CHECK(events.back().kind == Event::Run && events.back().status == RunStatus::Completed);
    CHECK(rig.loader->calls == 1);
}

CNX_TEST(changing_only_the_fit_parameters_reruns_from_fitbezier) {
    Rig rig;
    CHECK(rig.run() == RunStatus::Completed);
    (void)rig.rec->take();
    const auto before = rig.engine.snapshot();

    rig.edit([](cnx::EngineParameters& p) { p.fit.maxRounds = 7; });
    CHECK(rig.engine.firstDirtyStage() == Stage::FitBezier);
    CHECK(!rig.engine.needsToSegment() && rig.engine.needsToBezierFit());
    CHECK(rig.run() == RunStatus::Completed);
    const auto stages = finishedStages(rig.rec->take());
    CHECK(stages.size() == 7);
    for (std::size_t i = 0; i < stages.size(); ++i) CHECK_CTX(stages[i].second == (i < cnx::stageIndex(Stage::FitBezier)), std::to_string(i));
    const auto after = rig.engine.snapshot();
    // Copy-on-write: os artefatos de cima são os mesmos objetos; só os de baixo foram refeitos.
    CHECK(after->source == before->source && after->preprocessed == before->preprocessed && after->palette == before->palette);
    CHECK(after->segmentation == before->segmentation && after->smoothed == before->smoothed);
    CHECK(after->fitted != before->fitted);
    // As áreas das faces do ajustado vêm da fórmula de Green das cúbicas (não as da poligonal).
    cnx::VectorImage recomputed = *after->fitted;
    recomputed.updateFaceAreas();
    for (cnx::FaceId f = 0; f < static_cast<cnx::FaceId>(recomputed.faceCount()); ++f) CHECK(after->fitted->face(f).area == recomputed.face(f).area);
    CHECK(rig.loader->calls == 1);
    // O snapshot antigo continua válido e intacto.
    CHECK(before->fitted->hasCurves() && before->fitted->validate().ok);
}

CNX_TEST(each_parameter_group_dirties_exactly_its_stage) {
    struct Case {
        const char* name;
        std::function<void(cnx::EngineParameters&)> edit;
        Stage firstDirty;
    };
    const std::vector<Case> cases{
        {"preprocess", [](cnx::EngineParameters& p) { p.preprocess.blurSigmaPx = 0.7; }, Stage::Classify},
        {"classifier", [](cnx::EngineParameters& p) { p.classifier.autoClassify = false; p.classifier.forcedProfile = cnx::ImageProfile::LogoMed; }, Stage::Classify},
        {"seed", [](cnx::EngineParameters& p) { p.seed += 1; }, Stage::Classify},
        {"palette", [](cnx::EngineParameters& p) { p.palette.colorPenalty = 0.2; }, Stage::Palette},
        {"segment", [](cnx::EngineParameters& p) { p.segment.minRegionPixels = 7; }, Stage::Segment},
        {"smooth", [](cnx::EngineParameters& p) { p.smooth.priorStrength = 3.0; }, Stage::SmoothContours},
        {"fit", [](cnx::EngineParameters& p) { p.fit.newtonMaxIterations = 4; }, Stage::FitBezier},
    };
    for (const Case& c : cases) {
        Rig rig;
        CHECK(rig.run() == RunStatus::Completed);
        CHECK(rig.engine.firstDirtyStage() == std::nullopt);
        rig.edit(c.edit);
        CHECK_CTX(rig.engine.firstDirtyStage() == c.firstDirty, c.name);
        CHECK_CTX(rig.engine.needsToSegment() == (cnx::stageIndex(c.firstDirty) <= cnx::stageIndex(Stage::Segment)), c.name);
        CHECK_CTX(rig.engine.needsToBezierFit(), c.name);
        (void)rig.rec->take();
        CHECK(rig.run() == RunStatus::Completed);
        const auto stages = finishedStages(rig.rec->take());
        for (std::size_t i = 0; i < stages.size(); ++i) {
            CHECK_CTX(stages[i].second == (i < cnx::stageIndex(c.firstDirty)), std::string(c.name) + ": " + std::string(cnx::toString(stages[i].first)));
        }
        CHECK_CTX(rig.loader->calls == 1, c.name);  // o arquivo nunca é relido
    }
    // Entrada diferente suja o carregamento; a mesma entrada de novo, não.
    Rig rig;
    CHECK(rig.run() == RunStatus::Completed);
    rig.engine.setInput("logo.png");
    CHECK(rig.engine.firstDirtyStage() == std::nullopt);
    rig.engine.setInput("other.png");
    CHECK(rig.engine.firstDirtyStage() == Stage::LoadImage);
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(rig.loader->calls == 2);
    rig.engine.invalidateInput();
    CHECK(rig.engine.firstDirtyStage() == Stage::LoadImage);
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(rig.loader->calls == 3);
}

CNX_TEST(a_run_with_nothing_dirty_reuses_everything) {
    Rig rig;
    CHECK(rig.run() == RunStatus::Completed);
    (void)rig.rec->take();
    const auto before = rig.engine.snapshot();
    CHECK(rig.run() == RunStatus::Completed);
    const auto events = rig.rec->take();
    for (const auto& [stage, reused] : finishedStages(events)) CHECK_CTX(reused, std::string(cnx::toString(stage)));
    CHECK(rig.engine.snapshot()->fitted == before->fitted && rig.engine.snapshot()->source == before->source);
    CHECK(rig.loader->calls == 1);
    // Mesmo reaproveitado, o progresso chega a 100 %.
    double last = 0.0;
    for (const Event& e : events) {
        if (e.kind == Event::Progress) last = std::max(last, e.value);
    }
    CHECK(std::abs(last - 1.0) < 1e-12);
}

CNX_TEST(the_target_stage_limits_the_run_and_a_later_run_continues_from_the_cache) {
    Rig rig;
    CHECK(rig.run(Stage::Palette) == RunStatus::Completed);
    auto snap = rig.engine.snapshot();
    CHECK(snap->palette && !snap->segmentation && !snap->smoothed && !snap->fitted);
    CHECK(rig.engine.firstDirtyStage() == Stage::Segment);
    (void)rig.rec->take();
    CHECK(rig.run() == RunStatus::Completed);
    const auto stages = finishedStages(rig.rec->take());
    for (std::size_t i = 0; i < stages.size(); ++i) CHECK(stages[i].second == (i <= cnx::stageIndex(Stage::Palette)));
    CHECK(rig.engine.snapshot()->fitted != nullptr);
    CHECK(rig.engine.snapshot()->palette == snap->palette);
    // Refazer um estágio de cima, com alvo mais baixo, descarta os artefatos de baixo (estariam obsoletos).
    rig.edit([](cnx::EngineParameters& p) { p.palette.colorPenalty = 0.3; });
    CHECK(rig.run(Stage::Classify) == RunStatus::Completed);  // alvo acima do estágio sujo: nada a refazer
    CHECK(rig.engine.snapshot()->fitted != nullptr && rig.engine.firstDirtyStage() == Stage::Palette);
    CHECK(rig.run(Stage::Palette) == RunStatus::Completed);
    snap = rig.engine.snapshot();
    CHECK(snap->palette && !snap->segmentation && !snap->smoothed && !snap->fitted);
    CHECK(rig.engine.firstDirtyStage() == Stage::Segment);
}

CNX_TEST(cancelling_discards_the_partial_work_and_keeps_the_last_snapshot) {
    // (a) durante o carregamento
    {
        Rig rig;
        rig.loader->blockUntilStopped = true;
        rig.engine.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK(rig.engine.isRunning());
        rig.engine.cancel();
        CHECK(rig.engine.wait() == RunStatus::Cancelled);
        CHECK(rig.engine.snapshot() == nullptr);
        const auto events = rig.rec->take();
        CHECK(!events.empty() && events.back().kind == Event::Run && events.back().status == RunStatus::Cancelled);
        // O motor segue utilizável.
        rig.loader->blockUntilStopped = false;
        CHECK(rig.run() == RunStatus::Completed && rig.engine.snapshot() != nullptr);
    }
    // (b) entre estágios: o observador cancela quando a paleta termina; nada é publicado.
    {
        Rig rig;
        rig.rec->onFinishedHook = [&](Stage s) {
            if (s == Stage::Palette) rig.engine.cancel();
        };
        CHECK(rig.run() == RunStatus::Cancelled);
        CHECK(rig.engine.snapshot() == nullptr);
        for (const auto& [stage, reused] : finishedStages(rig.rec->take())) CHECK(cnx::stageIndex(stage) <= cnx::stageIndex(Stage::Palette));
    }
    // (c) dentro de um estágio longo: cancela no meio do suavizador (progresso > 20 %).
    {
        Rig rig;
        rig.loader->image = aaDisk();
        rig.edit([](cnx::EngineParameters& p) { p.smooth.maxCgIterations = 200; });
        std::atomic<bool> fired{false};
        rig.rec->onProgressHook = [&](Stage s, double g) {
            if (s == Stage::SmoothContours && g > 0.25 && !fired.exchange(true)) rig.engine.cancel();
        };
        CHECK(rig.run() == RunStatus::Cancelled);
        CHECK(fired && rig.engine.snapshot() == nullptr);
    }
    // (c2) cancelar quando o ajuste termina impede o estágio de saída, que não tem pontos de verificação.
    {
        Rig rig;
        auto out = std::make_shared<FakeOutput>();
        rig.engine.setOutputStage(out);
        cnx::ExportRequest req;
        req.file = "saida.svg";
        rig.engine.setExportRequests({req});
        rig.rec->onFinishedHook = [&](Stage s) {
            if (s == Stage::FitBezier) rig.engine.cancel();
        };
        CHECK(rig.run() == RunStatus::Cancelled);
        CHECK(out->writes == 0 && rig.engine.snapshot() == nullptr);
    }
    // (d) uma execução cancelada não estraga o cache anterior.
    {
        Rig rig;
        CHECK(rig.run() == RunStatus::Completed);
        const auto good = rig.engine.snapshot();
        rig.edit([](cnx::EngineParameters& p) { p.fit.maxRounds = 5; });
        rig.rec->onFinishedHook = [&](Stage s) {
            if (s == Stage::SmoothContours) rig.engine.cancel();
        };
        CHECK(rig.run() == RunStatus::Cancelled);
        CHECK(rig.engine.snapshot() == good);
        rig.rec->onFinishedHook = nullptr;
        CHECK(rig.run() == RunStatus::Completed);
        CHECK(rig.engine.snapshot() != good && rig.engine.snapshot()->source == good->source);
    }
}

CNX_TEST(starting_a_new_run_cancels_the_previous_one) {
    Rig rig;
    rig.loader->blockUntilStopped = true;
    rig.engine.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    rig.loader->blockUntilStopped = false;
    rig.engine.start();  // cancela a primeira e roda a segunda
    CHECK(rig.engine.wait() == RunStatus::Completed);
    const auto events = rig.rec->take();
    int cancelled = 0;
    int completed = 0;
    for (const Event& e : events) {
        if (e.kind != Event::Run) continue;
        cancelled += e.status == RunStatus::Cancelled ? 1 : 0;
        completed += e.status == RunStatus::Completed ? 1 : 0;
    }
    CHECK(cancelled == 1 && completed == 1);
    CHECK(rig.engine.snapshot() != nullptr);
}

CNX_TEST(failures_are_reported_and_leave_the_engine_usable) {
    Rig rig;
    rig.loader->failWith = "arquivo corrompido";
    CHECK(rig.run() == RunStatus::Failed);
    CHECK(rig.engine.lastMessage().find("arquivo corrompido") != std::string::npos);
    const auto events = rig.rec->take();
    CHECK(events.back().kind == Event::Run && events.back().status == RunStatus::Failed && events.back().message.find("corrompido") != std::string::npos);
    CHECK(rig.engine.snapshot() == nullptr);
    rig.loader->failWith.reset();
    CHECK(rig.run() == RunStatus::Completed && rig.engine.lastMessage().empty());

    // Imagem vazia.
    Rig empty;
    empty.loader->image = Array2D<Rgba8>();
    CHECK(empty.run() == RunStatus::Failed);
    CHECK(empty.engine.lastMessage().find("vazia") != std::string::npos);

    // Sem entrada configurada.
    auto loader = std::make_shared<FakeLoader>();
    CoreEngine bare(loader);
    bare.start();
    CHECK(bare.wait() == RunStatus::Failed && bare.lastMessage().find("entrada") != std::string::npos);
    // Sem carregador.
    CoreEngine none(nullptr);
    none.setInput("x.png");
    none.start();
    CHECK(none.wait() == RunStatus::Failed);

    // Parâmetro inválido num estágio posterior: a falha vem do estágio, com a mensagem dele.
    Rig bad;
    bad.edit([](cnx::EngineParameters& p) { p.fit.statThreshInitial = -1.0; });  // o ajustador rejeita
    CHECK(bad.run() == RunStatus::Failed);
    CHECK(bad.engine.snapshot() == nullptr);
}

CNX_TEST(observer_exceptions_do_not_break_a_run) {
    class Throwing final : public cnx::IEngineObserver {
    public:
        void onProgress(Stage, double) override { throw std::runtime_error("x"); }
        void onStageFinished(Stage, double, bool) override { throw std::runtime_error("y"); }
        void onRunFinished(RunStatus, const std::string&) override { throw std::runtime_error("z"); }
    };
    Rig rig;
    rig.engine.setObserver(std::make_shared<Throwing>());
    CHECK(rig.run() == RunStatus::Completed && rig.engine.snapshot() != nullptr);
}

CNX_TEST(the_profile_chosen_by_the_classifier_drives_the_later_stages) {
    // Logo com antialiasing: perfil LogoMedAA → modelo gerador no suavizador e SubPixelSegmenter na segmentação.
    {
        Rig rig;
        rig.loader->image = aaDisk();
        CHECK(rig.run() == RunStatus::Completed);
        const auto snap = rig.engine.snapshot();
        CHECK(snap->classification->profile == cnx::ImageProfile::LogoMedAA && snap->classification->automatic);
        CHECK(!snap->palette->exact && snap->palette->colors.size() == 2);
        CHECK(snap->segmentation->forest().rootCount() == 2);  // fundo e disco: as praias foram repartidas
        // A área do disco, pela fórmula de Green das cúbicas, é a do círculo (o suavizador gerador a acerta).
        double dark = 0.0;
        for (cnx::FaceId f = 1; f < static_cast<cnx::FaceId>(snap->fitted->faceCount()); ++f) {
            if (snap->fitted->face(f).color.r < 0.5f) dark = snap->fitted->face(f).area;
        }
        const double truth = std::numbers::pi * 14.0 * 14.0;
        std::printf("  disco AA pelo motor: área %.2f (círculo %.2f)\n", dark, truth);
        CHECK(std::abs(dark - truth) < 0.03 * truth);
    }
    // Perfil forçado ganha da classificação automática.
    {
        Rig rig;
        rig.edit([](cnx::EngineParameters& p) {
            p.classifier.autoClassify = false;
            p.classifier.forcedProfile = cnx::ImageProfile::PhotoHigh;
        });
        CHECK(rig.run() == RunStatus::Completed);
        const auto snap = rig.engine.snapshot();
        CHECK(snap->classification->profile == cnx::ImageProfile::PhotoHigh && !snap->classification->automatic);
    }
    // Foto: perfil PhotoMed, k-means no lugar da paleta exata e só PixelSegmenter.
    {
        Rig rig;
        rig.loader->image = photoLike();
        rig.edit([](cnx::EngineParameters& p) { p.smooth.maxCgIterations = 10; });
        CHECK(rig.run() == RunStatus::Completed);
        const auto snap = rig.engine.snapshot();
        CHECK(snap->classification->profile == cnx::ImageProfile::PhotoMed);
        CHECK(!snap->palette->exact && snap->fitted->hasCurves());
    }
    // Trocar o perfil (via forçado) suja os estágios que ele toca e reaproveita os outros.
    {
        Rig rig;
        CHECK(rig.run() == RunStatus::Completed);
        rig.edit([](cnx::EngineParameters& p) {
            p.classifier.autoClassify = false;
            p.classifier.forcedProfile = cnx::ImageProfile::LogoMed;  // o mesmo que a classificação automática escolheu
        });
        CHECK(rig.engine.firstDirtyStage() == Stage::Classify);  // o parâmetro mudou, ainda que o perfil resultante seja igual
        (void)rig.rec->take();
        CHECK(rig.run() == RunStatus::Completed);
        const auto stages = finishedStages(rig.rec->take());
        CHECK(!stages[1].second);
    }
}

CNX_TEST(a_one_pixel_diagonal_stroke_survives_the_whole_pipeline) {
    // Logo chapado + traço preto de 1 px em diagonal (15 px, só ligados pela quina): caminho da paleta exata.
    Array2D<Rgba8> img = flatLogo();
    for (int i = 0; i < 15; ++i) img(26 + i, 3 + i) = rgb(0, 0, 0);
    const auto darkArea = [](const cnx::VectorImage& v) {
        double area = 0.0;
        for (cnx::FaceId f = 1; f < static_cast<cnx::FaceId>(v.faceCount()); ++f) {
            if (v.face(f).color.r < 0.1f && v.face(f).color.b < 0.1f) area += v.face(f).area;
        }
        return area;
    };
    for (const bool link : {false, true}) {
        Rig rig;
        rig.loader->image = img;
        rig.edit([&](cnx::EngineParameters& p) { p.segment.linkStrokes = link; });
        CHECK(rig.run() == RunStatus::Completed);
        const auto snap = rig.engine.snapshot();
        CHECK(snap->classification->profile == cnx::ImageProfile::LogoMed && snap->palette->exact && snap->palette->colors.size() == 4);
        const double smoothed = darkArea(*snap->smoothed);
        const double fitted = darkArea(*snap->fitted);
        std::printf("  traço pelo motor (ligação %s): %zu regiões, área %.2f suavizada, %.2f em curvas\n", link ? "sim" : "não",
                    snap->segmentation->forest().rootCount(), smoothed, fitted);
        CHECK(snap->smoothed->validate().ok && snap->fitted->validate().ok);
        if (!link) {
            CHECK(smoothed == 0.0);  // o traço some (seus pixels viram fragmentos de outras cores)
        } else {
            // Fundo, vermelho, azul e o traço preto: 15 px de tinta + 14 pontes. As pontes não entram na cor
            // (senão a média cinza cairia no vermelho da paleta), e a suavização preserva a área (a mola).
            CHECK(snap->segmentation->forest().rootCount() == 4 && snap->smoothed->faceCount() == 5);
            CHECK(std::abs(smoothed - 29.0) < 3.0 && std::abs(fitted - 29.0) < 3.0);
        }
    }
}

CNX_TEST(manual_segmentation_edits_become_a_pseudo_parameter) {
    Rig rig;
    CHECK(rig.run() == RunStatus::Completed);
    const auto snap = rig.engine.snapshot();

    // Edição: funde as regiões vermelha e azul numa só (como uma pincelada) e compacta.
    cnx::PixelMap edited = *snap->segmentation;
    const cnx::RegionId red = edited.regionAtConst(10, 8);
    const cnx::RegionId blue = edited.regionAtConst(30, 25);
    CHECK(red != blue);
    (void)edited.forest().unite(red, blue);
    (void)edited.compact();
    CHECK(edited.forest().rootCount() == 2);
    auto editedPtr = std::make_shared<const cnx::PixelMap>(std::move(edited));
    cnx::Fingerprint hash;
    hash.add(std::string_view("pincelada-1"));

    rig.engine.replaceSegmentation(editedPtr, hash);
    CHECK(rig.engine.firstDirtyStage() == Stage::Segment);  // a edição é uma pseudo-parametrização do estágio
    (void)rig.rec->take();
    CHECK(rig.run() == RunStatus::Completed);
    auto stages = finishedStages(rig.rec->take());
    CHECK(stages[2].second);                       // a paleta não muda
    CHECK(!stages[3].second && !stages[4].second);  // segmentação (editada) e suavização refeitas
    CHECK(rig.engine.snapshot()->segmentation == editedPtr);
    // Faces são componentes conexas: exterior + fundo + os dois retângulos (agora da mesma região, mesma cor).
    const auto& edgedImg = *rig.engine.snapshot()->smoothed;
    CHECK(edgedImg.faceCount() == 4);
    CHECK(edgedImg.face(2).paletteIndex == edgedImg.face(3).paletteIndex || edgedImg.face(1).paletteIndex == edgedImg.face(2).paletteIndex ||
          edgedImg.face(1).paletteIndex == edgedImg.face(3).paletteIndex);
    CHECK(rig.engine.firstDirtyStage() == std::nullopt);

    // Outra edição (outro hash) suja de novo; a mesma edição outra vez não.
    cnx::Fingerprint hash2;
    hash2.add(std::string_view("pincelada-2"));
    rig.engine.replaceSegmentation(editedPtr, hash2);
    CHECK(rig.engine.firstDirtyStage() == Stage::Segment);
    CHECK(rig.run() == RunStatus::Completed);
    rig.engine.replaceSegmentation(editedPtr, hash2);
    CHECK(rig.engine.firstDirtyStage() == std::nullopt);

    // Mudar a paleta invalida a edição (era de outra paleta): a segmentação volta a ser calculada.
    rig.edit([](cnx::EngineParameters& p) { p.palette.colorPenalty = 0.123; });
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(rig.engine.snapshot()->segmentation != editedPtr && rig.engine.snapshot()->segmentation->forest().rootCount() == 3);

    // Argumentos inválidos.
    cnx::Fingerprint h;
    CHECK_THROWS(std::invalid_argument, rig.engine.replaceSegmentation(nullptr, h));
    CHECK_THROWS(std::invalid_argument, rig.engine.replaceSegmentation(std::make_shared<const cnx::PixelMap>(4, 4), h));  // não compactado
    Rig fresh;
    CHECK_THROWS(std::logic_error, fresh.engine.replaceSegmentation(editedPtr, h));  // sem paleta ainda
    // Reaplicar a entrada (mesmo caminho) também descarta a edição: a segmentação volta a ser calculada.
    rig.engine.replaceSegmentation(editedPtr, hash);
    CHECK(rig.run() == RunStatus::Completed && rig.engine.firstDirtyStage() == std::nullopt);
    rig.engine.setInput("logo.png");
    CHECK(rig.engine.firstDirtyStage() == Stage::Segment);
    CHECK(rig.run() == RunStatus::Completed && rig.engine.snapshot()->segmentation != editedPtr);
    // Trocar a entrada descarta a edição.
    rig.engine.replaceSegmentation(editedPtr, hash);
    rig.engine.setInput("another.png");
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(rig.engine.snapshot()->segmentation != editedPtr);
}

CNX_TEST(the_output_stage_runs_only_when_its_fingerprint_changes) {
    Rig rig;
    auto out = std::make_shared<FakeOutput>();
    rig.engine.setOutputStage(out);
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(out->writes == 0);  // sem pedidos de exportação: nada a escrever
    cnx::ExportRequest req;
    req.file = "saida.svg";
    req.format = cnx::ExportFormat::Svg;
    rig.engine.setExportRequests({req});
    CHECK(rig.engine.firstDirtyStage() == Stage::Output);
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(out->writes == 1 && out->lastRequests == 1 && out->sawFitted);
    (void)rig.rec->take();
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(out->writes == 1);  // em dia
    // Mudar a configuração do arquivo suja só a saída.
    req.settings.strokeWidthPt = 0.2;
    rig.engine.setExportRequests({req});
    CHECK(rig.engine.firstDirtyStage() == Stage::Output);
    (void)rig.rec->take();
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(out->writes == 2);
    const auto stages = finishedStages(rig.rec->take());
    for (std::size_t i = 0; i + 1 < stages.size(); ++i) CHECK(stages[i].second);
    CHECK(!stages.back().second);
    // Ocultar uma face (fundo) também muda a impressão digital da saída.
    req.settings.background.setHidden(1, true);
    rig.engine.setExportRequests({req});
    CHECK(rig.engine.firstDirtyStage() == Stage::Output);
    // Só o formato (mesmo arquivo) também suja a saída.
    {
        cnx::ExportRequest onlyFormat = req;
        onlyFormat.format = cnx::ExportFormat::Eps;
        rig.engine.setExportRequests({onlyFormat});
        CHECK(rig.run() == RunStatus::Completed);
        CHECK(out->writes == 3);
        rig.engine.setExportRequests({req});
        CHECK(rig.engine.firstDirtyStage() == Stage::Output);
        CHECK(rig.run() == RunStatus::Completed);
        CHECK(out->writes == 4);
    }
    // Mudar o formato idem; dois pedidos viram dois no estágio.
    cnx::ExportRequest second = req;
    second.format = cnx::ExportFormat::Pdf;
    second.file = "saida.pdf";
    rig.engine.setExportRequests({req, second});
    CHECK(rig.run() == RunStatus::Completed);
    CHECK(out->writes == 5 && out->lastRequests == 2);
    // Uma falha na escrita vira Failed e o snapshot anterior continua o publicado.
    class Broken final : public cnx::IOutputStage {
    public:
        void write(const cnx::PipelineSnapshot&, const std::vector<cnx::ExportRequest>&, cnx::TaskContext&) override { throw std::runtime_error("disco cheio"); }
    };
    const auto good = rig.engine.snapshot();
    rig.engine.setOutputStage(std::make_shared<Broken>());
    second.file = "outro.pdf";
    rig.engine.setExportRequests({req, second});
    CHECK(rig.run() == RunStatus::Failed && rig.engine.lastMessage().find("disco cheio") != std::string::npos);
    CHECK(rig.engine.snapshot() == good);
}

CNX_TEST(readers_are_safe_while_a_run_is_in_progress) {
    Rig rig;
    rig.loader->image = aaDisk();
    rig.engine.start();
    std::atomic<bool> stop{false};
    std::atomic<int> reads{0};
    std::thread reader([&] {
        while (!stop) {
            (void)rig.engine.snapshot();
            (void)rig.engine.parameters();
            (void)rig.engine.isRunning();
            (void)rig.engine.firstDirtyStage();
            ++reads;
        }
    });
    CHECK(rig.engine.wait() == RunStatus::Completed);
    stop = true;
    reader.join();
    CHECK(reads > 0 && rig.engine.snapshot() != nullptr);
    CHECK(cnx::toString(Stage::SmoothContours) == "SmoothContours" && cnx::kStageCount == 7);
}

int main() { return cnxtest::runAll(); }
