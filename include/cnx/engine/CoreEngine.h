// cnx/engine/CoreEngine.h — fachada do motor: pipeline de estágios, recálculo inteligente,
// cancelamento e progresso.
//
// Modelo de execução (docs/ARQUITETURA.md §0 item 1, §1.3 e §1.6): os estágios são
// sequencialmente dependentes, então um único orquestrador (std::jthread) os executa em ordem.
// O paralelismo é de dados, dentro de cada estágio. O núcleo não depende do Qt; o adaptador
// qt/EngineController roda esta fachada numa QThread e reemite os callbacks como sinais.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "cnx/engine/Fingerprint.h"
#include "cnx/engine/Parameters.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/export/Exporter.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

struct PreprocessedImage;    // SignalImage, EdgeImage, FlowField, perfil de AA
struct ImageClassification;  // 20 medidas + perfil escolhido
struct Palette;
class PixelMap;
class VectorImage;

/// Estágios na ordem de execução. Correspondência com a especificação: LoadImageThread,
/// ClassifyImageThread (inclui o Preprocessor), FindPalettesThread, SegmentationThread,
/// ContourSmoothingThread, BezierFittingThread, Save/VectorRender/ThumbsThread.
enum class Stage : std::uint8_t { LoadImage, Classify, Palette, Segment, SmoothContours, FitBezier, Output };

inline constexpr std::size_t kStageCount = 7;
constexpr std::size_t stageIndex(Stage s) noexcept { return static_cast<std::size_t>(s); }
std::string_view toString(Stage s) noexcept;

/// Faixas da barra de progresso global (§1 da especificação; paleta + segmentação = 0–20 %).
inline constexpr std::array<ProgressRange, kStageCount> kStageProgress{{
    {0.00, 0.00},  // LoadImage
    {0.00, 0.00},  // Classify
    {0.00, 0.05},  // Palette
    {0.05, 0.20},  // Segment
    {0.20, 0.80},  // SmoothContours
    {0.80, 0.90},  // FitBezier
    {0.90, 1.00},  // Output
}};

/// Imagem decodificada, 8 bits por canal. `formatHint` (".png", ".jpeg", ".bmp", ".tiff")
/// é uma das medidas do classificador.
struct SourceImage {
    sla::Array2D<sla::Rgba8> pixels;
    std::string formatHint;
    bool hasAlpha = false;
    double dpiX = 72.0;
    double dpiY = 72.0;
};

/// Decodificador plugável. O adaptador Qt implementa com QImage (plugins nativos, sem ImageMagick).
class IImageLoader {
public:
    virtual ~IImageLoader() = default;
    /// Lança std::runtime_error (ou derivada) em falha; deve respeitar `stop` em arquivos grandes.
    virtual SourceImage load(const std::filesystem::path& file, const std::stop_token& stop) = 0;
};

enum class RunStatus : std::uint8_t { Completed, Cancelled, Failed };

/// Callbacks chamados na thread do orquestrador. Uma GUI deve repassá-los à thread da interface
/// (ex.: sinal Qt com Qt::QueuedConnection). Implementações não devem lançar exceções.
class IEngineObserver {
public:
    virtual ~IEngineObserver() = default;
    virtual void onProgress(Stage stage, double globalFraction) = 0;
    virtual void onStageFinished(Stage stage, double elapsedSeconds, bool reusedFromCache) = 0;
    virtual void onRunFinished(RunStatus status, const std::string& message) = 0;
};

/// Artefatos imutáveis de uma execução completa. Uma nova execução copia os ponteiros do snapshot
/// anterior e substitui só os artefatos dos estágios sujos (copy-on-write). Nunca é publicado parcial.
struct PipelineSnapshot {
    std::shared_ptr<const SourceImage> source;
    std::shared_ptr<const PreprocessedImage> preprocessed;
    std::shared_ptr<const ImageClassification> classification;
    std::shared_ptr<const Palette> palette;
    std::shared_ptr<const PixelMap> segmentation;
    std::shared_ptr<const VectorImage> smoothed;  ///< poligonal, após ContourSmoother
    std::shared_ptr<const VectorImage> fitted;    ///< com curvas, após BezierFitter
    std::array<Fingerprint, kStageCount> fingerprints{};
};

class CoreEngine {
public:
    /// workerThreads = 0: std::thread::hardware_concurrency().
    explicit CoreEngine(std::shared_ptr<IImageLoader> loader, unsigned workerThreads = 0);
    /// Solicita cancelamento e aguarda o término.
    ~CoreEngine();

    CoreEngine(const CoreEngine&) = delete;
    CoreEngine& operator=(const CoreEngine&) = delete;

    // ---------------------------------------------------------------- configuração (thread-safe;
    //                                                                  vale a partir da próxima execução)
    void setObserver(std::shared_ptr<IEngineObserver> observer);
    void setInput(const std::filesystem::path& file);
    void setParameters(const EngineParameters& params);
    EngineParameters parameters() const;
    void setExportRequests(std::vector<ExportRequest> requests);

    // ---------------------------------------------------------------- recálculo inteligente
    /// Primeiro estágio cuja impressão digital mudou; nullopt = tudo em dia.
    std::optional<Stage> firstDirtyStage() const;
    bool needsToSegment() const;
    bool needsToBezierFit() const;

    // ---------------------------------------------------------------- execução
    /// Assíncrono. Executa os estágios sujos até `target`, inclusive. Se houver uma execução em
    /// curso, ela é cancelada e aguardada antes.
    void start(Stage target = Stage::Output);
    void cancel() noexcept;
    /// Bloqueia até o fim da execução corrente (ou retorna o status da última).
    RunStatus wait();
    bool isRunning() const noexcept;

    /// Último snapshot completo publicado; seguro de ler durante outra execução.
    std::shared_ptr<const PipelineSnapshot> snapshot() const;

    /// Edição manual (§4): substitui a segmentação. A impressão digital do estágio Segment passa a
    /// ser o hash das edições encadeado ao da paleta, o que suja SmoothContours em diante.
    void replaceSegmentation(std::shared_ptr<const PixelMap> edited, Fingerprint editsHash);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cnx
