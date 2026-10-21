// cnx/engine/TaskContext.h — cancelamento cooperativo e progresso por faixas.
#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <stop_token>
#include <utility>

namespace cnx {

/// Lançada por TaskContext::throwIfCancelled(); capturada pelo orquestrador do CoreEngine,
/// que descarta o snapshot parcial. Estágios não devem capturá-la.
class CancelTaskException final : public std::exception {
public:
    const char* what() const noexcept override { return "cnx: tarefa cancelada"; }
};

/// Faixa [begin, end] da barra de progresso global (ex.: suavização = [0.20, 0.80]).
struct ProgressRange {
    double begin = 0.0;
    double end = 1.0;

    /// Converte uma fração local ∈ [0, 1] (saturada) em fração global.
    constexpr double map(double local) const noexcept {
        const double c = local < 0.0 ? 0.0 : (local > 1.0 ? 1.0 : local);
        return begin + c * (end - begin);
    }
    /// Subfaixa [a, b] ⊂ [0, 1], relativa a esta faixa.
    constexpr ProgressRange sub(double a, double b) const noexcept { return {map(a), map(b)}; }
};

/// Contexto entregue a cada estágio.
///
/// Thread-safety: cancelled()/throwIfCancelled() podem ser chamados de qualquer thread do pool.
/// report()/checkpoint() só pela thread que executa o estágio (o throttling não é atômico);
/// tarefas paralelas usam throwIfCancelled() e o estágio agrega o progresso.
class TaskContext {
public:
    using ProgressFn = std::function<void(double globalFraction)>;

    TaskContext() = default;
    TaskContext(std::stop_token stop, std::shared_ptr<const ProgressFn> progress, ProgressRange range) noexcept
        : stop_(std::move(stop)), progress_(std::move(progress)), range_(range) {}

    const std::stop_token& stopToken() const noexcept { return stop_; }
    bool cancelled() const noexcept { return stop_.stop_requested(); }

    void throwIfCancelled() const {
        if (cancelled()) throw CancelTaskException{};
    }

    /// Publica progresso; descarta incrementos menores que kMinStep para não saturar a GUI.
    void report(double localFraction) {
        if (!progress_ || !*progress_) return;
        const double global = range_.map(localFraction);
        if (global - lastReported_ < kMinStep && localFraction < 1.0) return;
        lastReported_ = global;
        (*progress_)(global);
    }

    /// Ponto de verificação típico de um laço longo: cancela ou reporta.
    void checkpoint(double localFraction) {
        throwIfCancelled();
        report(localFraction);
    }

    /// Contexto filho para uma subetapa [a, b] deste estágio.
    TaskContext sub(double a, double b) const { return TaskContext(stop_, progress_, range_.sub(a, b)); }

    ProgressRange range() const noexcept { return range_; }

private:
    static constexpr double kMinStep = 1e-3;

    std::stop_token stop_;
    std::shared_ptr<const ProgressFn> progress_;
    ProgressRange range_{};
    double lastReported_ = -1.0;
};

}  // namespace cnx
