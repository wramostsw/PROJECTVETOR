// cnx/vector/ContourSmoother.h — suavização elástica dos contornos em 3 fases (§5, docs §1.13).
//
// Move os nós livres do VectorImage (todos menos Pinned e os em quarentena) para minimizar
//     E = λ_fid·(mola | E_gen | nada) + λ_ang·E_ang + λ_sp·E_sp + λ_air·E_air + λ_brd·E_brd
// por gradiente conjugado Polak–Ribière+ com Wolfe forte (Optimizer.h), cada fase com seus
// pesos (SmoothParams). Na fase `punctureOnPhase` os cantos reais são perfurados (flag Corner:
// o prior angular é desligado ali). Depois de cada iteração aceita, o fiscal anti-cruzamento
// confere as poligonais; se houver cruzamento, a iteração é desfeita, os nós envolvidos ficam em
// quarentena por `quarantineRounds` iterações e o CG recomeça.
//
// Invariantes: nenhuma iteração aceita deixa cruzamentos; E decresce a cada iteração aceita (a
// busca de Wolfe garante a queda; o recomeço após quarentena parte do ponto anterior); os nós
// Pinned nunca se movem; as arestas compartilhadas continuam compartilhadas (um nó, uma posição).
#pragma once

#include <array>

#include "cnx/engine/Parameters.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/vector/EnergyTerms.h"
#include "cnx/vector/GenerativeModel.h"
#include "cnx/vector/Optimizer.h"
#include "cnx/vector/VectorImage.h"

namespace cnx {

struct PhaseStats {
    int iterations = 0;       ///< iterações aceitas do CG
    int evaluations = 0;      ///< avaliações de energia (inclui a busca em linha)
    int crossingRejects = 0;  ///< iterações desfeitas pelo fiscal
    int puncturedNodes = 0;   ///< cantos perfurados nesta fase
    int freeNodes = 0;        ///< nós livres no início da fase
    double energyBegin = 0.0;
    double energyEnd = 0.0;
    EnergyBreakdown termsEnd;
    CgStatus status = CgStatus::Running;
};

struct SmoothResult {
    std::array<PhaseStats, SmoothParams::kPhases> phases;
};

class ContourSmoother {
public:
    /// `model` é opcional (fases com MeasurementType::Generative exigem um); deve sobreviver ao objeto.
    explicit ContourSmoother(SmoothParams params, const GenerativeModel* model = nullptr);

    /// Executa as 3 fases sobre `image` (modifica as posições, as flags Corner e a quarentena).
    /// Lança CancelTaskException se `ctx` for cancelado (as posições ficam no último estado aceito),
    /// std::invalid_argument se os contornos já se cruzarem na entrada ou os parâmetros forem
    /// inválidos. Ao final, Face::area é atualizada.
    SmoothResult run(VectorImage& image, TaskContext ctx = {}) const;

    /// Perfura os cantos: marca com NodeFlag::Corner os nós de arestas que isProbablyPuncturedNode
    /// aceita. Retorna quantos foram marcados agora. Exposta para testes.
    static int puncture(VectorImage& image, const SmoothTopology& topology, const SmoothParams& params);

private:
    SmoothParams params_;
    const GenerativeModel* model_;
};

}  // namespace cnx
