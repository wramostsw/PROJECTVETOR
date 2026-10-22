// cnx/vector/EnergyTerms.h — termos da energia de suavização e seus gradientes analíticos (§2.4).
//
//   E = λ_fid · (E_mola  ou  E_gen  ou  nada)
//     + λ_ang · E_ang + λ_sp · E_sp + λ_air · E_air + λ_brd · E_brd
//
// Cada termo soma o gradiente em `grad` (um Vec2 por nó, acumulando; span vazio = só a energia).
// As posições vêm de VectorImage::position e a escadinha original de originalPosition.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/vector/GenerativeModel.h"
#include "cnx/vector/VectorImage.h"
#include "sla/Vec.h"

namespace cnx {

/// Parte estática dos termos: depende só da topologia (arestas, laços, moldura), não das posições.
/// Calculada uma vez por fase.
struct SmoothTopology {
    /// Nó interior de uma aresta (ou o nó de uma aresta fechada) com seus dois vizinhos na cadeia.
    struct Triple {
        NodeId prev = kInvalidId;
        NodeId mid = kInvalidId;
        NodeId next = kInvalidId;
        EdgeId edge = kInvalidId;
        std::uint32_t index = 0;  ///< posição de `mid` em edgeNodes(edge)
    };
    /// Contorno pequeno (< airMaxNodes nós) sujeito à pressão do ar.
    struct AirLoop {
        std::vector<NodeId> nodes;  ///< em ordem de percurso, sem repetir o primeiro
        double originalArea = 0.0;  ///< área assinada na escadinha original
    };
    /// Nó sobre a moldura (não fixo): é puxado para a linha x = c ou y = c.
    struct BorderAnchor {
        NodeId node = kInvalidId;
        std::uint8_t axis = 0;  ///< 0: coordenada x, 1: coordenada y
        double coordinate = 0.0;
    };

    std::vector<Triple> triples;
    std::vector<AirLoop> airLoops;
    std::vector<BorderAnchor> borderAnchors;

    /// Faces ≥ 1 apenas (o exterior não tem contorno próprio). Os ângulos e o espaçamento não
    /// existem nas junções: ali a cadeia da aresta termina.
    static SmoothTopology build(const VectorImage& image, int airMaxNodes);
};

/// Pesos de uma fase.
struct EnergyWeights {
    MeasurementType measurement = MeasurementType::Spring;
    double fidelity = 1.0;
    double freeRadius = 0.0;
    double angular = 10.0;
    double spacing = 0.5;
    double air = 5.0;
    double border = 10.0;
};

/// Pesos da fase `phase` (0-based) a partir dos parâmetros. Lança std::invalid_argument se
/// phase estiver fora de [0, 3) ou priorType != 3.
EnergyWeights phaseWeights(const SmoothParams& params, int phase);

struct EnergyBreakdown {
    double spring = 0.0;      ///< já multiplicado por λ_fid
    double generative = 0.0;  ///< idem
    double angular = 0.0;
    double spacing = 0.0;
    double air = 0.0;
    double border = 0.0;
    double total() const noexcept { return spring + generative + angular + spacing + air + border; }
};

// ---- termos individuais; `weight` multiplica energia e gradiente ----

/// Σ_n max(0, ‖P_n − O_n‖ − r)². r = 0 é a mola simples; r > 0 é a zona livre (C¹).
double springEnergy(const VectorImage& image, double weight, double freeRadius, std::span<sla::Vec2> grad);

/// Σ θ², θ = atan2(u × v, u · v), u = P_k − P_{k−1}, v = P_{k+1} − P_k, nos nós de `topology.triples`
/// sem a flag Corner (canto perfurado). Segmentos de comprimento nulo são ignorados.
double angularEnergy(const VectorImage& image, const SmoothTopology& topology, double weight, std::span<sla::Vec2> grad);

/// Σ (‖Δ_k‖ − ‖Δ_{k−1}‖)² nos nós de `topology.triples`.
double spacingEnergy(const VectorImage& image, const SmoothTopology& topology, double weight, std::span<sla::Vec2> grad);

/// Σ_laços (A(P) − A₀)², A = área assinada do laço (shoelace).
double airPressureEnergy(const VectorImage& image, const SmoothTopology& topology, double weight, std::span<sla::Vec2> grad);

/// Σ (coordenada − âncora)² dos nós da moldura: eles deslizam ao longo dela, mas não saem dela.
double borderEnergy(const VectorImage& image, const SmoothTopology& topology, double weight, std::span<sla::Vec2> grad);

/// Energia total e gradiente (zerado e depois acumulado; span vazio = só a energia).
/// `model` é obrigatório se weights.measurement == Generative (senão std::invalid_argument).
/// Lança std::invalid_argument se grad não estiver vazio e tiver tamanho != nodeCount().
double evaluateEnergy(const VectorImage& image, const SmoothTopology& topology, const GenerativeModel* model,
                      const EnergyWeights& weights, std::span<sla::Vec2> grad, EnergyBreakdown* parts = nullptr);

// ---- canto perfurado ----

struct PunctureMeasure {
    double turn = 0.0;           ///< giro (rad, com sinal) entre as cordas dos braços, em P_k
    double deviationBack = 0.0;  ///< maior afastamento (px) de um nó do braço anterior à sua corda
    double deviationFwd = 0.0;
    double armBack = 0.0;        ///< comprimento real percorrido (≤ arm; menor se a cadeia termina antes)
    double armFwd = 0.0;
};

/// Filtro de caixa ao longo do comprimento de arco: cada nó vira a média da posição sobre [s − h, s + h] (integral
/// exata sobre a poligonal). Tira o ruído de "escadinha" de bordas inclinadas (degraus de 1 px) sem mudar o giro de
/// um canto, medido sobre braços de vários px. Numa cadeia aberta a janela encolhe de forma simétrica perto das
/// pontas (os extremos não se movem); numa fechada é periódica. h ≤ 0 devolve a cópia.
std::vector<sla::Vec2> smoothChain(std::span<const sla::Vec2> pts, bool closed, double h);

/// Mede o nó k da poligonal `pts` (fechada: o último não repete o primeiro). Cada braço segue a
/// poligonal por até `arm` px de comprimento de arco, parando no fim da cadeia aberta.
PunctureMeasure measureCorner(std::span<const sla::Vec2> pts, std::size_t k, bool closed, double arm);

/// `isProbablyPuncturedNode`: |giro| ≥ cornerThresh e os dois braços retos (desvio ≤ flatnessThresh).
/// cornerThresh ≤ 0 desativa (retorna falso).
bool isProbablyPuncturedNode(std::span<const sla::Vec2> pts, std::size_t k, bool closed, double arm, double cornerThresh,
                             double flatnessThresh);

}  // namespace cnx
