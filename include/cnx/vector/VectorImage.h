// cnx/vector/VectorImage.h — mapa planar por meia-aresta (DCEL) com fronteiras compartilhadas.
//
// Cada fronteira entre duas regiões é uma Edge armazenada uma única vez. Seus nós (variáveis do
// otimizador) e suas curvas (BezierFitter) pertencem à aresta, não às faces. Cada face só guarda
// meias-arestas: h = 2e percorre e no sentido direto (face `left`), h = 2e+1 no inverso (face
// `right`). Isso garante "zero frestas" da suavização até a exportação (docs/ARQUITETURA.md §1.4).
//
// Invariantes (validate()):
//   I1  todo nó interior pertence a exatamente uma aresta; junções só são referenciadas;
//   I2  laço externo de cada face tem área > 0, furos têm área < 0 (caminhos compostos com nonzero);
//   I3  V − E + F = 1 + C (C = componentes conexas do grafo de arestas);
//   I4  Σ_f área(f) = largura × altura (exterior excluído).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "sla/BezierCurve.h"
#include "sla/Vec.h"

namespace cnx {

using NodeId = std::int32_t;
using EdgeId = std::int32_t;
using HalfEdgeId = std::int32_t;
using FaceId = std::int32_t;

inline constexpr std::int32_t kInvalidId = -1;
/// A face 0 é sempre o exterior da imagem: cor transparente (= 0 no modelo gerador), nunca exportada.
inline constexpr FaceId kExteriorFace = 0;

namespace NodeFlag {
inline constexpr std::uint8_t Junction = 1u << 0;     ///< 3+ faces ou sela 2×2: quebra obrigatória do BezierFitter
inline constexpr std::uint8_t ImageBorder = 1u << 1;  ///< sobre a moldura (âncora de margem)
inline constexpr std::uint8_t Corner = 1u << 2;       ///< canto perfurado: sem prior de suavidade
inline constexpr std::uint8_t Pinned = 1u << 3;       ///< nunca se move (ex.: cantos da moldura)
inline constexpr std::uint8_t LoopAnchor = 1u << 4;   ///< vértice sintético de aresta fechada sem junções
}  // namespace NodeFlag

constexpr HalfEdgeId halfEdge(EdgeId e, bool reversed) noexcept { return 2 * e + (reversed ? 1 : 0); }
constexpr HalfEdgeId twin(HalfEdgeId h) noexcept { return h ^ 1; }
constexpr EdgeId edgeOf(HalfEdgeId h) noexcept { return h >> 1; }
constexpr bool isReversed(HalfEdgeId h) noexcept { return (h & 1) != 0; }

/// Cadeia de fronteira entre duas faces, de junção a junção.
/// Percorrida no sentido direto, a face `left` fica do lado de área positiva:
/// n = (Δy, −Δx) aponta para fora de `left` e para dentro de `right`.
struct Edge {
    std::uint32_t nodeBegin = 0;   ///< faixa em edgeNodes_; inclui as junções das pontas
    std::uint32_t nodeCount = 0;   ///< ≥ 2; aresta fechada: primeiro == último (LoopAnchor)
    FaceId left = kInvalidId;
    FaceId right = kInvalidId;
    std::uint32_t curveBegin = 0;  ///< faixa em curves_ (sentido direto); vazia antes do ajuste
    std::uint32_t curveCount = 0;
};

struct Face {
    sla::Rgba color{};
    std::int32_t paletteIndex = -1;
    HalfEdgeId outerLoop = kInvalidId;  ///< laço com área > 0 (kInvalidId no exterior)
    std::uint32_t holeBegin = 0;        ///< faixa em holeLoops_; cada laço com área < 0
    std::uint32_t holeCount = 0;
    FaceId parent = kInvalidId;         ///< face cujo furo contém esta (ordem de desenho, modo "bolo")
    double area = 0.0;                  ///< área assinada total (externo + furos); ver updateFaceAreas()
};

class VectorImage {
public:
    VectorImage() = default;
    /// Cria a face exterior (id 0).
    VectorImage(double width, double height);

    double width() const noexcept { return width_; }
    double height() const noexcept { return height_; }

    // ---------------------------------------------------------------- construção (BoundaryTracer)
    NodeId addNode(sla::Vec2 position, std::uint8_t flags = 0);
    FaceId addFace(const sla::Rgba& color, std::int32_t paletteIndex);
    /// nodes = junção inicial, nós interiores, junção final (sentido direto, `left` no lado positivo).
    EdgeId addEdge(std::span<const NodeId> nodes, FaceId left, FaceId right);
    /// Monta next_ por ordenação angular das meias-arestas em cada junção, separa laço externo e
    /// furos pelo sinal da área, preenche Face::outerLoop/holes/parent/area.
    void linkFaceLoops();

    // ---------------------------------------------------------------- nós (variáveis do otimizador)
    std::size_t nodeCount() const noexcept { return pos_.size(); }
    sla::Vec2 position(NodeId n) const noexcept { return pos_[ix(n)]; }
    void setPosition(NodeId n, sla::Vec2 p) noexcept { pos_[ix(n)] = p; }
    /// Posição na escadinha inicial (âncora da mola de fidelidade).
    sla::Vec2 originalPosition(NodeId n) const noexcept { return origin_[ix(n)]; }
    std::uint8_t flags(NodeId n) const noexcept { return flags_[ix(n)]; }
    void setFlags(NodeId n, std::uint8_t f) noexcept { flags_[ix(n)] = f; }

    std::span<sla::Vec2> positions() noexcept { return pos_; }
    std::span<const sla::Vec2> positions() const noexcept { return pos_; }
    std::span<const sla::Vec2> originalPositions() const noexcept { return origin_; }

    /// Quarentena (§5): nó congelado por `rounds` iterações do CG.
    void quarantine(NodeId n, std::uint8_t rounds) noexcept { quarantine_[ix(n)] = rounds; }
    bool isQuarantined(NodeId n) const noexcept { return quarantine_[ix(n)] != 0; }
    void tickQuarantine() noexcept;

    // ---------------------------------------------------------------- topologia
    std::size_t edgeCount() const noexcept { return edges_.size(); }
    std::size_t faceCount() const noexcept { return faces_.size(); }
    const Edge& edge(EdgeId e) const noexcept { return edges_[ix(e)]; }
    const Face& face(FaceId f) const noexcept { return faces_[ix(f)]; }
    Face& face(FaceId f) noexcept { return faces_[ix(f)]; }

    /// Nós da aresta no sentido direto.
    std::span<const NodeId> edgeNodes(EdgeId e) const noexcept {
        const Edge& ed = edges_[ix(e)];
        return {edgeNodes_.data() + ed.nodeBegin, ed.nodeCount};
    }
    HalfEdgeId next(HalfEdgeId h) const noexcept { return next_[ix(h)]; }
    FaceId faceOf(HalfEdgeId h) const noexcept {
        const Edge& ed = edges_[ix(edgeOf(h))];
        return isReversed(h) ? ed.right : ed.left;
    }
    NodeId origin(HalfEdgeId h) const noexcept {
        const auto nodes = edgeNodes(edgeOf(h));
        return isReversed(h) ? nodes.back() : nodes.front();
    }
    NodeId target(HalfEdgeId h) const noexcept { return origin(twin(h)); }
    std::span<const HalfEdgeId> holeLoops(FaceId f) const noexcept {
        const Face& fc = faces_[ix(f)];
        return {holeLoops_.data() + fc.holeBegin, fc.holeCount};
    }

    /// Área assinada de um laço. Por aresta: fórmula fechada de Green sobre as cúbicas, se a aresta
    /// tiver curvas; senão, shoelace sobre os nós.
    double loopArea(HalfEdgeId first) const;
    /// Recalcula Face::area a partir dos laços (após mover nós ou trocar curvas).
    void updateFaceAreas();

    // ---------------------------------------------------------------- curvas (BezierFitter)
    void clearCurves();
    /// Pré-condição: a cadeia começa em position(edgeNodes(e).front()) e termina em .back().
    void setEdgeCurves(EdgeId e, std::span<const sla::CurveSegment> curves);
    std::span<const sla::CurveSegment> edgeCurves(EdgeId e) const noexcept {
        const Edge& ed = edges_[ix(e)];
        return {curves_.data() + ed.curveBegin, ed.curveCount};
    }
    bool hasCurves() const noexcept { return !curves_.empty(); }

    // ---------------------------------------------------------------- vetor do otimizador
    // x = [x_0, y_0, x_1, y_1, …] dos nós livres. A máscara é fixada no início de cada fase do CG.
    /// Nós livres: sem Pinned e fora de quarentena, em ordem crescente de id (determinismo).
    std::vector<NodeId> collectFreeNodes() const;
    void gather(std::span<const NodeId> freeNodes, std::span<double> x) const;
    void scatter(std::span<const NodeId> freeNodes, std::span<const double> x);
    /// origin ← pos (opcional, entre fases).
    void resetOriginalPositions();

    // ---------------------------------------------------------------- invariantes
    struct ValidationReport {
        bool ok = true;
        std::string message;
    };
    /// Verifica I1–I4, a consistência de next (permutação, continuidade, mesma face), o registro de
    /// cada laço em exatamente uma face e a ausência de ciclos em Face::parent. As áreas são
    /// recalculadas (não usa Face::area); tolerância relativa a largura × altura.
    ValidationReport validate(double areaTolerance = 1e-9) const;

private:
    static constexpr std::size_t ix(std::int32_t i) noexcept { return static_cast<std::size_t>(i); }

    double width_ = 0.0;
    double height_ = 0.0;

    // nós (SoA): pos_ é o que o CG move; origin_ é a escadinha do BoundaryTracer
    std::vector<sla::Vec2> pos_;
    std::vector<sla::Vec2> origin_;
    std::vector<std::uint8_t> flags_;
    std::vector<std::uint8_t> quarantine_;

    std::vector<NodeId> edgeNodes_;
    std::vector<Edge> edges_;
    std::vector<HalfEdgeId> next_;  // 2·|E| entradas
    std::vector<Face> faces_;
    std::vector<HalfEdgeId> holeLoops_;
    std::vector<sla::CurveSegment> curves_;
};

}  // namespace cnx
