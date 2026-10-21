// src/vector/ContourSmoother.cpp — 3 fases de CG com perfuração de cantos e fiscal anti-cruzamento.
#include "cnx/vector/ContourSmoother.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cnx/vector/CrossingGuard.h"

namespace cnx {
namespace {

using sla::Vec2;

/// Energia da fase como função das coordenadas dos nós livres, x = [x_0, y_0, x_1, y_1, …].
/// Cada avaliação escreve as posições no VectorImage (o chamador restaura o ponto aceito).
class PhaseObjective final : public ObjectiveFunction {
public:
    /// `lockBorder`: a coordenada perpendicular dos nós da moldura vira constante (gradiente nulo).
    PhaseObjective(VectorImage& image, const SmoothTopology& topology, const GenerativeModel* model, EnergyWeights weights,
                   bool lockBorder)
        : image_(image), topology_(topology), model_(model), weights_(weights), full_(image.nodeCount()),
          lockedAxis_(image.nodeCount(), -1) {
        if (!lockBorder) return;
        for (const SmoothTopology::BorderAnchor& b : topology.borderAnchors) {
            lockedAxis_[static_cast<std::size_t>(b.node)] = static_cast<std::int8_t>(b.axis);
            // Coordenada exata na moldura: a posição inicial pode ter derivado de uma fase anterior sem trava.
            Vec2 p = image.position(b.node);
            (b.axis == 0 ? p.x : p.y) = b.coordinate;
            image.setPosition(b.node, p);
        }
    }

    void setFreeNodes(std::vector<NodeId> nodes) { free_ = std::move(nodes); }
    const std::vector<NodeId>& freeNodes() const noexcept { return free_; }

    double evaluate(std::span<const double> x, std::span<double> grad) override {
        image_.scatter(free_, x);
        const double e = evaluateEnergy(image_, topology_, model_, weights_, full_, nullptr);
        for (std::size_t i = 0; i < free_.size(); ++i) {
            const auto n = static_cast<std::size_t>(free_[i]);
            const Vec2 g = full_[n];
            grad[2 * i] = lockedAxis_[n] == 0 ? 0.0 : g.x;
            grad[2 * i + 1] = lockedAxis_[n] == 1 ? 0.0 : g.y;
        }
        return e;
    }

    /// Energia e termos no estado atual do VectorImage.
    double current(EnergyBreakdown* parts) {
        return evaluateEnergy(image_, topology_, model_, weights_, {}, parts);
    }

private:
    VectorImage& image_;
    const SmoothTopology& topology_;
    const GenerativeModel* model_;
    EnergyWeights weights_;
    std::vector<NodeId> free_;
    std::vector<Vec2> full_;
    std::vector<std::int8_t> lockedAxis_;  ///< eixo travado por nó (-1 = nenhum)
};

}  // namespace

ContourSmoother::ContourSmoother(SmoothParams params, const GenerativeModel* model) : params_(params), model_(model) {
    if (params_.maxCgIterations < 0) throw std::invalid_argument("ContourSmoother: maxCgIterations < 0");
    if (!(params_.maxNodeStepPx > 0.0)) throw std::invalid_argument("ContourSmoother: maxNodeStepPx <= 0");
    if (params_.quarantineRounds < 0) throw std::invalid_argument("ContourSmoother: quarantineRounds < 0");
    (void)phaseWeights(params_, 0);  // valida priorType
}

int ContourSmoother::puncture(VectorImage& image, const SmoothTopology& topology, const SmoothParams& params) {
    std::vector<NodeId> marked;
    std::vector<Vec2> pts;
    EdgeId currentEdge = kInvalidId;
    bool closed = false;
    for (const SmoothTopology::Triple& t : topology.triples) {
        if (t.edge != currentEdge) {  // as triplas de uma aresta são consecutivas
            currentEdge = t.edge;
            const auto nodes = image.edgeNodes(t.edge);
            closed = nodes.front() == nodes.back();
            pts.clear();
            for (std::size_t i = 0; i < nodes.size() - (closed ? 1 : 0); ++i) pts.push_back(image.position(nodes[i]));
        }
        if ((image.flags(t.mid) & NodeFlag::Corner) != 0) continue;
        if (isProbablyPuncturedNode(pts, t.index, closed, params.punctureArmPx, params.punctureCornerThresh,
                                    params.punctureFlatnessThresh)) {
            marked.push_back(t.mid);
        }
    }
    // Decide com as posições da fase anterior e só então marca (a ordem não influencia).
    int count = 0;
    for (const NodeId n : marked) {
        if ((image.flags(n) & NodeFlag::Corner) == 0) {
            image.setFlags(n, static_cast<std::uint8_t>(image.flags(n) | NodeFlag::Corner));
            ++count;
        }
    }
    return count;
}

SmoothResult ContourSmoother::run(VectorImage& image, TaskContext ctx) const {
    if (!CrossingGuard::find(image).empty()) {
        throw std::invalid_argument("ContourSmoother::run: os contornos já se cruzam na entrada");
    }
    if (!CrossingGuard::shortSegmentNodes(image, params_.minSegmentPx).empty()) {
        throw std::invalid_argument("ContourSmoother::run: há segmentos mais curtos que minSegmentPx na entrada");
    }
    SmoothResult result;
    const double nPhases = static_cast<double>(SmoothParams::kPhases);

    for (int phase = 0; phase < SmoothParams::kPhases; ++phase) {
        PhaseStats& st = result.phases[static_cast<std::size_t>(phase)];
        const EnergyWeights weights = phaseWeights(params_, phase);
        if (weights.measurement == MeasurementType::Generative && model_ == nullptr) {
            throw std::invalid_argument("ContourSmoother::run: fase generativa sem GenerativeModel");
        }
        const SmoothTopology topology = SmoothTopology::build(image, params_.airPressureMaxNodes);
        if (phase == params_.punctureOnPhase) st.puncturedNodes = puncture(image, topology, params_);

        PhaseObjective objective(image, topology, model_, weights, params_.borderHardConstraint);
        CgOptions options;
        options.maxIterations = params_.maxCgIterations + 1;  // o orçamento é controlado aqui
        options.gradientTol = params_.gradientTolerance;
        options.relEnergyTol = params_.energyTolerance;
        options.maxStep = params_.maxNodeStepPx;
        options.stride = 2;
        ConjugateGradient cg(objective, options);

        std::vector<double> x;
        int evaluations = 0;
        const auto rebuild = [&]() {
            evaluations += cg.evaluations();
            objective.setFreeNodes(image.collectFreeNodes());
            x.assign(2 * objective.freeNodes().size(), 0.0);
            image.gather(objective.freeNodes(), x);
            cg.reset(x);
        };
        rebuild();  // `evaluations` soma as avaliações de cada CG (a de reset() entra na reconstrução seguinte)
        st.freeNodes = static_cast<int>(objective.freeNodes().size());
        st.energyBegin = cg.value();
        st.status = CgStatus::MaxIterations;

        std::vector<double> before;
        for (int it = 0; it < params_.maxCgIterations; ++it) {
            ctx.checkpoint((static_cast<double>(phase) + static_cast<double>(it) / static_cast<double>(params_.maxCgIterations)) / nPhases);
            before.assign(cg.x().begin(), cg.x().end());
            const CgStatus s = cg.iterate();
            if (s == CgStatus::LineSearchFailed) {
                image.scatter(objective.freeNodes(), before);
                st.status = s;
                break;
            }
            image.scatter(objective.freeNodes(), cg.x());

            const std::vector<Crossing> crossings = CrossingGuard::find(image);
            std::vector<NodeId> offenders = CrossingGuard::offendingNodes(crossings);
            const std::vector<NodeId> collapsed = CrossingGuard::shortSegmentNodes(image, params_.minSegmentPx);
            offenders.insert(offenders.end(), collapsed.begin(), collapsed.end());
            if (!offenders.empty()) {
                // Desfaz a iteração, congela os nós envolvidos e recomeça o CG do ponto anterior.
                image.scatter(objective.freeNodes(), before);
                for (const NodeId n : offenders) {
                    image.quarantine(n, static_cast<std::uint8_t>(std::min(255, params_.quarantineRounds + 1)));
                }
                ++st.crossingRejects;
                rebuild();
            } else {
                ++st.iterations;
            }

            image.tickQuarantine();
            if (image.collectFreeNodes() != objective.freeNodes()) rebuild();  // alguém saiu da quarentena
            if (offenders.empty() && s != CgStatus::Running) {
                st.status = s;
                break;
            }
        }
        evaluations += cg.evaluations();
        st.evaluations = evaluations;

        // Estado final aceito = ponto do CG; a quarentena não vaza para a fase seguinte.
        image.scatter(objective.freeNodes(), cg.x());
        for (NodeId n = 0; n < static_cast<NodeId>(image.nodeCount()); ++n) image.quarantine(n, 0);
        st.energyEnd = objective.current(&st.termsEnd);
        ctx.checkpoint(static_cast<double>(phase + 1) / nPhases);
    }
    image.updateFaceAreas();
    return result;
}

}  // namespace cnx
