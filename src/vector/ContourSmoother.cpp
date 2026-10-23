// src/vector/ContourSmoother.cpp — 3 fases de CG com perfuração de cantos e fiscal anti-cruzamento.
#include "cnx/vector/ContourSmoother.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cnx/vector/CrossingGuard.h"

namespace cnx {
namespace {

using sla::Vec2;

constexpr double kStairFilterPx = 1.0;  ///< meia janela mínima do filtro de arco da detecção de cantos

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
    if (!(params_.punctureMinContrast >= 0.0)) throw std::invalid_argument("ContourSmoother: punctureMinContrast negativo ou NaN");
    if (!(params_.contrastFullLevels >= 0.0)) throw std::invalid_argument("ContourSmoother: contrastFullLevels negativo ou NaN");
    if (!(params_.jogMaxLedgePx >= 0.0)) throw std::invalid_argument("ContourSmoother: jogMaxLedgePx negativo ou NaN");
    if (!(params_.contrastFloor > 0.0 && params_.contrastFloor <= 1.0)) throw std::invalid_argument("ContourSmoother: contrastFloor fora de (0, 1]");
    (void)phaseWeights(params_, 0);  // valida priorType
}

int ContourSmoother::puncture(VectorImage& image, const SmoothTopology& topology, const SmoothParams& params, int* jogNodes) {
    // Cada canto candidato, com o giro (com sinal, no sentido da aresta, `left` do lado positivo) e as faces da aresta,
    // para separar os cantos de verdade dos degraus de bloco do JPEG (abaixo).
    struct Mark {
        NodeId node;
        Vec2 pos;
        FaceId left;
        FaceId right;
        double turn;
    };
    std::vector<Mark> marked;
    // Duas escalas de braço: o braço inteiro e a metade. Um fio de 2,4 px ou a ponta cega de um triângulo têm lados
    // mais curtos que o braço; o braço passaria do canto seguinte e reprovaria o teste de retidão.
    const std::array<double, 2> arms{params.punctureArmPx, params.punctureArmPx * 0.5};
    std::array<std::vector<Vec2>, 2> pts;
    EdgeId currentEdge = kInvalidId;
    bool closed = false;
    for (const SmoothTopology::Triple& t : topology.triples) {
        if (t.edge != currentEdge) {  // as triplas de uma aresta são consecutivas
            currentEdge = t.edge;
            const auto nodes = image.edgeNodes(t.edge);
            closed = nodes.front() == nodes.back();
            std::vector<Vec2> raw;
            for (std::size_t i = 0; i < nodes.size() - (closed ? 1 : 0); ++i) raw.push_back(image.position(nodes[i]));
            // Sobre a escadinha, os degraus de 1 px de uma borda inclinada reprovariam o teste de retidão (≥ 0,35 px
            // de desvio) e, na escala pequena, os degraus de um arco pareceriam cantos: mede-se na poligonal filtrada ao
            // longo do arco, com meia janela max(¼ do braço, 1 px) (o degrau do pixel é a escala mínima).
            for (std::size_t k = 0; k < arms.size(); ++k) pts[k] = smoothChain(raw, closed, std::max(arms[k] / 4.0, kStairFilterPx));
        }
        if ((image.flags(t.mid) & NodeFlag::Corner) != 0) continue;
        // Entre faixas parecidas não há canto: os degraus do JPEG não são quinas do desenho (docs §1.29).
        if (topology.nodeContrast[static_cast<std::size_t>(t.mid)] < params.punctureMinContrast) continue;
        for (std::size_t k = 0; k < arms.size(); ++k) {
            if (isProbablyPuncturedNode(pts[k], t.index, closed, arms[k], params.punctureCornerThresh, params.punctureFlatnessThresh)) {
                const Edge& edge = image.edge(t.edge);
                marked.push_back({t.mid, image.position(t.mid), edge.left, edge.right, measureCorner(pts[k], t.index, closed, arms[k]).turn});
                break;
            }
        }
    }
    // Degrau de bloco do JPEG (docs §1.32): dois cantos a no máximo `jogMaxLedgePx` um do outro, na fronteira de uma mesma
    // face, de convexidades opostas (um entra e o outro sai), formam uma saliência de 1 a 2 px numa borda que segue reta
    // dos dois lados. Um canto de verdade vira e continua virado; a saliência é o deslocamento de um bloco 8×8 da imagem
    // comprimida, e o Vector Magic a alisa. Perfurar os dois a preservaria, e ainda quebraria o ajuste de Bézier ali. A
    // busca vale entre arestas (a saliência atravessa uma junção quando uma faixa fina termina nela).
    std::vector<char> jog(marked.size(), 0);
    int jogs = 0;
    if (params.jogMaxLedgePx > 0.0 && marked.size() > 1) {
        const double cell = params.jogMaxLedgePx;
        using Cell = std::pair<long, long>;
        std::map<Cell, std::vector<std::size_t>> grid;
        const auto cellOf = [&](Vec2 p) { return Cell{static_cast<long>(std::floor(p.x / cell)), static_cast<long>(std::floor(p.y / cell))}; };
        for (std::size_t i = 0; i < marked.size(); ++i) grid[cellOf(marked[i].pos)].push_back(i);
        // Convexidade do canto em relação à face f: o giro vale com sinal contrário se f está do outro lado da aresta.
        const auto convexity = [](const Mark& m, FaceId f) { return m.turn * (m.left == f ? 1.0 : -1.0); };
        const auto shares = [](const Mark& a, const Mark& b, FaceId& common) {
            for (const FaceId f : {a.left, a.right}) {
                if (f == kExteriorFace) continue;
                if (f == b.left || f == b.right) {
                    common = f;
                    return true;
                }
            }
            return false;
        };
        for (std::size_t i = 0; i < marked.size(); ++i) {
            if (jog[i]) continue;
            const Cell c = cellOf(marked[i].pos);
            bool found = false;
            for (long dy = -1; dy <= 1 && !found; ++dy) {
                for (long dx = -1; dx <= 1 && !found; ++dx) {
                    const auto it = grid.find({c.first + dx, c.second + dy});
                    if (it == grid.end()) continue;
                    for (const std::size_t j : it->second) {
                        FaceId common = kExteriorFace;
                        if (j <= i || jog[j] || sla::norm(marked[i].pos - marked[j].pos) > params.jogMaxLedgePx) continue;
                        if (!shares(marked[i], marked[j], common)) continue;
                        if (convexity(marked[i], common) * convexity(marked[j], common) >= 0.0) continue;
                        jog[i] = jog[j] = 1;
                        jogs += 2;
                        found = true;
                        break;
                    }
                }
            }
        }
    }
    if (jogNodes != nullptr) *jogNodes = jogs;
    // Decide com as posições da fase anterior e só então marca (a ordem não influencia).
    int count = 0;
    for (std::size_t i = 0; i < marked.size(); ++i) {
        const NodeId n = marked[i].node;
        if (jog[i]) continue;
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
        const SmoothTopology topology = SmoothTopology::build(image, params_.airPressureMaxNodes,
                                                              {params_.contrastFullLevels, params_.contrastFloor});
        if (phase == params_.punctureOnPhase) st.puncturedNodes = puncture(image, topology, params_, &st.jogNodes);

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
        // A tolerância relativa de energia é medida sobre a energia total, dominada pela mola de fidelidade: numa escada de
        // 1 px o passo conjugado pode cair abaixo dela muito antes de o termo angular ter alisado o degrau (a direção
        // Polak–Ribière+ degenera para ~0 depois de uma rejeição do fiscal ou de um recomeço). Uma convergência declarada é
        // confirmada por UM recomeço com descida pelo gradiente puro antes de encerrar a fase; só vale se esse recomeço
        // também declarar convergência. O ponto aceito nunca piora: o CG parte do ponto atual.
        bool confirmed = false;
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
                if (!confirmed && s != CgStatus::MaxIterations) {
                    confirmed = true;
                    rebuild();  // zera a memória da direção conjugada no ponto aceito
                    continue;
                }
                st.status = s;
                break;
            }
            if (offenders.empty() && s == CgStatus::Running) confirmed = false;  // voltou a descer: a próxima parada se confirma de novo
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
