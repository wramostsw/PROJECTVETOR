// src/vector/VectorImage.cpp — mapa planar por meia-aresta: ligação dos laços, áreas, pais e invariantes.
#include "cnx/vector/VectorImage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cnx {
namespace {

constexpr std::size_t ux(std::int32_t i) noexcept { return static_cast<std::size_t>(i); }

std::int32_t toId(std::size_t index, const char* what) {
    if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) throw std::length_error(what);
    return static_cast<std::int32_t>(index);
}

std::uint32_t toU32(std::size_t value, const char* what) {
    if (value > std::numeric_limits<std::uint32_t>::max()) throw std::length_error(what);
    return static_cast<std::uint32_t>(value);
}

/// Contribuição exata de uma cúbica para a área de Green, ½∫₀¹ (B − O) × B' dt, com Q_i = P_i − O:
///   Σ_{i<j} K_ij (Q_i × Q_j),   K_01 = K_23 = 3/10,   K_02 = K_12 = K_13 = 3/20,   K_03 = 1/20.
/// (K_ij = ½∫(b_i b_j' − b_j b_i') dt, frações exatas das bases de Bernstein.) Num laço fechado a
/// origem O se cancela; ela só serve para condicionar as somas.
double cubicAreaTerm(const sla::BezierCurve& c, sla::Vec2 o) noexcept {
    const sla::Vec2 q0 = c.p[0] - o;
    const sla::Vec2 q1 = c.p[1] - o;
    const sla::Vec2 q2 = c.p[2] - o;
    const sla::Vec2 q3 = c.p[3] - o;
    return 0.3 * (sla::cross(q0, q1) + sla::cross(q2, q3)) +
           0.15 * (sla::cross(q0, q2) + sla::cross(q1, q2) + sla::cross(q1, q3)) + 0.05 * sla::cross(q0, q3);
}

/// Termo de área da aresta no sentido direto (o inverso é o negativo).
double forwardAreaTerm(const VectorImage& image, EdgeId e, sla::Vec2 o) {
    double sum = 0.0;
    const auto curves = image.edgeCurves(e);
    if (!curves.empty()) {
        for (const sla::CurveSegment& c : curves) sum += cubicAreaTerm(c.curve, o);
        return sum;
    }
    const auto nodes = image.edgeNodes(e);
    for (std::size_t k = 0; k + 1 < nodes.size(); ++k) {
        sum += 0.5 * sla::cross(image.position(nodes[k]) - o, image.position(nodes[k + 1]) - o);
    }
    return sum;
}

/// Ângulo da direção inicial da meia-aresta (primeiro nó com posição distinta da origem).
double initialAngle(const VectorImage& image, HalfEdgeId h) {
    const auto nodes = image.edgeNodes(edgeOf(h));
    const std::size_t n = nodes.size();
    const bool rev = isReversed(h);
    const sla::Vec2 o = image.position(rev ? nodes[n - 1] : nodes[0]);
    for (std::size_t k = 1; k < n; ++k) {
        const sla::Vec2 d = image.position(rev ? nodes[n - 1 - k] : nodes[k]) - o;
        if (d.x != 0.0 || d.y != 0.0) return std::atan2(d.y, d.x);
    }
    throw std::logic_error("VectorImage::linkFaceLoops: aresta de comprimento nulo");
}

/// Union-Find mínimo; a raiz é sempre o menor índice (resultado determinístico).
class DisjointSets {
public:
    explicit DisjointSets(std::size_t n) : parent_(n) { std::iota(parent_.begin(), parent_.end(), std::size_t{0}); }

    std::size_t find(std::size_t a) noexcept {
        while (parent_[a] != a) {
            parent_[a] = parent_[parent_[a]];
            a = parent_[a];
        }
        return a;
    }

    void unite(std::size_t a, std::size_t b) noexcept {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (b < a) std::swap(a, b);
        parent_[b] = a;
    }

private:
    std::vector<std::size_t> parent_;
};

}  // namespace

VectorImage::VectorImage(double width, double height) : width_(width), height_(height) {
    if (!std::isfinite(width) || !std::isfinite(height) || width < 0.0 || height < 0.0) {
        throw std::invalid_argument("VectorImage: dimensões devem ser finitas e não negativas");
    }
    faces_.push_back(Face{});  // exterior: cor transparente
}

NodeId VectorImage::addNode(sla::Vec2 position, std::uint8_t flags) {
    const NodeId id = toId(pos_.size(), "VectorImage: excesso de nós");
    pos_.push_back(position);
    origin_.push_back(position);
    flags_.push_back(flags);
    quarantine_.push_back(0);
    return id;
}

FaceId VectorImage::addFace(const sla::Rgba& color, std::int32_t paletteIndex) {
    if (faces_.empty()) {
        throw std::logic_error("VectorImage::addFace: sem face exterior (construa com largura e altura)");
    }
    const FaceId id = toId(faces_.size(), "VectorImage: excesso de faces");
    Face face;
    face.color = color;
    face.paletteIndex = paletteIndex;
    faces_.push_back(face);
    return id;
}

EdgeId VectorImage::addEdge(std::span<const NodeId> nodes, FaceId left, FaceId right) {
    if (nodes.size() < 2) throw std::invalid_argument("VectorImage::addEdge: a aresta precisa de pelo menos 2 nós");
    for (const NodeId n : nodes) {
        if (n < 0 || ux(n) >= pos_.size()) throw std::invalid_argument("VectorImage::addEdge: id de nó inválido");
    }
    for (std::size_t k = 0; k + 1 < nodes.size(); ++k) {
        if (nodes[k] == nodes[k + 1]) throw std::invalid_argument("VectorImage::addEdge: nós consecutivos repetidos");
    }
    if (nodes.front() == nodes.back() && nodes.size() < 4) {
        throw std::invalid_argument("VectorImage::addEdge: aresta fechada precisa de 3 nós distintos");
    }
    const auto validFace = [this](FaceId f) { return f >= 0 && ux(f) < faces_.size(); };
    if (!validFace(left) || !validFace(right)) throw std::invalid_argument("VectorImage::addEdge: face inválida");
    if (left == right) throw std::invalid_argument("VectorImage::addEdge: left e right devem ser faces distintas");

    const EdgeId id = toId(edges_.size(), "VectorImage: excesso de arestas");
    (void)toId(2 * edges_.size() + 1, "VectorImage: excesso de meias-arestas");
    Edge edge;
    edge.nodeBegin = toU32(edgeNodes_.size(), "VectorImage: excesso de nós de aresta");
    edge.nodeCount = toU32(nodes.size(), "VectorImage: aresta longa demais");
    (void)toU32(edgeNodes_.size() + nodes.size(), "VectorImage: excesso de nós de aresta");
    edge.left = left;
    edge.right = right;

    // `nodes` pode apontar para dentro de edgeNodes_ (ex.: edgeNodes() de outra aresta).
    const std::less<const NodeId*> before;
    const bool aliases = !edgeNodes_.empty() && !before(nodes.data(), edgeNodes_.data()) &&
                         before(nodes.data(), edgeNodes_.data() + edgeNodes_.size());
    if (aliases) {
        const std::vector<NodeId> copy(nodes.begin(), nodes.end());
        edgeNodes_.insert(edgeNodes_.end(), copy.begin(), copy.end());
    } else {
        edgeNodes_.insert(edgeNodes_.end(), nodes.begin(), nodes.end());
    }
    edges_.push_back(edge);
    next_.push_back(kInvalidId);  // as novas meias-arestas ficam sem laço até linkFaceLoops()
    next_.push_back(kInvalidId);
    return id;
}

void VectorImage::linkFaceLoops() {
    if (faces_.empty()) throw std::logic_error("VectorImage::linkFaceLoops: sem face exterior");
    const std::size_t numHalfEdges = 2 * edges_.size();
    next_.assign(numHalfEdges, kInvalidId);

    // 1. Raios: meias-arestas que partem de cada vértice, ordenadas pelo ângulo da direção inicial.
    struct Spoke {
        NodeId vertex;
        double angle;
        HalfEdgeId h;
    };
    std::vector<Spoke> spokes;
    spokes.reserve(numHalfEdges);
    for (std::size_t i = 0; i < numHalfEdges; ++i) {
        const auto h = static_cast<HalfEdgeId>(i);
        spokes.push_back({origin(h), initialAngle(*this, h), h});
    }
    std::sort(spokes.begin(), spokes.end(), [](const Spoke& a, const Spoke& b) {
        return std::tie(a.vertex, a.angle, a.h) < std::tie(b.vertex, b.angle, b.h);
    });

    // 2. Com a face no lado de área positiva, o laço que chega a um vértice segue pelo raio
    //    imediatamente anterior, em ângulo, ao raio de volta: next(twin(o_k)) = o_{k−1} (cíclico).
    for (std::size_t begin = 0; begin < spokes.size();) {
        std::size_t end = begin + 1;
        while (end < spokes.size() && spokes[end].vertex == spokes[begin].vertex) ++end;
        for (std::size_t k = begin; k < end; ++k) {
            if (k > begin && spokes[k].angle == spokes[k - 1].angle) {
                throw std::logic_error("VectorImage::linkFaceLoops: arestas sobrepostas na saída de um vértice");
            }
            const std::size_t prev = (k == begin) ? end - 1 : k - 1;
            next_[ux(twin(spokes[k].h))] = spokes[prev].h;
        }
        begin = end;
    }

    // 3. Laços = órbitas de next (next é uma permutação). Área > 0: laço externo; área ≤ 0: furo.
    for (Face& face : faces_) {
        face.outerLoop = kInvalidId;
        face.holeBegin = 0;
        face.holeCount = 0;
        face.parent = kInvalidId;
        face.area = 0.0;
    }
    std::vector<std::vector<HalfEdgeId>> holesOf(faces_.size());
    std::vector<std::uint8_t> visited(numHalfEdges, 0);
    std::vector<std::uint8_t> onHole(numHalfEdges, 0);
    for (std::size_t i = 0; i < numHalfEdges; ++i) {
        if (visited[i] != 0) continue;
        const auto first = static_cast<HalfEdgeId>(i);  // menor id do laço: representante determinístico
        const FaceId f = faceOf(first);
        HalfEdgeId h = first;
        do {
            if (faceOf(h) != f) throw std::logic_error("VectorImage::linkFaceLoops: laço atravessa faces diferentes");
            visited[ux(h)] = 1;
            h = next_[ux(h)];
        } while (h != first);

        const double area = loopArea(first);
        Face& face = faces_[ux(f)];
        face.area += area;
        if (area > 0.0) {
            if (f == kExteriorFace) throw std::logic_error("VectorImage::linkFaceLoops: exterior com laço de área positiva");
            if (face.outerLoop != kInvalidId) {
                throw std::logic_error("VectorImage::linkFaceLoops: face " + std::to_string(f) +
                                       " com mais de um laço externo (não conexa)");
            }
            face.outerLoop = first;
        } else {
            holesOf[ux(f)].push_back(first);
            do {
                onHole[ux(h)] = 1;
                h = next_[ux(h)];
            } while (h != first);
        }
    }
    holeLoops_.clear();
    for (std::size_t f = 0; f < faces_.size(); ++f) {
        if (f != 0 && faces_[f].outerLoop == kInvalidId) {
            throw std::logic_error("VectorImage::linkFaceLoops: face " + std::to_string(f) + " sem laço externo");
        }
        faces_[f].holeBegin = toU32(holeLoops_.size(), "VectorImage: excesso de furos");
        faces_[f].holeCount = toU32(holesOf[f].size(), "VectorImage: excesso de furos");
        holeLoops_.insert(holeLoops_.end(), holesOf[f].begin(), holesOf[f].end());
    }

    // 4. Pais. Se o laço externo de X encosta num furo de G, X está diretamente dentro de G.
    //    Se encosta no laço externo de G, X e G são irmãs (mesmo pai). As faces junto à moldura
    //    encostam no furo do exterior e ficam com pai 0.
    DisjointSets siblings(faces_.size());
    std::vector<FaceId> enclosing(faces_.size(), kInvalidId);
    for (std::size_t f = 1; f < faces_.size(); ++f) {
        const HalfEdgeId first = faces_[f].outerLoop;
        HalfEdgeId h = first;
        do {
            const HalfEdgeId t = twin(h);
            const FaceId g = faceOf(t);
            if (onHole[ux(t)] != 0) {
                if (enclosing[f] == kInvalidId) enclosing[f] = g;
            } else {
                siblings.unite(f, ux(g));
            }
            h = next_[ux(h)];
        } while (h != first);
    }
    std::vector<FaceId> groupParent(faces_.size(), kInvalidId);
    for (std::size_t f = 1; f < faces_.size(); ++f) {
        const std::size_t root = siblings.find(f);
        if (enclosing[f] != kInvalidId && groupParent[root] == kInvalidId) groupParent[root] = enclosing[f];
    }
    for (std::size_t f = 1; f < faces_.size(); ++f) faces_[f].parent = groupParent[siblings.find(f)];
}

void VectorImage::tickQuarantine() noexcept {
    for (std::uint8_t& q : quarantine_) {
        if (q != 0) --q;
    }
}

double VectorImage::loopArea(HalfEdgeId first) const {
    if (first < 0 || ux(first) >= next_.size()) throw std::out_of_range("VectorImage::loopArea: meia-aresta inválida");
    const sla::Vec2 o{0.5 * width_, 0.5 * height_};
    double area = 0.0;
    HalfEdgeId h = first;
    std::size_t steps = 0;
    do {
        const double term = forwardAreaTerm(*this, edgeOf(h), o);
        area += isReversed(h) ? -term : term;
        h = next_[ux(h)];
        if (h == kInvalidId) throw std::logic_error("VectorImage::loopArea: laços não ligados (chame linkFaceLoops)");
        if (++steps > next_.size()) throw std::logic_error("VectorImage::loopArea: laço não fecha");
    } while (h != first);
    return area;
}

void VectorImage::updateFaceAreas() {
    for (std::size_t f = 0; f < faces_.size(); ++f) {
        Face& face = faces_[f];
        double area = face.outerLoop != kInvalidId ? loopArea(face.outerLoop) : 0.0;
        for (const HalfEdgeId h : holeLoops(static_cast<FaceId>(f))) area += loopArea(h);
        face.area = area;
    }
}

void VectorImage::clearCurves() {
    curves_.clear();
    for (Edge& e : edges_) {
        e.curveBegin = 0;
        e.curveCount = 0;
    }
}

void VectorImage::setEdgeCurves(EdgeId e, std::span<const sla::CurveSegment> curves) {
    if (e < 0 || ux(e) >= edges_.size()) throw std::out_of_range("VectorImage::setEdgeCurves: aresta inválida");
    if (curves.empty()) throw std::invalid_argument("VectorImage::setEdgeCurves: cadeia vazia");
    const auto nodes = edgeNodes(e);
    const double tol = 1e-9 * (1.0 + std::max(width_, height_));
    if (sla::norm(curves.front().start() - position(nodes.front())) > tol ||
        sla::norm(curves.back().end() - position(nodes.back())) > tol) {
        throw std::invalid_argument("VectorImage::setEdgeCurves: a cadeia deve ligar as junções da aresta");
    }
    for (std::size_t k = 0; k + 1 < curves.size(); ++k) {
        if (sla::norm(curves[k].end() - curves[k + 1].start()) > tol) {
            throw std::invalid_argument("VectorImage::setEdgeCurves: cadeia descontínua");
        }
    }
    Edge& edge = edges_[ux(e)];
    if (curves.size() <= edge.curveCount) {
        std::copy(curves.begin(), curves.end(), curves_.begin() + static_cast<std::ptrdiff_t>(edge.curveBegin));
    } else {
        edge.curveBegin = toU32(curves_.size(), "VectorImage: excesso de curvas");
        (void)toU32(curves_.size() + curves.size(), "VectorImage: excesso de curvas");
        curves_.insert(curves_.end(), curves.begin(), curves.end());
    }
    edge.curveCount = static_cast<std::uint32_t>(curves.size());
}

std::vector<NodeId> VectorImage::collectFreeNodes() const {
    std::vector<NodeId> free;
    for (std::size_t n = 0; n < pos_.size(); ++n) {
        if ((flags_[n] & NodeFlag::Pinned) == 0 && quarantine_[n] == 0) free.push_back(static_cast<NodeId>(n));
    }
    return free;
}

void VectorImage::gather(std::span<const NodeId> freeNodes, std::span<double> x) const {
    if (x.size() != 2 * freeNodes.size()) throw std::invalid_argument("VectorImage::gather: x deve ter 2 entradas por nó");
    for (std::size_t k = 0; k < freeNodes.size(); ++k) {
        const sla::Vec2 p = pos_.at(ux(freeNodes[k]));
        x[2 * k] = p.x;
        x[2 * k + 1] = p.y;
    }
}

void VectorImage::scatter(std::span<const NodeId> freeNodes, std::span<const double> x) {
    if (x.size() != 2 * freeNodes.size()) throw std::invalid_argument("VectorImage::scatter: x deve ter 2 entradas por nó");
    for (std::size_t k = 0; k < freeNodes.size(); ++k) pos_.at(ux(freeNodes[k])) = {x[2 * k], x[2 * k + 1]};
}

void VectorImage::resetOriginalPositions() { origin_ = pos_; }

VectorImage::ValidationReport VectorImage::validate(double areaTolerance) const {
    ValidationReport report;
    int issues = 0;
    const auto fail = [&](const std::string& message) {
        if (issues < 8) {
            if (!report.message.empty()) report.message += "; ";
            report.message += message;
        } else if (issues == 8) {
            report.message += "; …";
        }
        ++issues;
        report.ok = false;
    };

    if (faces_.empty()) {
        fail("sem face exterior");
        return report;
    }
    const std::size_t numHalfEdges = 2 * edges_.size();
    if (next_.size() != numHalfEdges) {
        fail("next_ com tamanho incoerente");
        return report;
    }

    // Estrutura das arestas.
    for (std::size_t e = 0; e < edges_.size(); ++e) {
        const Edge& ed = edges_[e];
        const std::string tag = "aresta " + std::to_string(e);
        if (ed.nodeCount < 2 || std::size_t{ed.nodeBegin} + ed.nodeCount > edgeNodes_.size()) fail(tag + ": faixa de nós inválida");
        if (ed.left < 0 || ux(ed.left) >= faces_.size() || ed.right < 0 || ux(ed.right) >= faces_.size() ||
            ed.left == ed.right) {
            fail(tag + ": faces inválidas");
        }
        if (std::size_t{ed.curveBegin} + ed.curveCount > curves_.size()) fail(tag + ": faixa de curvas inválida");
    }
    for (const NodeId n : edgeNodes_) {
        if (n < 0 || ux(n) >= pos_.size()) fail("id de nó inválido em edgeNodes_");
    }
    if (!report.ok) return report;

    // next: permutação, continuidade geométrica (target(h) = origin(next(h))) e mesma face.
    std::vector<std::uint8_t> hit(numHalfEdges, 0);
    for (std::size_t i = 0; i < numHalfEdges; ++i) {
        const auto h = static_cast<HalfEdgeId>(i);
        const HalfEdgeId n = next_[i];
        if (n < 0 || ux(n) >= numHalfEdges) {
            fail("meia-aresta " + std::to_string(i) + " sem next (chame linkFaceLoops)");
            continue;
        }
        if (hit[ux(n)]++ != 0) fail("next não é permutação");
        if (target(h) != origin(n)) fail("laço descontínuo na meia-aresta " + std::to_string(i));
        if (faceOf(n) != faceOf(h)) fail("laço muda de face na meia-aresta " + std::to_string(i));
    }
    if (!report.ok) return report;

    // I1: cada nó é vértice (ponta de aresta) ou interior de exatamente uma aresta.
    std::vector<std::uint32_t> interiorUses(pos_.size(), 0);
    std::vector<std::uint8_t> isVertex(pos_.size(), 0);
    for (std::size_t e = 0; e < edges_.size(); ++e) {
        const auto nodes = edgeNodes(static_cast<EdgeId>(e));
        isVertex[ux(nodes.front())] = 1;
        isVertex[ux(nodes.back())] = 1;
        for (std::size_t k = 1; k + 1 < nodes.size(); ++k) ++interiorUses[ux(nodes[k])];
    }
    for (std::size_t n = 0; n < pos_.size(); ++n) {
        const std::string tag = "I1: nó " + std::to_string(n);
        if (isVertex[n] != 0 && interiorUses[n] > 0) {
            fail(tag + " é vértice e interior ao mesmo tempo");
        } else if (interiorUses[n] > 1) {
            fail(tag + " interior a " + std::to_string(interiorUses[n]) + " trechos de aresta");
        } else if (isVertex[n] == 0 && interiorUses[n] == 0) {
            fail(tag + " órfão");
        }
        if ((flags_[n] & NodeFlag::Junction) != 0 && isVertex[n] == 0) fail(tag + " marcado como junção fora das pontas");
    }

    // Laços: cada órbita de next registrada em exatamente uma face; I2 pelos sinais das áreas.
    std::vector<std::int32_t> loopOf(numHalfEdges, -1);
    std::vector<FaceId> loopFace;
    for (std::size_t i = 0; i < numHalfEdges; ++i) {
        if (loopOf[i] != -1) continue;
        const auto id = static_cast<std::int32_t>(loopFace.size());
        loopFace.push_back(faceOf(static_cast<HalfEdgeId>(i)));
        auto h = static_cast<HalfEdgeId>(i);
        do {
            loopOf[ux(h)] = id;
            h = next_[ux(h)];
        } while (h != static_cast<HalfEdgeId>(i));
    }
    std::vector<std::uint8_t> listed(loopFace.size(), 0);
    std::vector<double> faceArea(faces_.size(), 0.0);
    for (std::size_t f = 0; f < faces_.size(); ++f) {
        const Face& face = faces_[f];
        const std::string tag = "face " + std::to_string(f);
        const auto registerLoop = [&](HalfEdgeId h, bool outer) {
            if (h < 0 || ux(h) >= numHalfEdges) {
                fail(tag + ": laço inválido");
                return;
            }
            const auto loop = ux(loopOf[ux(h)]);
            if (listed[loop]++ != 0) fail(tag + ": laço registrado mais de uma vez");
            if (ux(loopFace[loop]) != f) fail(tag + ": laço pertence a outra face");
            const double a = loopArea(h);
            faceArea[f] += a;
            if (outer && !(a > 0.0)) fail("I2: " + tag + ": laço externo com área " + std::to_string(a));
            if (!outer && !(a < 0.0)) fail("I2: " + tag + ": furo com área " + std::to_string(a));
        };
        if (f == 0) {
            if (face.outerLoop != kInvalidId) fail("exterior com laço externo");
        } else if (face.outerLoop == kInvalidId) {
            fail(tag + ": sem laço externo");
        } else {
            registerLoop(face.outerLoop, true);
        }
        if (std::size_t{face.holeBegin} + face.holeCount > holeLoops_.size()) {
            fail(tag + ": faixa de furos inválida");
            continue;
        }
        for (const HalfEdgeId h : holeLoops(static_cast<FaceId>(f))) registerLoop(h, false);
    }
    for (std::size_t l = 0; l < listed.size(); ++l) {
        if (listed[l] == 0) fail("laço " + std::to_string(l) + " não registrado em nenhuma face");
    }

    // I3: V − E + F = 1 + C.
    DisjointSets components(pos_.size());
    for (std::size_t e = 0; e < edges_.size(); ++e) {
        const auto nodes = edgeNodes(static_cast<EdgeId>(e));
        components.unite(ux(nodes.front()), ux(nodes.back()));
    }
    long long vertices = 0;
    long long componentCount = 0;
    for (std::size_t n = 0; n < pos_.size(); ++n) {
        if (isVertex[n] == 0) continue;
        ++vertices;
        if (components.find(n) == n) ++componentCount;
    }
    const long long euler = vertices - static_cast<long long>(edges_.size()) + static_cast<long long>(faces_.size());
    if (euler != 1 + componentCount) {
        fail("I3: V − E + F = " + std::to_string(euler) + " ≠ 1 + C = " + std::to_string(1 + componentCount));
    }

    // I4: as faces cobrem a imagem exatamente; o exterior é o negativo.
    const double expected = width_ * height_;
    const double tol = areaTolerance * std::max(1.0, expected);
    double interior = 0.0;
    for (std::size_t f = 1; f < faces_.size(); ++f) {
        interior += faceArea[f];
        if (!(faceArea[f] > 0.0)) fail("I4: face " + std::to_string(f) + " com área " + std::to_string(faceArea[f]));
    }
    if (std::abs(interior - expected) > tol) {
        fail("I4: Σ áreas = " + std::to_string(interior) + " ≠ largura × altura = " + std::to_string(expected));
    }
    if (std::abs(faceArea[0] + expected) > tol) fail("I4: área do exterior = " + std::to_string(faceArea[0]));

    // Pais: válidos, sem ciclos, todas as cadeias terminam no exterior.
    for (std::size_t f = 1; f < faces_.size(); ++f) {
        FaceId p = faces_[f].parent;
        std::size_t steps = 0;
        while (p != kExteriorFace) {
            if (p < 0 || ux(p) >= faces_.size() || ++steps > faces_.size()) {
                fail("face " + std::to_string(f) + ": cadeia de pais inválida");
                break;
            }
            p = faces_[ux(p)].parent;
        }
    }
    return report;
}

}  // namespace cnx
