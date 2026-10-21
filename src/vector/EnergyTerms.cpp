// src/vector/EnergyTerms.cpp — termos da energia de suavização e gradientes analíticos.
#include "cnx/vector/EnergyTerms.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cnx {
namespace {

using sla::Vec2;

constexpr std::size_t ux(std::int32_t i) noexcept { return static_cast<std::size_t>(i); }

/// Nós de um laço de meias-arestas, em ordem de percurso e sem repetir o primeiro.
std::vector<NodeId> loopNodes(const VectorImage& image, HalfEdgeId first) {
    std::vector<NodeId> out;
    HalfEdgeId h = first;
    do {
        const auto nodes = image.edgeNodes(edgeOf(h));
        if (!isReversed(h)) {
            for (std::size_t i = 0; i + 1 < nodes.size(); ++i) out.push_back(nodes[i]);
        } else {
            for (std::size_t i = nodes.size() - 1; i >= 1; --i) out.push_back(nodes[i]);
        }
        h = image.next(h);
    } while (h != first);
    return out;
}

double shoelace(std::span<const Vec2> pos, const std::vector<NodeId>& nodes) {
    double a = 0.0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        a += sla::cross(pos[ux(nodes[i])], pos[ux(nodes[(i + 1) % nodes.size()])]);
    }
    return 0.5 * a;
}

}  // namespace

// ---------------------------------------------------------------- topologia

SmoothTopology SmoothTopology::build(const VectorImage& image, int airMaxNodes) {
    SmoothTopology t;
    for (EdgeId e = 0; e < static_cast<EdgeId>(image.edgeCount()); ++e) {
        const auto nodes = image.edgeNodes(e);
        const std::size_t m = nodes.size();
        const bool closed = nodes.front() == nodes.back();
        if (!closed) {
            for (std::size_t i = 1; i + 1 < m; ++i) {
                t.triples.push_back({nodes[i - 1], nodes[i], nodes[i + 1], e, static_cast<std::uint32_t>(i)});
            }
        } else if (m >= 4) {
            for (std::size_t i = 0; i + 1 < m; ++i) {
                const NodeId prev = i == 0 ? nodes[m - 2] : nodes[i - 1];
                t.triples.push_back({prev, nodes[i], nodes[i + 1], e, static_cast<std::uint32_t>(i)});
            }
        }
    }

    const std::span<const Vec2> origin = image.originalPositions();
    for (FaceId f = 1; f < static_cast<FaceId>(image.faceCount()); ++f) {
        const Face& face = image.face(f);
        std::vector<HalfEdgeId> loops;
        if (face.outerLoop != kInvalidId) loops.push_back(face.outerLoop);
        for (const HalfEdgeId h : image.holeLoops(f)) loops.push_back(h);
        for (const HalfEdgeId first : loops) {
            std::vector<NodeId> nodes = loopNodes(image, first);
            if (nodes.size() < 3 || static_cast<int>(nodes.size()) >= airMaxNodes) continue;
            const double a0 = shoelace(origin, nodes);
            t.airLoops.push_back({std::move(nodes), a0});
        }
    }

    for (NodeId n = 0; n < static_cast<NodeId>(image.nodeCount()); ++n) {
        const std::uint8_t fl = image.flags(n);
        if ((fl & NodeFlag::ImageBorder) == 0 || (fl & NodeFlag::Pinned) != 0) continue;
        const Vec2 o = image.originalPosition(n);
        if (o.x == 0.0) {
            t.borderAnchors.push_back({n, 0, 0.0});
        } else if (o.x == image.width()) {
            t.borderAnchors.push_back({n, 0, image.width()});
        } else if (o.y == 0.0) {
            t.borderAnchors.push_back({n, 1, 0.0});
        } else if (o.y == image.height()) {
            t.borderAnchors.push_back({n, 1, image.height()});
        }
    }
    return t;
}

EnergyWeights phaseWeights(const SmoothParams& p, int phase) {
    if (phase < 0 || phase >= SmoothParams::kPhases) throw std::invalid_argument("phaseWeights: fase fora de [0, 3)");
    if (p.priorType != 3) throw std::invalid_argument("phaseWeights: só prior_types = 3 (suavidade angular) está implementado");
    EnergyWeights w;
    w.measurement = p.measurementTypes[static_cast<std::size_t>(phase)];
    w.fidelity = p.fidelityWeight;
    w.freeRadius = p.fidelityFreeRadiusPx;
    w.angular = p.priorStrength;
    w.spacing = p.spacingWeight[static_cast<std::size_t>(phase)];
    w.air = p.airPressureWeight;
    w.border = p.borderAnchorWeight;
    return w;
}

// ---------------------------------------------------------------- termos

double springEnergy(const VectorImage& image, double weight, double freeRadius, std::span<Vec2> grad) {
    double energy = 0.0;
    const auto pos = image.positions();
    const auto origin = image.originalPositions();
    for (std::size_t n = 0; n < pos.size(); ++n) {
        const Vec2 d = pos[n] - origin[n];
        const double len = sla::norm(d);
        const double excess = len - freeRadius;
        if (excess <= 0.0) continue;
        energy += excess * excess;
        if (!grad.empty()) grad[n] += d * (weight * 2.0 * excess / len);
    }
    return weight * energy;
}

double angularEnergy(const VectorImage& image, const SmoothTopology& topology, double weight, std::span<Vec2> grad) {
    double energy = 0.0;
    const auto pos = image.positions();
    for (const SmoothTopology::Triple& t : topology.triples) {
        if ((image.flags(t.mid) & NodeFlag::Corner) != 0) continue;  // canto perfurado: sem suavidade
        const Vec2 u = pos[ux(t.mid)] - pos[ux(t.prev)];
        const Vec2 v = pos[ux(t.next)] - pos[ux(t.mid)];
        const double uu = sla::norm2(u);
        const double vv = sla::norm2(v);
        if (uu < 1e-18 || vv < 1e-18) continue;
        const double theta = std::atan2(sla::cross(u, v), sla::dot(u, v));
        energy += theta * theta;
        if (grad.empty()) continue;
        // θ = ângulo(v) − ângulo(u):  ∂θ/∂u = (u_y, −u_x)/‖u‖²,  ∂θ/∂v = (−v_y, v_x)/‖v‖².
        const Vec2 du = Vec2{u.y, -u.x} / uu;
        const Vec2 dv = Vec2{-v.y, v.x} / vv;
        const double s = weight * 2.0 * theta;
        grad[ux(t.prev)] += du * (-s);
        grad[ux(t.mid)] += (du - dv) * s;
        grad[ux(t.next)] += dv * s;
    }
    return weight * energy;
}

double spacingEnergy(const VectorImage& image, const SmoothTopology& topology, double weight, std::span<Vec2> grad) {
    double energy = 0.0;
    const auto pos = image.positions();
    for (const SmoothTopology::Triple& t : topology.triples) {
        const Vec2 a = pos[ux(t.mid)] - pos[ux(t.prev)];
        const Vec2 b = pos[ux(t.next)] - pos[ux(t.mid)];
        const double la = sla::norm(a);
        const double lb = sla::norm(b);
        const double s = lb - la;
        energy += s * s;
        if (grad.empty() || la < 1e-12 || lb < 1e-12) continue;
        // ∂s/∂P_{k+1} = b̂,  ∂s/∂P_k = −b̂ − â,  ∂s/∂P_{k−1} = â.
        const Vec2 ah = a / la;
        const Vec2 bh = b / lb;
        const double f = weight * 2.0 * s;
        grad[ux(t.prev)] += ah * f;
        grad[ux(t.mid)] += (bh + ah) * (-f);
        grad[ux(t.next)] += bh * f;
    }
    return weight * energy;
}

double airPressureEnergy(const VectorImage& image, const SmoothTopology& topology, double weight, std::span<Vec2> grad) {
    double energy = 0.0;
    const auto pos = image.positions();
    for (const SmoothTopology::AirLoop& loop : topology.airLoops) {
        const double diff = shoelace(pos, loop.nodes) - loop.originalArea;
        energy += diff * diff;
        if (grad.empty()) continue;
        const std::size_t n = loop.nodes.size();
        for (std::size_t i = 0; i < n; ++i) {
            const Vec2 prev = pos[ux(loop.nodes[(i + n - 1) % n])];
            const Vec2 next = pos[ux(loop.nodes[(i + 1) % n])];
            // ∂A/∂P_k = ½ (y_{k+1} − y_{k−1}, x_{k−1} − x_{k+1}).
            grad[ux(loop.nodes[i])] += Vec2{next.y - prev.y, prev.x - next.x} * (weight * 2.0 * diff * 0.5);
        }
    }
    return weight * energy;
}

double borderEnergy(const VectorImage& image, const SmoothTopology& topology, double weight, std::span<Vec2> grad) {
    double energy = 0.0;
    const auto pos = image.positions();
    for (const SmoothTopology::BorderAnchor& b : topology.borderAnchors) {
        const double c = b.axis == 0 ? pos[ux(b.node)].x : pos[ux(b.node)].y;
        const double d = c - b.coordinate;
        energy += d * d;
        if (grad.empty()) continue;
        (b.axis == 0 ? grad[ux(b.node)].x : grad[ux(b.node)].y) += weight * 2.0 * d;
    }
    return weight * energy;
}

double evaluateEnergy(const VectorImage& image, const SmoothTopology& topology, const GenerativeModel* model,
                      const EnergyWeights& w, std::span<Vec2> grad, EnergyBreakdown* parts) {
    if (!grad.empty()) {
        if (grad.size() != image.nodeCount()) throw std::invalid_argument("evaluateEnergy: grad.size() != nodeCount()");
        std::fill(grad.begin(), grad.end(), Vec2{});
    }
    EnergyBreakdown b;
    switch (w.measurement) {
        case MeasurementType::Spring:
            b.spring = springEnergy(image, w.fidelity, w.freeRadius, grad);
            break;
        case MeasurementType::Generative: {
            if (model == nullptr) throw std::invalid_argument("evaluateEnergy: medição generativa sem GenerativeModel");
            if (grad.empty()) {
                b.generative = w.fidelity * model->energy(image);
            } else {
                std::vector<Vec2> g(image.nodeCount());
                b.generative = w.fidelity * model->energyAndGradient(image, g);
                for (std::size_t n = 0; n < g.size(); ++n) grad[n] += g[n] * w.fidelity;
            }
            break;
        }
        case MeasurementType::None:
            break;
    }
    b.angular = angularEnergy(image, topology, w.angular, grad);
    b.spacing = spacingEnergy(image, topology, w.spacing, grad);
    b.air = airPressureEnergy(image, topology, w.air, grad);
    b.border = borderEnergy(image, topology, w.border, grad);
    if (parts != nullptr) *parts = b;
    return b.total();
}

// ---------------------------------------------------------------- canto perfurado

namespace {

struct Arm {
    double length = 0.0;
    double deviation = 0.0;
    Vec2 end;
};

/// Segue a poligonal a partir de k na direção dir (±1) por até `arm` de comprimento de arco.
Arm walkArm(std::span<const Vec2> pts, std::size_t k, bool closed, int dir, double arm) {
    const std::size_t n = pts.size();
    const Vec2 p0 = pts[k];
    Arm a;
    a.end = p0;
    std::vector<Vec2> passed;
    std::size_t cur = k;
    double len = 0.0;
    const std::size_t maxSteps = closed ? n - 1 : n;
    for (std::size_t step = 0; step < maxSteps; ++step) {
        long j = static_cast<long>(cur) + dir;
        if (closed) {
            j = (j + static_cast<long>(n)) % static_cast<long>(n);
        } else if (j < 0 || j >= static_cast<long>(n)) {
            break;
        }
        const Vec2 seg = pts[static_cast<std::size_t>(j)] - pts[cur];
        const double L = sla::norm(seg);
        if (L > 1e-12) {
            if (len + L >= arm) {
                a.end = pts[cur] + seg * ((arm - len) / L);
                len = arm;
                break;
            }
            len += L;
            passed.push_back(pts[static_cast<std::size_t>(j)]);
            a.end = pts[static_cast<std::size_t>(j)];
        }
        cur = static_cast<std::size_t>(j);
    }
    a.length = len;
    const Vec2 chord = a.end - p0;
    const double c = sla::norm(chord);
    if (c > 1e-12) {
        for (const Vec2& q : passed) a.deviation = std::max(a.deviation, std::abs(sla::cross(chord, q - p0)) / c);
    }
    return a;
}

}  // namespace

PunctureMeasure measureCorner(std::span<const Vec2> pts, std::size_t k, bool closed, double arm) {
    if (k >= pts.size()) throw std::out_of_range("measureCorner: k fora da poligonal");
    PunctureMeasure m;
    const Arm back = walkArm(pts, k, closed, -1, arm);
    const Arm fwd = walkArm(pts, k, closed, +1, arm);
    m.deviationBack = back.deviation;
    m.deviationFwd = fwd.deviation;
    m.armBack = back.length;
    m.armFwd = fwd.length;
    const Vec2 in = pts[k] - back.end;
    const Vec2 out = fwd.end - pts[k];
    if (sla::norm2(in) > 1e-18 && sla::norm2(out) > 1e-18) m.turn = std::atan2(sla::cross(in, out), sla::dot(in, out));
    return m;
}

bool isProbablyPuncturedNode(std::span<const Vec2> pts, std::size_t k, bool closed, double arm, double cornerThresh,
                             double flatnessThresh) {
    if (cornerThresh <= 0.0) return false;
    const PunctureMeasure m = measureCorner(pts, k, closed, arm);
    if (m.armBack <= 0.0 || m.armFwd <= 0.0) return false;
    return std::abs(m.turn) >= cornerThresh && m.deviationBack <= flatnessThresh && m.deviationFwd <= flatnessThresh;
}

}  // namespace cnx
