// src/vector/EdgeAnchors.cpp — posição sub-pixel das bordas pelo perfil de cobertura ao longo da normal.
#include "cnx/vector/EdgeAnchors.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "cnx/vector/EnergyTerms.h"

namespace cnx {
namespace {

using sla::Rgba;
using sla::Vec2;

constexpr double kFilterHalfWindowPx = 2.0;  ///< meia janela do filtro de arco que dá a tangente (a escadinha é o ruído)
constexpr long kCornerLookback = 3;          ///< nós de cada lado para medir o giro local
constexpr double kMaxResidual = 0.4;         ///< maior desvio de uma amostra à reta a–b, como fração do contraste
constexpr double kEndPure = 0.2;             ///< as pontas do perfil: t < 0,2 (lado A) e t > 0,8 (lado B)
constexpr double kStep = 0.5;                ///< passo da amostragem do perfil
constexpr double kMinReach = 1.5;            ///< menor alcance tentado (traços finos não têm 2,5 px limpos de cada lado)
constexpr double kArmNear = 2.0;             ///< braços de um canto: de 2 a 6 px de arco a partir do nó
constexpr double kArmMid = 4.0;
constexpr double kArmFar = 6.0;
constexpr double kArmFlat = 0.75;            ///< afastamento máximo do ponto médio do braço à sua corda (px)
constexpr double kMinCornerCross = 0.35;     ///< |sen| mínimo entre as normais dos dois braços (retas quase paralelas não se cruzam)
constexpr double kCornerShiftFactor = 1.5;   ///< o canto pode andar mais que um nó de borda: é a interseção de duas retas

/// Cor da imagem em (x, y) em coordenadas de borda de pixel (o centro do pixel i está em i + ½), bilinear, bordas replicadas.
Rgba sampleBilinear(const sla::Array2D<sla::Rgba8>& img, double x, double y) {
    const double fx = x - 0.5;
    const double fy = y - 0.5;
    const double x0 = std::floor(fx);
    const double y0 = std::floor(fy);
    const double tx = fx - x0;
    const double ty = fy - y0;
    const auto clampX = [&](double v) { return std::clamp(static_cast<int>(v), 0, img.width() - 1); };
    const auto clampY = [&](double v) { return std::clamp(static_cast<int>(v), 0, img.height() - 1); };
    const int xa = clampX(x0), xb = clampX(x0 + 1.0), ya = clampY(y0), yb = clampY(y0 + 1.0);
    const Rgba c00 = sla::toRgba(img(xa, ya));
    const Rgba c10 = sla::toRgba(img(xb, ya));
    const Rgba c01 = sla::toRgba(img(xa, yb));
    const Rgba c11 = sla::toRgba(img(xb, yb));
    const auto w00 = static_cast<float>((1 - tx) * (1 - ty));
    const auto w10 = static_cast<float>(tx * (1 - ty));
    const auto w01 = static_cast<float>((1 - tx) * ty);
    const auto w11 = static_cast<float>(tx * ty);
    return c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11;
}

/// Posição da borda em `p` ao longo de `normal`, para o par de faces da aresta, em qualquer ordem das cores e com o
/// alcance reduzido de `reach` até kMinReach se o perfil mais largo não for limpo (um traço fino, uma borda vizinha).
std::optional<double> profileShift(const sla::Array2D<sla::Rgba8>& source, Vec2 p, Vec2 normal, const Rgba& cl, const Rgba& cr,
                                   double reach) {
    for (double r = reach; r >= kMinReach - 1e-9; r -= kStep) {
        if (const auto s = edgeOffset(source, p, normal, cl, cr, r, kStep)) return s;
        // Qual face fica do lado negativo da normal não se sabe pela orientação da aresta: o perfil diz. Mesma
        // convenção na ordem inversa: positivo = para o lado positivo da normal.
        if (const auto s = edgeOffset(source, p, normal, cr, cl, r, kStep)) return s;
    }
    return std::nullopt;
}

/// Ponto a `dist` px de arco do nó i, no sentido dir (±1). Falso se a cadeia aberta acaba antes.
bool arcPoint(const std::vector<Vec2>& pts, bool closed, std::size_t i, int dir, double dist, Vec2& out) {
    const long n = static_cast<long>(pts.size());
    double walked = 0.0;
    long cur = static_cast<long>(i);
    for (long step = 0; step < n; ++step) {
        long j = cur + dir;
        if (closed) {
            j = ((j % n) + n) % n;
        } else if (j < 0 || j >= n) {
            return false;
        }
        const Vec2 seg = pts[static_cast<std::size_t>(j)] - pts[static_cast<std::size_t>(cur)];
        const double len = sla::norm(seg);
        if (len > 1e-12) {
            if (walked + len >= dist) {
                out = pts[static_cast<std::size_t>(cur)] + seg * ((dist - walked) / len);
                return true;
            }
            walked += len;
        }
        cur = j;
    }
    return false;
}

/// Maior afastamento, à reta que passa por `origin` com direção `d`, dos nós da cadeia que ficam a menos de `dist` px de
/// arco do nó i no sentido dir. Um canto vizinho dentro do braço aparece aqui, mesmo quando o ponto médio ainda cai na reta.
double nodeDeviation(const std::vector<Vec2>& pts, bool closed, std::size_t i, int dir, double dist, Vec2 origin, Vec2 d) {
    const long n = static_cast<long>(pts.size());
    double walked = 0.0;
    double worst = 0.0;
    long cur = static_cast<long>(i);
    for (long step = 0; step < n; ++step) {
        long j = cur + dir;
        if (closed) {
            j = ((j % n) + n) % n;
        } else if (j < 0 || j >= n) {
            break;
        }
        walked += sla::norm(pts[static_cast<std::size_t>(j)] - pts[static_cast<std::size_t>(cur)]);
        if (walked >= dist) break;
        worst = std::max(worst, std::abs(sla::cross(pts[static_cast<std::size_t>(j)] - origin, d)));
        cur = j;
    }
    return worst;
}

/// Reta n·p = c de um braço reto de uma aresta que sai do nó i no sentido dir, com a posição sub-pixel medida.
struct ArmLine {
    Vec2 n;
    double c = 0.0;
};

std::optional<ArmLine> armLine(const sla::Array2D<sla::Rgba8>& source, const std::vector<Vec2>& raw, bool closed, std::size_t i, int dir,
                               const Rgba& cl, const Rgba& cr, double reach, double maxShift) {
    Vec2 near, mid, far;
    if (!arcPoint(raw, closed, i, dir, kArmNear, near) || !arcPoint(raw, closed, i, dir, kArmMid, mid) ||
        !arcPoint(raw, closed, i, dir, kArmFar, far)) {
        return std::nullopt;  // braço curto demais (a aresta acaba perto do canto)
    }
    const Vec2 chord = far - near;
    const double len = sla::norm(chord);
    if (len < 1e-9) return std::nullopt;
    const Vec2 d = chord / len;
    if (std::abs(sla::cross(mid - near, d)) > kArmFlat ||
        nodeDeviation(raw, closed, i, dir, kArmFar, near, d) > kArmFlat) {
        return std::nullopt;  // braço curvo, ou com outro canto dentro dele: sem reta para medir
    }
    const Vec2 center = (near + far) * 0.5;
    const Vec2 normal{-d.y, d.x};
    const auto shift = profileShift(source, center, normal, cl, cr, reach);
    if (!shift) return std::nullopt;
    return ArmLine{normal, sla::dot(normal, center) + std::clamp(*shift, -maxShift, maxShift)};
}

}  // namespace

std::optional<double> edgeOffset(const sla::Array2D<sla::Rgba8>& source, Vec2 p, Vec2 normal, const Rgba& a, const Rgba& b,
                                 double reach, double step) {
    if (!(reach > 0.0) || !(step > 0.0) || step > reach) throw std::invalid_argument("edgeOffset: reach/step inválidos");
    if (source.width() <= 0 || source.height() <= 0) throw std::invalid_argument("edgeOffset: imagem vazia");
    const Rgba v = b - a;
    const double vv = sla::dot(v, v);
    if (!(vv > 1e-12)) return std::nullopt;  // as duas cores coincidem: sem borda para medir

    const int n = static_cast<int>(std::lround(2.0 * reach / step)) + 1;
    std::vector<double> t(static_cast<std::size_t>(n));
    double residual = 0.0;  // maior desvio² de uma amostra à reta a–b (uma única amostra de outra cor já invalida o perfil)
    for (int k = 0; k < n; ++k) {
        const double s = -reach + step * k;
        const Rgba c = sampleBilinear(source, p.x + normal.x * s, p.y + normal.y * s);
        const double raw = sla::dot(c - a, v) / vv;
        const double tk = std::clamp(raw, 0.0, 1.0);  // o anel do JPEG passa de 0..1 e não é cobertura
        const Rgba off = (c - a) - v * static_cast<float>(tk);
        residual = std::max(residual, static_cast<double>(sla::dot(off, off)));
        t[static_cast<std::size_t>(k)] = tk;
    }
    // Só vale a transição limpa de a para b: pontas puras dos dois lados e as amostras sobre a reta a–b.
    const double head = 0.5 * (t[0] + t[1]);
    const double tail = 0.5 * (t[static_cast<std::size_t>(n - 1)] + t[static_cast<std::size_t>(n - 2)]);
    if (head > kEndPure || tail < 1.0 - kEndPure) return std::nullopt;
    if (std::sqrt(residual) > kMaxResidual * std::sqrt(vv)) return std::nullopt;

    double integral = 0.0;  // trapézio
    for (int k = 0; k + 1 < n; ++k) integral += 0.5 * (t[static_cast<std::size_t>(k)] + t[static_cast<std::size_t>(k + 1)]) * step;
    return reach - integral;
}

AnchorStats refineEdgeAnchors(VectorImage& image, const sla::Array2D<sla::Rgba8>& source, const SmoothParams& params) {
    if (static_cast<double>(source.width()) != image.width() || static_cast<double>(source.height()) != image.height()) {
        throw std::invalid_argument("refineEdgeAnchors: a imagem tem dimensões diferentes do VectorImage");
    }
    AnchorStats stats;
    const double reach = params.anchorReachPx;
    const double maxShift = params.anchorMaxShiftPx;
    const double cornerTurn = params.punctureCornerThresh > 0.0 ? params.punctureCornerThresh : 0.8;
    double shiftSum = 0.0;
    constexpr std::uint8_t kSkipFlags = NodeFlag::Junction | NodeFlag::ImageBorder | NodeFlag::Corner | NodeFlag::Pinned;

    for (EdgeId e = 0; e < static_cast<EdgeId>(image.edgeCount()); ++e) {
        const Edge& edge = image.edge(e);
        if (edge.left == kExteriorFace || edge.right == kExteriorFace) continue;  // a moldura tem a sua própria âncora
        const Rgba cl = image.face(edge.left).color;
        const Rgba cr = image.face(edge.right).color;
        const auto nodes = image.edgeNodes(e);
        const bool lowContrast = 255.0 * std::sqrt(sla::distance2(cl, cr)) < params.anchorMinContrast;

        const bool closed = nodes.front() == nodes.back();
        const std::size_t m = closed ? nodes.size() - 1 : nodes.size();  // nós distintos na cadeia
        if (m < 3) continue;
        std::vector<Vec2> raw(m);
        for (std::size_t i = 0; i < m; ++i) raw[i] = image.originalPosition(nodes[i]);
        const std::vector<Vec2> filtered = smoothChain(raw, closed, kFilterHalfWindowPx);

        // Índice vizinho a `k` passos: periódico numa cadeia fechada; preso às pontas numa aberta.
        const auto at = [&](std::size_t i, long k) {
            const long mm = static_cast<long>(m);
            long j = static_cast<long>(i) + k;
            j = closed ? ((j % mm) + mm) % mm : std::clamp(j, 0L, mm - 1);
            return static_cast<std::size_t>(j);
        };
        const std::size_t first = closed ? 0 : 1;
        const std::size_t last = closed ? m : m - 1;  // exclusivo
        for (std::size_t i = first; i < last; ++i) {
            const NodeId node = nodes[i];
            if ((image.flags(node) & kSkipFlags) != 0) continue;
            if (lowContrast) {
                ++stats.lowContrast;
                continue;
            }
            ++stats.candidates;
            const Vec2 u = filtered[i] - filtered[at(i, -kCornerLookback)];
            const Vec2 w = filtered[at(i, kCornerLookback)] - filtered[i];
            const bool turning = sla::norm2(u) < 1e-18 || sla::norm2(w) < 1e-18 ||
                                 std::abs(std::atan2(sla::cross(u, w), sla::dot(u, w))) > cornerTurn;
            if (turning) {
                // Canto (ou curva fechada): a tangente do filtro não é a da borda. Um canto é a interseção das retas
                // dos seus dois braços, cada uma com a sua posição sub-pixel; sem dois braços retos e medidos, a
                // âncora fica na escadinha.
                const auto back = armLine(source, raw, closed, i, -1, cl, cr, reach, maxShift);
                const auto fwd = armLine(source, raw, closed, i, +1, cl, cr, reach, maxShift);
                const double det = back && fwd ? back->n.x * fwd->n.y - back->n.y * fwd->n.x : 0.0;
                if (!back || !fwd || std::abs(det) < kMinCornerCross) {
                    ++stats.skippedCorner;
                    continue;
                }
                const Vec2 corner{(back->c * fwd->n.y - back->n.y * fwd->c) / det, (back->n.x * fwd->c - back->c * fwd->n.x) / det};
                const double moved = sla::norm(corner - raw[i]);
                if (moved > kCornerShiftFactor * maxShift) {
                    ++stats.skippedCorner;
                    continue;
                }
                image.setOriginalPosition(node, corner);
                ++stats.moved;
                ++stats.cornersMoved;
                shiftSum += moved;
                continue;
            }
            Vec2 d = filtered[at(i, 1)] - filtered[at(i, -1)];
            const double len = sla::norm(d);
            if (len < 1e-9) {
                ++stats.skippedCorner;
                continue;
            }
            d = d / len;
            const Vec2 normal{-d.y, d.x};
            const auto shift = profileShift(source, raw[i], normal, cl, cr, reach);
            if (!shift) {
                ++stats.skippedProfile;
                continue;
            }
            const double s = std::clamp(*shift, -maxShift, maxShift);
            image.setOriginalPosition(node, raw[i] + normal * s);
            ++stats.moved;
            shiftSum += std::abs(s);
        }
    }
    stats.meanShiftPx = stats.moved > 0 ? shiftSum / stats.moved : 0.0;
    return stats;
}

}  // namespace cnx
