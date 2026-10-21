// src/vector/GenerativeModel.cpp — cobertura exata por área, energia esparsa e gradiente analítico.
#include "cnx/vector/GenerativeModel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cnx {
namespace detail {

/// Incremento fundido do acumulador de área: na linha y, a partir da coluna x, a soma prefixada
/// da linha passa a valer `d` a mais (4 canais).
struct Group {
    std::int32_t y = 0;
    std::int32_t x = 0;
    Color4 d{};
};

}  // namespace detail

namespace {

using detail::Group;

// ---------------------------------------------------------------- tipos internos

struct Segment {
    sla::Vec2 a;
    sla::Vec2 b;
    NodeId na = kInvalidId;
    NodeId nb = kInvalidId;
    Color4 dc{};  ///< c_esquerda − c_direita
};

/// Migalha de pão: o trecho [t0, t1] do segmento `seg` cai inteiro no pixel (x, y).
struct Piece {
    std::uint32_t seg = 0;
    std::int32_t x = 0;
    std::int32_t y = 0;
    double t0 = 0.0;
    double t1 = 0.0;
};

struct Event {
    std::int32_t y = 0;
    std::int32_t x = 0;
    Color4 d{};
};

constexpr std::size_t ux(std::int32_t i) noexcept { return static_cast<std::size_t>(i); }

Color4 toColor4(const sla::Rgba& c) noexcept { return {c.r, c.g, c.b, c.a}; }

double dot4(const Color4& a, const Color4& b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }

bool isZero(const Color4& c) noexcept { return c[0] == 0.0 && c[1] == 0.0 && c[2] == 0.0 && c[3] == 0.0; }

/// Cor do modelo: o exterior da imagem é transparente (0).
Color4 faceColor(const VectorImage& image, FaceId f) {
    return f == kExteriorFace ? Color4{} : toColor4(image.face(f).color);
}

// ---------------------------------------------------------------- migalhas de pão (§2.2.1)

/// Recorta o parâmetro t ∈ [t0, t1] de p + t·d ao intervalo [lo, hi]. Falso se sobra comprimento nulo.
bool clip1d(double p, double d, double lo, double hi, double& t0, double& t1) noexcept {
    if (d == 0.0) return p >= lo && p <= hi && t0 < t1;
    double ta = (lo - p) / d;
    double tb = (hi - p) / d;
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
    return t0 < t1;
}

/// Cruzamentos de linhas inteiras de uma coordenada ao longo de t, em ordem crescente (DDA).
struct LineWalker {
    double p = 0.0;      ///< coordenada inicial do segmento
    double d = 0.0;      ///< variação total
    double m = 0.0;      ///< próxima linha inteira
    double step = 0.0;   ///< ±1
    bool active = false;

    LineWalker(double p_, double d_, double u0, bool enabled) : p(p_), d(d_), active(enabled && d_ != 0.0) {
        if (!active) return;
        const double s = p + u0 * d;
        if (d > 0.0) {
            m = std::floor(s) + 1.0;
            step = 1.0;
        } else {
            m = std::ceil(s) - 1.0;
            step = -1.0;
        }
    }
    double t() const noexcept { return active ? (m - p) / d : std::numeric_limits<double>::infinity(); }
    void advance() noexcept { m += step; }
};

/// Chama fn(ta, tb) para cada subintervalo de [u0, u1] sem cruzar linha inteira (x se useX, y sempre).
template <class Fn>
void forEachCrumb(double ax, double ay, double dx, double dy, double u0, double u1, bool useX, Fn&& fn) {
    LineWalker wx(ax, dx, u0, useX);
    LineWalker wy(ay, dy, u0, true);
    double prev = u0;
    for (;;) {
        const double tx = wx.t();
        const double ty = wy.t();
        const double t = std::min(tx, ty);
        if (!(t < u1)) break;
        if (t > prev) {
            fn(prev, t);
            prev = t;
        }
        if (tx == t) wx.advance();
        if (ty == t) wy.advance();
    }
    if (u1 > prev) fn(prev, u1);
}

/// Gera os incrementos do acumulador (e, se pedido, as migalhas) de um segmento.
///
/// Pedaço dentro do pixel (i, j): α = −δy·((i+1) − x̄) vai para a coluna i e γ − α, com γ = −δy,
/// para a coluna i+1 (§2.2.2). A parte do segmento à esquerda da imagem (x < 0) cobre, na sua
/// linha, tudo o que está à direita: soma γ na coluna 0. Partes com y fora de [0, H] ou x > W não
/// afetam nenhum pixel.
void scanSegment(const Segment& s, std::uint32_t index, int W, int H, bool keepZeroEvents, std::vector<Event>& events,
                 std::vector<Piece>* pieces, std::size_t& pieceCount) {
    const double ax = s.a.x;
    const double ay = s.a.y;
    const double dx = s.b.x - s.a.x;
    const double dy = s.b.y - s.a.y;
    if (dx == 0.0 && dy == 0.0) return;

    double t0 = 0.0;
    double t1 = 1.0;
    if (!clip1d(ay, dy, 0.0, static_cast<double>(H), t0, t1)) return;

    // Parte com x ∈ [0, W] e parte com x < 0.
    double i0 = t0;
    double i1 = t1;
    bool hasInside = clip1d(ax, dx, 0.0, static_cast<double>(W), i0, i1);
    double l0 = 0.0;
    double l1 = -1.0;  // vazio
    if (dx == 0.0) {
        if (ax < 0.0) {
            l0 = t0;
            l1 = t1;
        }
    } else {
        const double tx = (0.0 - ax) / dx;  // instante em que x = 0
        if (dx > 0.0) {
            l0 = t0;
            l1 = std::min(t1, tx);
        } else {
            l0 = std::max(t0, tx);
            l1 = t1;
        }
    }

    const auto emitInside = [&](double ta, double tb) {
        const double xm = ax + 0.5 * (ta + tb) * dx;
        const double ym = ay + 0.5 * (ta + tb) * dy;
        int col = static_cast<int>(std::floor(xm));
        int row = static_cast<int>(std::floor(ym));
        col = std::max(col, 0);
        row = std::max(row, 0);
        if (col >= W || row >= H) return;  // sobre a borda direita/inferior: nenhum pixel dentro
        ++pieceCount;
        if (pieces != nullptr) pieces->push_back({index, col, row, ta, tb});
        const double deltaY = (tb - ta) * dy;
        if (deltaY == 0.0 && !keepZeroEvents) return;
        const double gamma = -deltaY;
        const double alpha = -deltaY * (static_cast<double>(col + 1) - xm);
        Event e;
        e.y = row;
        e.x = col;
        for (std::size_t c = 0; c < 4; ++c) e.d[c] = s.dc[c] * alpha;
        events.push_back(e);
        if (col + 1 < W) {
            e.x = col + 1;
            for (std::size_t c = 0; c < 4; ++c) e.d[c] = s.dc[c] * (gamma - alpha);
            events.push_back(e);
        }
    };

    if (hasInside) forEachCrumb(ax, ay, dx, dy, i0, i1, true, emitInside);

    if (l1 > l0) {
        forEachCrumb(ax, ay, dx, dy, l0, l1, false, [&](double ta, double tb) {
            const int row = static_cast<int>(std::floor(ay + 0.5 * (ta + tb) * dy));
            if (row < 0 || row >= H) return;
            const double gamma = -(tb - ta) * dy;
            if (gamma == 0.0) return;
            Event e;
            e.y = row;
            e.x = 0;
            for (std::size_t c = 0; c < 4; ++c) e.d[c] = s.dc[c] * gamma;
            events.push_back(e);
        });
    }
}

/// Ordena os incrementos por (linha, coluna) e soma os repetidos.
std::vector<Group> mergeEvents(std::vector<Event>& events) {
    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) { return a.y != b.y ? a.y < b.y : a.x < b.x; });
    std::vector<Group> groups;
    groups.reserve(events.size());
    for (const Event& e : events) {
        if (!groups.empty() && groups.back().y == e.y && groups.back().x == e.x) {
            for (std::size_t c = 0; c < 4; ++c) groups.back().d[c] += e.d[c];
        } else {
            groups.push_back({e.y, e.x, e.d});
        }
    }
    return groups;
}

/// Segmentos de uma poligonal percorrida de `from` a `to` (nós em ordem).
void appendChain(const VectorImage& image, std::span<const NodeId> nodes, bool reversed, const Color4& dc,
                 std::vector<Segment>& out) {
    const std::size_t n = nodes.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const NodeId from = reversed ? nodes[n - 1 - i] : nodes[i];
        const NodeId to = reversed ? nodes[n - 2 - i] : nodes[i + 1];
        out.push_back({image.position(from), image.position(to), from, to, dc});
    }
}

/// Um segmento por trecho de aresta, com dc = c_esq − c_dir (exterior = 0). Arestas sem diferença
/// de cor não contribuem em nada.
std::vector<Segment> buildSegments(const VectorImage& image) {
    std::vector<Segment> segs;
    for (EdgeId e = 0; e < static_cast<EdgeId>(image.edgeCount()); ++e) {
        const Edge& ed = image.edge(e);
        const Color4 l = faceColor(image, ed.left);
        const Color4 r = faceColor(image, ed.right);
        const Color4 dc{l[0] - r[0], l[1] - r[1], l[2] - r[2], l[3] - r[3]};
        if (isZero(dc)) continue;
        appendChain(image, image.edgeNodes(e), false, dc, segs);
    }
    return segs;
}

/// Segmentos dos laços de uma face, orientados como os laços (dc = indicador da face no canal 0).
std::vector<Segment> buildFaceSegments(const VectorImage& image, FaceId f) {
    std::vector<Segment> segs;
    const Color4 one{1.0, 0.0, 0.0, 0.0};
    const auto loop = [&](HalfEdgeId first) {
        HalfEdgeId h = first;
        do {
            appendChain(image, image.edgeNodes(edgeOf(h)), isReversed(h), one, segs);
            h = image.next(h);
        } while (h != first);
    };
    const Face& fc = image.face(f);
    if (fc.outerLoop != kInvalidId) loop(fc.outerLoop);
    for (const HalfEdgeId h : image.holeLoops(f)) loop(h);
    return segs;
}

/// Preenche `visit(x, y, ĉ)` para todo pixel da imagem a partir dos incrementos fundidos.
template <class Fn>
void forEachPixel(const std::vector<Group>& groups, int W, int H, Fn&& visit) {
    std::size_t g = 0;
    for (int y = 0; y < H; ++y) {
        Color4 c{};
        for (int x = 0; x < W; ++x) {
            if (g < groups.size() && groups[g].y == y && groups[g].x == x) {
                for (std::size_t k = 0; k < 4; ++k) c[k] += groups[g].d[k];
                ++g;
            }
            visit(x, y, c);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------- construção

GenerativeModel::GenerativeModel(sla::Array2D<sla::Rgba> target) : target_(std::move(target)) {
    if (target_.width() <= 0 || target_.height() <= 0) throw std::invalid_argument("GenerativeModel: imagem alvo vazia");
    const std::size_t stride = static_cast<std::size_t>(target_.width()) + 1;
    const std::size_t rows = static_cast<std::size_t>(target_.height());
    prefS1_.assign(stride * rows * 4, 0.0);
    prefS2_.assign(stride * rows, 0.0);
    if (!weights_.empty()) prefW_.assign(stride * rows, 0.0);
    for (int y = 0; y < target_.height(); ++y) {
        const std::size_t base = static_cast<std::size_t>(y) * stride;
        for (int x = 0; x < target_.width(); ++x) {
            const double w = pixelWeight(x, y);
            const Color4 i = toColor4(target_(x, y));
            const std::size_t a = base + static_cast<std::size_t>(x);
            for (std::size_t c = 0; c < 4; ++c) prefS1_[(a + 1) * 4 + c] = prefS1_[a * 4 + c] + w * i[c];
            prefS2_[a + 1] = prefS2_[a] + w * dot4(i, i);
            if (!prefW_.empty()) prefW_[a + 1] = prefW_[a] + w;
        }
    }
}

GenerativeModel::GenerativeModel(sla::Array2D<sla::Rgba> target, const sla::Array2D<float>& weights)
    : target_(std::move(target)), weights_(weights) {
    if (target_.width() <= 0 || target_.height() <= 0) throw std::invalid_argument("GenerativeModel: imagem alvo vazia");
    if (weights_.width() != target_.width() || weights_.height() != target_.height()) {
        throw std::invalid_argument("GenerativeModel: pesos com dimensões diferentes do alvo");
    }
    for (int y = 0; y < weights_.height(); ++y) {
        for (int x = 0; x < weights_.width(); ++x) {
            const float w = weights_(x, y);
            if (!(w >= 0.0f) || !std::isfinite(w)) throw std::invalid_argument("GenerativeModel: peso negativo ou não finito");
        }
    }
    const std::size_t stride = static_cast<std::size_t>(target_.width()) + 1;
    const std::size_t rows = static_cast<std::size_t>(target_.height());
    prefS1_.assign(stride * rows * 4, 0.0);
    prefS2_.assign(stride * rows, 0.0);
    prefW_.assign(stride * rows, 0.0);
    for (int y = 0; y < target_.height(); ++y) {
        const std::size_t base = static_cast<std::size_t>(y) * stride;
        for (int x = 0; x < target_.width(); ++x) {
            const double w = pixelWeight(x, y);
            const Color4 i = toColor4(target_(x, y));
            const std::size_t a = base + static_cast<std::size_t>(x);
            for (std::size_t c = 0; c < 4; ++c) prefS1_[(a + 1) * 4 + c] = prefS1_[a * 4 + c] + w * i[c];
            prefS2_[a + 1] = prefS2_[a] + w * dot4(i, i);
            prefW_[a + 1] = prefW_[a] + w;
        }
    }
}

sla::Array2D<sla::Rgba> GenerativeModel::targetFrom(const sla::Array2D<sla::Rgba8>& image) {
    sla::Array2D<sla::Rgba> out(image.width(), image.height());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) out(x, y) = sla::toRgba(image(x, y));
    }
    return out;
}

void GenerativeModel::checkDimensions(const VectorImage& image) const {
    if (image.width() != static_cast<double>(width()) || image.height() != static_cast<double>(height())) {
        throw std::invalid_argument("GenerativeModel: VectorImage com dimensões diferentes do alvo");
    }
}

// ---------------------------------------------------------------- energia esparsa (§2.2.3)

double GenerativeModel::runEnergy(int y, int x0, int x1, const Color4& c) const {
    if (x1 <= x0) return 0.0;
    const std::size_t stride = static_cast<std::size_t>(width()) + 1;
    const std::size_t a = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x0);
    const std::size_t b = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x1);
    const double n = prefW_.empty() ? static_cast<double>(x1 - x0) : prefW_[b] - prefW_[a];
    Color4 s1;
    for (std::size_t k = 0; k < 4; ++k) s1[k] = prefS1_[b * 4 + k] - prefS1_[a * 4 + k];
    const double s2 = prefS2_[b] - prefS2_[a];
    // E_run = n‖c‖² − 2 c·S¹ + S²; o valor verdadeiro é ≥ 0, então corta o resíduo de arredondamento.
    return std::max(0.0, n * dot4(c, c) - 2.0 * dot4(c, s1) + s2);
}

double GenerativeModel::sweep(const std::vector<Group>& groups, std::vector<Color4>* groupColors) const {
    const int W = width();
    const int H = height();
    if (groupColors != nullptr) groupColors->assign(groups.size(), Color4{});
    double total = 0.0;
    std::size_t g = 0;
    for (int y = 0; y < H; ++y) {
        Color4 c{};
        int cursor = 0;
        for (; g < groups.size() && groups[g].y == y; ++g) {
            const int x = groups[g].x;
            total += runEnergy(y, cursor, x, c);  // pixels não cruzados antes da coluna x: cor constante
            for (std::size_t k = 0; k < 4; ++k) c[k] += groups[g].d[k];
            if (groupColors != nullptr) (*groupColors)[g] = c;
            const Color4 i = toColor4(target_(x, y));
            const double r0 = c[0] - i[0];
            const double r1 = c[1] - i[1];
            const double r2 = c[2] - i[2];
            const double r3 = c[3] - i[3];
            total += pixelWeight(x, y) * (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);
            cursor = x + 1;
        }
        total += runEnergy(y, cursor, W, c);
    }
    return total;
}

double GenerativeModel::energy(const VectorImage& image, GenerativeStats* stats) const {
    checkDimensions(image);
    const std::vector<Segment> segs = buildSegments(image);
    std::vector<Event> events;
    std::size_t pieceCount = 0;
    for (std::size_t s = 0; s < segs.size(); ++s) {
        scanSegment(segs[s], static_cast<std::uint32_t>(s), width(), height(), false, events, nullptr, pieceCount);
    }
    const std::size_t eventCount = events.size();
    const std::vector<Group> groups = mergeEvents(events);
    if (stats != nullptr) *stats = {segs.size(), pieceCount, eventCount, groups.size()};
    return sweep(groups, nullptr);
}

// ---------------------------------------------------------------- gradiente (§2.3)

double GenerativeModel::energyAndGradient(const VectorImage& image, std::span<sla::Vec2> gradient,
                                          GenerativeStats* stats) const {
    checkDimensions(image);
    if (gradient.size() != image.nodeCount()) {
        throw std::invalid_argument("GenerativeModel::energyAndGradient: gradient.size() != nodeCount()");
    }
    std::fill(gradient.begin(), gradient.end(), sla::Vec2{});

    const std::vector<Segment> segs = buildSegments(image);
    std::vector<Event> events;
    std::vector<Piece> pieces;
    std::size_t pieceCount = 0;
    // Eventos nulos também são mantidos: marcam os pixels cruzados por segmentos horizontais, que
    // têm gradiente (a normal não é nula) embora não alterem a cobertura.
    for (std::size_t s = 0; s < segs.size(); ++s) {
        scanSegment(segs[s], static_cast<std::uint32_t>(s), width(), height(), true, events, &pieces, pieceCount);
    }
    const std::size_t eventCount = events.size();
    const std::vector<Group> groups = mergeEvents(events);
    std::vector<Color4> groupColors;
    const double total = sweep(groups, &groupColors);
    if (stats != nullptr) *stats = {segs.size(), pieceCount, eventCount, groups.size()};

    for (const Piece& p : pieces) {
        const Segment& s = segs[p.seg];
        // ĉ do pixel cruzado: o grupo (y, x) existe porque a migalha gerou um incremento ali.
        const auto it = std::lower_bound(groups.begin(), groups.end(), std::make_pair(p.y, p.x),
                                         [](const Group& gr, const std::pair<std::int32_t, std::int32_t>& key) {
                                             return gr.y != key.first ? gr.y < key.first : gr.x < key.second;
                                         });
        const Color4& c = groupColors[static_cast<std::size_t>(it - groups.begin())];
        const sla::Rgba& t = target_(p.x, p.y);
        const Color4 r{c[0] - t.r, c[1] - t.g, c[2] - t.b, c[3] - t.a};
        const double scale = 2.0 * pixelWeight(p.x, p.y) * dot4(s.dc, r);
        if (scale == 0.0) continue;
        // n_k·dt: normal externa da face esquerda, sem normalizar (‖n‖ = comprimento do segmento).
        const sla::Vec2 n = sla::outwardNormal(s.b - s.a);
        const double q = 0.5 * (p.t1 * p.t1 - p.t0 * p.t0);
        const double wStart = (p.t1 - p.t0) - q;  // ∫(1−t) dt
        const double wEnd = q;                    // ∫ t dt
        gradient[ux(s.na)] += n * (scale * wStart);
        gradient[ux(s.nb)] += n * (scale * wEnd);
    }
    return total;
}

// ---------------------------------------------------------------- imagem sintetizada densa

void GenerativeModel::render(const VectorImage& image, sla::Array2D<Color4>& out) const {
    checkDimensions(image);
    const std::vector<Segment> segs = buildSegments(image);
    std::vector<Event> events;
    std::size_t pieceCount = 0;
    for (std::size_t s = 0; s < segs.size(); ++s) {
        scanSegment(segs[s], static_cast<std::uint32_t>(s), width(), height(), false, events, nullptr, pieceCount);
    }
    const std::vector<Group> groups = mergeEvents(events);
    out = sla::Array2D<Color4>(width(), height());
    forEachPixel(groups, width(), height(), [&](int x, int y, const Color4& c) { out(x, y) = c; });
}

void GenerativeModel::faceCoverage(const VectorImage& image, FaceId face, sla::Array2D<double>& out) const {
    checkDimensions(image);
    if (face < 0 || ux(face) >= image.faceCount()) throw std::out_of_range("GenerativeModel::faceCoverage: face inválida");
    if (face == kExteriorFace) {
        // Os laços da face exterior são a moldura percorrida ao contrário (área −W·H): a soma de Green
        // dá −1 em cada pixel. O exterior é o complemento da imagem, então sua cobertura dentro dela é 0.
        out = sla::Array2D<double>(width(), height());
        return;
    }
    const std::vector<Segment> segs = buildFaceSegments(image, face);
    std::vector<Event> events;
    std::size_t pieceCount = 0;
    for (std::size_t s = 0; s < segs.size(); ++s) {
        scanSegment(segs[s], static_cast<std::uint32_t>(s), width(), height(), false, events, nullptr, pieceCount);
    }
    const std::vector<Group> groups = mergeEvents(events);
    out = sla::Array2D<double>(width(), height());
    forEachPixel(groups, width(), height(), [&](int x, int y, const Color4& c) { out(x, y) = c[0]; });
}

double GenerativeModel::denseEnergy(const VectorImage& image) const {
    sla::Array2D<Color4> synth;
    render(image, synth);
    double total = 0.0;
    for (int y = 0; y < height(); ++y) {
        for (int x = 0; x < width(); ++x) {
            const Color4 i = toColor4(target_(x, y));
            double e = 0.0;
            for (std::size_t c = 0; c < 4; ++c) e += (synth(x, y)[c] - i[c]) * (synth(x, y)[c] - i[c]);
            total += pixelWeight(x, y) * e;
        }
    }
    return total;
}

}  // namespace cnx
