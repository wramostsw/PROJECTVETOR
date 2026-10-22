// src/preprocess/AADetector.cpp — perfis de 6 pixels através das cristas: degrau, rampa ou outro.
#include "cnx/preprocess/AADetector.h"

#include <cmath>
#include <stdexcept>

namespace cnx {
namespace {

using Color4d = std::array<double, 4>;

Color4d toColor(const sla::Rgba& c) { return {c.r, c.g, c.b, c.a}; }

double dot4(const Color4d& a, const Color4d& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }

int roundToPixel(double v) { return static_cast<int>(std::floor(v + 0.5)); }

}  // namespace

EdgeProfile analyzeEdgeProfile(const std::array<sla::Rgba, 6>& samples, const AAParams& params) {
    EdgeProfile out;
    const Color4d a = toColor(samples[0]);
    const Color4d b = toColor(samples[5]);
    Color4d d{};
    for (std::size_t c = 0; c < 4; ++c) d[c] = b[c] - a[c];
    const double d2 = dot4(d, d);
    if (!(std::sqrt(d2) >= params.minContrast)) return out;  // Other: sem degrau entre as pontas

    double previous = 0.0;  // t₀ = 0
    int intermediate = 0;
    for (std::size_t k = 1; k <= 5; ++k) {
        double t = 1.0;  // t₅ = 1
        if (k < 5) {
            const Color4d v = toColor(samples[k]);
            Color4d s{};
            for (std::size_t c = 0; c < 4; ++c) s[c] = v[c] - a[c];
            t = dot4(s, d) / d2;
            double r2 = 0.0;
            for (std::size_t c = 0; c < 4; ++c) {
                const double e = s[c] - t * d[c];
                r2 += e * e;
            }
            if (std::sqrt(r2 / d2) > params.lineTolerance) return out;  // terceira cor
            if (t < -params.monotoneTolerance || t > 1.0 + params.monotoneTolerance) return out;  // ringing
            if (t > params.stepTolerance && t < 1.0 - params.stepTolerance) ++intermediate;
        }
        if (t < previous - params.monotoneTolerance) return out;  // vai e volta: textura, não borda
        previous = t;
    }
    out.kind = intermediate == 0 ? EdgeProfileKind::Step : EdgeProfileKind::Ramp;
    out.intermediate = intermediate;
    return out;
}

double AAStats::meanTransitionWidth() const noexcept {
    if (valid() == 0) return 0.0;
    double sum = 0.0;
    for (std::size_t k = 0; k < rampWidths.size(); ++k) sum += static_cast<double>(k) * static_cast<double>(rampWidths[k]);
    return sum / static_cast<double>(valid());
}

AAStats detectAntialiasing(const sla::Array2D<sla::Rgba8>& image, const EdgeImage& edges, const AAParams& params) {
    if (image.width() != edges.width() || image.height() != edges.height()) {
        throw std::invalid_argument("detectAntialiasing: EdgeImage com dimensões diferentes da imagem");
    }
    AAStats stats;
    for (const EdgePoint& p : edges.ridges(params.minEdgeMagnitude)) {
        ++stats.ridges;
        std::array<sla::Rgba, 6> samples{};
        bool inside = true;
        for (int k = 0; k < 6 && inside; ++k) {
            const double off = static_cast<double>(k - 3);
            const int x = roundToPixel(p.x + off * p.normal.x);
            const int y = roundToPixel(p.y + off * p.normal.y);
            inside = image.inBounds(x, y);
            if (inside) samples[static_cast<std::size_t>(k)] = sla::toRgba(image(x, y));
        }
        if (!inside) {
            ++stats.others;
            continue;
        }
        const EdgeProfile profile = analyzeEdgeProfile(samples, params);
        switch (profile.kind) {
            case EdgeProfileKind::Step: ++stats.steps; break;
            case EdgeProfileKind::Ramp: ++stats.ramps; break;
            case EdgeProfileKind::Other: ++stats.others; break;
        }
        if (profile.kind != EdgeProfileKind::Other) ++stats.rampWidths[static_cast<std::size_t>(profile.intermediate)];
    }
    return stats;
}

}  // namespace cnx
