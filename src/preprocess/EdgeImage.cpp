// src/preprocess/EdgeImage.cpp — Sobel nos 4 canais, tensor de estrutura e cristas.
#include "cnx/preprocess/EdgeImage.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cnx {
namespace {

int roundToPixel(double v) { return static_cast<int>(std::floor(v + 0.5)); }

}  // namespace

EdgeImage EdgeImage::compute(const sla::Array2D<sla::Rgba8>& image) {
    const int w = image.width();
    const int h = image.height();
    EdgeImage out;
    out.magnitude_ = sla::Array2D<float>(w, h);
    out.normalX_ = sla::Array2D<float>(w, h);
    out.normalY_ = sla::Array2D<float>(w, h);
    if (w == 0 || h == 0) return out;
    // Em inteiros sobre os 8 bits: exato (região chapada dá 0, não 1e-17) e independente da ordem das somas.
    const auto px = [&](int x, int y) -> const sla::Rgba8& { return image(std::clamp(x, 0, w - 1), std::clamp(y, 0, h - 1)); };
    const auto channel = [](const sla::Rgba8& p, std::size_t c) -> int { return c == 0 ? p.r : c == 1 ? p.g : c == 2 ? p.b : p.a; };
    constexpr double kScale = 1.0 / (4.0 * 255.0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::array<const sla::Rgba8*, 9> n{&px(x - 1, y - 1), &px(x, y - 1), &px(x + 1, y - 1), &px(x - 1, y), &px(x, y),
                                                     &px(x + 1, y),     &px(x - 1, y + 1), &px(x, y + 1), &px(x + 1, y + 1)};
            std::int64_t jxx = 0;
            std::int64_t jyy = 0;
            std::int64_t jxy = 0;
            for (std::size_t c = 0; c < 4; ++c) {
                const auto v = [&](std::size_t k) { return channel(*n[k], c); };
                // Sobel: gx = [-1 0 1; -2 0 2; -1 0 1], gy = gxᵀ (|g| ≤ 4·255 por canal).
                const std::int64_t gx = (v(2) - v(0)) + 2 * (v(5) - v(3)) + (v(8) - v(6));
                const std::int64_t gy = (v(6) - v(0)) + 2 * (v(7) - v(1)) + (v(8) - v(2));
                jxx += gx * gx;
                jyy += gy * gy;
                jxy += gx * gy;
            }
            const std::int64_t sum = jxx + jyy;
            out.magnitude_(x, y) = static_cast<float>(std::sqrt(static_cast<double>(sum)) * kScale);
            if (sum > 0) {
                const double theta = 0.5 * std::atan2(2.0 * static_cast<double>(jxy), static_cast<double>(jxx - jyy));
                out.normalX_(x, y) = static_cast<float>(std::cos(theta));
                out.normalY_(x, y) = static_cast<float>(std::sin(theta));
            }
        }
    }
    return out;
}

sla::Array2D<float> EdgeImage::sharpnessWeights(double edgeScale) const {
    if (!(edgeScale > 0.0) || !std::isfinite(edgeScale)) throw std::invalid_argument("EdgeImage::sharpnessWeights: edgeScale <= 0 ou não finito");
    sla::Array2D<float> out(width(), height());
    for (int y = 0; y < height(); ++y) {
        for (int x = 0; x < width(); ++x) out(x, y) = static_cast<float>(1.0 / (1.0 + static_cast<double>(magnitude_(x, y)) / edgeScale));
    }
    return out;
}

std::vector<EdgePoint> EdgeImage::ridges(float minMagnitude) const {
    std::vector<EdgePoint> out;
    const auto at = [&](double fx, double fy) {
        const int ix = roundToPixel(fx);
        const int iy = roundToPixel(fy);
        return magnitude_.inBounds(ix, iy) ? magnitude_(ix, iy) : 0.0f;
    };
    for (int y = 0; y < height(); ++y) {
        for (int x = 0; x < width(); ++x) {
            const float m = magnitude_(x, y);
            if (!(m >= minMagnitude) || !(m > 0.0f)) continue;
            const sla::Vec2 n = normal(x, y);
            const float before = at(x - n.x, y - n.y);
            const float after = at(x + n.x, y + n.y);
            if (m > before && m >= after) out.push_back({x, y, m, n});
        }
    }
    return out;
}

}  // namespace cnx
