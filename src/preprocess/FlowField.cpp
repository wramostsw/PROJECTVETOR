// src/preprocess/FlowField.cpp — tensor de estrutura integrado por gaussiana separável.
#include "cnx/preprocess/FlowField.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cnx {
namespace {

using Tensor3 = std::array<double, 3>;  // xx, xy, yy

/// Anisotropia λ₁ − λ₂ = √((Jxx − Jyy)² + 4·Jxy²).
double anisotropy(const StructureTensor& t) noexcept {
    const double d = static_cast<double>(t.xx) - static_cast<double>(t.yy);
    const double c = 2.0 * static_cast<double>(t.xy);
    return std::sqrt(d * d + c * c);
}

}  // namespace

FlowField FlowField::compute(const sla::Array2D<sla::Rgba8>& image, double sigma) {
    if (!(sigma >= 0.0) || !std::isfinite(sigma)) throw std::invalid_argument("FlowField::compute: sigma < 0 ou não finito");
    const int w = image.width();
    const int h = image.height();
    FlowField out;
    out.sigma_ = sigma;
    const sla::Array2D<StructureTensor> raw = sobelTensors(image);
    out.tensor_ = sla::Array2D<StructureTensor>(w, h);
    if (w == 0 || h == 0) return out;

    constexpr double kUnit = kSobelScale * kSobelScale;
    if (sigma == 0.0) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const StructureTensor& r = raw(x, y);
                out.tensor_(x, y) = {static_cast<float>(r.xx * kUnit), static_cast<float>(r.xy * kUnit), static_cast<float>(r.yy * kUnit)};
            }
        }
        return out;
    }

    const int radius = static_cast<int>(std::ceil(3.0 * sigma));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    double norm = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        const double v = std::exp(-0.5 * k * k / (sigma * sigma));
        kernel[static_cast<std::size_t>(k + radius)] = v;
        norm += v;
    }
    for (double& v : kernel) v /= norm;

    // Horizontal sobre o tensor bruto (em double), depois vertical; bordas replicadas. Ordem fixa das somas.
    std::vector<Tensor3> tmp(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    const auto at = [w](int x, int y) { return static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x); };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            Tensor3 s{};
            for (int k = -radius; k <= radius; ++k) {
                const StructureTensor& r = raw(std::clamp(x + k, 0, w - 1), y);
                const double g = kernel[static_cast<std::size_t>(k + radius)];
                s[0] += g * r.xx;
                s[1] += g * r.xy;
                s[2] += g * r.yy;
            }
            tmp[at(x, y)] = s;
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            Tensor3 s{};
            for (int k = -radius; k <= radius; ++k) {
                const Tensor3& r = tmp[at(x, std::clamp(y + k, 0, h - 1))];
                const double g = kernel[static_cast<std::size_t>(k + radius)];
                for (std::size_t c = 0; c < 3; ++c) s[c] += g * r[c];
            }
            out.tensor_(x, y) = {static_cast<float>(s[0] * kUnit), static_cast<float>(s[1] * kUnit), static_cast<float>(s[2] * kUnit)};
        }
    }
    return out;
}

double FlowField::energy(int x, int y) const noexcept {
    const StructureTensor& t = tensor_(x, y);
    return static_cast<double>(t.xx) + static_cast<double>(t.yy);
}

double FlowField::coherence(int x, int y) const noexcept { return coherenceOf(tensor_(x, y)); }

double FlowField::coherenceOf(const StructureTensor& t) noexcept {
    const double e = static_cast<double>(t.xx) + static_cast<double>(t.yy);
    if (!(e > 0.0)) return 0.0;
    return std::min(1.0, anisotropy(t) / e);
}

sla::Vec2 FlowField::tangent(int x, int y) const noexcept {
    const StructureTensor& t = tensor_(x, y);
    if (!(anisotropy(t) > 0.0)) return {};
    // Normal = autovetor de λ₁, no ângulo θ = ½·atan2(2Jxy, Jxx − Jyy); a tangente é a normal girada de 90°.
    const double theta = 0.5 * std::atan2(2.0 * static_cast<double>(t.xy), static_cast<double>(t.xx) - static_cast<double>(t.yy));
    return {-std::sin(theta), std::cos(theta)};
}

sla::Vec2 FlowField::normal(int x, int y) const noexcept {
    const sla::Vec2 t = tangent(x, y);
    return {t.y, -t.x};
}

double FlowField::alignment(int x, int y, sla::Vec2 direction) const noexcept { return alignmentOf(tensor_(x, y), direction); }

double FlowField::alignmentOf(const StructureTensor& t, sla::Vec2 direction) noexcept {
    const double a = anisotropy(t);
    const double dd = sla::norm2(direction);
    if (!(a > 0.0) || !(dd > 0.0)) return 0.0;
    // Com a normal no ângulo θ e a direção no ângulo ψ: cos 2α = cos(2(θ + 90° − ψ)) = −cos(2θ − 2ψ), onde
    // (cos 2θ, sin 2θ) = (Jxx − Jyy, 2Jxy)/a e (cos 2ψ, sin 2ψ) = (dx² − dy², 2·dx·dy)/‖d‖².
    const double c2 = direction.x * direction.x - direction.y * direction.y;
    const double s2 = 2.0 * direction.x * direction.y;
    const double v = -((static_cast<double>(t.xx) - static_cast<double>(t.yy)) * c2 + 2.0 * static_cast<double>(t.xy) * s2) / (a * dd);
    return std::clamp(v, -1.0, 1.0);
}

}  // namespace cnx
