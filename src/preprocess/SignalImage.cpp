// src/preprocess/SignalImage.cpp — sinal pré-multiplicado: média de área, bilinear e gaussiana separáveis.
#include "cnx/preprocess/SignalImage.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cnx {
namespace {

using Acc = std::array<double, 4>;

struct Tap {
    int index = 0;
    double weight = 0.0;
};
using Taps = std::vector<std::vector<Tap>>;  // por índice de saída

/// Média de área: o intervalo de saída o cobre [o·r, (o+1)·r) da entrada, r = in/out; os pesos somam 1.
Taps areaTaps(int in, int out) {
    Taps taps(static_cast<std::size_t>(out));
    const double r = static_cast<double>(in) / static_cast<double>(out);
    for (int o = 0; o < out; ++o) {
        const double a = o * r;
        const double b = (o + 1) * r;
        const int first = static_cast<int>(std::floor(a));
        const int last = std::min(in - 1, static_cast<int>(std::ceil(b)) - 1);
        for (int i = first; i <= last; ++i) {
            const double cover = std::min(b, i + 1.0) - std::max(a, static_cast<double>(i));
            if (cover > 0.0) taps[static_cast<std::size_t>(o)].push_back({i, cover / r});
        }
    }
    return taps;
}

/// Bilinear com centros de pixel: f = (o + ½)·in/out − ½, entre ⌊f⌋ e ⌊f⌋ + 1 (bordas replicadas).
Taps bilinearTaps(int in, int out) {
    Taps taps(static_cast<std::size_t>(out));
    for (int o = 0; o < out; ++o) {
        const double f = (o + 0.5) * in / out - 0.5;
        const int i0 = std::clamp(static_cast<int>(std::floor(f)), 0, in - 1);
        const int i1 = std::min(i0 + 1, in - 1);
        const double t = std::clamp(f - i0, 0.0, 1.0);
        taps[static_cast<std::size_t>(o)] = {{i0, 1.0 - t}, {i1, t}};
    }
    return taps;
}

/// Gaussiana (σ > 0): raio ⌈3σ⌉ ≥ 1, pesos exp(−k²/2σ²) normalizados, índices replicados na borda.
Taps gaussianTaps(int n, double sigma) {
    const int radius = static_cast<int>(std::ceil(3.0 * sigma));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        const double v = std::exp(-0.5 * k * k / (sigma * sigma));
        kernel[static_cast<std::size_t>(k + radius)] = v;
        sum += v;
    }
    Taps taps(static_cast<std::size_t>(n));
    for (int o = 0; o < n; ++o) {
        for (int k = -radius; k <= radius; ++k) taps[static_cast<std::size_t>(o)].push_back({std::clamp(o + k, 0, n - 1), kernel[static_cast<std::size_t>(k + radius)] / sum});
    }
    return taps;
}

/// Filtro separável: horizontal (tx, largura w) e depois vertical (ty, altura h), acumulando em double.
sla::Array2D<sla::Rgba> separable(const sla::Array2D<sla::Rgba>& in, const Taps& tx, const Taps& ty) {
    const int w = static_cast<int>(tx.size());
    const int h = static_cast<int>(ty.size());
    std::vector<Acc> mid(static_cast<std::size_t>(w) * static_cast<std::size_t>(in.height()));
    for (int y = 0; y < in.height(); ++y) {
        for (int x = 0; x < w; ++x) {
            Acc s{};
            for (const Tap& t : tx[static_cast<std::size_t>(x)]) {
                const sla::Rgba& c = in(t.index, y);
                s[0] += t.weight * c.r;
                s[1] += t.weight * c.g;
                s[2] += t.weight * c.b;
                s[3] += t.weight * c.a;
            }
            mid[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] = s;
        }
    }
    sla::Array2D<sla::Rgba> out(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            Acc s{};
            for (const Tap& t : ty[static_cast<std::size_t>(y)]) {
                const Acc& c = mid[static_cast<std::size_t>(t.index) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)];
                for (std::size_t k = 0; k < 4; ++k) s[k] += t.weight * c[k];
            }
            out(x, y) = {static_cast<float>(s[0]), static_cast<float>(s[1]), static_cast<float>(s[2]), static_cast<float>(s[3])};
        }
    }
    return out;
}

}  // namespace

SignalImage SignalImage::fromStraight(const sla::Array2D<sla::Rgba8>& image) {
    SignalImage out;
    out.signal_ = sla::Array2D<sla::Rgba>(image.width(), image.height());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const sla::Rgba8 c = image(x, y);
            const double a = c.a / 255.0;
            out.signal_(x, y) = {static_cast<float>(c.r / 255.0 * a), static_cast<float>(c.g / 255.0 * a), static_cast<float>(c.b / 255.0 * a),
                                 static_cast<float>(a)};
        }
    }
    return out;
}

sla::Rgba SignalImage::straight(int x, int y) const noexcept {
    const sla::Rgba& p = signal_(x, y);
    if (!(p.a > 0.0f)) return {0.0f, 0.0f, 0.0f, 0.0f};
    const auto u = [&](float v) { return static_cast<float>(std::clamp(static_cast<double>(v) / p.a, 0.0, 1.0)); };
    return {u(p.r), u(p.g), u(p.b), std::min(p.a, 1.0f)};
}

sla::Array2D<sla::Rgba8> SignalImage::toStraight8() const {
    sla::Array2D<sla::Rgba8> out(width(), height());
    const auto q = [](double v) { return static_cast<std::uint8_t>(std::clamp(std::lround(255.0 * v), 0L, 255L)); };
    for (int y = 0; y < height(); ++y) {
        for (int x = 0; x < width(); ++x) {
            const sla::Rgba& p = signal_(x, y);
            const std::uint8_t a8 = q(p.a);
            if (a8 == 0) {
                out(x, y) = {0, 0, 0, 0};
                continue;
            }
            const double a = p.a;
            out(x, y) = {q(p.r / a), q(p.g / a), q(p.b / a), a8};
        }
    }
    return out;
}

SignalImage SignalImage::resampledArea(int w, int h) const {
    if (w < 1 || h < 1) throw std::invalid_argument("SignalImage::resampledArea: tamanho de saída < 1");
    if (w > width() || h > height()) throw std::invalid_argument("SignalImage::resampledArea: só reduz (use resampledBilinear)");
    SignalImage out;
    out.signal_ = separable(signal_, areaTaps(width(), w), areaTaps(height(), h));
    return out;
}

SignalImage SignalImage::resampledBilinear(int w, int h) const {
    if (w < 1 || h < 1) throw std::invalid_argument("SignalImage::resampledBilinear: tamanho de saída < 1");
    if (signal_.empty()) throw std::invalid_argument("SignalImage::resampledBilinear: sinal vazio");
    SignalImage out;
    out.signal_ = separable(signal_, bilinearTaps(width(), w), bilinearTaps(height(), h));
    return out;
}

SignalImage SignalImage::blurred(double sigma) const {
    if (std::isinf(sigma)) throw std::invalid_argument("SignalImage::blurred: sigma infinito");
    if (!(sigma > 0.0) || signal_.empty()) return *this;
    SignalImage out;
    out.signal_ = separable(signal_, gaussianTaps(width(), sigma), gaussianTaps(height(), sigma));
    return out;
}

}  // namespace cnx
