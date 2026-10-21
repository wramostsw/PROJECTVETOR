// src/preprocess/Preprocessor.cpp — alfa, reamostragem por média de área e desfoque gaussiano.
#include "cnx/preprocess/Preprocessor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cnx {
namespace {

using sla::Rgba8;

struct Tap {
    int index = 0;
    float weight = 0.0f;
};

/// Para cada coluna (ou linha) de saída, os pixels de entrada que ela cobre e a fração coberta de cada um.
/// O intervalo de saída i cobre [i·r, (i+1)·r) da entrada, com r = in/out; os pesos somam 1.
std::vector<std::vector<Tap>> areaTaps(int in, int out) {
    std::vector<std::vector<Tap>> taps(static_cast<std::size_t>(out));
    const double r = static_cast<double>(in) / static_cast<double>(out);
    for (int o = 0; o < out; ++o) {
        const double a = o * r;
        const double b = (o + 1) * r;
        const int first = static_cast<int>(std::floor(a));
        const int last = std::min(in - 1, static_cast<int>(std::ceil(b)) - 1);
        for (int i = first; i <= last; ++i) {
            const double cover = std::min(b, i + 1.0) - std::max(a, static_cast<double>(i));
            if (cover > 0.0) taps[static_cast<std::size_t>(o)].push_back({i, static_cast<float>(cover / r)});
        }
    }
    return taps;
}

Rgba8 quantize(const std::array<float, 4>& v) {
    const auto q = [](float x) { return static_cast<std::uint8_t>(std::clamp(static_cast<int>(std::lround(x)), 0, 255)); };
    return {q(v[0]), q(v[1]), q(v[2]), q(v[3])};
}

std::array<float, 4> asFloats(Rgba8 c) { return {static_cast<float>(c.r), static_cast<float>(c.g), static_cast<float>(c.b), static_cast<float>(c.a)}; }

}  // namespace

sla::Array2D<Rgba8> resampleArea(const sla::Array2D<Rgba8>& in, int w, int h) {
    if (w < 1 || h < 1) throw std::invalid_argument("resampleArea: tamanho de saída < 1");
    if (w > in.width() || h > in.height()) throw std::invalid_argument("resampleArea: só reduz (use a ampliação bilinear)");
    const auto tx = areaTaps(in.width(), w);
    const auto ty = areaTaps(in.height(), h);
    // Horizontal (float), depois vertical.
    std::vector<std::array<float, 4>> mid(static_cast<std::size_t>(w) * static_cast<std::size_t>(in.height()));
    for (int y = 0; y < in.height(); ++y) {
        for (int x = 0; x < w; ++x) {
            std::array<float, 4> s{};
            for (const Tap& t : tx[static_cast<std::size_t>(x)]) {
                const auto c = asFloats(in(t.index, y));
                for (std::size_t k = 0; k < 4; ++k) s[k] += t.weight * c[k];
            }
            mid[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] = s;
        }
    }
    sla::Array2D<Rgba8> out(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::array<float, 4> s{};
            for (const Tap& t : ty[static_cast<std::size_t>(y)]) {
                const auto& c = mid[static_cast<std::size_t>(t.index) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)];
                for (std::size_t k = 0; k < 4; ++k) s[k] += t.weight * c[k];
            }
            out(x, y) = quantize(s);
        }
    }
    return out;
}

namespace {

sla::Array2D<Rgba8> resampleBilinear(const sla::Array2D<Rgba8>& in, int w, int h) {
    sla::Array2D<Rgba8> out(w, h);
    for (int y = 0; y < h; ++y) {
        const double fy = (y + 0.5) * in.height() / h - 0.5;
        const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, in.height() - 1);
        const int y1 = std::min(y0 + 1, in.height() - 1);
        const float ty = static_cast<float>(std::clamp(fy - y0, 0.0, 1.0));
        for (int x = 0; x < w; ++x) {
            const double fx = (x + 0.5) * in.width() / w - 0.5;
            const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, in.width() - 1);
            const int x1 = std::min(x0 + 1, in.width() - 1);
            const float tx = static_cast<float>(std::clamp(fx - x0, 0.0, 1.0));
            const auto a = asFloats(in(x0, y0)), b = asFloats(in(x1, y0)), c = asFloats(in(x0, y1)), d = asFloats(in(x1, y1));
            std::array<float, 4> v{};
            for (std::size_t k = 0; k < 4; ++k) v[k] = (a[k] * (1 - tx) + b[k] * tx) * (1 - ty) + (c[k] * (1 - tx) + d[k] * tx) * ty;
            out(x, y) = quantize(v);
        }
    }
    return out;
}

}  // namespace

sla::Array2D<Rgba8> gaussianBlur(const sla::Array2D<Rgba8>& in, double sigma) {
    if (!(sigma > 0.0)) return in;
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<float> kernel(static_cast<std::size_t>(2 * radius + 1));
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        const float v = static_cast<float>(std::exp(-0.5 * k * k / (sigma * sigma)));
        kernel[static_cast<std::size_t>(k + radius)] = v;
        sum += v;
    }
    for (float& v : kernel) v /= sum;
    const int w = in.width();
    const int h = in.height();
    std::vector<std::array<float, 4>> tmp(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::array<float, 4> s{};
            for (int k = -radius; k <= radius; ++k) {
                const auto c = asFloats(in(std::clamp(x + k, 0, w - 1), y));
                for (std::size_t d = 0; d < 4; ++d) s[d] += kernel[static_cast<std::size_t>(k + radius)] * c[d];
            }
            tmp[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] = s;
        }
    }
    sla::Array2D<Rgba8> out(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::array<float, 4> s{};
            for (int k = -radius; k <= radius; ++k) {
                const auto& c = tmp[static_cast<std::size_t>(std::clamp(y + k, 0, h - 1)) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)];
                for (std::size_t d = 0; d < 4; ++d) s[d] += kernel[static_cast<std::size_t>(k + radius)] * c[d];
            }
            out(x, y) = quantize(s);
        }
    }
    return out;
}

PreprocessedImage preprocess(const SourceImage& source, const PreprocessParams& params, TaskContext& ctx) {
    if (source.pixels.width() <= 0 || source.pixels.height() <= 0) throw std::invalid_argument("preprocess: imagem vazia");
    if (!(params.targetAreaPx > 0.0)) throw std::invalid_argument("preprocess: targetAreaPx <= 0");
    if (params.blurSigmaPx < 0.0) throw std::invalid_argument("preprocess: blurSigmaPx < 0");
    ctx.throwIfCancelled();

    PreprocessedImage out;
    sla::Array2D<Rgba8> px = source.pixels;

    // 1. Alfa constante carrega zero informação: torna-se opaco.
    if (params.flattenAlpha) {
        bool constant = true;
        const std::uint8_t a0 = px(0, 0).a;
        for (int y = 0; y < px.height() && constant; ++y) {
            for (int x = 0; x < px.width(); ++x) {
                if (px(x, y).a != a0) {
                    constant = false;
                    break;
                }
            }
        }
        if (constant && a0 != 255) {
            for (int y = 0; y < px.height(); ++y) {
                for (int x = 0; x < px.width(); ++x) px(x, y).a = 255;
            }
            out.alphaFlattened = true;
        }
    }
    ctx.checkpoint(0.25);

    // 2. Reamostragem para a área-alvo.
    const double area = static_cast<double>(px.width()) * static_cast<double>(px.height());
    const double s = std::sqrt(params.targetAreaPx / area);
    if (s < 1.0) {
        const int w = std::max(1, static_cast<int>(std::lround(px.width() * s)));
        const int h = std::max(1, static_cast<int>(std::lround(px.height() * s)));
        if (w != px.width() || h != px.height()) {
            px = resampleArea(px, w, h);
            out.scale = static_cast<double>(w) / static_cast<double>(source.pixels.width());
        }
    } else if (s > 1.0 && params.allowUpscale) {
        const int w = std::max(1, static_cast<int>(std::lround(px.width() * s)));
        const int h = std::max(1, static_cast<int>(std::lround(px.height() * s)));
        px = resampleBilinear(px, w, h);
        out.scale = static_cast<double>(w) / static_cast<double>(source.pixels.width());
    }
    ctx.checkpoint(0.6);

    // 3. Cópia desfocada para a paleta e a segmentação.
    out.image = px;
    out.smooth = params.blurSigmaPx > 0.0 ? gaussianBlur(px, params.blurSigmaPx) : std::move(px);
    ctx.report(1.0);
    return out;
}

}  // namespace cnx
