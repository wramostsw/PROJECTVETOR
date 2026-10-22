// src/preprocess/Preprocessor.cpp — alfa, reamostragem e desfoque, todos sobre o sinal pré-multiplicado.
#include "cnx/preprocess/Preprocessor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cnx {

sla::Array2D<sla::Rgba8> resampleArea(const sla::Array2D<sla::Rgba8>& in, int w, int h) {
    return SignalImage::fromStraight(in).resampledArea(w, h).toStraight8();
}

sla::Array2D<sla::Rgba8> gaussianBlur(const sla::Array2D<sla::Rgba8>& in, double sigma) {
    if (!(sigma > 0.0)) return in;
    return SignalImage::fromStraight(in).blurred(sigma).toStraight8();
}

PreprocessedImage preprocess(const SourceImage& source, const PreprocessParams& params, TaskContext& ctx) {
    if (source.pixels.width() <= 0 || source.pixels.height() <= 0) throw std::invalid_argument("preprocess: imagem vazia");
    if (!(params.targetAreaPx > 0.0)) throw std::invalid_argument("preprocess: targetAreaPx <= 0");
    if (!(params.blurSigmaPx >= 0.0) || !std::isfinite(params.blurSigmaPx)) throw std::invalid_argument("preprocess: blurSigmaPx < 0 ou não finito");
    ctx.throwIfCancelled();

    PreprocessedImage out;
    sla::Array2D<sla::Rgba8> px = source.pixels;

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
    SignalImage signal = SignalImage::fromStraight(px);
    ctx.checkpoint(0.25);

    // 2. Reamostragem para a área-alvo, no sinal pré-multiplicado (sem quantizar).
    const double area = static_cast<double>(px.width()) * static_cast<double>(px.height());
    const double s = std::sqrt(params.targetAreaPx / area);
    if (s < 1.0) {
        const int w = std::max(1, static_cast<int>(std::lround(px.width() * s)));
        const int h = std::max(1, static_cast<int>(std::lround(px.height() * s)));
        if (w != px.width() || h != px.height()) {
            signal = signal.resampledArea(w, h);
            out.scale = static_cast<double>(w) / static_cast<double>(source.pixels.width());
        }
    } else if (s > 1.0 && params.allowUpscale) {
        const int w = std::max(1, static_cast<int>(std::lround(px.width() * s)));
        const int h = std::max(1, static_cast<int>(std::lround(px.height() * s)));
        signal = signal.resampledBilinear(w, h);
        out.scale = static_cast<double>(w) / static_cast<double>(source.pixels.width());
    }
    ctx.checkpoint(0.6);

    // 3. As imagens de 8 bits saem do sinal; a desfocada (paleta e segmentação) também é filtrada nele.
    out.image = signal.toStraight8();
    out.smooth = params.blurSigmaPx > 0.0 ? signal.blurred(params.blurSigmaPx).toStraight8() : out.image;
    out.signal = std::move(signal);
    ctx.report(1.0);
    return out;
}

}  // namespace cnx
