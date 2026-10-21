// src/palette/PaletteFinder.cpp — histograma ponderado, k-means K = 2..12, nota e refinamento.
#include "cnx/palette/PaletteFinder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "cnx/palette/ColorHistogram.h"
#include "cnx/palette/KMeans.h"

namespace cnx {
namespace {

constexpr std::size_t kSeedMix = 0x9E3779B97F4A7C15ull;

Color4d toColor(sla::Rgba8 c) { return {c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0}; }

sla::Rgba toRgba(const Color4d& c) {
    return {static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2]), static_cast<float>(c[3])};
}

std::uint32_t pack(sla::Rgba8 c) {
    return (static_cast<std::uint32_t>(c.r) << 24) | (static_cast<std::uint32_t>(c.g) << 16) | (static_cast<std::uint32_t>(c.b) << 8) | c.a;
}

sla::Rgba8 unpack(std::uint32_t v) {
    return {static_cast<std::uint8_t>(v >> 24), static_cast<std::uint8_t>(v >> 16), static_cast<std::uint8_t>(v >> 8), static_cast<std::uint8_t>(v)};
}

}  // namespace

PaletteFinder::PaletteFinder(PaletteParams params, std::uint64_t seed) : params_(params), seed_(seed) {
    if (params_.minColors < 1) throw std::invalid_argument("PaletteFinder: minColors < 1");
    if (params_.maxColors < params_.minColors) throw std::invalid_argument("PaletteFinder: maxColors < minColors");
    if (params_.histogramBins < 2 || params_.histogramBins > 256) throw std::invalid_argument("PaletteFinder: histogramBins fora de [2, 256]");
    if (params_.colorPenalty < 0.0) throw std::invalid_argument("PaletteFinder: colorPenalty < 0");
    if (!(params_.edgeScale > 0.0)) throw std::invalid_argument("PaletteFinder: edgeScale <= 0");
    if (params_.forcedNumColors && *params_.forcedNumColors < 1) throw std::invalid_argument("PaletteFinder: forcedNumColors < 1");
}

double PaletteFinder::score(double error, int colors, double penalty) {
    return std::log(error + 1.0) + static_cast<double>(colors - 2) * penalty;
}

sla::Array2D<float> PaletteFinder::edgeMagnitude(const sla::Array2D<sla::Rgba8>& image) {
    const int w = image.width();
    const int h = image.height();
    sla::Array2D<float> out(w, h);
    const auto px = [&](int x, int y) { return toColor(image(std::clamp(x, 0, w - 1), std::clamp(y, 0, h - 1))); };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;
            // Sobel: gx = [-1 0 1; -2 0 2; -1 0 1], gy = gxᵀ, por canal.
            const std::array<Color4d, 9> n{px(x - 1, y - 1), px(x, y - 1), px(x + 1, y - 1), px(x - 1, y), px(x, y),
                                           px(x + 1, y),     px(x - 1, y + 1), px(x, y + 1), px(x + 1, y + 1)};
            for (std::size_t c = 0; c < 4; ++c) {
                const double gx = -n[0][c] + n[2][c] - 2.0 * n[3][c] + 2.0 * n[5][c] - n[6][c] + n[8][c];
                const double gy = -n[0][c] - 2.0 * n[1][c] - n[2][c] + n[6][c] + 2.0 * n[7][c] + n[8][c];
                sum += gx * gx + gy * gy;
            }
            out(x, y) = static_cast<float>(std::sqrt(sum) / 4.0);
        }
    }
    return out;
}

sla::Array2D<float> PaletteFinder::pixelWeights(const sla::Array2D<sla::Rgba8>& image) const {
    sla::Array2D<float> edge = edgeMagnitude(image);
    for (int y = 0; y < edge.height(); ++y) {
        for (int x = 0; x < edge.width(); ++x) edge(x, y) = static_cast<float>(1.0 / (1.0 + static_cast<double>(edge(x, y)) / params_.edgeScale));
    }
    return edge;
}

PaletteResult PaletteFinder::find(const sla::Array2D<sla::Rgba8>& image) const { return find(image, pixelWeights(image)); }

PaletteResult PaletteFinder::find(const sla::Array2D<sla::Rgba8>& image, const sla::Array2D<float>& weights) const {
    if (image.width() <= 0 || image.height() <= 0) throw std::invalid_argument("PaletteFinder::find: imagem vazia");
    if (weights.width() != image.width() || weights.height() != image.height()) {
        throw std::invalid_argument("PaletteFinder::find: pesos com dimensões diferentes da imagem");
    }
    const std::size_t total = static_cast<std::size_t>(image.width()) * static_cast<std::size_t>(image.height());

    // ---- cores exatas: imagem chapada? ----
    std::unordered_map<std::uint32_t, std::uint64_t> exact;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) ++exact[pack(image(x, y))];
    }
    PaletteResult result;
    result.distinctColors = exact.size();
    std::vector<std::pair<std::uint64_t, std::uint32_t>> significant;  // (contagem, cor)
    std::uint64_t covered = 0;
    for (const auto& [color, count] : exact) {
        if (static_cast<double>(count) >= params_.exactColorMinShare * static_cast<double>(total)) {
            significant.emplace_back(count, color);
            covered += count;
        }
    }
    if (!params_.forcedNumColors && significant.size() <= static_cast<std::size_t>(params_.maxColors) &&
        static_cast<double>(covered) >= 0.99 * static_cast<double>(total)) {
        std::sort(significant.begin(), significant.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        for (const auto& [count, color] : significant) {
            result.palette.colors.push_back(sla::toRgba(unpack(color)));
            result.fractions.push_back(static_cast<double>(count) / static_cast<double>(covered));
        }
        result.exact = true;
        result.palette.exact = true;
        result.chosenK = static_cast<int>(significant.size());
        return result;
    }

    // ---- histograma ponderado e coreset ----
    ColorHistogram hist(params_.histogramBins);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) hist.add(image(x, y), static_cast<double>(weights(x, y)));
    }
    if (!(hist.totalWeight() > 0.0)) throw std::invalid_argument("PaletteFinder::find: pesos todos nulos");
    std::vector<KMeansItem> items;
    for (const HistogramCell& c : hist.cells()) {
        if (c.weight > 0.0) items.push_back({c.mean, c.weight, c.scatter});
    }

    // ---- K = minColors..maxColors ----
    const int kMax = std::min<int>(params_.maxColors, static_cast<int>(items.size()));
    int kMin = std::min(params_.minColors, kMax);
    int kLast = kMax;
    if (params_.forcedNumColors) {
        kMin = std::min(*params_.forcedNumColors, kMax);
        kLast = kMin;
    }
    const double separation = 2.0 / static_cast<double>(params_.histogramBins);
    KMeansResult best;
    double bestScore = std::numeric_limits<double>::infinity();
    int bestK = 0;
    for (int k = kMin; k <= kLast; ++k) {
        sla::myPrng rng(seed_ ^ (static_cast<std::uint64_t>(k) * kSeedMix));
        KMeansResult r = kmeans(items, chooseSeeds(items, static_cast<std::size_t>(k), separation, rng), params_.kmeansMaxRounds);
        PaletteCandidate cand;
        cand.requested = k;
        cand.colors = static_cast<int>(r.centers.size());
        cand.error = r.error;
        cand.score = score(r.error, cand.colors, params_.colorPenalty);
        result.candidates.push_back(cand);
        // Menor nota; empate → menos cores (a busca crescente já cuida do resto).
        if (cand.score < bestScore - 1e-12 || (std::abs(cand.score - bestScore) <= 1e-12 && cand.colors < static_cast<int>(best.centers.size()))) {
            bestScore = cand.score;
            best = std::move(r);
            bestK = k;
        }
    }
    result.chosenK = bestK;
    result.palette.score = bestScore;

    // ---- refinamento sobre os pixels, só para o K escolhido ----
    std::vector<Color4d> centers = best.centers;
    const std::size_t n = centers.size();
    std::vector<int> assign(total, 0);
    std::vector<double> pw(n, 0.0);
    for (int round = 0; round < std::max(0, params_.refineRounds) + 1; ++round) {
        std::vector<Color4d> sum(n, Color4d{});
        std::fill(pw.begin(), pw.end(), 0.0);
        std::size_t i = 0;
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x, ++i) {
                const Color4d c = toColor(image(x, y));
                std::size_t arg = 0;
                double bd = distance2(c, centers[0]);
                for (std::size_t j = 1; j < n; ++j) {
                    const double d = distance2(c, centers[j]);
                    if (d < bd) {
                        bd = d;
                        arg = j;
                    }
                }
                assign[i] = static_cast<int>(arg);
                const double w = weights(x, y);
                pw[arg] += w;
                for (std::size_t d = 0; d < 4; ++d) sum[arg][d] += w * c[d];
            }
        }
        if (round == std::max(0, params_.refineRounds)) break;  // última passada só mede
        for (std::size_t j = 0; j < n; ++j) {
            if (pw[j] > 0.0) {
                for (std::size_t d = 0; d < 4; ++d) centers[j][d] = sum[j][d] / pw[j];
            }
        }
    }
    // Erro final nos pixels, descarte de centros vazios e ordenação por peso.
    double err = 0.0;
    double totalW = 0.0;
    {
        std::size_t i = 0;
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x, ++i) {
                const double w = weights(x, y);
                err += w * distance2(toColor(image(x, y)), centers[static_cast<std::size_t>(assign[i])]);
                totalW += w;
            }
        }
    }
    std::vector<std::size_t> order;
    for (std::size_t j = 0; j < n; ++j) {
        if (pw[j] > 0.0) order.push_back(j);
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return pw[a] > pw[b]; });
    double sumW = 0.0;
    for (const std::size_t j : order) sumW += pw[j];
    for (const std::size_t j : order) {
        result.palette.colors.push_back(toRgba(centers[j]));
        result.fractions.push_back(pw[j] / sumW);
    }
    result.error = err / totalW;
    return result;
}

}  // namespace cnx
