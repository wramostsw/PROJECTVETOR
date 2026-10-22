// src/classify/Classifier.cpp — as 20 medidas (cores, bordas, AA, erro, segmentos, arquivo) e a árvore.
#include "cnx/classify/Classifier.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cnx/palette/ColorHistogram.h"
#include "cnx/preprocess/EdgeImage.h"
#include "cnx/segment/QuadTree.h"

namespace cnx {
namespace {

constexpr int kHistogramBins = 16;
constexpr int kQuadTreeBlock = 16;
constexpr double kQuadTreeVariance = 1.0e-3;  // o padrão de SegmentParams::quadTreeVariance

bool isLossy(const std::string& hint) {
    std::string h = hint;
    std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return h == ".jpg" || h == ".jpeg" || h == "jpg" || h == "jpeg";
}

std::uint32_t pack(sla::Rgba8 c) {
    return (static_cast<std::uint32_t>(c.r) << 24) | (static_cast<std::uint32_t>(c.g) << 16) | (static_cast<std::uint32_t>(c.b) << 8) | c.a;
}

Color4d toColor(sla::Rgba8 c) { return {c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0}; }

double distance2(const Color4d& a, const Color4d& b) {
    double s = 0.0;
    for (std::size_t k = 0; k < 4; ++k) s += (a[k] - b[k]) * (a[k] - b[k]);
    return s;
}

/// c está sobre o segmento entre dois picos distintos, longe das pontas (uma mistura das duas tintas)?
bool isBetweenPeaks(const Color4d& c, const std::vector<Color4d>& peaks, const MeasureOptions& opt) {
    const double minSep2 = opt.minPeakSeparation * opt.minPeakSeparation;
    const double maxDist2 = opt.betweenMaxDistance * opt.betweenMaxDistance;
    for (std::size_t i = 0; i < peaks.size(); ++i) {
        for (std::size_t j = i + 1; j < peaks.size(); ++j) {
            Color4d seg{};
            Color4d rel{};
            double len2 = 0.0;
            double proj = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                seg[k] = peaks[j][k] - peaks[i][k];
                rel[k] = c[k] - peaks[i][k];
                len2 += seg[k] * seg[k];
                proj += rel[k] * seg[k];
            }
            if (len2 < minSep2) continue;
            const double t = proj / len2;
            if (t < opt.betweenEndMargin || t > 1.0 - opt.betweenEndMargin) continue;
            double d2 = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                const double e = rel[k] - t * seg[k];
                d2 += e * e;
            }
            if (d2 <= maxDist2) return true;
        }
    }
    return false;
}

}  // namespace

ImageMeasures measureImage(const PreprocessedImage& pre, const SourceImage& source, const PaletteParams& palette, TaskContext& ctx,
                           const MeasureOptions& opt) {
    const sla::Array2D<sla::Rgba8>& img = pre.image;
    if (img.width() <= 0 || img.height() <= 0) throw std::invalid_argument("measureImage: imagem vazia");
    const int w = img.width();
    const int h = img.height();
    const double total = static_cast<double>(w) * static_cast<double>(h);
    ImageMeasures m;

    // ---- cores exatas
    std::unordered_map<std::uint32_t, std::uint64_t> counts;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) ++counts[pack(img(x, y))];
    }
    std::uint64_t covered = 0;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> significant;  // (pixels, cor), por pixels decrescente
    for (const auto& [color, n] : counts) {
        if (static_cast<double>(n) >= palette.exactColorMinShare * total) {
            ++m.significantColors;
            covered += n;
            significant.emplace_back(n, color);
        }
    }
    std::sort(significant.begin(), significant.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    m.flatShare = static_cast<double>(covered) / total;
    m.distinctColorRatio = static_cast<double>(counts.size()) / total;
    ctx.checkpoint(0.15);

    // ---- picos do histograma 16⁴ e pixels "entre" picos
    ColorHistogram hist(kHistogramBins);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) hist.add(img(x, y), 1.0);
    }
    // Pico = célula cheia (≥ peakMinShare) a mais de peakRadius de todo pico anterior; células cheias mais
    // perto que isso são a mesma tinta repartida entre faixas vizinhas (ruído leve perto de um limite de faixa).
    const std::vector<HistogramCell>& cells = hist.cells();  // por peso (= pixels) decrescente
    const double radius2 = opt.peakRadius * opt.peakRadius;
    const auto nearestPeak2 = [](const Color4d& c, const std::vector<Color4d>& peaks) {
        double best = std::numeric_limits<double>::infinity();
        for (const Color4d& p : peaks) best = std::min(best, distance2(c, p));
        return best;
    };
    std::vector<Color4d> allPeaks;
    for (const HistogramCell& cell : cells) {
        if (static_cast<double>(cell.pixels) < opt.peakMinShare * total) break;
        if (allPeaks.size() >= static_cast<std::size_t>(std::max(1, opt.maxPeaks))) break;
        if (nearestPeak2(cell.mean, allPeaks) > radius2) allPeaks.push_back(cell.mean);
    }
    if (allPeaks.empty()) allPeaks.push_back(cells.front().mean);  // nenhuma célula chega a ser pico: a mais cheia
    m.mainPeaks = static_cast<int>(allPeaks.size());
    // As cores da "paleta de picos": os mais pesados (a ordem de allPeaks é a de peso).
    const std::size_t paletteCap = static_cast<std::size_t>(std::max(1, opt.maxPalettePeaks));
    const std::vector<Color4d> peaks(allPeaks.begin(), allPeaks.begin() + static_cast<std::ptrdiff_t>(std::min(allPeaks.size(), paletteCap)));
    // Tintas = picos + cores exatas significativas longe de todo pico (uma tinta de área pequena, < peakMinShare,
    // continua sendo tinta numa imagem sem perdas; senão o erro dela pareceria "espalhado").
    std::vector<Color4d> inks = peaks;
    std::vector<Color4d> covering = allPeaks;
    for (const auto& [n, packed] : significant) {
        if (inks.size() >= 2 * paletteCap) break;
        const Color4d c = toColor(sla::Rgba8{static_cast<std::uint8_t>(packed >> 24), static_cast<std::uint8_t>(packed >> 16),
                                             static_cast<std::uint8_t>(packed >> 8), static_cast<std::uint8_t>(packed)});
        if (nearestPeak2(c, covering) > radius2) {
            inks.push_back(c);
            covering.push_back(c);
        }
    }
    std::uint64_t peakPixels = 0;
    std::uint64_t between = 0;
    for (const HistogramCell& cell : cells) {
        if (nearestPeak2(cell.mean, covering) <= radius2) {
            peakPixels += cell.pixels;
        } else if (isBetweenPeaks(cell.mean, peaks, opt)) {
            between += cell.pixels;
        }
    }
    m.peakShare = static_cast<double>(peakPixels) / total;
    m.betweenShare = static_cast<double>(between) / total;
    ctx.checkpoint(0.35);

    // ---- bordas e erro da paleta de picos
    const EdgeImage edges = EdgeImage::compute(img);
    ctx.checkpoint(0.55);
    double errTotal = 0.0;
    double errEdge = 0.0;
    double magSum = 0.0;
    double weakSq = 0.0;
    std::size_t edgePixels = 0;
    std::size_t weakPixels = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const Color4d c = toColor(img(x, y));
            double best = distance2(c, inks[0]);
            for (std::size_t k = 1; k < inks.size(); ++k) best = std::min(best, distance2(c, inks[k]));
            const float mag = edges.magnitude(x, y);
            errTotal += best;
            magSum += mag;
            if (mag > opt.edgeMagnitude) {
                ++edgePixels;
                errEdge += best;
            }
            if (mag < opt.aa.minEdgeMagnitude) {
                ++weakPixels;
                weakSq += static_cast<double>(mag) * mag;
            }
        }
    }
    m.quantizationRmse = std::sqrt(errTotal / total);
    m.edgeErrorShare = errTotal > 1e-12 ? errEdge / errTotal : 0.0;
    m.edgeFraction = static_cast<double>(edgePixels) / total;
    m.meanEdgeMagnitude = magSum / total;
    m.interiorNoise = weakPixels == 0 ? 0.0 : std::sqrt(weakSq / static_cast<double>(weakPixels));
    ctx.checkpoint(0.7);

    // ---- antialiasing: perfis através das cristas
    const AAStats aa = detectAntialiasing(img, edges, opt.aa);
    m.strongEdgeFraction = static_cast<double>(aa.ridges) / total;
    m.edgeProfiles = aa.valid();
    m.stepShare = aa.valid() == 0 ? 0.0 : static_cast<double>(aa.steps) / static_cast<double>(aa.valid());
    m.rampShare = aa.rampShare();
    m.otherShare = aa.ridges == 0 ? 0.0 : static_cast<double>(aa.others) / static_cast<double>(aa.ridges);
    m.meanTransitionWidth = aa.meanTransitionWidth();
    ctx.checkpoint(0.85);

    // ---- segmentos preliminares
    const std::size_t leaves = quadTreeSplit(img, kQuadTreeBlock, kQuadTreeVariance).size();
    const double blocks = std::ceil(w / static_cast<double>(kQuadTreeBlock)) * std::ceil(h / static_cast<double>(kQuadTreeBlock));
    m.quadTreeLeavesPerBlock = static_cast<double>(leaves) / blocks;

    // ---- arquivo
    m.lossyFormat = isLossy(source.formatHint);
    m.hasAlpha = source.hasAlpha;
    ctx.report(1.0);
    return m;
}

ImageProfile decideProfile(const ImageMeasures& m, int maxColors, const ClassificationRules& rules) {
    if (m.mainPeaks > maxColors) return ImageProfile::PhotoMed;
    const double coverage = m.peakShare + m.betweenShare;
    if (coverage < (m.lossyFormat ? rules.minLogoCoverageLossy : rules.minLogoCoverage)) return ImageProfile::PhotoMed;
    if (m.quantizationRmse > rules.maxFlatRmse) {
        const double lift = m.edgeFraction > 0.0 ? m.edgeErrorShare / m.edgeFraction : 0.0;
        if (lift < rules.minEdgeErrorLift) return ImageProfile::PhotoMed;
    }
    const bool antialiased = m.betweenShare >= rules.minBetweenShare || (m.edgeProfiles >= rules.minAAProfiles && m.rampShare >= rules.minRampShare);
    return antialiased ? ImageProfile::LogoMedAA : ImageProfile::LogoMed;
}

ImageClassification classify(const PreprocessedImage& pre, const SourceImage& source, const ClassifierParams& params,
                             const PaletteParams& palette, TaskContext& ctx) {
    ImageClassification out;
    if (!params.autoClassify) {
        out.profile = params.forcedProfile;
        out.automatic = false;
        return out;
    }
    out.measures = measureImage(pre, source, palette, ctx);
    out.profile = decideProfile(out.measures, palette.maxColors);
    return out;
}

ImageClassification classify(const PreprocessedImage& pre, const SourceImage& source, const ClassifierParams& params,
                             const PaletteParams& palette) {
    TaskContext ctx;
    return classify(pre, source, params, palette, ctx);
}

}  // namespace cnx
