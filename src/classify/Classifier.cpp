// src/classify/Classifier.cpp — medidas simples e regra de três degraus (provisória).
#include "cnx/classify/Classifier.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>

#include "cnx/palette/PaletteFinder.h"

namespace cnx {
namespace {

bool isLossy(const std::string& hint) {
    std::string h = hint;
    std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return h == ".jpg" || h == ".jpeg" || h == "jpg" || h == "jpeg";
}

std::uint32_t pack(sla::Rgba8 c) {
    return (static_cast<std::uint32_t>(c.r) << 24) | (static_cast<std::uint32_t>(c.g) << 16) | (static_cast<std::uint32_t>(c.b) << 8) | c.a;
}

}  // namespace

ImageClassification classify(const PreprocessedImage& pre, const SourceImage& source, const ClassifierParams& params,
                             const PaletteParams& palette) {
    ImageClassification out;
    if (!params.autoClassify) {
        out.profile = params.forcedProfile;
        out.automatic = false;
        return out;
    }
    const sla::Array2D<sla::Rgba8>& img = pre.image;
    const std::size_t total = static_cast<std::size_t>(img.width()) * static_cast<std::size_t>(img.height());

    std::unordered_map<std::uint32_t, std::uint64_t> counts;
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) ++counts[pack(img(x, y))];
    }
    std::uint64_t covered = 0;
    int significant = 0;
    for (const auto& [color, n] : counts) {
        if (static_cast<double>(n) >= palette.exactColorMinShare * static_cast<double>(total)) {
            ++significant;
            covered += n;
        }
    }
    const sla::Array2D<float> edge = PaletteFinder::edgeMagnitude(img);
    std::size_t edgePixels = 0;
    for (int y = 0; y < edge.height(); ++y) {
        for (int x = 0; x < edge.width(); ++x) edgePixels += edge(x, y) > 0.05f ? 1 : 0;
    }

    out.measures.significantColors = significant;
    out.measures.flatShare = static_cast<double>(covered) / static_cast<double>(total);
    out.measures.edgeFraction = static_cast<double>(edgePixels) / static_cast<double>(total);
    out.measures.lossyFormat = isLossy(source.formatHint);
    out.measures.hasAlpha = source.hasAlpha;

    const ImageMeasures& m = out.measures;
    if (m.lossyFormat || m.significantColors > palette.maxColors || m.flatShare < 0.80) {
        out.profile = ImageProfile::PhotoMed;
    } else if (m.flatShare >= 0.99) {
        out.profile = ImageProfile::LogoMed;
    } else {
        out.profile = ImageProfile::LogoMedAA;
    }
    return out;
}

}  // namespace cnx
