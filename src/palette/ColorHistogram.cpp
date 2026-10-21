// src/palette/ColorHistogram.cpp — histograma esparso ponderado.
#include "cnx/palette/ColorHistogram.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cnx {

ColorHistogram::ColorHistogram(int bins) : bins_(bins) {
    if (bins < 2 || bins > 256) throw std::invalid_argument("ColorHistogram: bins fora de [2, 256]");
}

std::uint32_t ColorHistogram::keyOf(sla::Rgba8 c) const noexcept {
    const auto bin = [this](std::uint8_t v) { return static_cast<std::uint32_t>(v) * static_cast<std::uint32_t>(bins_) / 256u; };
    const auto b = static_cast<std::uint32_t>(bins_);
    return ((bin(c.r) * b + bin(c.g)) * b + bin(c.b)) * b + bin(c.a);
}

void ColorHistogram::add(sla::Rgba8 color, double weight) {
    if (!(weight >= 0.0) || !std::isfinite(weight)) throw std::invalid_argument("ColorHistogram::add: peso negativo ou não finito");
    Accum& a = accum_[keyOf(color)];
    const std::array<double, 4> v{color.r / 255.0, color.g / 255.0, color.b / 255.0, color.a / 255.0};
    a.weight += weight;
    ++a.pixels;
    for (std::size_t k = 0; k < 4; ++k) a.sum[k] += weight * v[k];
    a.sumSq += weight * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
    ++pixels_;
    totalWeight_ += weight;
    dirty_ = true;
}

std::size_t ColorHistogram::cellCount() const { return accum_.size(); }

const std::vector<HistogramCell>& ColorHistogram::cells() const {
    if (!dirty_) return cells_;
    cells_.clear();
    cells_.reserve(accum_.size());
    for (const auto& [key, a] : accum_) {
        HistogramCell c;
        c.key = key;
        c.weight = a.weight;
        c.pixels = a.pixels;
        if (a.weight > 0.0) {
            double sq = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                c.mean[k] = a.sum[k] / a.weight;
                sq += a.sum[k] * a.sum[k];
            }
            c.scatter = std::max(0.0, a.sumSq - sq / a.weight);
        } else {
            // Todos os pixels da célula têm peso 0: a média simples mantém a posição significativa.
            c.mean = {0.0, 0.0, 0.0, 0.0};
        }
        cells_.push_back(c);
    }
    std::sort(cells_.begin(), cells_.end(), [](const HistogramCell& x, const HistogramCell& y) {
        return x.weight != y.weight ? x.weight > y.weight : x.key < y.key;
    });
    dirty_ = false;
    return cells_;
}

}  // namespace cnx
