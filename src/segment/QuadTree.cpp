// src/segment/QuadTree.cpp — pré-corte em blocos por variância.
#include "cnx/segment/QuadTree.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace cnx {
namespace {

void split(const sla::Array2D<sla::Rgba8>& image, const QuadBlock& b, double variance, std::vector<QuadBlock>& out) {
    if ((b.w == 1 && b.h == 1) || blockColorVariance(image, b) <= variance) {
        out.push_back(b);
        return;
    }
    const int w0 = b.w > 1 ? b.w / 2 : b.w;
    const int h0 = b.h > 1 ? b.h / 2 : b.h;
    const int w1 = b.w > 1 ? b.w - w0 : 0;
    const int h1 = b.h > 1 ? b.h - h0 : 0;
    split(image, {b.x, b.y, w0, h0}, variance, out);
    if (w1 > 0) split(image, {b.x + w0, b.y, w1, h0}, variance, out);
    if (h1 > 0) split(image, {b.x, b.y + h0, w0, h1}, variance, out);
    if (w1 > 0 && h1 > 0) split(image, {b.x + w0, b.y + h0, w1, h1}, variance, out);
}

}  // namespace

double blockColorVariance(const sla::Array2D<sla::Rgba8>& image, const QuadBlock& b) {
    // Somas inteiras exatas (≤ 255² · 4 · w · h); a variância sai de E[v²] − E[v]² sem cancelamento
    // relevante porque os acumuladores são inteiros.
    std::array<std::uint64_t, 4> sum{};
    std::array<std::uint64_t, 4> sq{};
    for (int y = b.y; y < b.y + b.h; ++y) {
        for (int x = b.x; x < b.x + b.w; ++x) {
            const sla::Rgba8 c = image(x, y);
            const std::array<std::uint64_t, 4> v{c.r, c.g, c.b, c.a};
            for (std::size_t ch = 0; ch < 4; ++ch) {
                sum[ch] += v[ch];
                sq[ch] += v[ch] * v[ch];
            }
        }
    }
    const double n = static_cast<double>(b.w) * static_cast<double>(b.h);
    double total = 0.0;
    for (std::size_t ch = 0; ch < 4; ++ch) {
        // n·Σv² − (Σv)² é inteiro e ≥ 0: variância = (n·Σv² − (Σv)²) / n².
        const double num = n * static_cast<double>(sq[ch]) - static_cast<double>(sum[ch]) * static_cast<double>(sum[ch]);
        total += num / (n * n);
    }
    return total / (255.0 * 255.0);
}

std::vector<QuadBlock> quadTreeSplit(const sla::Array2D<sla::Rgba8>& image, int maxBlock, double variance) {
    if (maxBlock < 1) throw std::invalid_argument("quadTreeSplit: maxBlock < 1");
    if (image.width() <= 0 || image.height() <= 0) throw std::invalid_argument("quadTreeSplit: imagem vazia");
    std::vector<QuadBlock> leaves;
    leaves.reserve(static_cast<std::size_t>(image.width() / maxBlock + 1) * static_cast<std::size_t>(image.height() / maxBlock + 1));
    for (int y = 0; y < image.height(); y += maxBlock) {
        for (int x = 0; x < image.width(); x += maxBlock) {
            const int w = image.width() - x < maxBlock ? image.width() - x : maxBlock;
            const int h = image.height() - y < maxBlock ? image.height() - y : maxBlock;
            split(image, {x, y, w, h}, variance, leaves);
        }
    }
    return leaves;
}

}  // namespace cnx
