// src/io/PngDecoder.cpp — chunks, inflate, desfiltragem, Adam7 e conversão para RGBA de 8 bits.
#include "cnx/io/PngDecoder.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "cnx/io/Inflate.h"

namespace cnx {
namespace {

constexpr std::array<std::uint8_t, 8> kSignature{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};

// Adam7: origem e passo de cada uma das 7 passadas.
constexpr std::array<int, 7> kStartX{0, 4, 0, 2, 0, 1, 0};
constexpr std::array<int, 7> kStartY{0, 0, 4, 0, 2, 0, 1};
constexpr std::array<int, 7> kStepX{8, 8, 4, 4, 2, 2, 1};
constexpr std::array<int, 7> kStepY{8, 8, 8, 4, 4, 2, 2};

std::uint32_t be32(const std::uint8_t* p) noexcept {
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) | (static_cast<std::uint32_t>(p[2]) << 8) | p[3];
}
unsigned be16(const std::uint8_t* p) noexcept { return (static_cast<unsigned>(p[0]) << 8) | p[1]; }

struct Header {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    int depth = 0;
    int color = 0;
    bool interlaced = false;
    int channels() const noexcept { return color == 0 || color == 3 ? 1 : color == 4 ? 2 : color == 2 ? 3 : 4; }
    int bitsPerPixel() const noexcept { return depth * channels(); }
    std::uint64_t rowBytes(std::uint64_t w) const noexcept { return (w * static_cast<std::uint64_t>(bitsPerPixel()) + 7) / 8; }
};

Header parseHeader(const std::uint8_t* d, const DecodeOptions& options) {
    Header h;
    h.width = be32(d);
    h.height = be32(d + 4);
    h.depth = d[8];
    h.color = d[9];
    if (h.width == 0 || h.height == 0 || h.width > 0x7FFFFFFFu || h.height > 0x7FFFFFFFu) throw ImageDecodeError("PNG: dimensões inválidas no IHDR");
    detail::checkDimensions(h.width, h.height, options, "PNG");
    const bool ok = (h.color == 0 && (h.depth == 1 || h.depth == 2 || h.depth == 4 || h.depth == 8 || h.depth == 16)) ||
                    (h.color == 3 && (h.depth == 1 || h.depth == 2 || h.depth == 4 || h.depth == 8)) ||
                    ((h.color == 2 || h.color == 4 || h.color == 6) && (h.depth == 8 || h.depth == 16));
    if (!ok) throw ImageDecodeError("PNG: combinação de tipo de cor " + std::to_string(h.color) + " e profundidade " + std::to_string(h.depth) + " inválida");
    if (d[10] != 0) throw ImageDecodeError("PNG: método de compressão desconhecido");
    if (d[11] != 0) throw ImageDecodeError("PNG: método de filtragem desconhecido");
    if (d[12] > 1) throw ImageDecodeError("PNG: método de entrelaçamento desconhecido");
    h.interlaced = d[12] == 1;
    return h;
}

/// Tamanho de uma passada (ou da imagem inteira) em pixels.
std::uint64_t passSize(std::uint64_t full, int start, int step) noexcept {
    return full > static_cast<std::uint64_t>(start) ? (full - static_cast<std::uint64_t>(start) + static_cast<std::uint64_t>(step) - 1) / static_cast<std::uint64_t>(step) : 0;
}

int paeth(int a, int b, int c) noexcept {
    const int p = a + b - c;
    const int pa = std::abs(p - a);
    const int pb = std::abs(p - b);
    const int pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    return pb <= pc ? b : c;
}

/// Desfaz o filtro de uma linha (in place); `prior` é a linha anterior já desfiltrada (zeros na primeira).
void unfilter(int type, std::uint8_t* row, const std::uint8_t* prior, std::size_t n, std::size_t bpp) {
    switch (type) {
        case 0: return;
        case 1:
            for (std::size_t i = bpp; i < n; ++i) row[i] = static_cast<std::uint8_t>(row[i] + row[i - bpp]);
            return;
        case 2:
            for (std::size_t i = 0; i < n; ++i) row[i] = static_cast<std::uint8_t>(row[i] + prior[i]);
            return;
        case 3:
            for (std::size_t i = 0; i < n; ++i) {
                const int left = i >= bpp ? row[i - bpp] : 0;
                row[i] = static_cast<std::uint8_t>(row[i] + ((left + prior[i]) >> 1));
            }
            return;
        case 4:
            for (std::size_t i = 0; i < n; ++i) {
                const int left = i >= bpp ? row[i - bpp] : 0;
                const int upLeft = i >= bpp ? prior[i - bpp] : 0;
                row[i] = static_cast<std::uint8_t>(row[i] + paeth(left, prior[i], upLeft));
            }
            return;
        default: throw ImageDecodeError("PNG: tipo de filtro inválido (" + std::to_string(type) + ")");
    }
}

/// Amostra i de uma linha desfiltrada, na profundidade original.
unsigned sampleAt(const std::uint8_t* row, std::size_t i, int depth) noexcept {
    switch (depth) {
        case 16: return be16(row + 2 * i);
        case 8: return row[i];
        default: {
            const std::size_t bit = i * static_cast<std::size_t>(depth);
            return (row[bit / 8] >> (8 - depth - static_cast<int>(bit % 8))) & ((1u << depth) - 1u);
        }
    }
}

std::uint8_t to8(unsigned v, int depth) noexcept {
    if (depth == 16) return static_cast<std::uint8_t>((v * 255u + 32767u) / 65535u);
    if (depth == 8) return static_cast<std::uint8_t>(v);
    return static_cast<std::uint8_t>(v * 255u / ((1u << depth) - 1u));
}

}  // namespace

SourceImage decodePng(std::span<const std::uint8_t> data, const DecodeOptions& options) {
    if (data.size() < kSignature.size() || !std::equal(kSignature.begin(), kSignature.end(), data.begin())) throw ImageDecodeError("PNG: assinatura inválida");
    Header h;
    bool haveHeader = false;
    bool idatDone = false;
    bool ended = false;
    std::vector<std::uint8_t> idat;
    std::vector<sla::Rgba8> palette;
    std::vector<std::uint8_t> paletteAlpha;
    bool trns = false;
    std::array<unsigned, 3> key{};
    SourceImage out;

    std::size_t pos = kSignature.size();
    while (!ended) {
        if (data.size() - pos < 12) throw ImageDecodeError("PNG: arquivo truncado (sem IEND)");
        const std::uint32_t len = be32(&data[pos]);
        if (len > 0x7FFFFFFFu || len > data.size() - pos - 12) throw ImageDecodeError("PNG: chunk truncado ou com tamanho inválido");
        const std::uint8_t* type = &data[pos + 4];
        const std::uint8_t* body = type + 4;
        const std::string name(reinterpret_cast<const char*>(type), 4);
        const bool critical = (type[0] & 0x20u) == 0;
        const bool crcOk = crc32(data.subspan(pos + 4, 4 + std::size_t{len})) == be32(body + len);
        pos += 12 + std::size_t{len};
        if (!crcOk) {
            if (critical) throw ImageDecodeError("PNG: CRC errado no chunk " + name);
            continue;  // ancilar corrompido: ignorado
        }
        if (!haveHeader && name != "IHDR") throw ImageDecodeError("PNG: o primeiro chunk não é IHDR");
        if (!idat.empty() && name != "IDAT") idatDone = true;

        if (name == "IHDR") {
            if (haveHeader) throw ImageDecodeError("PNG: IHDR repetido");
            if (len != 13) throw ImageDecodeError("PNG: IHDR com tamanho inválido");
            h = parseHeader(body, options);
            haveHeader = true;
        } else if (name == "PLTE") {
            if (h.color == 0 || h.color == 4) throw ImageDecodeError("PNG: PLTE numa imagem em tons de cinza");
            if (len % 3 != 0 || len == 0 || len / 3 > 256) throw ImageDecodeError("PNG: PLTE com tamanho inválido");
            if (!idat.empty()) throw ImageDecodeError("PNG: PLTE depois de IDAT");
            palette.clear();
            for (std::uint32_t i = 0; i < len; i += 3) palette.push_back({body[i], body[i + 1], body[i + 2], 255});
        } else if (name == "tRNS") {
            if (h.color == 0 && len == 2) {
                key[0] = be16(body) & ((1u << h.depth) - 1u);
                trns = true;
            } else if (h.color == 2 && len == 6) {
                const unsigned mask = h.depth == 16 ? 0xFFFFu : 0xFFu;
                key = {be16(body) & mask, be16(body + 2) & mask, be16(body + 4) & mask};
                trns = true;
            } else if (h.color == 3) {
                if (palette.empty() || len > palette.size()) throw ImageDecodeError("PNG: tRNS sem PLTE ou maior que a paleta");
                paletteAlpha.assign(body, body + len);
                trns = true;
            }  // tipos com canal alfa: tRNS não se aplica e é ignorado
        } else if (name == "IDAT") {
            if (idatDone) throw ImageDecodeError("PNG: chunks IDAT não consecutivos");
            idat.insert(idat.end(), body, body + len);
        } else if (name == "pHYs") {
            if (len == 9 && body[8] == 1) {  // pixels por metro
                out.dpiX = be32(body) * 0.0254;
                out.dpiY = be32(body + 4) * 0.0254;
            }
        } else if (name == "IEND") {
            ended = true;
        } else if (critical) {
            throw ImageDecodeError("PNG: chunk crítico desconhecido " + name);
        }
    }
    if (idat.empty()) throw ImageDecodeError("PNG: sem IDAT");
    if (h.color == 3 && palette.empty()) throw ImageDecodeError("PNG: imagem de paleta sem PLTE");

    // Tamanho exato dos dados filtrados: por passada, altura × (1 byte de filtro + bytes da linha).
    const int passes = h.interlaced ? 7 : 1;
    std::uint64_t expected = 0;
    for (int p = 0; p < passes; ++p) {
        const std::uint64_t pw = h.interlaced ? passSize(h.width, kStartX[static_cast<std::size_t>(p)], kStepX[static_cast<std::size_t>(p)]) : h.width;
        const std::uint64_t ph = h.interlaced ? passSize(h.height, kStartY[static_cast<std::size_t>(p)], kStepY[static_cast<std::size_t>(p)]) : h.height;
        if (pw != 0 && ph != 0) expected += ph * (1 + h.rowBytes(pw));
    }
    if (expected > std::numeric_limits<std::size_t>::max()) throw ImageDecodeError("PNG: imagem grande demais");
    std::vector<std::uint8_t> raw = inflateZlib(idat, static_cast<std::size_t>(expected), options.stop);

    out.pixels = sla::Array2D<sla::Rgba8>(static_cast<int>(h.width), static_cast<int>(h.height));
    out.formatHint = ".png";
    out.hasAlpha = h.color == 4 || h.color == 6 || trns;
    const std::size_t bpp = static_cast<std::size_t>(std::max(1, h.bitsPerPixel() / 8));
    const int ch = h.channels();
    std::size_t offset = 0;
    for (int p = 0; p < passes; ++p) {
        const int x0 = h.interlaced ? kStartX[static_cast<std::size_t>(p)] : 0;
        const int y0 = h.interlaced ? kStartY[static_cast<std::size_t>(p)] : 0;
        const int dx = h.interlaced ? kStepX[static_cast<std::size_t>(p)] : 1;
        const int dy = h.interlaced ? kStepY[static_cast<std::size_t>(p)] : 1;
        const auto pw = static_cast<std::size_t>(h.interlaced ? passSize(h.width, x0, dx) : h.width);
        const auto ph = static_cast<std::size_t>(h.interlaced ? passSize(h.height, y0, dy) : h.height);
        if (pw == 0 || ph == 0) continue;
        const auto n = static_cast<std::size_t>(h.rowBytes(pw));
        std::vector<std::uint8_t> zero(n, 0);
        const std::uint8_t* prior = zero.data();
        for (std::size_t r = 0; r < ph; ++r) {
            detail::checkStop(options.stop);
            std::uint8_t* row = raw.data() + offset + 1;
            unfilter(raw[offset], row, prior, n, bpp);
            prior = row;
            offset += 1 + n;
            const int y = y0 + static_cast<int>(r) * dy;
            for (std::size_t c = 0; c < pw; ++c) {
                const int x = x0 + static_cast<int>(c) * dx;
                std::array<unsigned, 4> s{};
                for (int k = 0; k < ch; ++k) s[static_cast<std::size_t>(k)] = sampleAt(row, c * static_cast<std::size_t>(ch) + static_cast<std::size_t>(k), h.depth);
                sla::Rgba8& px = out.pixels(x, y);
                switch (h.color) {
                    case 0: {
                        const std::uint8_t g = to8(s[0], h.depth);
                        px = {g, g, g, static_cast<std::uint8_t>(trns && s[0] == key[0] ? 0 : 255)};
                        break;
                    }
                    case 2:
                        px = {to8(s[0], h.depth), to8(s[1], h.depth), to8(s[2], h.depth),
                              static_cast<std::uint8_t>(trns && s[0] == key[0] && s[1] == key[1] && s[2] == key[2] ? 0 : 255)};
                        break;
                    case 3:
                        if (s[0] >= palette.size()) throw ImageDecodeError("PNG: índice fora da paleta");
                        px = palette[s[0]];
                        if (s[0] < paletteAlpha.size()) px.a = paletteAlpha[s[0]];
                        break;
                    case 4: {
                        const std::uint8_t g = to8(s[0], h.depth);
                        px = {g, g, g, to8(s[1], h.depth)};
                        break;
                    }
                    default: px = {to8(s[0], h.depth), to8(s[1], h.depth), to8(s[2], h.depth), to8(s[3], h.depth)}; break;
                }
            }
        }
    }
    return out;
}

}  // namespace cnx
