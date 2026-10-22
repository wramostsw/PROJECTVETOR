// src/io/JpegDecoder.cpp — marcadores, Huffman, scans sequenciais e progressivos, IDCT, croma e cor.
#include "cnx/io/JpegDecoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

namespace cnx {
namespace {

/// Posição natural (linha·8 + coluna) do k-ésimo coeficiente em zigue-zague.
constexpr std::array<std::uint8_t, 64> kZigzag{0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                                               41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                                               30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
constexpr int kFastBits = 9;

unsigned be16(const std::uint8_t* p) noexcept { return (static_cast<unsigned>(p[0]) << 8) | p[1]; }

// ---------------------------------------------------------------- Huffman

struct HuffTable {
    bool present = false;
    std::array<std::uint8_t, 256> values{};
    std::array<std::int32_t, 18> maxCode{};  ///< maior código de cada comprimento (−1 se nenhum); [17] = sentinela
    std::array<std::int32_t, 17> valPtr{};   ///< índice em values do primeiro código de cada comprimento
    std::array<std::int32_t, 17> minCode{};
    std::array<std::uint16_t, 1u << kFastBits> fast{};  ///< comprimento << 8 | valor; 0 = código mais longo
};

void buildHuffman(HuffTable& t, const std::uint8_t* counts, const std::uint8_t* values, int total) {
    t = HuffTable{};
    t.present = true;
    std::copy(values, values + total, t.values.begin());
    std::int32_t code = 0;
    int k = 0;
    for (int len = 1; len <= 16; ++len) {
        const int n = counts[len - 1];
        t.valPtr[static_cast<std::size_t>(len)] = k;
        t.minCode[static_cast<std::size_t>(len)] = code;
        for (int i = 0; i < n; ++i, ++k, ++code) {
            // Antes de usar o código: um código ≥ 2^len (tabela superdimensionada) indexaria fora da tabela rápida.
            if (code >= (1 << len)) throw ImageDecodeError("JPEG: tabela de Huffman superdimensionada");
            if (len <= kFastBits) {
                const int shift = kFastBits - len;
                for (int j = 0; j < (1 << shift); ++j) t.fast[static_cast<std::size_t>((code << shift) | j)] = static_cast<std::uint16_t>((len << 8) | values[k]);
            }
        }
        t.maxCode[static_cast<std::size_t>(len)] = n > 0 ? code - 1 : -1;
        code <<= 1;
    }
    t.maxCode[17] = 0x7FFFFFFF;
}

/// Leitor de bits MSB-primeiro dos dados entrópicos: tira o enchimento FF 00 e para num marcador. Depois do
/// marcador (ou do fim) fornece zeros; consumir algum deles é truncamento.
class BitReader {
public:
    explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}

    std::uint32_t peek16() {
        fill();
        return static_cast<std::uint32_t>(acc_ >> (count_ - 16)) & 0xFFFFu;
    }
    void consume(int n) {
        count_ -= n;
        if (count_ < fake_) throw ImageDecodeError("JPEG: dados entrópicos truncados");
    }
    std::uint32_t bits(int n) {
        if (n == 0) return 0;
        fill();
        const auto v = static_cast<std::uint32_t>(acc_ >> (count_ - n)) & ((1u << n) - 1u);
        consume(n);
        return v;
    }
    /// Valor com sinal de s bits (EXTEND da norma).
    int receiveExtend(int s) {
        if (s == 0) return 0;
        if (s > 16) throw ImageDecodeError("JPEG: categoria de magnitude inválida");
        const int v = static_cast<int>(bits(s));
        return v < (1 << (s - 1)) ? v - (1 << s) + 1 : v;
    }
    /// Reinício: descarta os bits restantes e espera RSTn.
    void restart() {
        acc_ = 0;
        count_ = 0;
        fake_ = 0;
        marker_ = false;
        if (pos_ + 1 < data_.size() && data_[pos_] == 0xFF && data_[pos_ + 1] >= 0xD0 && data_[pos_ + 1] <= 0xD7) {
            pos_ += 2;
            return;
        }
        throw ImageDecodeError("JPEG: marcador de reinício ausente");
    }

private:
    void fill() {
        while (count_ <= 48) {
            std::uint8_t byte = 0;
            bool real = false;
            if (!marker_ && pos_ < data_.size()) {
                if (data_[pos_] != 0xFF) {
                    byte = data_[pos_++];
                    real = true;
                } else if (pos_ + 1 < data_.size() && data_[pos_ + 1] == 0x00) {
                    byte = 0xFF;
                    pos_ += 2;
                    real = true;
                } else {
                    marker_ = true;  // um marcador (RSTn ou o fim do scan): daqui em diante, zeros
                }
            }
            acc_ = (acc_ << 8) | byte;
            count_ += 8;
            if (!real) fake_ += 8;
        }
    }

    std::span<const std::uint8_t> data_;
    std::size_t pos_ = 0;
    std::uint64_t acc_ = 0;
    int count_ = 0;
    int fake_ = 0;  ///< bits de enchimento artificial no fim do acumulador
    bool marker_ = false;
};

int decodeHuffman(BitReader& br, const HuffTable& t) {
    const std::uint32_t peek = br.peek16();
    const std::uint16_t e = t.fast[peek >> (16 - kFastBits)];
    if (e != 0) {
        br.consume(e >> 8);
        return e & 0xFF;
    }
    for (int len = kFastBits + 1; len <= 16; ++len) {
        const auto code = static_cast<std::int32_t>(peek >> (16 - len));
        if (code <= t.maxCode[static_cast<std::size_t>(len)]) {
            br.consume(len);
            return t.values[static_cast<std::size_t>(t.valPtr[static_cast<std::size_t>(len)] + code - t.minCode[static_cast<std::size_t>(len)])];
        }
    }
    throw ImageDecodeError("JPEG: código de Huffman inválido");
}

// ---------------------------------------------------------------- quadro e componentes

struct Component {
    int id = 0;
    int h = 1;
    int v = 1;
    int tq = 0;
    int blocksW = 0;  ///< blocos por linha no buffer (mcusX·h)
    int blocksH = 0;
    int width = 0;  ///< amostras reais: ⌈W·h/hmax⌉
    int height = 0;
    bool latched = false;  ///< a tabela de quantização já foi fixada (primeiro scan do componente)
    std::array<std::uint16_t, 64> quant{};  ///< em ordem natural
    std::vector<std::int16_t> coef;  ///< blocksW·blocksH·64, ordem natural
    int pred = 0;
    std::int16_t* block(int bx, int by) { return coef.data() + (static_cast<std::size_t>(by) * static_cast<std::size_t>(blocksW) + static_cast<std::size_t>(bx)) * 64; }
};

struct Frame {
    bool present = false;
    bool progressive = false;
    int width = 0;
    int height = 0;
    int hmax = 1;
    int vmax = 1;
    int mcusX = 0;
    int mcusY = 0;
    std::vector<Component> comps;
};

struct Scan {
    std::vector<int> comps;  ///< índices em Frame::comps
    std::array<int, 4> td{};
    std::array<int, 4> ta{};
    int ss = 0;
    int se = 63;
    int ah = 0;
    int al = 0;
};

class Decoder {
public:
    Decoder(std::span<const std::uint8_t> data, const DecodeOptions& options) : data_(data), options_(options) {}

    SourceImage run() {
        if (data_.size() < 4 || data_[0] != 0xFF || data_[1] != 0xD8) throw ImageDecodeError("JPEG: sem o marcador SOI");
        std::size_t pos = 2;
        bool scans = false;
        for (;;) {
            // Próximo marcador. Bytes estranhos antes dele são pulados (o libjpeg avisa e segue); bytes FF de
            // enchimento são permitidos.
            while (pos < data_.size() && data_[pos] != 0xFF) ++pos;
            if (pos >= data_.size()) {
                if (scans) break;  // sem EOI, mas com os scans completos: aceito
                throw ImageDecodeError("JPEG: arquivo truncado");
            }
            while (pos < data_.size() && data_[pos] == 0xFF) ++pos;
            if (pos >= data_.size()) {
                if (scans) break;
                throw ImageDecodeError("JPEG: arquivo truncado");
            }
            const std::uint8_t m = data_[pos++];
            if (m == 0xD9) break;                       // EOI
            if (m == 0x01 || (m >= 0xD0 && m <= 0xD7)) continue;  // TEM, RSTn soltos
            if (m == 0xD8) throw ImageDecodeError("JPEG: SOI repetido");
            if (data_.size() - pos < 2) throw ImageDecodeError("JPEG: segmento truncado");
            const unsigned len = be16(&data_[pos]);
            if (len < 2 || len > data_.size() - pos) throw ImageDecodeError("JPEG: segmento truncado ou com tamanho inválido");
            const std::span<const std::uint8_t> seg = data_.subspan(pos + 2, len - 2);
            pos += len;
            switch (m) {
                case 0xC0:
                case 0xC1:
                case 0xC2: frame(seg, m == 0xC2); break;
                case 0xC3: throw ImageDecodeError("JPEG: modo sem perdas não suportado");
                case 0xC5:
                case 0xC6:
                case 0xC7: throw ImageDecodeError("JPEG: modo hierárquico não suportado");
                case 0xC9:
                case 0xCA:
                case 0xCB:
                case 0xCD:
                case 0xCE:
                case 0xCF:
                case 0xCC: throw ImageDecodeError("JPEG: codificação aritmética não suportada");
                case 0xC4: huffman(seg); break;
                case 0xDB: quant(seg); break;
                case 0xDD:
                    if (seg.size() != 2) throw ImageDecodeError("JPEG: DRI com tamanho inválido");
                    restartInterval_ = static_cast<int>(be16(seg.data()));
                    break;
                case 0xDC: throw ImageDecodeError("JPEG: DNL não suportado");
                case 0xE0: jfif(seg); break;
                case 0xE1: exif(seg); break;
                case 0xEE: adobe(seg); break;
                case 0xDA: pos = scan(seg, pos); scans = true; break;
                default: break;  // APPn, COM e outros: ignorados
            }
        }
        if (!frame_.present || !scans) throw ImageDecodeError("JPEG: sem quadro ou sem scans");
        return finish();
    }

private:
    // ---- segmentos

    void frame(std::span<const std::uint8_t> s, bool progressive) {
        if (frame_.present) throw ImageDecodeError("JPEG: mais de um quadro (SOF)");
        if (s.size() < 6) throw ImageDecodeError("JPEG: SOF curto");
        if (s[0] != 8) throw ImageDecodeError("JPEG: precisão de " + std::to_string(s[0]) + " bits não suportada");
        frame_.height = static_cast<int>(be16(&s[1]));
        frame_.width = static_cast<int>(be16(&s[3]));
        if (frame_.height == 0) throw ImageDecodeError("JPEG: altura 0 (DNL) não suportada");
        detail::checkDimensions(static_cast<std::uint64_t>(frame_.width), static_cast<std::uint64_t>(frame_.height), options_, "JPEG");
        const int n = s[5];
        if (n < 1 || n > 4 || s.size() != 6 + 3 * static_cast<std::size_t>(n)) throw ImageDecodeError("JPEG: número de componentes inválido");
        frame_.progressive = progressive;
        for (int i = 0; i < n; ++i) {
            Component c;
            c.id = s[6 + 3 * static_cast<std::size_t>(i)];
            c.h = s[7 + 3 * static_cast<std::size_t>(i)] >> 4;
            c.v = s[7 + 3 * static_cast<std::size_t>(i)] & 15;
            c.tq = s[8 + 3 * static_cast<std::size_t>(i)];
            if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.tq > 3) throw ImageDecodeError("JPEG: amostragem ou tabela de componente inválida");
            for (const Component& o : frame_.comps) {
                if (o.id == c.id) throw ImageDecodeError("JPEG: id de componente repetido");
            }
            frame_.hmax = std::max(frame_.hmax, c.h);
            frame_.vmax = std::max(frame_.vmax, c.v);
            frame_.comps.push_back(c);
        }
        frame_.mcusX = (frame_.width + 8 * frame_.hmax - 1) / (8 * frame_.hmax);
        frame_.mcusY = (frame_.height + 8 * frame_.vmax - 1) / (8 * frame_.vmax);
        for (Component& c : frame_.comps) {
            c.blocksW = frame_.mcusX * c.h;
            c.blocksH = frame_.mcusY * c.v;
            c.width = (frame_.width * c.h + frame_.hmax - 1) / frame_.hmax;
            c.height = (frame_.height * c.v + frame_.vmax - 1) / frame_.vmax;
            c.coef.assign(static_cast<std::size_t>(c.blocksW) * static_cast<std::size_t>(c.blocksH) * 64, 0);
        }
        frame_.present = true;
    }

    void huffman(std::span<const std::uint8_t> s) {
        std::size_t p = 0;
        while (p < s.size()) {
            if (s.size() - p < 17) throw ImageDecodeError("JPEG: DHT curto");
            const int tc = s[p] >> 4;
            const int th = s[p] & 15;
            if (tc > 1 || th > 3) throw ImageDecodeError("JPEG: classe ou destino de tabela de Huffman inválido");
            int total = 0;
            for (int i = 0; i < 16; ++i) total += s[p + 1 + static_cast<std::size_t>(i)];
            if (total > 256 || s.size() - p - 17 < static_cast<std::size_t>(total)) throw ImageDecodeError("JPEG: DHT truncado");
            buildHuffman(huff_[static_cast<std::size_t>(tc)][static_cast<std::size_t>(th)], &s[p + 1], &s[p + 17], total);
            p += 17 + static_cast<std::size_t>(total);
        }
    }

    void quant(std::span<const std::uint8_t> s) {
        std::size_t p = 0;
        while (p < s.size()) {
            const int pq = s[p] >> 4;
            const int tq = s[p] & 15;
            if (pq > 1 || tq > 3) throw ImageDecodeError("JPEG: tabela de quantização inválida");
            const std::size_t n = pq == 0 ? 64 : 128;
            if (s.size() - p - 1 < n) throw ImageDecodeError("JPEG: DQT truncado");
            for (std::size_t k = 0; k < 64; ++k) {
                qt_[static_cast<std::size_t>(tq)][kZigzag[k]] = static_cast<std::uint16_t>(pq == 0 ? s[p + 1 + k] : be16(&s[p + 1 + 2 * k]));
            }
            qtPresent_[static_cast<std::size_t>(tq)] = true;
            p += 1 + n;
        }
    }

    void jfif(std::span<const std::uint8_t> s) {
        if (s.size() < 12 || !std::equal(s.begin(), s.begin() + 5, "JFIF")) return;  // "JFIF\0"
        const int units = s[7];
        const double xd = be16(&s[8]);
        const double yd = be16(&s[10]);
        if ((units == 1 || units == 2) && xd > 0 && yd > 0) {
            const double f = units == 1 ? 1.0 : 2.54;
            jfifDpi_ = {xd * f, yd * f};
            haveJfifDpi_ = true;
        }
    }

    void adobe(std::span<const std::uint8_t> s) {
        if (s.size() < 12 || !std::equal(s.begin(), s.begin() + 5, "Adobe")) return;
        adobe_ = true;
        adobeTransform_ = s[11];
    }

    /// EXIF: orientação (0x0112), resolução (0x011A, 0x011B) e unidade (0x0128) do IFD0. Metadado: um
    /// bloco malformado é ignorado, não é erro.
    void exif(std::span<const std::uint8_t> s) {
        static constexpr std::array<std::uint8_t, 6> kExif{'E', 'x', 'i', 'f', 0, 0};
        if (s.size() < 14 || !std::equal(kExif.begin(), kExif.end(), s.begin())) return;
        const std::span<const std::uint8_t> t = s.subspan(6);
        const bool le = t[0] == 'I' && t[1] == 'I';
        if (!le && !(t[0] == 'M' && t[1] == 'M')) return;
        const auto u16 = [&](std::size_t o) -> unsigned { return le ? (t[o] | (t[o + 1] << 8)) : ((t[o] << 8) | t[o + 1]); };
        const auto u32 = [&](std::size_t o) -> std::uint32_t {
            return le ? (static_cast<std::uint32_t>(t[o]) | (static_cast<std::uint32_t>(t[o + 1]) << 8) | (static_cast<std::uint32_t>(t[o + 2]) << 16) | (static_cast<std::uint32_t>(t[o + 3]) << 24))
                      : ((static_cast<std::uint32_t>(t[o]) << 24) | (static_cast<std::uint32_t>(t[o + 1]) << 16) | (static_cast<std::uint32_t>(t[o + 2]) << 8) | t[o + 3]);
        };
        if (u16(2) != 42) return;
        const std::uint32_t ifd = u32(4);
        if (ifd > t.size() - 2) return;
        const unsigned n = u16(ifd);
        double xres = 0.0;
        double yres = 0.0;
        unsigned unit = 2;
        const auto rational = [&](std::uint32_t off) {
            if (off > t.size() - 8) return 0.0;
            const std::uint32_t den = u32(off + 4);
            return den == 0 ? 0.0 : static_cast<double>(u32(off)) / den;
        };
        for (unsigned i = 0; i < n; ++i) {
            const std::size_t e = ifd + 2 + 12 * static_cast<std::size_t>(i);
            if (e + 12 > t.size()) return;
            const unsigned tag = u16(e);
            if (tag == 0x0112) {
                const unsigned o = u16(e + 8);
                if (o >= 1 && o <= 8) orientation_ = static_cast<int>(o);
            } else if (tag == 0x011A) {
                xres = rational(u32(e + 8));
            } else if (tag == 0x011B) {
                yres = rational(u32(e + 8));
            } else if (tag == 0x0128) {
                unit = u16(e + 8);
            }
        }
        if (xres > 0.0 && yres > 0.0 && (unit == 2 || unit == 3)) {
            const double f = unit == 2 ? 1.0 : 2.54;
            exifDpi_ = {xres * f, yres * f};
            haveExifDpi_ = true;
        }
    }

    // ---- scans

    std::size_t scan(std::span<const std::uint8_t> s, std::size_t dataStart) {
        if (!frame_.present) throw ImageDecodeError("JPEG: SOS antes do SOF");
        if (s.empty()) throw ImageDecodeError("JPEG: SOS curto");
        Scan sc;
        const int ns = s[0];
        if (ns < 1 || ns > 4 || s.size() != 4 + 2 * static_cast<std::size_t>(ns)) throw ImageDecodeError("JPEG: SOS com tamanho inválido");
        int blocksInMcu = 0;
        for (int i = 0; i < ns; ++i) {
            const int id = s[1 + 2 * static_cast<std::size_t>(i)];
            const int tables = s[2 + 2 * static_cast<std::size_t>(i)];
            int idx = -1;
            for (std::size_t c = 0; c < frame_.comps.size(); ++c) {
                if (frame_.comps[c].id == id) idx = static_cast<int>(c);
            }
            if (idx < 0) throw ImageDecodeError("JPEG: SOS cita um componente que não está no quadro");
            sc.comps.push_back(idx);
            sc.td[static_cast<std::size_t>(i)] = tables >> 4;
            sc.ta[static_cast<std::size_t>(i)] = tables & 15;
            if (sc.td[static_cast<std::size_t>(i)] > 3 || sc.ta[static_cast<std::size_t>(i)] > 3) throw ImageDecodeError("JPEG: tabela de Huffman inválida no SOS");
            blocksInMcu += frame_.comps[static_cast<std::size_t>(idx)].h * frame_.comps[static_cast<std::size_t>(idx)].v;
        }
        if (ns > 1 && blocksInMcu > 10) throw ImageDecodeError("JPEG: MCU com mais de 10 blocos");
        const std::size_t p = 1 + 2 * static_cast<std::size_t>(ns);
        sc.ss = s[p];
        sc.se = s[p + 1];
        sc.ah = s[p + 2] >> 4;
        sc.al = s[p + 2] & 15;
        if (frame_.progressive) {
            if (sc.se > 63 || sc.ss > sc.se || (sc.ss == 0 && sc.se != 0) || (sc.ss > 0 && ns != 1) || sc.ah > 13 || sc.al > 13) {
                throw ImageDecodeError("JPEG: parâmetros de scan progressivo inválidos");
            }
        } else if (sc.ss != 0 || sc.se != 63 || sc.ah != 0 || sc.al != 0) {
            throw ImageDecodeError("JPEG: parâmetros de scan sequencial inválidos");
        }
        const bool needDc = sc.ss == 0 && sc.ah == 0;
        const bool needAc = !frame_.progressive || sc.ss > 0;
        for (int i = 0; i < ns; ++i) {
            if (needDc && !huff_[0][static_cast<std::size_t>(sc.td[static_cast<std::size_t>(i)])].present) throw ImageDecodeError("JPEG: tabela de Huffman DC ausente");
            if (needAc && !huff_[1][static_cast<std::size_t>(sc.ta[static_cast<std::size_t>(i)])].present) throw ImageDecodeError("JPEG: tabela de Huffman AC ausente");
            Component& c = frame_.comps[static_cast<std::size_t>(sc.comps[static_cast<std::size_t>(i)])];
            if (!c.latched) {
                if (!qtPresent_[static_cast<std::size_t>(c.tq)]) throw ImageDecodeError("JPEG: tabela de quantização ausente");
                c.quant = qt_[static_cast<std::size_t>(c.tq)];
                c.latched = true;
            }
            c.pred = 0;
        }

        // Fim dos dados entrópicos: o primeiro FF seguido de algo que não é 00 nem RSTn.
        std::size_t end = dataStart;
        while (end + 1 < data_.size() && !(data_[end] == 0xFF && data_[end + 1] != 0x00 && !(data_[end + 1] >= 0xD0 && data_[end + 1] <= 0xD7))) ++end;
        if (end + 1 >= data_.size()) end = data_.size();
        BitReader br(data_.subspan(dataStart, end - dataStart));
        eobrun_ = 0;

        const auto restartIfDue = [&](long long done, long long total) {
            if (restartInterval_ > 0 && done % restartInterval_ == 0 && done < total) {
                br.restart();
                for (const int ci : sc.comps) frame_.comps[static_cast<std::size_t>(ci)].pred = 0;
                eobrun_ = 0;
            }
        };
        if (ns == 1) {
            // Não entrelaçado: os blocos do componente em ordem raster, cada um é uma MCU.
            Component& c = frame_.comps[static_cast<std::size_t>(sc.comps[0])];
            const int bw = (c.width + 7) / 8;
            const int bh = (c.height + 7) / 8;
            const long long total = static_cast<long long>(bw) * bh;
            long long done = 0;
            for (int by = 0; by < bh; ++by) {
                detail::checkStop(options_.stop);
                for (int bx = 0; bx < bw; ++bx) {
                    decodeBlock(br, sc, 0, c, c.block(bx, by));
                    restartIfDue(++done, total);
                }
            }
        } else {
            const long long total = static_cast<long long>(frame_.mcusX) * frame_.mcusY;
            long long done = 0;
            for (int my = 0; my < frame_.mcusY; ++my) {
                detail::checkStop(options_.stop);
                for (int mx = 0; mx < frame_.mcusX; ++mx) {
                    for (int i = 0; i < ns; ++i) {
                        Component& c = frame_.comps[static_cast<std::size_t>(sc.comps[static_cast<std::size_t>(i)])];
                        for (int v = 0; v < c.v; ++v) {
                            for (int h = 0; h < c.h; ++h) decodeBlock(br, sc, i, c, c.block(mx * c.h + h, my * c.v + v));
                        }
                    }
                    restartIfDue(++done, total);
                }
            }
        }
        return end;
    }

    void decodeBlock(BitReader& br, const Scan& sc, int i, Component& c, std::int16_t* blk) {
        const HuffTable& dc = huff_[0][static_cast<std::size_t>(sc.td[static_cast<std::size_t>(i)])];
        const HuffTable& ac = huff_[1][static_cast<std::size_t>(sc.ta[static_cast<std::size_t>(i)])];
        if (!frame_.progressive) {
            c.pred += br.receiveExtend(decodeHuffman(br, dc));
            blk[0] = static_cast<std::int16_t>(c.pred);
            for (int k = 1; k < 64;) {
                const int rs = decodeHuffman(br, ac);
                const int r = rs >> 4;
                const int s = rs & 15;
                if (s == 0) {
                    if (r != 15) break;  // EOB
                    k += 16;             // ZRL
                    continue;
                }
                k += r;
                if (k > 63) throw ImageDecodeError("JPEG: coeficiente além do bloco");
                blk[kZigzag[static_cast<std::size_t>(k)]] = static_cast<std::int16_t>(br.receiveExtend(s));
                ++k;
            }
            return;
        }
        if (sc.ss == 0) {  // DC progressivo
            if (sc.ah == 0) {
                c.pred += br.receiveExtend(decodeHuffman(br, dc));
                blk[0] = static_cast<std::int16_t>(c.pred * (1 << sc.al));
            } else if (br.bits(1) != 0) {
                blk[0] = static_cast<std::int16_t>(blk[0] | (1 << sc.al));
            }
            return;
        }
        if (sc.ah == 0) {  // AC, primeira passada
            if (eobrun_ > 0) {
                --eobrun_;
                return;
            }
            for (int k = sc.ss; k <= sc.se;) {
                const int rs = decodeHuffman(br, ac);
                const int r = rs >> 4;
                const int s = rs & 15;
                if (s == 0) {
                    if (r < 15) {
                        eobrun_ = (1 << r) - 1 + static_cast<int>(br.bits(r));
                        break;
                    }
                    k += 16;
                    continue;
                }
                k += r;
                if (k > sc.se) throw ImageDecodeError("JPEG: coeficiente além da faixa do scan");
                blk[kZigzag[static_cast<std::size_t>(k)]] = static_cast<std::int16_t>(br.receiveExtend(s) * (1 << sc.al));
                ++k;
            }
            return;
        }
        refineAc(br, sc, ac, blk);
    }

    /// AC, refinamento por aproximações sucessivas (T.81 G.1.2.3; a mesma estrutura do libjpeg).
    void refineAc(BitReader& br, const Scan& sc, const HuffTable& ac, std::int16_t* blk) {
        const int p1 = 1 << sc.al;
        const int m1 = -p1;
        const auto refine = [&](std::int16_t& coef) {
            if (br.bits(1) != 0 && (coef & p1) == 0) coef = static_cast<std::int16_t>(coef + (coef >= 0 ? p1 : m1));
        };
        int k = sc.ss;
        if (eobrun_ == 0) {
            for (; k <= sc.se; ++k) {
                const int rs = decodeHuffman(br, ac);
                int r = rs >> 4;
                int s = rs & 15;
                if (s != 0) {
                    if (s != 1) throw ImageDecodeError("JPEG: refinamento AC com magnitude diferente de 1");
                    s = br.bits(1) != 0 ? p1 : m1;
                } else if (r != 15) {
                    eobrun_ = (1 << r) + static_cast<int>(br.bits(r));
                    break;
                }  // r = 15, s = 0: ZRL, pula 16 coeficientes ainda nulos
                for (; k <= sc.se; ++k) {
                    std::int16_t& coef = blk[kZigzag[static_cast<std::size_t>(k)]];
                    if (coef != 0) {
                        refine(coef);
                    } else {
                        if (r == 0) break;  // o coeficiente nulo alvo
                        --r;
                    }
                }
                if (s != 0) {
                    if (k > sc.se) throw ImageDecodeError("JPEG: coeficiente além da faixa do scan");
                    blk[kZigzag[static_cast<std::size_t>(k)]] = static_cast<std::int16_t>(s);
                }
            }
        }
        if (eobrun_ > 0) {
            for (; k <= sc.se; ++k) {
                std::int16_t& coef = blk[kZigzag[static_cast<std::size_t>(k)]];
                if (coef != 0) refine(coef);
            }
            --eobrun_;
        }
    }

    // ---- reconstrução

    static const std::array<std::array<double, 8>, 8>& idctTable() {
        static const std::array<std::array<double, 8>, 8> t = [] {
            std::array<std::array<double, 8>, 8> c{};
            for (int x = 0; x < 8; ++x) {
                for (int u = 0; u < 8; ++u) c[static_cast<std::size_t>(x)][static_cast<std::size_t>(u)] = (u == 0 ? std::sqrt(0.5) : 1.0) * 0.5 * std::cos((2 * x + 1) * u * std::numbers::pi / 16.0);
            }
            return c;
        }();
        return t;
    }

    /// Plano de amostras do componente (blocksW·8 × blocksH·8), pela IDCT de cada bloco desquantizado.
    std::vector<std::uint8_t> reconstruct(Component& c) {
        const auto& t = idctTable();
        const std::size_t stride = static_cast<std::size_t>(c.blocksW) * 8;
        std::vector<std::uint8_t> plane(stride * static_cast<std::size_t>(c.blocksH) * 8);
        for (int by = 0; by < c.blocksH; ++by) {
            detail::checkStop(options_.stop);
            for (int bx = 0; bx < c.blocksW; ++bx) {
                const std::int16_t* blk = c.block(bx, by);
                std::array<double, 64> f{};
                for (std::size_t k = 0; k < 64; ++k) f[k] = static_cast<double>(blk[k]) * c.quant[k];
                std::array<double, 64> tmp{};  // linhas: tmp[v][x] = Σ_u c[x][u]·F[v][u]
                for (std::size_t v = 0; v < 8; ++v) {
                    for (std::size_t x = 0; x < 8; ++x) {
                        double s = 0.0;
                        for (std::size_t u = 0; u < 8; ++u) s += t[x][u] * f[v * 8 + u];
                        tmp[v * 8 + x] = s;
                    }
                }
                for (std::size_t y = 0; y < 8; ++y) {
                    for (std::size_t x = 0; x < 8; ++x) {
                        double s = 0.0;
                        for (std::size_t v = 0; v < 8; ++v) s += t[y][v] * tmp[v * 8 + x];
                        const long q = std::lround(s + 128.0);
                        plane[(static_cast<std::size_t>(by) * 8 + y) * stride + static_cast<std::size_t>(bx) * 8 + x] = static_cast<std::uint8_t>(std::clamp(q, 0L, 255L));
                    }
                }
            }
        }
        return plane;
    }

    SourceImage finish() {
        const int w = frame_.width;
        const int h = frame_.height;
        const std::size_t nc = frame_.comps.size();
        std::vector<std::vector<std::uint8_t>> planes;
        for (Component& c : frame_.comps) planes.push_back(reconstruct(c));
        // Amostra do componente c no pixel de saída (x, y): direta, ou interpolação linear centrada.
        struct Tap {
            int i0, i1;
            double t;
        };
        const auto taps = [](int n, int full, int factor, int fmax) {
            std::vector<Tap> out(static_cast<std::size_t>(full));
            for (int o = 0; o < full; ++o) {
                const double f = std::clamp((o + 0.5) * factor / fmax - 0.5, 0.0, static_cast<double>(n - 1));
                const int i0 = static_cast<int>(f);
                out[static_cast<std::size_t>(o)] = {i0, std::min(i0 + 1, n - 1), f - i0};
            }
            return out;
        };
        std::vector<std::vector<Tap>> tx;
        std::vector<std::vector<Tap>> ty;
        for (const Component& c : frame_.comps) {
            tx.push_back(taps(c.width, w, c.h, frame_.hmax));
            ty.push_back(taps(c.height, h, c.v, frame_.vmax));
        }
        int mode = 0;  // 0 cinza, 1 YCbCr, 2 RGB, 3 CMYK, 4 YCCK
        if (nc == 3) {
            const bool rgbIds = frame_.comps[0].id == 'R' && frame_.comps[1].id == 'G' && frame_.comps[2].id == 'B';
            mode = adobe_ ? (adobeTransform_ == 0 ? 2 : 1) : (rgbIds ? 2 : 1);
        } else if (nc == 4) {
            mode = adobe_ && adobeTransform_ == 2 ? 4 : 3;
        } else if (nc == 2) {
            throw ImageDecodeError("JPEG: 2 componentes não suportados");
        }
        SourceImage out;
        out.formatHint = ".jpeg";
        out.pixels = sla::Array2D<sla::Rgba8>(w, h);
        const auto q = [](double v) { return static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L)); };
        std::array<double, 4> s{};
        for (int y = 0; y < h; ++y) {
            detail::checkStop(options_.stop);
            for (int x = 0; x < w; ++x) {
                for (std::size_t c = 0; c < nc; ++c) {
                    const Component& comp = frame_.comps[c];
                    const std::size_t stride = static_cast<std::size_t>(comp.blocksW) * 8;
                    const std::vector<std::uint8_t>& pl = planes[c];
                    if (comp.h == frame_.hmax && comp.v == frame_.vmax) {
                        s[c] = pl[static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x)];
                        continue;
                    }
                    const Tap& a = tx[c][static_cast<std::size_t>(x)];
                    const Tap& b = ty[c][static_cast<std::size_t>(y)];
                    const auto at = [&](int xi, int yi) { return static_cast<double>(pl[static_cast<std::size_t>(yi) * stride + static_cast<std::size_t>(xi)]); };
                    const double top = at(a.i0, b.i0) * (1.0 - a.t) + at(a.i1, b.i0) * a.t;
                    const double bottom = at(a.i0, b.i1) * (1.0 - a.t) + at(a.i1, b.i1) * a.t;
                    s[c] = top * (1.0 - b.t) + bottom * b.t;
                }
                sla::Rgba8& px = out.pixels(x, y);
                const auto ycc = [&](double yy, double cb, double cr, double& r, double& g, double& bl) {
                    r = yy + 1.402 * (cr - 128.0);
                    g = yy - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0);
                    bl = yy + 1.772 * (cb - 128.0);
                };
                switch (mode) {
                    case 0: px = {q(s[0]), q(s[0]), q(s[0]), 255}; break;
                    case 1: {
                        double r = 0, g = 0, b = 0;
                        ycc(s[0], s[1], s[2], r, g, b);
                        px = {q(r), q(g), q(b), 255};
                        break;
                    }
                    case 2: px = {q(s[0]), q(s[1]), q(s[2]), 255}; break;
                    case 3:  // CMYK Adobe (invertido): R = C·K/255
                        px = {q(s[0] * s[3] / 255.0), q(s[1] * s[3] / 255.0), q(s[2] * s[3] / 255.0), 255};
                        break;
                    default: {  // YCCK: YCbCr → C, M, Y invertidos, depois como CMYK
                        double r = 0, g = 0, b = 0;
                        ycc(s[0], s[1], s[2], r, g, b);
                        const auto cl = [](double v) { return std::clamp(v, 0.0, 255.0); };
                        px = {q((255.0 - cl(r)) * s[3] / 255.0), q((255.0 - cl(g)) * s[3] / 255.0), q((255.0 - cl(b)) * s[3] / 255.0), 255};
                        break;
                    }
                }
            }
        }
        if (haveJfifDpi_) {
            out.dpiX = jfifDpi_[0];
            out.dpiY = jfifDpi_[1];
        } else if (haveExifDpi_) {
            out.dpiX = exifDpi_[0];
            out.dpiY = exifDpi_[1];
        }
        orient(out);
        return out;
    }

    /// Orientação EXIF: 2 espelho horizontal, 3 giro de 180°, 4 espelho vertical, 5 transposição, 6 giro de 90°
    /// horário, 7 transversa, 8 giro de 90° anti-horário. 5–8 trocam largura e altura (e os DPI).
    void orient(SourceImage& img) const {
        if (orientation_ == 1) return;
        const sla::Array2D<sla::Rgba8>& in = img.pixels;
        const int w = in.width();
        const int h = in.height();
        const bool swap = orientation_ >= 5;
        sla::Array2D<sla::Rgba8> out(swap ? h : w, swap ? w : h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int nx = x;
                int ny = y;
                switch (orientation_) {
                    case 2: nx = w - 1 - x; break;
                    case 3: nx = w - 1 - x; ny = h - 1 - y; break;
                    case 4: ny = h - 1 - y; break;
                    case 5: nx = y; ny = x; break;
                    case 6: nx = h - 1 - y; ny = x; break;
                    case 7: nx = h - 1 - y; ny = w - 1 - x; break;
                    default: nx = y; ny = w - 1 - x; break;  // 8
                }
                out(nx, ny) = in(x, y);
            }
        }
        img.pixels = std::move(out);
        if (swap) std::swap(img.dpiX, img.dpiY);
    }

    std::span<const std::uint8_t> data_;
    const DecodeOptions& options_;
    Frame frame_;
    std::array<std::array<HuffTable, 4>, 2> huff_{};
    std::array<std::array<std::uint16_t, 64>, 4> qt_{};
    std::array<bool, 4> qtPresent_{};
    int restartInterval_ = 0;
    int eobrun_ = 0;
    bool adobe_ = false;
    int adobeTransform_ = 0;
    int orientation_ = 1;
    bool haveJfifDpi_ = false;
    bool haveExifDpi_ = false;
    std::array<double, 2> jfifDpi_{};
    std::array<double, 2> exifDpi_{};
};

}  // namespace

SourceImage decodeJpeg(std::span<const std::uint8_t> data, const DecodeOptions& options) { return Decoder(data, options).run(); }

}  // namespace cnx
