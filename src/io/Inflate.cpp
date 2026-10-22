// src/io/Inflate.cpp — zlib/DEFLATE: blocos armazenados, Huffman fixo e dinâmico, Adler-32 e CRC-32.
#include "cnx/io/Inflate.h"

#include <algorithm>
#include <array>
#include <cstring>

#include "cnx/io/ImageDecode.h"

namespace cnx {
namespace {

constexpr int kFastBits = 9;
constexpr std::size_t kStopEvery = std::size_t{1} << 16;

/// Leitor de bits LSB-primeiro (a ordem do DEFLATE). Os bits além do fim lidos por peek valem 0; consumir
/// qualquer um deles é erro de truncamento.
class BitReader {
public:
    explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}

    std::uint32_t bits(int n) {
        if (n == 0) return 0;
        if (count_ < n) fill();
        if (count_ < n) throw ImageDecodeError("DEFLATE: dados truncados");
        const auto v = static_cast<std::uint32_t>(buffer_ & ((std::uint64_t{1} << n) - 1));
        drop(n);
        return v;
    }
    std::uint32_t peek(int n) {
        if (count_ < n) fill();
        return static_cast<std::uint32_t>(buffer_ & ((std::uint64_t{1} << n) - 1));
    }
    int available() const noexcept { return count_; }
    void drop(int n) noexcept {
        buffer_ >>= n;
        count_ -= n;
    }
    void alignToByte() noexcept { drop(count_ % 8); }
    /// Copia n bytes alinhados (bloco armazenado): primeiro os do acumulador, depois direto da entrada.
    void copyBytes(std::size_t n, std::vector<std::uint8_t>& out) {
        while (n > 0 && count_ >= 8) {
            out.push_back(static_cast<std::uint8_t>(bits(8)));
            --n;
        }
        if (n > data_.size() - pos_) throw ImageDecodeError("DEFLATE: bloco armazenado truncado");
        out.insert(out.end(), data_.begin() + static_cast<std::ptrdiff_t>(pos_), data_.begin() + static_cast<std::ptrdiff_t>(pos_ + n));
        pos_ += n;
    }

private:
    void fill() noexcept {
        while (count_ <= 56 && pos_ < data_.size()) {
            buffer_ |= static_cast<std::uint64_t>(data_[pos_++]) << count_;
            count_ += 8;
        }
    }

    std::span<const std::uint8_t> data_;
    std::size_t pos_ = 0;
    std::uint64_t buffer_ = 0;
    int count_ = 0;
};

/// Código de Huffman canônico: contagem por comprimento, símbolos em ordem de código e a tabela rápida
/// (índice = os 9 próximos bits; entrada = símbolo | comprimento << 9; 0 = código mais longo ou inválido).
struct Huffman {
    std::array<std::uint16_t, 16> count{};
    std::array<std::uint16_t, 288> symbol{};
    std::array<std::uint16_t, 1u << kFastBits> fast{};
};

/// Monta o código a partir dos comprimentos. Devolve 0 se completo, > 0 se incompleto, < 0 se superdimensionado.
int build(Huffman& h, const std::uint8_t* length, int n) {
    h.count.fill(0);
    h.fast.fill(0);
    for (int s = 0; s < n; ++s) ++h.count[length[s]];
    if (h.count[0] == n) return 0;  // nenhum código: completo, mas decode falha se for usado
    int left = 1;
    for (int len = 1; len < 16; ++len) {
        left <<= 1;
        left -= h.count[static_cast<std::size_t>(len)];
        if (left < 0) return left;
    }
    std::array<std::uint16_t, 16> offs{};
    for (int len = 1; len < 15; ++len) offs[static_cast<std::size_t>(len + 1)] = static_cast<std::uint16_t>(offs[static_cast<std::size_t>(len)] + h.count[static_cast<std::size_t>(len)]);
    for (int s = 0; s < n; ++s) {
        if (length[s] != 0) h.symbol[offs[length[s]]++] = static_cast<std::uint16_t>(s);
    }
    // Códigos canônicos (RFC 1951 §3.2.2) para a tabela rápida, com os bits invertidos (LSB primeiro).
    std::array<std::uint32_t, 16> next{};
    std::uint32_t code = 0;
    for (int len = 1; len < 16; ++len) {
        next[static_cast<std::size_t>(len)] = code;
        code = (code + h.count[static_cast<std::size_t>(len)]) << 1;
    }
    for (int s = 0; s < n; ++s) {
        const int len = length[s];
        if (len == 0) continue;
        const std::uint32_t c = next[static_cast<std::size_t>(len)]++;
        if (len > kFastBits) continue;
        std::uint32_t rev = 0;
        for (int b = 0; b < len; ++b) rev |= ((c >> b) & 1u) << (len - 1 - b);
        for (std::uint32_t i = rev; i < (1u << kFastBits); i += 1u << len) h.fast[i] = static_cast<std::uint16_t>(s | (len << kFastBits));
    }
    return left;
}

int decode(BitReader& br, const Huffman& h) {
    const std::uint16_t e = h.fast[br.peek(kFastBits)];
    if (e != 0) {
        const int len = e >> kFastBits;
        if (len > br.available()) throw ImageDecodeError("DEFLATE: dados truncados");
        br.drop(len);
        return e & ((1 << kFastBits) - 1);
    }
    // Bit a bit (puff.c): código mais longo que a tabela rápida, ou inválido.
    int code = 0;
    int first = 0;
    int index = 0;
    for (int len = 1; len < 16; ++len) {
        code |= static_cast<int>(br.bits(1));
        const int count = h.count[static_cast<std::size_t>(len)];
        if (code - count < first) return h.symbol[static_cast<std::size_t>(index + (code - first))];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    throw ImageDecodeError("DEFLATE: código de Huffman inválido");
}

constexpr std::array<std::uint16_t, 29> kLengthBase{3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr std::array<std::uint8_t, 29> kLengthExtra{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::array<std::uint16_t, 30> kDistBase{1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
                                                  1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr std::array<std::uint8_t, 30> kDistExtra{0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

class Inflater {
public:
    Inflater(BitReader& br, std::size_t maxSize, const std::stop_token& stop) : br_(br), max_(maxSize), stop_(stop) {}

    std::vector<std::uint8_t> run() {
        bool last = false;
        while (!last) {
            last = br_.bits(1) != 0;
            switch (br_.bits(2)) {
                case 0: stored(); break;
                case 1: codes(fixedTables().first, fixedTables().second); break;
                case 2: dynamic(); break;
                default: throw ImageDecodeError("DEFLATE: tipo de bloco inválido (3)");
            }
        }
        return std::move(out_);
    }

private:
    void grow(std::size_t n) {
        if (n > max_ - out_.size()) throw ImageDecodeError("DEFLATE: a saída excede o tamanho esperado");
        if (out_.size() / kStopEvery != (out_.size() + n) / kStopEvery) detail::checkStop(stop_);
    }

    void stored() {
        br_.alignToByte();
        const std::uint32_t len = br_.bits(16);
        const std::uint32_t nlen = br_.bits(16);
        if ((len ^ 0xFFFFu) != nlen) throw ImageDecodeError("DEFLATE: tamanho de bloco armazenado inconsistente");
        grow(len);
        br_.copyBytes(len, out_);
    }

    static const std::pair<Huffman, Huffman>& fixedTables() {
        static const std::pair<Huffman, Huffman> t = [] {
            std::array<std::uint8_t, 288> l{};
            for (int s = 0; s < 144; ++s) l[static_cast<std::size_t>(s)] = 8;
            for (int s = 144; s < 256; ++s) l[static_cast<std::size_t>(s)] = 9;
            for (int s = 256; s < 280; ++s) l[static_cast<std::size_t>(s)] = 7;
            for (int s = 280; s < 288; ++s) l[static_cast<std::size_t>(s)] = 8;
            std::pair<Huffman, Huffman> p;
            (void)build(p.first, l.data(), 288);
            std::array<std::uint8_t, 30> d{};
            d.fill(5);
            (void)build(p.second, d.data(), 30);  // incompleto (30 de 32), como a RFC define
            return p;
        }();
        return t;
    }

    void dynamic() {
        static constexpr std::array<std::uint8_t, 19> kOrder{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
        const int nlen = static_cast<int>(br_.bits(5)) + 257;
        const int ndist = static_cast<int>(br_.bits(5)) + 1;
        const int ncode = static_cast<int>(br_.bits(4)) + 4;
        if (nlen > 286 || ndist > 30) throw ImageDecodeError("DEFLATE: contagem de códigos inválida");
        std::array<std::uint8_t, 320> lengths{};
        for (int i = 0; i < ncode; ++i) lengths[kOrder[static_cast<std::size_t>(i)]] = static_cast<std::uint8_t>(br_.bits(3));
        Huffman lencode;
        if (build(lencode, lengths.data(), 19) != 0) throw ImageDecodeError("DEFLATE: código dos comprimentos incompleto ou superdimensionado");
        int index = 0;
        while (index < nlen + ndist) {
            int sym = decode(br_, lencode);
            if (sym < 16) {
                lengths[static_cast<std::size_t>(index++)] = static_cast<std::uint8_t>(sym);
                continue;
            }
            std::uint8_t len = 0;
            if (sym == 16) {
                if (index == 0) throw ImageDecodeError("DEFLATE: repetição sem comprimento anterior");
                len = lengths[static_cast<std::size_t>(index - 1)];
                sym = 3 + static_cast<int>(br_.bits(2));
            } else if (sym == 17) {
                sym = 3 + static_cast<int>(br_.bits(3));
            } else {
                sym = 11 + static_cast<int>(br_.bits(7));
            }
            if (index + sym > nlen + ndist) throw ImageDecodeError("DEFLATE: comprimentos demais");
            while (sym-- > 0) lengths[static_cast<std::size_t>(index++)] = len;
        }
        if (lengths[256] == 0) throw ImageDecodeError("DEFLATE: sem o código de fim de bloco");
        Huffman lit;
        Huffman dist;
        // Incompleto só no caso permitido: um único código de comprimento 1.
        int err = build(lit, lengths.data(), nlen);
        if (err != 0 && (err < 0 || nlen != lit.count[0] + lit.count[1])) throw ImageDecodeError("DEFLATE: código de literais inválido");
        err = build(dist, lengths.data() + nlen, ndist);
        if (err != 0 && (err < 0 || ndist != dist.count[0] + dist.count[1])) throw ImageDecodeError("DEFLATE: código de distâncias inválido");
        codes(lit, dist);
    }

    void codes(const Huffman& lit, const Huffman& dist) {
        for (;;) {
            int sym = decode(br_, lit);
            if (sym < 256) {
                grow(1);
                out_.push_back(static_cast<std::uint8_t>(sym));
                continue;
            }
            if (sym == 256) return;
            sym -= 257;
            if (sym >= 29) throw ImageDecodeError("DEFLATE: símbolo de comprimento inválido");
            const std::size_t len = kLengthBase[static_cast<std::size_t>(sym)] + br_.bits(kLengthExtra[static_cast<std::size_t>(sym)]);
            // Só existem 30 códigos de distância (o fixo tem 30 de 32 e o dinâmico ≤ 30): 30 e 31 já falham no
            // decode como código de Huffman inválido.
            const int dsym = decode(br_, dist);
            const std::size_t d = kDistBase[static_cast<std::size_t>(dsym)] + br_.bits(kDistExtra[static_cast<std::size_t>(dsym)]);
            if (d > out_.size()) throw ImageDecodeError("DEFLATE: distância antes do início dos dados");
            grow(len);
            const std::size_t from = out_.size() - d;
            for (std::size_t i = 0; i < len; ++i) out_.push_back(out_[from + i]);  // cópia com sobreposição
        }
    }

    BitReader& br_;
    std::size_t max_;
    const std::stop_token& stop_;
    std::vector<std::uint8_t> out_;
};

constexpr std::array<std::uint32_t, 256> kCrcTable = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t n = 0; n < 256; ++n) {
        std::uint32_t c = n;
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        t[n] = c;
    }
    return t;
}();

}  // namespace

std::uint32_t adler32(std::span<const std::uint8_t> data) noexcept {
    std::uint32_t a = 1;
    std::uint32_t b = 0;
    for (std::size_t i = 0; i < data.size();) {
        const std::size_t end = std::min(data.size(), i + 5552);  // maior bloco sem estourar 32 bits
        for (; i < end; ++i) {
            a += data[i];
            b += a;
        }
        a %= 65521u;
        b %= 65521u;
    }
    return (b << 16) | a;
}

std::uint32_t crc32(std::span<const std::uint8_t> data, std::uint32_t crc) noexcept {
    std::uint32_t c = crc ^ 0xFFFFFFFFu;
    for (const std::uint8_t byte : data) c = kCrcTable[(c ^ byte) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

std::vector<std::uint8_t> inflateRaw(std::span<const std::uint8_t> data, std::size_t maxSize, const std::stop_token& stop) {
    BitReader br(data);
    return Inflater(br, maxSize, stop).run();
}

std::vector<std::uint8_t> inflateZlib(std::span<const std::uint8_t> data, std::size_t expectedSize, const std::stop_token& stop) {
    if (data.size() < 6) throw ImageDecodeError("zlib: fluxo curto demais");
    const std::uint8_t cmf = data[0];
    const std::uint8_t flg = data[1];
    if ((cmf & 0x0Fu) != 8) throw ImageDecodeError("zlib: método de compressão não é DEFLATE");
    if ((cmf >> 4) > 7) throw ImageDecodeError("zlib: janela maior que 32 KiB");
    if (((static_cast<unsigned>(cmf) << 8) | flg) % 31u != 0) throw ImageDecodeError("zlib: cabeçalho corrompido (FCHECK)");
    if ((flg & 0x20u) != 0) throw ImageDecodeError("zlib: dicionário predefinido não suportado");
    BitReader br(data.subspan(2));
    std::vector<std::uint8_t> out = Inflater(br, expectedSize, stop).run();
    if (out.size() != expectedSize) throw ImageDecodeError("zlib: dados descomprimidos menores que o esperado");
    br.alignToByte();
    std::uint32_t stored = 0;
    for (int k = 0; k < 4; ++k) stored = (stored << 8) | br.bits(8);
    if (stored != adler32(out)) throw ImageDecodeError("zlib: Adler-32 não confere");
    return out;
}

}  // namespace cnx
