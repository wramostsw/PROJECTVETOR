// tests/unit/image_decoder_test.cpp — decodificadores próprios de PNG, JPEG e zlib contra os arquivos de
// referência de tests/data (codificadores independentes em Python e o GDI+ do Windows), arquivos corrompidos
// montados aqui, limites, cancelamento e o carregador no CoreEngine.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stop_token>
#include <string>
#include <vector>

#include "cnx/classify/Classifier.h"
#include "cnx/engine/CoreEngine.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/io/ImageDecode.h"
#include "cnx/io/ImageFileLoader.h"
#include "cnx/io/Inflate.h"
#include "cnx/io/JpegDecoder.h"
#include "cnx/io/PngDecoder.h"
#include "cnx/segment/PixelMap.h"
#include "cnx/vector/VectorImage.h"
#include "test_harness.h"

using Bytes = std::vector<std::uint8_t>;
namespace fs = std::filesystem;

namespace {

const fs::path kData = CNX_TEST_DATA;

Bytes readBytes(const fs::path& p) { return cnx::ImageFileLoader::readFile(p, std::uintmax_t{1} << 30); }

struct Case {
    std::string kind, name;
    int width = 0, height = 0;
    double dpiX = 0, dpiY = 0;
    bool alpha = false;
    int tolerance = 0;
    std::size_t rawSize = 0;
};

std::vector<Case> manifest(const char* file) {
    std::vector<Case> out;
    std::ifstream in(kData / file);
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        std::istringstream s(line);
        Case c;
        s >> c.kind >> c.name;
        if (c.kind == "zlib" || c.kind == "deflate") {
            s >> c.rawSize;
        } else {
            int a = 0;
            s >> c.width >> c.height >> c.dpiX >> c.dpiY >> a >> c.tolerance;
            c.alpha = a != 0;
        }
        out.push_back(c);
    }
    return out;
}

/// Mensagem de ImageDecodeError (vazia se não lançou ou lançou outra coisa).
std::string decodeError(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const cnx::ImageDecodeError& e) {
        return e.what();
    } catch (...) {
        return "<outra exceção>";
    }
    return {};
}
bool mentions(const std::string& message, const char* what) { return message.find(what) != std::string::npos; }

void put32(Bytes& b, std::uint32_t v) {
    for (int k = 3; k >= 0; --k) b.push_back(static_cast<std::uint8_t>(v >> (8 * k)));
}

/// zlib com blocos armazenados (sem compressão) e o Adler-32 certo.
Bytes zlibStored(const Bytes& raw) {
    Bytes z{0x78, 0x01};
    std::size_t pos = 0;
    do {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - pos);
        z.push_back(pos + n == raw.size() ? 1 : 0);
        z.push_back(static_cast<std::uint8_t>(n));
        z.push_back(static_cast<std::uint8_t>(n >> 8));
        z.push_back(static_cast<std::uint8_t>(~n));
        z.push_back(static_cast<std::uint8_t>(~n >> 8));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos), raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
        pos += n;
    } while (pos < raw.size());
    put32(z, cnx::adler32(raw));
    return z;
}

Bytes pngChunk(const char* type, const Bytes& body) {
    Bytes c;
    put32(c, static_cast<std::uint32_t>(body.size()));
    Bytes typed(type, type + 4);
    typed.insert(typed.end(), body.begin(), body.end());
    c.insert(c.end(), typed.begin(), typed.end());
    put32(c, cnx::crc32(typed));
    return c;
}

/// PNG montado à mão: IHDR, chunks extras antes do IDAT, IDAT com `filtered` (zlib armazenado), IEND.
Bytes makePng(std::uint32_t w, std::uint32_t h, int depth, int color, const Bytes& filtered, const std::vector<Bytes>& extra = {}) {
    Bytes png{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    Bytes ihdr;
    put32(ihdr, w);
    put32(ihdr, h);
    ihdr.insert(ihdr.end(), {static_cast<std::uint8_t>(depth), static_cast<std::uint8_t>(color), 0, 0, 0});
    const Bytes c = pngChunk("IHDR", ihdr);
    png.insert(png.end(), c.begin(), c.end());
    for (const Bytes& e : extra) png.insert(png.end(), e.begin(), e.end());
    const Bytes idat = pngChunk("IDAT", zlibStored(filtered));
    png.insert(png.end(), idat.begin(), idat.end());
    const Bytes end = pngChunk("IEND", {});
    png.insert(png.end(), end.begin(), end.end());
    return png;
}

/// Escritor de bits LSB-primeiro (DEFLATE); `huff` escreve um código de Huffman (MSB do código primeiro).
struct DeflateBits {
    Bytes out;
    int n = 0;
    void bits(std::uint32_t v, int count) {
        for (int i = 0; i < count; ++i) {
            if (n % 8 == 0) out.push_back(0);
            out.back() = static_cast<std::uint8_t>(out.back() | (((v >> i) & 1u) << (n % 8)));
            ++n;
        }
    }
    void huff(std::uint32_t code, int len) {
        for (int i = len - 1; i >= 0; --i) bits((code >> i) & 1u, 1);
    }
};

/// Posição do primeiro segmento JPEG com o marcador m (o índice do FF).
std::size_t findMarker(const Bytes& j, std::uint8_t m) {
    for (std::size_t i = 2; i + 1 < j.size(); ++i) {
        if (j[i] == 0xFF && j[i + 1] == m) return i;
    }
    return j.size();
}
Bytes withoutSegment(Bytes j, std::uint8_t m) {
    const std::size_t at = findMarker(j, m);
    const std::size_t len = (static_cast<std::size_t>(j[at + 2]) << 8) | j[at + 3];
    j.erase(j.begin() + static_cast<std::ptrdiff_t>(at), j.begin() + static_cast<std::ptrdiff_t>(at + 2 + len));
    return j;
}

}  // namespace

// ================================================================ arquivos de referência

CNX_TEST(every_reference_file_decodes_to_its_expected_pixels) {
    std::vector<Case> cases = manifest("manifest.txt");
    for (const Case& c : manifest("manifest_gdiplus.txt")) cases.push_back(c);
    int images = 0;
    cnx::ImageFileLoader loader;
    for (const Case& c : cases) {
        if (c.kind != "png" && c.kind != "jpeg") continue;
        ++images;
        const fs::path file = kData / (c.name + (c.kind == "png" ? ".png" : ".jpg"));
        cnx::SourceImage img;
        const std::string err = decodeError([&] { img = loader.load(file, {}); });
        if (!err.empty()) {
            CHECK_CTX(false, c.name + ": " + err);
            continue;
        }
        CHECK_CTX(img.pixels.width() == c.width && img.pixels.height() == c.height, c.name + " dimensões");
        CHECK_CTX(img.formatHint == (c.kind == "png" ? ".png" : ".jpeg") && img.hasAlpha == c.alpha, c.name + " formato/alfa");
        CHECK_CTX(std::abs(img.dpiX - c.dpiX) < 1e-3 * c.dpiX && std::abs(img.dpiY - c.dpiY) < 1e-3 * c.dpiY,
                  c.name + " DPI " + std::to_string(img.dpiX) + " × " + std::to_string(img.dpiY));
        if (img.pixels.width() != c.width || img.pixels.height() != c.height) continue;
        const Bytes expected = readBytes(kData / (c.name + ".rgba"));
        int worst = 0;
        std::size_t differ = 0;
        double sum = 0.0;
        for (int y = 0; y < c.height; ++y) {
            for (int x = 0; x < c.width; ++x) {
                const sla::Rgba8 p = img.pixels(x, y);
                const std::size_t o = 4 * (static_cast<std::size_t>(y) * static_cast<std::size_t>(c.width) + static_cast<std::size_t>(x));
                const std::array<int, 4> got{p.r, p.g, p.b, p.a};
                for (std::size_t k = 0; k < 4; ++k) {
                    const int d = std::abs(got[k] - expected[o + k]);
                    worst = std::max(worst, d);
                    differ += d != 0 ? 1 : 0;
                    sum += d;
                }
            }
        }
        if (c.kind == "jpeg") {
            std::printf("  %-34s %3d×%-3d  diferença máx %d, %zu canais diferentes, média %.4f\n", c.name.c_str(), c.width, c.height, worst, differ,
                        sum / (4.0 * c.width * c.height));
        }
        CHECK_CTX(worst <= c.tolerance, c.name + ": diferença " + std::to_string(worst));
    }
    std::printf("  %d imagens de referência\n", images);
    CHECK(images == 72);  // 36 PNG e 36 JPEG (os do gerador Python e os 2 do GDI+)
}

CNX_TEST(zlib_and_deflate_reference_streams_inflate_exactly) {
    int n = 0;
    for (const Case& c : manifest("manifest.txt")) {
        if (c.kind != "zlib" && c.kind != "deflate") continue;
        ++n;
        const Bytes z = readBytes(kData / (c.name + (c.kind == "zlib" ? ".z" : ".deflate")));
        const Bytes raw = readBytes(kData / (c.name + ".raw"));
        Bytes got;
        const std::string err = decodeError([&] { got = c.kind == "zlib" ? cnx::inflateZlib(z, c.rawSize) : cnx::inflateRaw(z, c.rawSize); });
        CHECK_CTX(err.empty() && got == raw, c.name + ": " + err);
    }
    CHECK(n == 11);
    // Valores conhecidos das somas de verificação.
    const std::string nine = "123456789";
    const Bytes b(nine.begin(), nine.end());
    CHECK(cnx::crc32(b) == 0xCBF43926u);
    const std::string wiki = "Wikipedia";
    CHECK(cnx::adler32(Bytes(wiki.begin(), wiki.end())) == 0x11E60398u);
    CHECK(cnx::adler32({}) == 1u && cnx::crc32({}) == 0u);
    // CRC incremental = CRC do todo.
    CHECK(cnx::crc32(Bytes(b.begin() + 4, b.end()), cnx::crc32(Bytes(b.begin(), b.begin() + 4))) == 0xCBF43926u);
}

// ================================================================ fluxos inválidos

CNX_TEST(invalid_zlib_and_deflate_streams_are_rejected_with_the_reason) {
    const Bytes text = readBytes(kData / "zlib_text_l9.raw");
    const Bytes good = readBytes(kData / "zlib_text_l9.z");
    const auto zerr = [&](Bytes z, std::size_t size) { return decodeError([&] { (void)cnx::inflateZlib(z, size); }); };
    CHECK(zerr(good, text.size()).empty());
    Bytes cm = good;
    cm[0] = 0x79;  // CM = 9
    CHECK(mentions(zerr(cm, text.size()), "DEFLATE"));
    Bytes fcheck = good;
    fcheck[1] ^= 1;
    CHECK(mentions(zerr(fcheck, text.size()), "FCHECK"));
    Bytes dict{0x78, 0xBB};  // FDICT = 1 (FCHECK válido: 0x78BB % 31 == 0)
    dict.insert(dict.end(), good.begin() + 2, good.end());
    CHECK(mentions(zerr(dict, text.size()), "dicionário"));
    Bytes adler = good;
    adler.back() ^= 0x40;
    CHECK(mentions(zerr(adler, text.size()), "Adler"));
    CHECK(mentions(zerr(Bytes(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(good.size() / 2)), text.size()), "truncados"));
    CHECK(mentions(zerr(good, text.size() - 1), "excede"));
    CHECK(mentions(zerr(good, text.size() + 1), "menores"));
    CHECK(mentions(zerr(Bytes{0x78, 0x01}, 0), "curto"));
    {
        // CINFO = 8 (janela de 64 KiB), com um FLG que satisfaz o FCHECK e sem dicionário.
        Bytes window = good;
        window[0] = 0x88;
        for (unsigned flg = 0; flg < 0x20; ++flg) {
            if ((0x8800u | flg) % 31u == 0) window[1] = static_cast<std::uint8_t>(flg);
        }
        CHECK(mentions(zerr(window, text.size()), "janela"));
    }
    // Blocos montados bit a bit (DEFLATE cru).
    const auto raw = [](const DeflateBits& d, std::size_t max = 100) { return decodeError([&] { (void)cnx::inflateRaw(d.out, max); }); };
    DeflateBits type3;
    type3.bits(1, 1);
    type3.bits(3, 2);
    CHECK(mentions(raw(type3), "tipo de bloco"));
    DeflateBits stored;
    stored.bits(1, 1);
    stored.bits(0, 2);
    stored.bits(0, 5);  // alinha
    stored.bits(4, 16);
    stored.bits(0x1234, 16);  // NLEN errado
    CHECK(mentions(raw(stored), "inconsistente"));
    // Huffman fixo: comprimento 3 (código 257 = 0000001, 7 bits) com distância 1 e saída vazia.
    DeflateBits far;
    far.bits(1, 1);
    far.bits(1, 2);
    far.huff(0x01, 7);
    far.huff(0x00, 5);
    CHECK(mentions(raw(far), "distância antes"));
    // Literal 'A' (0x41 → código 0x30 + 0x41, 8 bits) e o símbolo 286 (11000110, inválido).
    DeflateBits bad286;
    bad286.bits(1, 1);
    bad286.bits(1, 2);
    bad286.huff(0x30 + 0x41, 8);
    bad286.huff(0xC6, 8);
    CHECK(mentions(raw(bad286), "comprimento inválido"));
    // Distância 30 (código 11110): fora do código fixo, que só tem 30 distâncias.
    DeflateBits dist30;
    dist30.bits(1, 1);
    dist30.bits(1, 2);
    dist30.huff(0x30 + 0x41, 8);
    dist30.huff(0x01, 7);
    dist30.huff(30, 5);
    dist30.bits(0xFFFF, 16);  // bits a seguir: o decode bit a bit esgota os 15 comprimentos sem achar o código
    CHECK(mentions(raw(dist30), "código de Huffman inválido"));
    // Literal, depois EOB (0000000): "A"; e o mesmo sem EOB: truncado.
    DeflateBits ok;
    ok.bits(1, 1);
    ok.bits(1, 2);
    ok.huff(0x30 + 0x41, 8);
    ok.huff(0x00, 7);
    CHECK(cnx::inflateRaw(ok.out, 10) == Bytes{'A'});
    CHECK(mentions(raw(ok, 0), "excede"));
    // Dinâmico com os 19 comprimentos do código de comprimentos iguais a 1: superdimensionado.
    DeflateBits over;
    over.bits(1, 1);
    over.bits(2, 2);
    over.bits(0, 5);
    over.bits(0, 5);
    over.bits(15, 4);
    for (int i = 0; i < 19; ++i) over.bits(1, 3);
    CHECK(mentions(raw(over), "comprimentos incompleto ou superdimensionado"));
    // Dinâmico: HLIT = 287 símbolos (> 286).
    DeflateBits many;
    many.bits(1, 1);
    many.bits(2, 2);
    many.bits(30, 5);
    many.bits(0, 5);
    many.bits(0, 4);
    CHECK(mentions(raw(many), "contagem"));
    // Repetição (16) sem comprimento anterior. Código dos comprimentos: só os símbolos 16 e 0, ambos com
    // comprimento 1 (completo); canônico: 0 → "0", 16 → "1". O primeiro símbolo lido é 16.
    DeflateBits rep0;
    rep0.bits(1, 1);
    rep0.bits(2, 2);
    rep0.bits(0, 5);
    rep0.bits(0, 5);
    rep0.bits(0, 4);
    rep0.bits(1, 3);
    rep0.bits(0, 3);
    rep0.bits(0, 3);
    rep0.bits(1, 3);
    rep0.huff(1, 1);  // 16 como primeiro comprimento
    CHECK(mentions(raw(rep0), "repetição sem comprimento anterior"));
    // Código de literais incompleto que não é o caso permitido: só o EOB, com comprimento 2. Código dos
    // comprimentos: os símbolos 0 e 2, comprimento 1 cada (0 → "0", 2 → "1"); HCLEN = 16 cobre a posição de 2.
    DeflateBits incomplete;
    incomplete.bits(1, 1);
    incomplete.bits(2, 2);
    incomplete.bits(0, 5);
    incomplete.bits(0, 5);
    incomplete.bits(12, 4);
    for (int i = 0; i < 16; ++i) incomplete.bits(i == 3 || i == 15 ? 1 : 0, 3);  // ordem 16 17 18 0 8 7 9 6 10 5 11 4 12 3 13 2
    for (int i = 0; i < 256; ++i) incomplete.huff(0, 1);  // literais 0..255: comprimento 0
    incomplete.huff(1, 1);                                // EOB: comprimento 2
    incomplete.huff(0, 1);                                // a única distância: 0
    CHECK(mentions(raw(incomplete), "código de literais inválido"));
}

CNX_TEST(invalid_png_files_are_rejected_with_the_reason) {
    const auto perr = [](const Bytes& b, cnx::DecodeOptions o = {}) { return decodeError([&] { (void)cnx::decodePng(b, o); }); };
    // Um PNG RGB 2×2 montado aqui: decodifica.
    const Bytes rows{0, 255, 0, 0, 0, 255, 0, 1, 0, 0, 255, 10, 10, 10};  // linha 0 filtro 0, linha 1 filtro Sub
    const Bytes png = makePng(2, 2, 8, 2, rows);
    const cnx::SourceImage ok = cnx::decodePng(png);
    CHECK((ok.pixels(0, 0) == sla::Rgba8{255, 0, 0, 255} && ok.pixels(1, 0) == sla::Rgba8{0, 255, 0, 255}));
    CHECK((ok.pixels(0, 1) == sla::Rgba8{0, 0, 255, 255} && ok.pixels(1, 1) == sla::Rgba8{10, 10, 9, 255}));  // Sub: (10,10,10) + (0,0,255) mod 256
    CHECK(!ok.hasAlpha && ok.dpiX == 72.0);
    // Assinatura, IHDR, CRC crítico e ancilar.
    Bytes sig = png;
    sig[1] = 'Q';
    CHECK(mentions(perr(sig), "assinatura"));
    Bytes crc = png;
    crc[29] ^= 1;  // CRC do IHDR
    CHECK(mentions(perr(crc), "CRC errado no chunk IHDR"));
    Bytes body = png;
    body[16] ^= 1;  // dentro do IHDR (largura), CRC agora errado
    CHECK(mentions(perr(body), "CRC"));
    CHECK(mentions(perr(makePng(0, 2, 8, 2, rows)), "dimensões"));
    CHECK(mentions(perr(makePng(2, 2, 3, 2, rows)), "combinação"));
    CHECK(mentions(perr(makePng(2, 2, 16, 3, rows)), "combinação"));
    CHECK(mentions(perr(makePng(2, 2, 8, 5, rows)), "combinação"));
    // Truncado, dados a mais e a menos, filtro inválido.
    CHECK(mentions(perr(Bytes(png.begin(), png.end() - 20)), "trunc"));
    Bytes extraRow = rows;
    extraRow.push_back(0);
    CHECK(mentions(perr(makePng(2, 2, 8, 2, extraRow)), "excede"));
    CHECK(mentions(perr(makePng(2, 2, 8, 2, Bytes(rows.begin(), rows.end() - 1))), "menores"));
    Bytes filter5 = rows;
    filter5[7] = 5;
    CHECK(mentions(perr(makePng(2, 2, 8, 2, filter5)), "filtro inválido"));
    // Chunks: crítico desconhecido, PLTE ausente, índice fora da paleta, IDAT não consecutivos, PLTE em cinza.
    CHECK(mentions(perr(makePng(2, 2, 8, 2, rows, {pngChunk("ABCD", {1, 2, 3})})), "crítico desconhecido ABCD"));
    CHECK(perr(makePng(2, 2, 8, 2, rows, {pngChunk("abCd", {1, 2, 3}), pngChunk("tEXt", {'a', 0, 'b'})})).empty());
    const Bytes idx{0, 0, 1, 0, 2, 1};  // paleta 8 bits, 2×2: índices 0 1 / 2 1
    CHECK(mentions(perr(makePng(2, 2, 8, 3, idx)), "sem PLTE"));
    CHECK(mentions(perr(makePng(2, 2, 8, 3, idx, {pngChunk("PLTE", {1, 2, 3, 4, 5, 6})})), "fora da paleta"));
    const cnx::SourceImage pal = cnx::decodePng(makePng(2, 2, 8, 3, idx, {pngChunk("PLTE", {1, 2, 3, 4, 5, 6, 7, 8, 9}), pngChunk("tRNS", {0, 128})}));
    CHECK((pal.pixels(0, 0) == sla::Rgba8{1, 2, 3, 0} && pal.pixels(1, 0) == sla::Rgba8{4, 5, 6, 128} && pal.pixels(0, 1) == sla::Rgba8{7, 8, 9, 255} && pal.hasAlpha));
    CHECK(mentions(perr(makePng(2, 2, 8, 3, idx, {pngChunk("PLTE", {1, 2, 3}), pngChunk("tRNS", {0, 1})})), "tRNS"));
    {
        // Chave tRNS de RGB: só o pixel com os três canais iguais à chave fica transparente.
        const Bytes key{0, 10, 0, 20, 0, 30};
        const cnx::SourceImage k = cnx::decodePng(makePng(3, 1, 8, 2, Bytes{0, 10, 20, 30, 10, 99, 30, 10, 20, 31}, {pngChunk("tRNS", key)}));
        CHECK(k.pixels(0, 0).a == 0 && k.pixels(1, 0).a == 255 && k.pixels(2, 0).a == 255 && k.hasAlpha);
    }
    CHECK(mentions(perr(makePng(2, 2, 8, 0, Bytes{0, 1, 2, 0, 3, 4}, {pngChunk("PLTE", {1, 2, 3})})), "PLTE numa imagem em tons de cinza"));
    {
        // IDAT, tEXt, IDAT.
        Bytes split{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
        Bytes ihdr;
        put32(ihdr, 2);
        put32(ihdr, 2);
        ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
        const Bytes z = zlibStored(rows);
        for (const Bytes& c : {pngChunk("IHDR", ihdr), pngChunk("IDAT", Bytes(z.begin(), z.begin() + 5)), pngChunk("tEXt", {'a', 0, 'b'}),
                               pngChunk("IDAT", Bytes(z.begin() + 5, z.end())), pngChunk("IEND", {})}) {
            split.insert(split.end(), c.begin(), c.end());
        }
        CHECK(mentions(perr(split), "não consecutivos"));
    }
    // Limite de pixels, conferido no IHDR (antes de alocar ou descomprimir).
    cnx::DecodeOptions small;
    small.maxPixels = 3;
    CHECK(mentions(perr(png, small), "limite"));
    CHECK(mentions(perr(makePng(70000, 70000, 8, 2, rows)), "limite"));
}

CNX_TEST(invalid_jpeg_files_are_rejected_with_the_reason) {
    const Bytes good = readBytes(kData / "jpeg_ycc420.jpg");
    const auto jerr = [](const Bytes& b, cnx::DecodeOptions o = {}) { return decodeError([&] { (void)cnx::decodeJpeg(b, o); }); };
    CHECK(jerr(good).empty());
    CHECK(mentions(jerr(Bytes(good.begin() + 2, good.end())), "SOI"));
    const std::size_t sof = findMarker(good, 0xC0);
    for (const auto& [marker, why] : std::vector<std::pair<int, const char*>>{
             {0xC3, "sem perdas"}, {0xC5, "hierárquico"}, {0xC9, "aritmética"}, {0xCA, "aritmética"}}) {
        Bytes b = good;
        b[sof + 1] = static_cast<std::uint8_t>(marker);
        CHECK_CTX(mentions(jerr(b), why), why);
    }
    Bytes p12 = good;
    p12[sof + 4] = 12;
    CHECK(mentions(jerr(p12), "12 bits"));
    Bytes dnl = good;
    dnl[sof + 5] = 0;
    dnl[sof + 6] = 0;
    CHECK(mentions(jerr(dnl), "DNL"));
    CHECK(mentions(jerr(withoutSegment(good, 0xC4)), "Huffman"));
    {
        // DHT superdimensionado: 3 códigos de comprimento 1 (o total de valores fica igual).
        Bytes over = good;
        const std::size_t at = findMarker(over, 0xC4);
        std::uint8_t* counts = &over[at + 5];
        int delta = 3 - counts[0];
        counts[0] = 3;
        for (int j = 15; j > 0 && delta > 0; --j) {
            const int take = std::min<int>(delta, counts[j]);
            counts[j] = static_cast<std::uint8_t>(counts[j] - take);
            delta -= take;
        }
        CHECK(mentions(jerr(over), "superdimensionada"));
    }
    {
        // A tabela AC 0 levada para o destino 3: o scan cita uma tabela AC que não existe.
        Bytes noAc = good;
        const std::size_t at = findMarker(noAc, 0xC4);
        const std::size_t end = at + 2 + ((static_cast<std::size_t>(noAc[at + 2]) << 8) | noAc[at + 3]);
        for (std::size_t p = at + 4; p < end;) {
            int total = 0;
            for (std::size_t i = 0; i < 16; ++i) total += noAc[p + 1 + i];
            if (noAc[p] == 0x10) noAc[p] = 0x13;
            p += 17 + static_cast<std::size_t>(total);
        }
        CHECK(mentions(jerr(noAc), "AC ausente"));
    }
    CHECK(mentions(jerr(withoutSegment(good, 0xDB)), "quantização"));
    CHECK(mentions(jerr(withoutSegment(good, 0xC0)), "SOS antes do SOF"));
    // Truncado no meio dos dados entrópicos.
    const std::size_t sos = findMarker(good, 0xDA);
    CHECK(mentions(jerr(Bytes(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(sos + (good.size() - sos) / 2))), "truncados"));
    // Componente do SOS que não está no quadro.
    Bytes comp = good;
    comp[sos + 5] = 99;
    CHECK(mentions(jerr(comp), "componente"));
    // Dois quadros.
    {
        Bytes two = good;
        const std::size_t len = (static_cast<std::size_t>(good[sof + 2]) << 8) | good[sof + 3];
        two.insert(two.begin() + static_cast<std::ptrdiff_t>(sof), good.begin() + static_cast<std::ptrdiff_t>(sof), good.begin() + static_cast<std::ptrdiff_t>(sof + 2 + len));
        CHECK(mentions(jerr(two), "mais de um quadro"));
    }
    // Reinício: sem o marcador RST esperado.
    {
        Bytes r = readBytes(kData / "jpeg_ycc420_restart.jpg");
        CHECK(jerr(r).empty());
        const std::size_t rst = findMarker(r, 0xD0);
        CHECK(rst < r.size());
        r.erase(r.begin() + static_cast<std::ptrdiff_t>(rst), r.begin() + static_cast<std::ptrdiff_t>(rst + 2));
        CHECK(mentions(jerr(r), "reinício"));
    }
    // Limite de pixels.
    cnx::DecodeOptions small;
    small.maxPixels = 100;
    CHECK(mentions(jerr(good, small), "limite"));
    // Formato desconhecido.
    CHECK(mentions(decodeError([] { (void)cnx::decodeImage(Bytes{'G', 'I', 'F', '8', '9', 'a'}); }), "não reconhecido"));
    CHECK(cnx::sniffImageFormat(good) == cnx::ImageFormat::Jpeg && !cnx::sniffImageFormat(Bytes{0xFF, 0xD8}) && !cnx::sniffImageFormat(Bytes{0xFF, 0xD8, 0x00}));
    CHECK(!cnx::sniffImageFormat(Bytes{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x00}));
}

CNX_TEST(corrupted_files_fail_cleanly_and_never_crash) {
    // Fuzz determinístico: bytes trocados, cortes e marcadores FF inseridos nos arquivos de referência. Só
    // ImageDecodeError é aceito (no Debug, um índice fora dos limites aborta pela asserção da biblioteca).
    std::uint64_t state = 0xC0FFEEull;
    const auto next = [&] {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    int decoded = 0;
    int rejected = 0;
    int other = 0;
    for (const char* f : {"png_rgba8_adam7.png", "png_pal2_trns.png", "png_gray16_trns.png", "jpeg_ycc420.jpg", "jpeg_ycc420_progressive_restart.jpg",
                          "jpeg_gray_progressive.jpg", "jpeg_cmyk.jpg", "jpeg_orient6_dpi.jpg", "gdiplus_photo.jpg"}) {
        const Bytes original = readBytes(kData / f);
        for (int trial = 0; trial < 400; ++trial) {
            Bytes b = original;
            const int edits = 1 + static_cast<int>(next() % 4);
            for (int e = 0; e < edits && !b.empty(); ++e) {
                const std::size_t at = static_cast<std::size_t>(next() % b.size());
                switch (next() % 4) {
                    case 0: b[at] = static_cast<std::uint8_t>(next()); break;
                    case 1: b[at] ^= static_cast<std::uint8_t>(1u << (next() % 8)); break;
                    case 2: b.resize(std::max<std::size_t>(at, 8)); break;
                    default: b.insert(b.begin() + static_cast<std::ptrdiff_t>(at), 0xFF); break;
                }
            }
            try {
                cnx::DecodeOptions o;
                o.maxPixels = 1u << 22;  // um IHDR/SOF corrompido não pode pedir gigabytes
                (void)cnx::decodeImage(b, o);
                ++decoded;
            } catch (const cnx::ImageDecodeError&) {
                ++rejected;
            } catch (...) {
                ++other;
            }
        }
    }
    std::printf("  fuzz: %d decodificados, %d recusados, %d outras exceções\n", decoded, rejected, other);
    CHECK(other == 0 && rejected > 1000);
}

// ================================================================ carregador, cancelamento, motor

CNX_TEST(the_loader_uses_the_signature_honors_limits_and_cancellation) {
    const fs::path tmp = fs::temp_directory_path() / "cnx_image_decoder_test";
    fs::create_directories(tmp);
    // Um PNG com extensão .jpg é lido como PNG (o classificador recebe ".png").
    fs::copy_file(kData / "png_rgb8.png", tmp / "disfarcado.jpg", fs::copy_options::overwrite_existing);
    cnx::ImageFileLoader loader;
    const cnx::SourceImage s = loader.load(tmp / "disfarcado.jpg", {});
    CHECK(s.formatHint == ".png" && s.pixels.width() == 17);
    CHECK(mentions(decodeError([&] { (void)loader.load(tmp / "nao_existe.png", {}); }), "não foi possível"));
    cnx::ImageFileLoader tiny(cnx::ImageFileLoader::Limits{100, std::uint64_t{1} << 28});
    CHECK(mentions(decodeError([&] { (void)tiny.load(kData / "png_rgba8_large.png", {}); }), "limite"));
    cnx::ImageFileLoader fewPixels(cnx::ImageFileLoader::Limits{std::uintmax_t{1} << 30, 1000});
    CHECK(mentions(decodeError([&] { (void)fewPixels.load(kData / "png_rgba8_large.png", {}); }), "limite"));
    // Cancelamento: na leitura, no PNG, no JPEG e no inflate (≥ 64 KiB).
    std::stop_source stop;
    stop.request_stop();
    for (const char* f : {"png_rgba8_large.png", "jpeg_ycc420.jpg"}) {
        bool cancelled = false;
        try {
            (void)loader.load(kData / f, stop.get_token());
        } catch (const cnx::CancelTaskException&) {
            cancelled = true;
        }
        CHECK_CTX(cancelled, f);
    }
    cnx::DecodeOptions opt;
    opt.stop = stop.get_token();
    bool cancelled = false;
    try {
        (void)cnx::decodePng(readBytes(kData / "png_rgba8_large.png"), opt);
    } catch (const cnx::CancelTaskException&) {
        cancelled = true;
    }
    CHECK(cancelled);
    cancelled = false;
    try {
        (void)cnx::inflateZlib(readBytes(kData / "zlib_zeros.z"), 70000, stop.get_token());
    } catch (const cnx::CancelTaskException&) {
        cancelled = true;
    }
    CHECK(cancelled);
    fs::remove_all(tmp);
}

CNX_TEST(the_core_engine_vectorizes_a_png_file_read_by_the_loader) {
    // Logo chapado 48×36 (fundo branco, retângulo vermelho, retângulo azul) gravado como PNG RGB com Paeth.
    const int w = 48;
    const int h = 36;
    Bytes filtered;
    std::vector<std::uint8_t> prior(static_cast<std::size_t>(3 * w), 0);
    for (int y = 0; y < h; ++y) {
        std::vector<std::uint8_t> row;
        for (int x = 0; x < w; ++x) {
            std::array<int, 3> c{255, 255, 255};
            if (x >= 5 && x < 24 && y >= 4 && y < 18) c = {220, 30, 30};
            if (x >= 26 && x < 43 && y >= 20 && y < 32) c = {30, 30, 220};
            for (int k : c) row.push_back(static_cast<std::uint8_t>(k));
        }
        filtered.push_back(4);
        for (std::size_t i = 0; i < row.size(); ++i) {
            const int a = i >= 3 ? row[i - 3] : 0;
            const int b = prior[i];
            const int cc = i >= 3 ? prior[i - 3] : 0;
            const int p = a + b - cc;
            const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - cc);
            const int pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : cc;
            filtered.push_back(static_cast<std::uint8_t>(row[i] - pred));
        }
        prior = row;
    }
    Bytes phys;
    put32(phys, 11811);
    put32(phys, 11811);
    phys.push_back(1);
    const Bytes png = makePng(w, h, 8, 2, filtered, {pngChunk("pHYs", phys)});
    const fs::path file = fs::temp_directory_path() / "cnx_logo_test.png";
    {
        std::ofstream o(file, std::ios::binary);
        o.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    }
    cnx::CoreEngine engine(std::make_shared<cnx::ImageFileLoader>());
    engine.setInput(file);
    cnx::EngineParameters p = engine.parameters();
    p.smooth.maxCgIterations = 40;
    engine.setParameters(p);
    engine.start(cnx::Stage::Output);
    CHECK(engine.wait() == cnx::RunStatus::Completed);
    const auto snap = engine.snapshot();
    CHECK(snap && snap->source && snap->source->formatHint == ".png" && std::abs(snap->source->dpiX - 299.9994) < 1e-3);
    CHECK(snap->classification->profile == cnx::ImageProfile::LogoMed && snap->palette->exact && snap->palette->colors.size() == 3);
    CHECK(snap->segmentation->forest().rootCount() == 3 && snap->fitted->hasCurves());
    // Arquivo inválido: a execução falha com a mensagem do decodificador, e o motor continua usável.
    {
        std::ofstream o(file, std::ios::binary);
        o << "isto não é uma imagem";
    }
    engine.start(cnx::Stage::Output);
    CHECK(engine.wait() == cnx::RunStatus::Failed && mentions(engine.lastMessage(), "não reconhecido"));
    fs::remove(file);
}

int main() { return cnxtest::runAll(); }
