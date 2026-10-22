#!/usr/bin/env python3
"""Gera os arquivos de referência dos decodificadores de imagem (tests/data, docs §1.23).

Tudo aqui é independente do código C++ do motor: um codificador PNG (todos os tipos de cor e profundidades, os
5 filtros, Adam7, IDAT repartido, chunks ancilares) e um codificador JPEG (DCT direta, tabelas de Huffman
ótimas pelo anexo K.2, baseline, scans separados, reinícios e progressivo com aproximações sucessivas, pelo
algoritmo de codificação do libjpeg), mais EXIF, JFIF e Adobe. O "esperado" de cada arquivo:
  - PNG: as amostras de origem convertidas para RGBA de 8 bits pelas regras da norma;
  - JPEG: a reconstrução dos coeficientes quantizados que o próprio codificador produziu (IDCT em double,
    croma por interpolação linear centrada, YCbCr do JFIF), e a orientação EXIF aplicada por rotações.
O zlib do Python gera os fluxos DEFLATE (níveis e estratégias diferentes).

Saída: <nome>.png/.jpg/.z/.deflate, <nome>.rgba (ou .raw) e manifest.txt, uma linha por caso:
    png|jpeg <nome> <largura> <altura> <dpiX> <dpiY> <temAlfa> <tolerância>
    zlib|deflate <nome> <tamanho descomprimido>
Uso: python -I make_fixtures.py   (determinístico: os mesmos bytes a cada execução)
"""

import math
import os
import struct
import zlib

OUT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = []


class Rng:
    """splitmix64, o mesmo gerador dos testes C++ (só para ter dados reproduzíveis)."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n


def write(name, data):
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)


# =============================================================================================== PNG

def chunk(kind, body, corrupt=False):
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    if corrupt:
        crc ^= 0x1
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def pack_row(samples, depth):
    if depth == 8:
        return bytes(samples)
    if depth == 16:
        return b"".join(struct.pack(">H", s) for s in samples)
    out = bytearray()
    acc = 0
    nbits = 0
    for s in samples:
        acc = (acc << depth) | s
        nbits += depth
        if nbits == 8:
            out.append(acc)
            acc = 0
            nbits = 0
    if nbits:
        out.append(acc << (8 - nbits))
    return bytes(out)


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_row(kind, row, prior, bpp):
    out = bytearray()
    for i, x in enumerate(row):
        a = row[i - bpp] if i >= bpp else 0
        b = prior[i]
        c = prior[i - bpp] if i >= bpp else 0
        pred = [0, a, b, (a + b) // 2, paeth(a, b, c)][kind]
        out.append((x - pred) & 0xFF)
    return bytes([kind]) + bytes(out)


def encode_png_data(samples, w, h, color, depth, interlace):
    """samples[y][x] = tupla de amostras do pixel. Filtro da linha r de cada passada = r mod 5."""
    ch = CHANNELS[color]
    bpp = max(1, ch * depth // 8)
    raw = bytearray()
    passes = ADAM7 if interlace else [(0, 0, 1, 1)]
    for x0, y0, dx, dy in passes:
        xs = list(range(x0, w, dx))
        ys = list(range(y0, h, dy))
        if not xs or not ys:
            continue
        prior = bytes(len(pack_row([0] * (len(xs) * ch), depth)))
        for r, y in enumerate(ys):
            flat = [s for x in xs for s in samples[y][x]]
            row = pack_row(flat, depth)
            raw += filter_row(r % 5, row, prior, bpp)
            prior = row
    return bytes(raw)


def to8(v, depth):
    if depth == 16:
        return (v * 255 + 32767) // 65535
    if depth == 8:
        return v
    return v * 255 // ((1 << depth) - 1)


def png_case(name, w, h, color, depth, seed, interlace=False, palette=None, trns=None, phys=None, level=6, strategy=zlib.Z_DEFAULT_STRATEGY,
             idat_split=None, extra=(), corrupt_text=False):
    rng = Rng(seed)
    ch = CHANNELS[color]
    maxv = (1 << depth) - 1
    nindex = len(palette) if palette else 0
    samples = [[tuple(rng.below(nindex) if color == 3 else rng.below(maxv + 1) for _ in range(ch)) for _ in range(w)] for _ in range(h)]
    # Garante pixels com a chave de transparência, se houver.
    if trns is not None and color in (0, 2):
        samples[0][0] = tuple(trns)
        samples[h - 1][w - 1] = tuple(trns)
    expected = bytearray()
    for y in range(h):
        for x in range(w):
            s = samples[y][x]
            if color == 0:
                g = to8(s[0], depth)
                a = 0 if trns is not None and s[0] == trns[0] else 255
                expected += bytes([g, g, g, a])
            elif color == 2:
                a = 0 if trns is not None and tuple(s) == tuple(trns) else 255
                expected += bytes([to8(s[0], depth), to8(s[1], depth), to8(s[2], depth), a])
            elif color == 3:
                r, g, b = palette[s[0]]
                a = trns[s[0]] if trns is not None and s[0] < len(trns) else 255
                expected += bytes([r, g, b, a])
            elif color == 4:
                g = to8(s[0], depth)
                expected += bytes([g, g, g, to8(s[1], depth)])
            else:
                expected += bytes(to8(v, depth) for v in s)
    raw = encode_png_data(samples, w, h, color, depth, interlace)
    comp = zlib.compressobj(level, zlib.DEFLATED, 15, 9, strategy)
    z = comp.compress(raw) + comp.flush()
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, 1 if interlace else 0))
    out += chunk(b"gAMA", struct.pack(">I", 45455))
    out += chunk(b"tEXt", b"Comment\x00cnx fixture", corrupt=corrupt_text)
    if palette:
        out += chunk(b"PLTE", b"".join(bytes(c) for c in palette))
    if trns is not None:
        if color == 0:
            out += chunk(b"tRNS", struct.pack(">H", trns[0]))
        elif color == 2:
            out += chunk(b"tRNS", struct.pack(">HHH", *trns))
        else:
            out += chunk(b"tRNS", bytes(trns))
    dpi = (72.0, 72.0)
    if phys:
        out += chunk(b"pHYs", struct.pack(">IIB", phys[0], phys[1], 1))
        dpi = (phys[0] * 0.0254, phys[1] * 0.0254)
    for kind, body in extra:
        out += chunk(kind, body)
    if idat_split:
        parts = [z[i:i + idat_split] for i in range(0, len(z), idat_split)]
        out += chunk(b"IDAT", b"")  # IDAT vazio também é válido
        for p in parts:
            out += chunk(b"IDAT", p)
    else:
        out += chunk(b"IDAT", z)
    out += chunk(b"IEND", b"")
    write(name + ".png", out)
    write(name + ".rgba", bytes(expected))
    alpha = 1 if color in (4, 6) or trns is not None else 0
    MANIFEST.append("png %s %d %d %.6f %.6f %d 0" % (name, w, h, dpi[0], dpi[1], alpha))


def make_pngs():
    pal16 = [(17 * i, 255 - 13 * i, (37 * i) % 256) for i in range(16)]
    pal4 = pal16[:4]
    pal2 = [(0, 0, 0), (250, 120, 10)]
    pal200 = [((i * 7) % 256, (i * 11) % 256, (i * 13) % 256) for i in range(200)]
    for d in (1, 2, 4, 8, 16):
        png_case("png_gray%d" % d, 13, 7, 0, d, 100 + d)
    png_case("png_gray4_trns", 11, 6, 0, 4, 201, trns=(9,))
    png_case("png_gray16_trns", 9, 5, 0, 16, 202, trns=(40000,))
    png_case("png_rgb8", 17, 9, 2, 8, 203)
    png_case("png_rgb16", 10, 6, 2, 16, 204)
    png_case("png_rgb8_trns", 12, 8, 2, 8, 205, trns=(10, 20, 30))
    png_case("png_rgb16_trns", 7, 7, 2, 16, 206, trns=(1000, 2000, 65535))
    png_case("png_pal1", 19, 5, 3, 1, 207, palette=pal2)
    png_case("png_pal2_trns", 14, 6, 3, 2, 208, palette=pal4, trns=(0, 128))  # tRNS mais curto que a paleta
    png_case("png_pal4", 15, 7, 3, 4, 209, palette=pal16)
    png_case("png_pal8_trns", 21, 11, 3, 8, 210, palette=pal200, trns=tuple((i * 5) % 256 for i in range(150)))
    png_case("png_graya8", 9, 9, 4, 8, 211)
    png_case("png_graya16", 8, 5, 4, 16, 212)
    png_case("png_rgba8", 23, 13, 6, 8, 213)
    png_case("png_rgba16", 6, 9, 6, 16, 214)
    # Adam7, inclusive tamanhos com passadas vazias.
    png_case("png_rgba8_adam7", 33, 19, 6, 8, 215, interlace=True)
    png_case("png_pal2_adam7", 13, 11, 3, 2, 216, palette=pal4, interlace=True)
    png_case("png_gray16_adam7", 9, 10, 0, 16, 217, interlace=True)
    png_case("png_gray1_adam7", 5, 3, 0, 1, 218, interlace=True)
    png_case("png_rgb8_adam7_1x1", 1, 1, 2, 8, 219, interlace=True)
    png_case("png_rgba8_adam7_3x2", 3, 2, 6, 8, 220, interlace=True)
    png_case("png_rgb8_1x1", 1, 1, 2, 8, 221)
    # Compressão: armazenado, Huffman fixo, só Huffman, RLE, nível 9; IDAT repartido; pHYs; ancilar corrompido.
    png_case("png_rgb8_stored", 31, 17, 2, 8, 222, level=0)
    png_case("png_rgb8_fixed", 31, 17, 2, 8, 223, strategy=zlib.Z_FIXED)
    png_case("png_rgb8_huffonly", 31, 17, 2, 8, 224, strategy=zlib.Z_HUFFMAN_ONLY)
    png_case("png_rgb8_rle", 31, 17, 2, 8, 225, strategy=zlib.Z_RLE)
    png_case("png_rgba8_split", 40, 30, 6, 8, 226, level=9, idat_split=97)
    png_case("png_rgb8_phys", 12, 12, 2, 8, 227, phys=(11811, 5906))  # ≈ 300 × 150 dpi
    png_case("png_rgb8_badtext", 8, 8, 2, 8, 228, corrupt_text=True)
    png_case("png_rgba8_ancillary", 10, 10, 6, 8, 229, extra=[(b"zzZz", b"desconhecido mas ancilar"), (b"tIME", bytes(7))])
    png_case("png_rgba8_large", 300, 200, 6, 8, 230, level=9)


# =============================================================================================== zlib

def make_zlib():
    rng = Rng(300)
    text = (b"vectorization " * 600) + bytes(rng.below(256) for _ in range(5000))
    zeros = bytes(70000)
    noise = bytes(rng.below(256) for _ in range(40000))
    runs = b"".join(bytes([rng.below(4)]) * (1 + rng.below(300)) for _ in range(400))
    cases = [
        ("zlib_empty", b"", 6, zlib.Z_DEFAULT_STRATEGY, 15),
        ("zlib_text_l1", text, 1, zlib.Z_DEFAULT_STRATEGY, 15),
        ("zlib_text_l9", text, 9, zlib.Z_DEFAULT_STRATEGY, 15),
        ("zlib_zeros", zeros, 9, zlib.Z_DEFAULT_STRATEGY, 15),
        ("zlib_noise_stored", noise, 0, zlib.Z_DEFAULT_STRATEGY, 15),
        ("zlib_noise_l6", noise, 6, zlib.Z_DEFAULT_STRATEGY, 15),
        ("zlib_runs_fixed", runs, 6, zlib.Z_FIXED, 15),
        ("zlib_runs_rle", runs, 6, zlib.Z_RLE, 15),
        ("zlib_text_huffonly", text, 6, zlib.Z_HUFFMAN_ONLY, 15),
        ("zlib_text_window9", text, 9, zlib.Z_DEFAULT_STRATEGY, 9),
    ]
    for name, data, level, strategy, wbits in cases:
        c = zlib.compressobj(level, zlib.DEFLATED, wbits, 8, strategy)
        write(name + ".z", c.compress(data) + c.flush())
        write(name + ".raw", data)
        MANIFEST.append("zlib %s %d" % (name, len(data)))
    c = zlib.compressobj(9, zlib.DEFLATED, -15, 8, zlib.Z_DEFAULT_STRATEGY)
    write("deflate_text.deflate", c.compress(text) + c.flush())
    write("deflate_text.raw", text)
    MANIFEST.append("deflate deflate_text %d" % len(text))


# =============================================================================================== JPEG

ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
          35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
LUMA_Q = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
          18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
CHROMA_Q = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32
COS = [[(math.sqrt(0.5) if u == 0 else 1.0) * 0.5 * math.cos((2 * x + 1) * u * math.pi / 16) for u in range(8)] for x in range(8)]


def scaled_table(base, quality):
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return [min(255, max(1, (b * s + 50) // 100)) for b in base]


def fdct(block):
    """block[y][x] (já com −128) → F[v][u] em ordem natural, pela definição (separável)."""
    tmp = [[sum(COS[x][u] * block[y][x] for x in range(8)) for u in range(8)] for y in range(8)]
    return [sum(COS[y][v] * tmp[y][u] for y in range(8)) for v in range(8) for u in range(8)]


def idct(F):
    """F em ordem natural → amostras 8×8 (com +128, arredondadas e limitadas): a definição da norma."""
    tmp = [[sum(COS[x][u] * F[v * 8 + u] for u in range(8)) for x in range(8)] for v in range(8)]
    out = []
    for y in range(8):
        row = []
        for x in range(8):
            s = sum(COS[y][v] * tmp[v][x] for v in range(8))
            row.append(min(255, max(0, math.floor(s + 128.0 + 0.5))))
        out.append(row)
    return out


def source_image(w, h, seed):
    """Imagem suave com formas (para o JPEG ter AC, mas boa reconstrução): RGB."""
    rng = Rng(seed)
    cx, cy = w * 0.4, h * 0.55
    img = []
    for y in range(h):
        row = []
        for x in range(w):
            r = 128 + 100 * math.sin(x * 0.21 + y * 0.05)
            g = 40 + 3.0 * y + 60 * math.cos(x * 0.13)
            b = 200 - 2.5 * x
            if (x - cx) ** 2 + (y - cy) ** 2 < (min(w, h) * 0.3) ** 2:
                r, g, b = 230, 40, 50
            if (x // 6 + y // 6) % 5 == 0:
                g += 40
            row.append(tuple(min(255, max(0, int(v) + rng.below(5) - 2)) for v in (r, g, b)))
        img.append(row)
    return img


def rgb_to_ycc(p):
    r, g, b = p
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return (y, cb, cr)


class Comp:
    def __init__(self, cid, h, v, tq):
        self.id, self.h, self.v, self.tq = cid, h, v, tq


def build_planes(img, w, h, comps, space):
    """Planos de amostras de cada componente na resolução própria (média de caixa para o croma)."""
    hmax = max(c.h for c in comps)
    vmax = max(c.v for c in comps)
    if space == "ycc":
        full = [[rgb_to_ycc(p) for p in row] for row in img]
    elif space == "rgb":
        full = [[tuple(float(v) for v in p) for p in row] for row in img]
    elif space == "gray":
        full = [[(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2],) for p in row] for row in img]
    elif space == "cmyk":  # convenção Adobe, valores invertidos: armazenado = 255 − tinta
        full = []
        for row in img:
            out = []
            for r, g, b in row:
                k = max(r, g, b)
                if k == 0:
                    out.append((0.0, 0.0, 0.0, 0.0))
                else:
                    out.append((r * 255.0 / k, g * 255.0 / k, b * 255.0 / k, float(k)))
            full.append(out)
    else:  # ycck: YCbCr de (255 − C, ...) ... armazena Y, Cb, Cr de RGB' = 255 − (C, M, Y) "não invertidos"
        full = []
        for row in img:
            out = []
            for r, g, b in row:
                k = max(r, g, b)
                c = (r * 255.0 / k, g * 255.0 / k, b * 255.0 / k) if k else (0.0, 0.0, 0.0)
                y, cb, cr = rgb_to_ycc(tuple(255.0 - v for v in c))
                out.append((y, cb, cr, float(k)))
            full.append(out)
    planes = []
    for ci, c in enumerate(comps):
        cw = (w * c.h + hmax - 1) // hmax
        chh = (h * c.v + vmax - 1) // vmax
        fx, fy = hmax // c.h, vmax // c.v
        plane = []
        for y in range(chh):
            row = []
            for x in range(cw):
                acc, n = 0.0, 0
                for yy in range(y * fy, min(h, (y + 1) * fy)):
                    for xx in range(x * fx, min(w, (x + 1) * fx)):
                        acc += full[yy][xx][ci]
                        n += 1
                row.append(acc / n)
            plane.append(row)
        planes.append(plane)
    return planes


def quantize_component(plane, c, mcus_x, mcus_y, q):
    """Coeficientes quantizados (ordem natural) da grade de blocos acolchoada (bordas replicadas)."""
    bw, bh = mcus_x * c.h, mcus_y * c.v
    ch, cw = len(plane), len(plane[0])
    blocks = []
    for by in range(bh):
        row = []
        for bx in range(bw):
            block = [[plane[min(ch - 1, by * 8 + y)][min(cw - 1, bx * 8 + x)] - 128.0 for x in range(8)] for y in range(8)]
            F = fdct(block)
            row.append([int(round(F[k] / q[k])) for k in range(64)])
        blocks.append(row)
    return blocks


def bitlen(v):
    return 0 if v == 0 else v.bit_length()


def huffman_lengths(freq):
    """Anexo K.2: comprimentos ótimos limitados a 16 bits, com o símbolo reservado 256."""
    freq = list(freq) + [1]
    others = [-1] * 257
    codesize = [0] * 257
    while True:
        v1 = -1
        for i in range(257):
            if freq[i] > 0 and (v1 < 0 or freq[i] <= freq[v1]):
                v1 = i
        v2 = -1
        for i in range(257):
            if i != v1 and freq[i] > 0 and (v2 < 0 or freq[i] <= freq[v2]):
                v2 = i
        if v2 < 0:
            break
        freq[v1] += freq[v2]
        freq[v2] = 0
        codesize[v1] += 1
        while others[v1] >= 0:
            v1 = others[v1]
            codesize[v1] += 1
        others[v1] = v2
        codesize[v2] += 1
        while others[v2] >= 0:
            v2 = others[v2]
            codesize[v2] += 1
    bits = [0] * 33
    for i in range(257):
        if codesize[i]:
            bits[codesize[i]] += 1
    for i in range(32, 16, -1):
        while bits[i] > 0:
            j = i - 2
            while bits[j] == 0:
                j -= 1
            bits[i] -= 2
            bits[i - 1] += 1
            bits[j + 1] += 2
            bits[j] -= 1
    i = 16
    while bits[i] == 0:
        i -= 1
    bits[i] -= 1
    values = []
    for size in range(1, 33):
        for s in range(256):
            if codesize[s] == size:
                values.append(s)
    return bits[1:17], values


def huffman_codes(bits, values):
    codes = {}
    code, k = 0, 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            codes[values[k]] = (code, length)
            code += 1
            k += 1
        code <<= 1
    return codes


class Tokens:
    """Sequência de símbolos de Huffman (por tabela) e bits crus de um scan, para tabelas ótimas em 2 passadas."""

    def __init__(self):
        self.items = []

    def sym(self, table, value):
        self.items.append(("s", table, value))

    def bits(self, value, n):
        if n:
            self.items.append(("b", value & ((1 << n) - 1), n))

    def restart(self, index):
        self.items.append(("r", index, 0))


class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, value, n):
        for i in range(n - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0)
                self.acc, self.n = 0, 0

    def flush(self):
        if self.n:
            self.put((1 << (8 - self.n)) - 1, 8 - self.n)  # enchimento com 1s


def serialize(tokens):
    """Tabelas ótimas por chave de tabela e os bytes entrópicos."""
    freqs = {}
    for kind, a, b in tokens.items:
        if kind == "s":
            freqs.setdefault(a, [0] * 256)[b] += 1
    tables = {key: huffman_lengths(f) for key, f in freqs.items()}
    codes = {key: huffman_codes(*t) for key, t in tables.items()}
    bw = BitWriter()
    for kind, a, b in tokens.items:
        if kind == "s":
            code, length = codes[a][b]
            bw.put(code, length)
        elif kind == "b":
            bw.put(a, b)
        else:
            bw.flush()
            bw.out += bytes([0xFF, 0xD0 + (a % 8)])
    bw.flush()
    return tables, bytes(bw.out)


def dht(tables):
    body = b""
    for (cls, ident), (bits, values) in sorted(tables.items()):
        body += bytes([(cls << 4) | ident]) + bytes(bits) + bytes(values)
    return b"\xff\xc4" + struct.pack(">H", 2 + len(body)) + body


class ScanEncoder:
    def __init__(self, comps, coefs, mcus_x, mcus_y, w, h, restart, progressive):
        self.comps, self.coefs = comps, coefs
        self.mcus_x, self.mcus_y, self.w, self.h = mcus_x, mcus_y, w, h
        self.restart = restart
        self.progressive = progressive

    def units(self, scan_comps):
        """Lista de MCUs: cada uma, lista de (índice do componente no scan, bloco)."""
        hmax = max(c.h for c in self.comps)
        vmax = max(c.v for c in self.comps)
        if len(scan_comps) == 1:
            ci = scan_comps[0]
            c = self.comps[ci]
            cw = (self.w * c.h + hmax - 1) // hmax
            chh = (self.h * c.v + vmax - 1) // vmax
            return [[(0, self.coefs[ci][by][bx])] for by in range((chh + 7) // 8) for bx in range((cw + 7) // 8)]
        out = []
        for my in range(self.mcus_y):
            for mx in range(self.mcus_x):
                mcu = []
                for i, ci in enumerate(scan_comps):
                    c = self.comps[ci]
                    for v in range(c.v):
                        for hh in range(c.h):
                            mcu.append((i, self.coefs[ci][my * c.v + v][mx * c.h + hh]))
                mcu_list = mcu
                out.append(mcu_list)
        return out

    def encode(self, scan_comps, ss, se, ah, al):
        t = Tokens()
        units = self.units(scan_comps)
        dc_key = lambda i: (0, 0 if scan_comps[i] == 0 else 1)
        ac_key = lambda i: (1, 0 if scan_comps[i] == 0 else 1)
        state = {"pred": [0] * len(scan_comps), "eobrun": 0, "be": []}

        def emit_eobrun(key):
            if state["eobrun"] > 0:
                n = bitlen(state["eobrun"]) - 1
                t.sym(key, n << 4)
                t.bits(state["eobrun"], n)
                state["eobrun"] = 0
                for bit in state["be"]:
                    t.bits(bit, 1)
                state["be"] = []

        last_key = [None]
        for u, mcu in enumerate(units):
            if self.restart and u > 0 and u % self.restart == 0:
                if last_key[0] is not None:
                    emit_eobrun(last_key[0])
                t.restart(u // self.restart - 1)
                state["pred"] = [0] * len(scan_comps)
            for i, blk in mcu:
                last_key[0] = ac_key(i)
                if not self.progressive:
                    self.sequential(t, blk, i, state, dc_key(i), ac_key(i))
                elif ss == 0:
                    if ah == 0:
                        v = blk[0] >> al  # deslocamento aritmético, como o libjpeg
                        diff = v - state["pred"][i]
                        state["pred"][i] = v
                        self.put_value(t, dc_key(i), diff, 0)
                    else:
                        t.bits((blk[0] >> al) & 1, 1)
                elif ah == 0:
                    self.ac_first(t, blk, ss, se, al, ac_key(i), state, emit_eobrun)
                else:
                    self.ac_refine(t, blk, ss, se, al, ac_key(i), state, emit_eobrun)
        if last_key[0] is not None:
            emit_eobrun(last_key[0])
        return t

    @staticmethod
    def put_value(t, key, v, run):
        n = bitlen(abs(v))
        t.sym(key, (run << 4) | n)
        t.bits(v if v >= 0 else v - 1, n)

    def sequential(self, t, blk, i, state, dkey, akey):
        diff = blk[0] - state["pred"][i]
        state["pred"][i] = blk[0]
        self.put_value(t, dkey, diff, 0)
        r = 0
        for k in range(1, 64):
            v = blk[ZIGZAG[k]]
            if v == 0:
                r += 1
                continue
            while r > 15:
                t.sym(akey, 0xF0)
                r -= 16
            self.put_value(t, akey, v, r)
            r = 0
        if r > 0:
            t.sym(akey, 0x00)

    def ac_first(self, t, blk, ss, se, al, key, state, emit_eobrun):
        r = 0
        for k in range(ss, se + 1):
            v = blk[ZIGZAG[k]]
            mag = abs(v) >> al
            if mag == 0:
                r += 1
                continue
            emit_eobrun(key)
            while r > 15:
                t.sym(key, 0xF0)
                r -= 16
            self.put_value(t, key, mag if v > 0 else -mag, r)
            r = 0
        if r > 0:
            state["eobrun"] += 1
            if state["eobrun"] == 0x7FFF:
                emit_eobrun(key)

    def ac_refine(self, t, blk, ss, se, al, key, state, emit_eobrun):
        absval = {k: abs(blk[ZIGZAG[k]]) >> al for k in range(ss, se + 1)}
        eob = 0
        for k in range(ss, se + 1):
            if absval[k] == 1:
                eob = k
        r = 0
        br = []
        for k in range(ss, se + 1):
            v = absval[k]
            if v == 0:
                r += 1
                continue
            while r > 15 and k <= eob:
                emit_eobrun(key)
                t.sym(key, 0xF0)
                r -= 16
                for bit in br:
                    t.bits(bit, 1)
                br = []
            if v > 1:
                br.append(v & 1)
                continue
            emit_eobrun(key)
            t.sym(key, (r << 4) | 1)
            t.bits(1 if blk[ZIGZAG[k]] > 0 else 0, 1)
            for bit in br:
                t.bits(bit, 1)
            br = []
            r = 0
        if r > 0 or br:
            state["eobrun"] += 1
            state["be"] += br
            if state["eobrun"] == 0x7FFF or len(state["be"]) > 937:
                emit_eobrun(key)


def reconstruct(comps, coefs, quants, w, h, mode):
    """A reconstrução de referência: IDCT dos coeficientes desquantizados, croma linear centrado, cor."""
    hmax = max(c.h for c in comps)
    vmax = max(c.v for c in comps)
    planes = []
    for ci, c in enumerate(comps):
        q = quants[c.tq]
        rows = []
        for brow in coefs[ci]:
            blocks = [idct([blk[k] * q[k] for k in range(64)]) for blk in brow]
            for y in range(8):
                rows.append([blocks[bx][y][x] for bx in range(len(brow)) for x in range(8)])
        planes.append(rows)

    def taps(n, full, f, fmax):
        out = []
        for o in range(full):
            v = min(max((o + 0.5) * f / fmax - 0.5, 0.0), n - 1.0)
            i0 = int(v)
            out.append((i0, min(i0 + 1, n - 1), v - i0))
        return out

    tx = [taps((w * c.h + hmax - 1) // hmax, w, c.h, hmax) for c in comps]
    ty = [taps((h * c.v + vmax - 1) // vmax, h, c.v, vmax) for c in comps]
    rnd = lambda v: min(255, max(0, math.floor(v + 0.5)))
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            s = []
            for ci, c in enumerate(comps):
                p = planes[ci]
                if c.h == hmax and c.v == vmax:
                    s.append(float(p[y][x]))
                    continue
                x0, x1, a = tx[ci][x]
                y0, y1, b = ty[ci][y]
                top = p[y0][x0] * (1 - a) + p[y0][x1] * a
                bottom = p[y1][x0] * (1 - a) + p[y1][x1] * a
                s.append(top * (1 - b) + bottom * b)
            if mode == "gray":
                g = rnd(s[0])
                row.append((g, g, g))
            elif mode == "rgb":
                row.append(tuple(rnd(v) for v in s))
            else:
                yy, cb, cr = s[0], s[1], s[2]
                r = yy + 1.402 * (cr - 128)
                g = yy - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
                bl = yy + 1.772 * (cb - 128)
                if mode == "ycc":
                    row.append((rnd(r), rnd(g), rnd(bl)))
                elif mode == "cmyk":
                    k = s[3]
                    row.append((rnd(s[0] * k / 255), rnd(s[1] * k / 255), rnd(s[2] * k / 255)))
                else:  # ycck
                    k = s[3]
                    cl = lambda v: min(255.0, max(0.0, v))
                    row.append((rnd((255 - cl(r)) * k / 255), rnd((255 - cl(g)) * k / 255), rnd((255 - cl(bl)) * k / 255)))
        out.append(row)
    return out


def orient(img, o):
    """Orientação EXIF por composição de giros e espelhos (independente do mapeamento direto do C++)."""
    rot = lambda m: [[m[len(m) - 1 - y][x] for y in range(len(m))] for x in range(len(m[0]))]  # 90° horário
    lr = lambda m: [list(reversed(r)) for r in m]
    ud = lambda m: list(reversed(m))
    return {1: lambda m: m, 2: lr, 3: lambda m: rot(rot(m)), 4: ud, 5: lambda m: lr(rot(m)), 6: rot, 7: lambda m: ud(rot(m)),
            8: lambda m: rot(rot(rot(m)))}[o](img)


def exif_segment(orientation=None, xres=None, unit=2, big_endian=False):
    e = ">" if big_endian else "<"
    entries = []
    if orientation is not None:
        entries.append((0x0112, 3, 1, struct.pack(e + "HH", orientation, 0)))
    data_off = 8 + 2 + 12 * 4 + 4
    blobs = b""
    if xres is not None:
        for tag in (0x011A, 0x011B):
            entries.append((tag, 5, 1, struct.pack(e + "I", data_off + len(blobs))))
            blobs += struct.pack(e + "II", int(xres * 1000), 1000)
        entries.append((0x0128, 3, 1, struct.pack(e + "HH", unit, 0)))
    ifd = struct.pack(e + "H", len(entries))
    for tag, typ, count, value in entries:
        ifd += struct.pack(e + "HHI", tag, typ, count) + value
    ifd += b"\x00" * (12 * (4 - len(entries)))  # completa até o deslocamento fixo dos dados
    ifd += struct.pack(e + "I", 0)
    tiff = (b"MM" if big_endian else b"II") + struct.pack(e + "HI", 42, 8) + ifd + blobs
    body = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + struct.pack(">H", 2 + len(body)) + body


def jpeg_case(name, w, h, sampling, seed, space="ycc", progressive=False, restart=0, separate_scans=False, quality=90, q16=False,
              jfif=True, jfif_density=None, adobe=None, exif=None, scans=None, rgb_ids=False, fill_bytes=False, stray_bytes=False,
              no_eoi=False, redefine_dqt=False, tolerance=0):
    img = source_image(w, h, seed)
    ncomp = len(sampling)
    ids = [ord("R"), ord("G"), ord("B")] if rgb_ids else list(range(1, ncomp + 1))
    comps = [Comp(ids[i], sampling[i][0], sampling[i][1], 0 if i == 0 or space in ("rgb", "cmyk") else 1) for i in range(ncomp)]
    quants = [scaled_table(LUMA_Q, quality), scaled_table(CHROMA_Q, quality)]
    hmax = max(c.h for c in comps)
    vmax = max(c.v for c in comps)
    mcus_x = (w + 8 * hmax - 1) // (8 * hmax)
    mcus_y = (h + 8 * vmax - 1) // (8 * vmax)
    planes = build_planes(img, w, h, comps, space)
    coefs = [quantize_component(planes[i], comps[i], mcus_x, mcus_y, quants[comps[i].tq]) for i in range(ncomp)]
    # Os coeficientes têm de caber na faixa da norma (o DC é ≤ 2047 com as tabelas acima).
    out = b"\xff\xd8"
    density = (72.0, 72.0)
    if jfif:
        units, xd, yd = (0, 1, 1) if jfif_density is None else jfif_density
        out += b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + bytes([1, 2, units]) + struct.pack(">HH", xd, yd) + b"\x00\x00"
        if units in (1, 2):
            f = 1.0 if units == 1 else 2.54
            density = (xd * f, yd * f)
    if exif is not None:
        out += exif_segment(**exif)
        if exif.get("xres") is not None and not (jfif and jfif_density and jfif_density[0] in (1, 2)):
            f = 1.0 if exif.get("unit", 2) == 2 else 2.54
            x = int(exif["xres"] * 1000) / 1000.0
            density = (x * f, x * f)
    if adobe is not None:
        out += b"\xff\xee" + struct.pack(">H", 14) + b"Adobe" + struct.pack(">HHH", 100, 0, 0) + bytes([adobe])
    for tq, q in enumerate(quants[: 1 if space in ("rgb", "cmyk", "gray") else 2]):
        if q16:
            out += b"\xff\xdb" + struct.pack(">H", 2 + 1 + 128) + bytes([0x10 | tq]) + b"".join(struct.pack(">H", q[ZIGZAG[k]]) for k in range(64))
        else:
            out += b"\xff\xdb" + struct.pack(">H", 2 + 1 + 64) + bytes([tq]) + bytes(q[ZIGZAG[k]] for k in range(64))
    if stray_bytes:
        out += b"\x00\x13\x37"  # bytes estranhos entre segmentos: o decodificador pula até o próximo FF
    sof = 0xC2 if progressive else (0xC1 if q16 else 0xC0)
    body = bytes([8]) + struct.pack(">HH", h, w) + bytes([ncomp]) + b"".join(bytes([c.id, (c.h << 4) | c.v, c.tq]) for c in comps)
    out += bytes([0xFF, sof]) + struct.pack(">H", 2 + len(body)) + body
    if restart:
        out += b"\xff\xdd" + struct.pack(">HH", 4, restart)
    enc = ScanEncoder(comps, coefs, mcus_x, mcus_y, w, h, restart, progressive)
    if scans is None:
        if progressive:
            raise ValueError("progressivo exige o roteiro de scans")
        scans = [([i], 0, 63, 0, 0) for i in range(ncomp)] if separate_scans else [(list(range(ncomp)), 0, 63, 0, 0)]
    for index, (comp_list, ss, se, ah, al) in enumerate(scans):
        if redefine_dqt and index == 1:
            # Tabelas trocadas depois do 1º scan: a desquantização usa as do 1º scan de cada componente (libjpeg).
            for tq in range(2):
                out += b"\xff\xdb" + struct.pack(">H", 2 + 1 + 64) + bytes([tq]) + bytes([1] * 64)
        tokens = enc.encode(comp_list, ss, se, ah, al)
        tables, data = serialize(tokens)
        if tables:
            out += dht(tables)
        body = bytes([len(comp_list)])
        for ci in comp_list:
            tsel = 0 if ci == 0 else 1
            body += bytes([comps[ci].id, (tsel << 4) | tsel])
        body += bytes([ss, se, (ah << 4) | al])
        if fill_bytes:
            out += b"\xff\xff"  # bytes de enchimento antes do marcador
        out += b"\xff\xda" + struct.pack(">H", 2 + len(body)) + body + data
    if not no_eoi:
        out += b"\xff\xd9"
    mode = {"ycc": "ycc", "rgb": "rgb", "gray": "gray", "cmyk": "cmyk", "ycck": "ycck"}[space]
    rec = reconstruct(comps, coefs, quants, w, h, mode)
    o = exif.get("orientation") if exif else None
    if o:
        rec = orient(rec, o)
        if o >= 5:
            density = (density[1], density[0])
    ow, oh = len(rec[0]), len(rec)
    write(name + ".jpg", out)
    write(name + ".rgba", b"".join(bytes([r, g, b, 255]) for row in rec for (r, g, b) in row))
    MANIFEST.append("jpeg %s %d %d %.6f %.6f 0 %d" % (name, ow, oh, density[0], density[1], tolerance))


# Roteiro progressivo do libjpeg para YCbCr (jcparam.c, jpeg_simple_progression) e um só para cinza.
PROG_YCC = [([0, 1, 2], 0, 0, 0, 1), ([0], 1, 5, 0, 2), ([2], 1, 63, 0, 1), ([1], 1, 63, 0, 1), ([0], 6, 63, 0, 2), ([0], 1, 63, 2, 1),
            ([0, 1, 2], 0, 0, 1, 0), ([2], 1, 63, 1, 0), ([1], 1, 63, 1, 0), ([0], 1, 63, 1, 0)]
PROG_GRAY = [([0], 0, 0, 0, 1), ([0], 1, 5, 0, 2), ([0], 6, 63, 0, 2), ([0], 1, 63, 2, 1), ([0], 0, 0, 1, 0), ([0], 1, 63, 1, 0)]
PROG_SPECTRAL = [([0, 1, 2], 0, 0, 0, 0), ([0], 1, 9, 0, 0), ([0], 10, 63, 0, 0), ([1], 1, 63, 0, 0), ([2], 1, 63, 0, 0)]


def make_jpegs():
    S444 = [(1, 1), (1, 1), (1, 1)]
    S420 = [(2, 2), (1, 1), (1, 1)]
    jpeg_case("jpeg_ycc444", 32, 24, S444, 400)
    jpeg_case("jpeg_ycc422", 37, 23, [(2, 1), (1, 1), (1, 1)], 401)
    jpeg_case("jpeg_ycc420", 37, 23, S420, 402)
    jpeg_case("jpeg_ycc411", 45, 17, [(4, 1), (1, 1), (1, 1)], 403)
    jpeg_case("jpeg_ycc440", 21, 30, [(1, 2), (1, 1), (1, 1)], 404)
    jpeg_case("jpeg_ycc_mixed", 29, 27, [(2, 2), (1, 1), (1, 2)], 405)
    jpeg_case("jpeg_gray", 41, 17, [(1, 1)], 406, space="gray")
    jpeg_case("jpeg_gray_restart", 41, 17, [(1, 1)], 407, space="gray", restart=5)
    jpeg_case("jpeg_ycc420_restart", 53, 35, S420, 408, restart=3)
    jpeg_case("jpeg_ycc420_separate", 37, 23, S420, 409, separate_scans=True)
    jpeg_case("jpeg_ycc420_q16", 24, 16, S420, 410, q16=True, quality=97)
    jpeg_case("jpeg_ycc420_progressive", 53, 35, S420, 411, progressive=True, scans=PROG_YCC)
    jpeg_case("jpeg_ycc420_progressive_restart", 53, 35, S420, 412, progressive=True, scans=PROG_YCC, restart=2)
    jpeg_case("jpeg_gray_progressive", 64, 48, [(1, 1)], 413, space="gray", progressive=True, scans=PROG_GRAY, quality=60)
    jpeg_case("jpeg_ycc444_spectral", 32, 32, S444, 414, progressive=True, scans=PROG_SPECTRAL)
    jpeg_case("jpeg_adobe_rgb", 24, 16, S444, 415, space="rgb", adobe=0)
    jpeg_case("jpeg_rgb_ids", 24, 16, S444, 416, space="rgb", rgb_ids=True, jfif=False)
    jpeg_case("jpeg_cmyk", 24, 16, [(1, 1)] * 4, 417, space="cmyk", adobe=0, jfif=False)
    jpeg_case("jpeg_ycck", 24, 16, [(2, 2), (1, 1), (1, 1), (2, 2)], 418, space="ycck", adobe=2, jfif=False)
    jpeg_case("jpeg_jfif_dpi", 16, 16, S444, 419, jfif_density=(1, 300, 150))
    jpeg_case("jpeg_jfif_dpcm", 16, 16, S444, 420, jfif_density=(2, 118, 118))
    jpeg_case("jpeg_exif_dpi", 16, 16, S444, 421, jfif=False, exif={"xres": 240.0})
    jpeg_case("jpeg_exif_dpcm_be", 16, 16, S444, 422, jfif=False, exif={"xres": 100.0, "unit": 3, "big_endian": True})
    for o in range(1, 9):
        jpeg_case("jpeg_orient%d" % o, 13, 7, S420, 430 + o, exif={"orientation": o, "big_endian": o % 2 == 0})
    jpeg_case("jpeg_fill_stray_noeoi", 24, 16, S420, 440, fill_bytes=True, stray_bytes=True, no_eoi=True)
    jpeg_case("jpeg_orient6_dpi", 13, 7, S420, 441, jfif_density=(1, 300, 150), exif={"orientation": 6})
    jpeg_case("jpeg_jfif_over_exif", 16, 16, S444, 442, jfif_density=(1, 300, 300), exif={"xres": 96.0})
    jpeg_case("jpeg_progressive_dqt_redefined", 32, 24, S420, 443, progressive=True, scans=PROG_YCC, redefine_dqt=True)


def main():
    make_pngs()
    make_zlib()
    make_jpegs()
    with open(os.path.join(OUT, "manifest.txt"), "w", newline="\n") as f:
        f.write("\n".join(MANIFEST) + "\n")
    print("%d casos em %s" % (len(MANIFEST), OUT))


if __name__ == "__main__":
    main()
