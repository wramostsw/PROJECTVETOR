// tests/unit/flow_field_test.cpp — FlowField (tensor de estrutura integrado: energia, coerência, tangente,
// alinhamento) e a ligação de traços finos diagonais na segmentação (PixelSegmenter::linkStrokeSweep).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "cnx/preprocess/EdgeImage.h"
#include "cnx/preprocess/FlowField.h"
#include "cnx/segment/PixelSegmenter.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::FlowField;
using cnx::PixelMap;
using cnx::PixelSegmenter;
using cnx::RegionId;
using sla::Array2D;
using sla::Rgba8;
using sla::Vec2;

namespace {

Rgba8 rgb(int r, int g, int b) { return {static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b), 255}; }
const Rgba8 kWhite = rgb(255, 255, 255);
const Rgba8 kBlack = rgb(0, 0, 0);

using Pixels = std::vector<std::array<int, 2>>;

/// Segmento de Bresenham de (x0, y0) a (x1, y1), 8-conexo.
Pixels bresenham(int x0, int y0, int x1, int y1) {
    Pixels out;
    const int dx = std::abs(x1 - x0);
    const int dy = -std::abs(y1 - y0);
    const int sx = x0 < x1 ? 1 : -1;
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        out.push_back({x0, y0});
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
    return out;
}

Array2D<Rgba8> draw(int w, int h, const Pixels& pixels, Rgba8 ink = kBlack, Rgba8 bg = kWhite) {
    Array2D<Rgba8> img(w, h, bg);
    for (const auto& p : pixels) img(p[0], p[1]) = ink;
    return img;
}

/// Mapa inicial com uma região por componente 4-conexa de cor idêntica (o modo "paleta exata").
PixelMap exactMap(const Array2D<Rgba8>& img) {
    PixelMap map(img.width(), img.height());
    std::vector<std::array<int, 2>> stack;
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            if (map.regionAtConst(x, y) != cnx::kNoRegion) continue;
            const RegionId id = map.createRegion();
            map.setLabel(x, y, id, img(x, y));
            stack.assign(1, {x, y});
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + cnxtest::kDx4[d];
                    const int ny = cy + cnxtest::kDy4[d];
                    if (!img.inBounds(nx, ny) || map.regionAtConst(nx, ny) != cnx::kNoRegion || !(img(nx, ny) == img(x, y))) continue;
                    map.setLabel(nx, ny, id, img(nx, ny));
                    stack.push_back({nx, ny});
                }
            }
        }
    }
    return map;
}

cnx::SegmentationResult segment(const Array2D<Rgba8>& img, cnx::SegmentParams params = {}) { return PixelSegmenter(img, params).run(); }

/// Todos os pixels na mesma região (mapa compactado), e só eles com essa cor de região.
bool oneRegion(const PixelMap& map, const Pixels& pixels) {
    const RegionId r = map.compactLabels()(pixels[0][0], pixels[0][1]);
    return std::all_of(pixels.begin(), pixels.end(), [&](const auto& p) { return map.compactLabels()(p[0], p[1]) == r; });
}

std::size_t pixelsOf(const PixelMap& map, RegionId r) {
    std::size_t n = 0;
    for (int y = 0; y < map.height(); ++y) {
        for (int x = 0; x < map.width(); ++x) n += map.compactLabels()(x, y) == r ? 1 : 0;
    }
    return n;
}

double angleBetweenAxes(Vec2 a, Vec2 b) {  // em graus, ∈ [0, 90]
    const double c = std::abs(sla::dot(a, b)) / (sla::norm(a) * sla::norm(b));
    return std::acos(std::min(1.0, c)) * 180.0 / std::numbers::pi;
}

}  // namespace

// ================================================================ FlowField

CNX_TEST(flow_field_of_flat_empty_and_invalid_input) {
    const FlowField flat = FlowField::compute(Array2D<Rgba8>(9, 7, rgb(40, 80, 120)), 1.5);
    CHECK(flat.width() == 9 && flat.height() == 7 && flat.sigma() == 1.5);
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 9; ++x) {
            CHECK(flat.energy(x, y) == 0.0 && flat.coherence(x, y) == 0.0);
            CHECK(flat.tangent(x, y) == Vec2(0, 0) && flat.normal(x, y) == Vec2(0, 0) && flat.alignment(x, y, {1, 1}) == 0.0);
        }
    }
    const FlowField empty = FlowField::compute(Array2D<Rgba8>(), 1.0);
    CHECK(empty.width() == 0 && empty.height() == 0);
    const Array2D<Rgba8> img(4, 4);
    CHECK_THROWS(std::invalid_argument, (void)FlowField::compute(img, -0.5));
    CHECK_THROWS(std::invalid_argument, (void)FlowField::compute(img, std::numeric_limits<double>::quiet_NaN()));
    CHECK_THROWS(std::invalid_argument, (void)FlowField::compute(img, std::numeric_limits<double>::infinity()));
}

CNX_TEST(the_raw_field_is_the_sobel_tensor_of_the_edge_image) {
    cnxtest::SplitMix64 rng{5101};
    Array2D<Rgba8> img(19, 13);
    for (int y = 0; y < 13; ++y) {
        for (int x = 0; x < 19; ++x) img(x, y) = {static_cast<std::uint8_t>(rng.below(256)), static_cast<std::uint8_t>(rng.below(256)),
                                                   static_cast<std::uint8_t>(rng.below(256)), static_cast<std::uint8_t>(rng.below(256))};
    }
    const FlowField raw = FlowField::compute(img, 0.0);
    const cnx::EdgeImage edges = cnx::EdgeImage::compute(img);
    const Array2D<cnx::StructureTensor> sobel = cnx::sobelTensors(img);
    constexpr double kUnit = cnx::kSobelScale * cnx::kSobelScale;
    for (int y = 0; y < 13; ++y) {
        for (int x = 0; x < 19; ++x) {
            const cnx::StructureTensor& t = raw.tensor(x, y);
            CHECK(t.xx == static_cast<float>(sobel(x, y).xx * kUnit) && t.xy == static_cast<float>(sobel(x, y).xy * kUnit) &&
                  t.yy == static_cast<float>(sobel(x, y).yy * kUnit));
            // Energia = m², e a tangente é perpendicular à normal da borda.
            const double m = edges.magnitude(x, y);
            CHECK(std::abs(raw.energy(x, y) - m * m) <= 1e-6 * (1.0 + m * m));
            if (raw.coherence(x, y) > 1e-3) {
                CHECK(std::abs(sla::dot(raw.tangent(x, y), edges.normal(x, y))) < 1e-5);
                CHECK(std::abs(sla::norm(raw.tangent(x, y)) - 1.0) < 1e-12);
                CHECK(std::abs(sla::dot(raw.tangent(x, y), raw.normal(x, y))) < 1e-12);
            }
            CHECK(raw.coherence(x, y) >= 0.0 && raw.coherence(x, y) <= 1.0);
        }
    }
}

CNX_TEST(integration_is_a_normalized_gaussian_with_exponential_weights) {
    // Degrau horizontal em y = 10: tensor bruto só nas linhas 9 e 10, (0, 0, 3) — m = √3.
    Array2D<Rgba8> img(20, 30, kBlack);
    for (int y = 10; y < 30; ++y) {
        for (int x = 0; x < 20; ++x) img(x, y) = kWhite;
    }
    const double sigma = 1.5;
    const FlowField f = FlowField::compute(img, sigma);
    const int radius = 5;  // ⌈3σ⌉
    double norm = 0.0;
    for (int k = -radius; k <= radius; ++k) norm += std::exp(-0.5 * k * k / (sigma * sigma));
    const auto g = [&](int k) { return std::abs(k) > radius ? 0.0 : std::exp(-0.5 * k * k / (sigma * sigma)) / norm; };
    double column = 0.0;
    for (int y = 0; y < 30; ++y) {
        const double expected = 3.0 * (g(y - 9) + g(y - 10));
        CHECK_CTX(std::abs(f.energy(7, y) - expected) <= 1e-6 * (1.0 + expected), "linha " + std::to_string(y));
        CHECK(f.tensor(7, y).xx == 0.0f && f.tensor(7, y).xy == 0.0f);
        if (f.energy(7, y) > 0.0) {
            CHECK(f.coherence(7, y) == 1.0);
            CHECK(std::abs(f.tangent(7, y).x) == 1.0 && std::abs(f.tangent(7, y).y) < 1e-15);
        }
        column += f.energy(7, y);
    }
    CHECK(f.energy(7, 3) == 0.0 && f.energy(7, 16) == 0.0 && f.energy(7, 4) > 0.0 && f.energy(7, 15) > 0.0);  // raio exato
    CHECK(std::abs(column - 6.0) < 1e-5);  // a integração conserva a energia
    // Bordas replicadas (não circulares): degrau vertical em x = 2, tensor bruto só nas colunas 1 e 2.
    Array2D<Rgba8> left(20, 6, kWhite);
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 2; ++x) left(x, y) = kBlack;
    }
    const FlowField fl = FlowField::compute(left, sigma);
    CHECK(std::abs(fl.energy(0, 3) - 3.0 * (g(1) + g(2))) < 1e-6);
    CHECK(fl.energy(19, 3) == 0.0 && fl.energy(15, 3) == 0.0 && fl.energy(7, 3) > 0.0);
}

CNX_TEST(the_center_of_a_one_pixel_line_inherits_its_direction) {
    Pixels line;
    for (int i = 0; i < 24; ++i) line.push_back({i, i});
    const Array2D<Rgba8> img = draw(24, 24, line);
    // Sem integração, o centro do traço não tem gradiente (os dois flancos se cancelam no Sobel).
    const FlowField raw = FlowField::compute(img, 0.0);
    CHECK(raw.energy(12, 12) == 0.0 && raw.tangent(12, 12) == Vec2(0, 0));
    const FlowField f = FlowField::compute(img, 1.0);
    std::printf("  traço diagonal: coerência %.4f, tangente (%.4f, %.4f)\n", f.coherence(12, 12), f.tangent(12, 12).x, f.tangent(12, 12).y);
    CHECK(f.coherence(12, 12) > 0.99);
    CHECK(angleBetweenAxes(f.tangent(12, 12), {1, 1}) < 0.5);
    CHECK(f.alignment(12, 12, {1, 1}) > 0.999 && f.alignment(12, 12, {-2, -2}) > 0.999);
    CHECK(f.alignment(12, 12, {1, -1}) < -0.999);
    CHECK(std::abs(f.alignment(12, 12, {1, 0})) < 0.01 && f.alignment(12, 12, {0, 0}) == 0.0);
    // alignment = cos 2α = 2·(t·d̂)² − 1 em qualquer pixel anisotrópico.
    cnxtest::SplitMix64 rng{5102};
    for (int k = 0; k < 200; ++k) {
        const int x = rng.below(24);
        const int y = rng.below(24);
        if (!(f.coherence(x, y) > 1e-3)) continue;
        const double a = cnxtest::uniform01(rng) * 2.0 * std::numbers::pi;
        const Vec2 d{std::cos(a) * 3.0, std::sin(a) * 3.0};
        const double c = sla::dot(f.tangent(x, y), d) / 3.0;
        CHECK(std::abs(f.alignment(x, y, d) - (2.0 * c * c - 1.0)) < 1e-6);
    }
}

CNX_TEST(the_tangent_follows_edges_at_any_angle_and_around_a_circle) {
    double worst = 0.0;
    for (const double deg : {0.0, 10.0, 22.5, 37.0, 45.0, 60.0, 80.0, 100.0, 135.0, 170.0}) {
        // Semiplano com AA de 8×8 cuja borda passa pelo centro com a direção `deg`.
        const double a = deg * std::numbers::pi / 180.0;
        const Vec2 t{std::cos(a), std::sin(a)};
        Array2D<Rgba8> img(33, 33);
        for (int y = 0; y < 33; ++y) {
            for (int x = 0; x < 33; ++x) {
                int hits = 0;
                for (int j = 0; j < 8; ++j) {
                    for (int i = 0; i < 8; ++i) hits += sla::cross(t, Vec2{x + (i + 0.5) / 8 - 16.5, y + (j + 0.5) / 8 - 16.5}) < 0.0 ? 1 : 0;
                }
                const auto v = static_cast<std::uint8_t>(std::lround(255.0 * (1.0 - hits / 64.0)));
                img(x, y) = rgb(v, v, v);
            }
        }
        const FlowField f = FlowField::compute(img, 1.0);
        const double err = angleBetweenAxes(f.tangent(16, 16), t);
        worst = std::max(worst, err);
        CHECK_CTX(err < 1.5 && f.coherence(16, 16) > 0.95, std::to_string(deg) + "°");
    }
    std::printf("  borda reta: maior erro de orientação %.3f°\n", worst);
    // Disco: a tangente é perpendicular ao raio em todo pixel de borda.
    Array2D<Rgba8> disk(48, 48, kWhite);
    for (int y = 0; y < 48; ++y) {
        for (int x = 0; x < 48; ++x) {
            if (std::hypot(x + 0.5 - 24.2, y + 0.5 - 23.7) <= 15.0) disk(x, y) = kBlack;
        }
    }
    const FlowField fd = FlowField::compute(disk, 1.0);
    int checked = 0;
    double worstDisk = 90.0;
    double weakest = 1.0;
    for (int y = 0; y < 48; ++y) {
        for (int x = 0; x < 48; ++x) {
            const double r = std::hypot(x + 0.5 - 24.2, y + 0.5 - 23.7);
            if (std::abs(r - 15.0) > 1.0) continue;
            ++checked;
            worstDisk = std::min(worstDisk, angleBetweenAxes(fd.tangent(x, y), {x + 0.5 - 24.2, y + 0.5 - 23.7}));
            weakest = std::min(weakest, fd.coherence(x, y));
        }
    }
    std::printf("  disco: %d pixels, menor ângulo tangente–raio %.2f°, menor coerência %.3f\n", checked, worstDisk, weakest);
    CHECK(checked > 150 && worstDisk > 75.0 && weakest > 0.75);  // escadinha dura: menos precisa que a borda com AA
}

CNX_TEST(corners_crossings_and_checkerboards_are_not_coherent) {
    // Xadrez de 1 px: o Sobel é exatamente nulo longe da moldura (onde a borda replicada quebra o padrão).
    Array2D<Rgba8> chess(20, 20);
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) chess(x, y) = (x + y) % 2 == 0 ? kBlack : kWhite;
    }
    const FlowField fc = FlowField::compute(chess, 1.0);
    for (int y = 5; y < 15; ++y) {
        for (int x = 5; x < 15; ++x) CHECK(fc.energy(x, y) == 0.0 && fc.coherence(x, y) == 0.0);
    }
    // Cruz de dois traços de 3 px: no cruzamento a orientação é indefinida.
    Array2D<Rgba8> cross(25, 25, kWhite);
    for (int k = 0; k < 25; ++k) {
        for (int j = 11; j <= 13; ++j) {
            cross(k, j) = kBlack;
            cross(j, k) = kBlack;
        }
    }
    const FlowField fx = FlowField::compute(cross, 1.0);
    std::printf("  coerência: cruzamento %.3f, braço %.3f\n", fx.coherence(12, 12), fx.coherence(12, 4));
    CHECK(fx.coherence(12, 12) < 0.1 && fx.coherence(12, 4) > 0.9);
    // Canto de um quadrado: coerência baixa no vértice, alta no meio dos lados.
    Array2D<Rgba8> square(24, 24, kWhite);
    for (int y = 6; y < 18; ++y) {
        for (int x = 6; x < 18; ++x) square(x, y) = kBlack;
    }
    const FlowField fs = FlowField::compute(square, 1.0);
    std::printf("  coerência: canto %.3f, lado %.3f\n", fs.coherence(6, 6), fs.coherence(12, 6));
    CHECK(fs.coherence(6, 6) < 0.5 && fs.coherence(12, 6) > 0.95);
}

// ================================================================ traços finos na segmentação

CNX_TEST(a_one_pixel_diagonal_line_survives_segmentation) {
    Pixels diag;
    Pixels anti;
    for (int i = 3; i <= 28; ++i) {
        diag.push_back({i, i});
        anti.push_back({31 - i, i});
    }
    for (const Pixels* line : {&diag, &anti}) {
        const Array2D<Rgba8> img = draw(32, 32, *line);
        cnx::SegmentParams off;
        off.linkStrokes = false;
        const cnx::SegmentationResult without = segment(img, off);
        CHECK(without.stats.finalRegions == 1 && without.stats.strokeLinks == 0);  // o traço some, pixel a pixel
        const cnx::SegmentationResult with = segment(img);
        std::printf("  traço %s: %d regiões, %d ligações\n", line == &diag ? "\\" : "/", with.stats.finalRegions, with.stats.strokeLinks);
        CHECK(with.stats.finalRegions == 2 && with.stats.strokeLinks == 25);
        CHECK(oneRegion(with.map, *line));
        const RegionId r = with.map.compactLabels()((*line)[0][0], (*line)[0][1]);
        CHECK(pixelsOf(with.map, r) == 26 + 25);  // o traço e uma ponte por ligação
        CHECK(with.map.regionColor(r) == sla::toRgba(kBlack));
    }
}

CNX_TEST(lines_of_many_slopes_and_a_thin_circle_stay_connected) {
    struct Case {
        const char* name;
        Pixels pixels;
        int regions;
    };
    std::vector<Case> cases{
        {"inclinação 1/3", bresenham(2, 4, 37, 16), 2},  {"inclinação 2/3", bresenham(2, 3, 37, 26), 2},
        {"inclinação 3", bresenham(5, 2, 16, 35), 2},    {"inclinação -1/2", bresenham(2, 30, 37, 12), 2},
        {"inclinação 1/8", bresenham(1, 10, 38, 14), 2},
    };
    Pixels ring;
    for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 40; ++x) {
            if (std::abs(std::hypot(x - 19.5, y - 19.5) - 12.0) < 0.5) ring.push_back({x, y});
        }
    }
    cases.push_back({"círculo r = 12", ring, 3});
    for (const Case& c : cases) {
        const cnx::SegmentationResult r = segment(draw(40, 40, c.pixels));
        std::printf("  %s: %zu px, %d regiões, %d ligações\n", c.name, c.pixels.size(), r.stats.finalRegions, r.stats.strokeLinks);
        CHECK_CTX(r.stats.finalRegions == c.regions && oneRegion(r.map, c.pixels), c.name);
    }
}

CNX_TEST(strokes_are_not_invented_where_the_field_or_the_colors_say_no) {
    const auto links = [](const Array2D<Rgba8>& img, cnx::SegmentParams params = {}) {
        PixelSegmenter seg(img, params, exactMap(img));
        return seg.linkStrokeSweep();
    };
    Pixels diag;
    for (int i = 2; i < 22; ++i) diag.push_back({i, i});
    CHECK(links(draw(24, 24, diag)) == 19);
    cnx::SegmentParams off;
    off.linkStrokes = false;
    CHECK(links(draw(24, 24, diag), off) == 0);
    // Coerência mínima apertada: as pontas (e só elas) deixam de ligar.
    cnx::SegmentParams strict;
    strict.strokeMinCoherence = 0.95;
    const int tight = links(draw(24, 24, diag), strict);
    std::printf("  coerência ≥ 0,95: %d ligações\n", tight);
    CHECK(tight > 0 && tight < 19);
    // Pontes quase da cor do traço (cinza 20, a 0,018 do preto): o traço já tem 2 px, não há o que ligar.
    Array2D<Rgba8> thick = draw(24, 24, diag);
    for (int i = 2; i < 21; ++i) thick(i + 1, i) = rgb(20, 20, 20);
    CHECK(links(thick) == 0);
    // Xadrez: o campo é nulo — nenhum par diagonal vira traço.
    Array2D<Rgba8> chess(16, 16);
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) chess(x, y) = (x + y) % 2 == 0 ? kBlack : kWhite;
    }
    CHECK(links(chess) == 0);
    // Dois quadrados que se tocam pela quina: o campo ali não é coerente.
    Array2D<Rgba8> corner(24, 24, kWhite);
    for (int y = 4; y < 12; ++y) {
        for (int x = 4; x < 12; ++x) {
            corner(x, y) = kBlack;
            corner(x + 8, y + 8) = kBlack;
        }
    }
    CHECK(links(corner) == 0);
    // Cores que não combinam ao longo da diagonal: preto e vermelho alternados.
    Array2D<Rgba8> mixed = draw(24, 24, diag);
    for (std::size_t k = 0; k < diag.size(); k += 2) mixed(diag[k][0], diag[k][1]) = rgb(230, 0, 0);
    CHECK(links(mixed) == 0);
    // Escadinha de AA: a cadeia cinza entre o preto e o branco é mistura das duas pontes, não traço.
    Array2D<Rgba8> aa(24, 24);
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 24; ++x) aa(x, y) = x + y < 23 ? kBlack : x + y == 23 ? rgb(128, 128, 128) : kWhite;
    }
    CHECK(links(aa) == 0);
    // O mesmo cinza entre duas tintas das quais ele não é mistura (vermelho e azul) é traço.
    Array2D<Rgba8> outline(24, 24);
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 24; ++x) outline(x, y) = x + y < 23 ? rgb(220, 30, 30) : x + y == 23 ? rgb(128, 128, 128) : rgb(30, 30, 220);
    }
    CHECK(links(outline) > 15);
    // Inclinação 1/3 (≈ 26,6° da diagonal): ligada com o ângulo padrão (0,7 rad ≈ 40°), não com 0,3 rad.
    const Array2D<Rgba8> shallow = draw(40, 20, bresenham(1, 3, 37, 15));
    cnx::SegmentParams narrow;
    narrow.strokeMaxAngle = 0.3;
    CHECK(links(shallow) == 12 && links(shallow, narrow) == 0);
}

CNX_TEST(parallel_diagonal_lines_are_linked_along_and_never_across) {
    // x − y = 0 e x − y = 2: cada par (i, i)–(i + 1, i − 1) é diagonal, mas atravessa o campo (α = 90°).
    Pixels a;
    Pixels b;
    for (int i = 3; i < 25; ++i) {
        a.push_back({i, i});
        b.push_back({i + 2, i});
    }
    Pixels both = a;
    both.insert(both.end(), b.begin(), b.end());
    const Array2D<Rgba8> img = draw(30, 30, both);
    PixelSegmenter seg(img, {}, exactMap(img));
    const int n = seg.linkStrokeSweep();
    PixelMap& map = seg.map();
    CHECK(n == 42);
    const RegionId ra = map.regionAt(a[0][0], a[0][1]);
    const RegionId rb = map.regionAt(b[0][0], b[0][1]);
    CHECK(ra != rb);
    for (const auto& p : a) CHECK(map.regionAt(p[0], p[1]) == ra);
    for (const auto& p : b) CHECK(map.regionAt(p[0], p[1]) == rb);
    // As pontes escolhidas não encostam na outra linha: a fileira entre as duas (x − y = 1) continua branca.
    for (int i = 3; i < 25; ++i) CHECK(map.regionAt(i + 1, i) != ra && map.regionAt(i + 1, i) != rb);
}

CNX_TEST(the_bridge_with_more_ink_is_preferred) {
    // Traço preto em x = y; as pontes de baixo (x − y = −1) são cinza-claro (a mesma região do branco, depois
    // da fusão), as de cima são brancas: a ponte vai para o lado com mais tinta, contra a ordem padrão (a horizontal).
    Pixels line;
    for (int i = 3; i <= 28; ++i) line.push_back({i, i});
    Array2D<Rgba8> img = draw(32, 32, line);
    for (int i = 3; i < 28; ++i) img(i, i + 1) = rgb(235, 235, 235);
    const cnx::SegmentationResult r = segment(img);
    CHECK(r.stats.finalRegions == 2 && r.stats.strokeLinks == 25 && oneRegion(r.map, line));
    const RegionId stroke = r.map.compactLabels()(3, 3);
    for (int i = 3; i < 28; ++i) CHECK(r.map.compactLabels()(i, i + 1) == stroke && r.map.compactLabels()(i + 1, i) != stroke);
}

CNX_TEST(the_bridge_that_touches_another_region_is_avoided) {
    // Um quadradinho quase branco (outra região, campo quase intacto) encosta na ponte horizontal de um único par:
    // ali a ponte vai para baixo, para o traço não ganhar um contato novo; nos outros pares fica a horizontal.
    Pixels diag;
    for (int i = 2; i < 22; ++i) diag.push_back({i, i});
    Array2D<Rgba8> img = draw(24, 24, diag);
    for (int y = 9; y <= 10; ++y) {
        for (int x = 12; x <= 13; ++x) img(x, y) = rgb(245, 245, 250);
    }
    PixelSegmenter seg(img, {}, exactMap(img));
    CHECK(seg.linkStrokeSweep() == 19);
    PixelMap& map = seg.map();
    const RegionId stroke = map.regionAt(2, 2);
    CHECK(map.regionAt(11, 10) != stroke && map.regionAt(10, 11) == stroke);  // o par (10,10)–(11,11)
    CHECK(map.regionAt(10, 9) == stroke && map.regionAt(9, 10) != stroke);    // o par (9,9)–(10,10): horizontal
}

CNX_TEST(bridges_count_for_topology_but_not_for_the_stroke_color) {
    Pixels diag;
    for (int i = 2; i < 22; ++i) diag.push_back({i, i});
    const Array2D<Rgba8> img = draw(24, 24, diag);
    PixelSegmenter seg(img, {}, exactMap(img));
    CHECK(seg.linkStrokeSweep() == 19);
    PixelMap& map = seg.map();
    const RegionId r = map.regionAt(2, 2);
    CHECK(map.forest().pixelCount(r) == 39 && map.forest().stats(r).samples.count() == 20);
    CHECK(map.forest().stats(r).samples.mean() == sla::toRgba(kBlack));
    int excluded = 0;
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 24; ++x) excluded += map.isExcludedFromColorStats(x, y) ? 1 : 0;
    }
    CHECK(excluded == 19);
    // Com a paleta, a cor vem só da tinta: a média com as pontes (cinza ≈ 0,49) cairia no vermelho.
    cnx::Palette pal;
    pal.colors = {sla::toRgba(kWhite), sla::toRgba(kBlack), sla::Rgba{0.86f, 0.12f, 0.12f, 1.0f}};
    const PixelMap done = seg.finish(&pal);
    CHECK(done.regionColor(done.compactLabels()(2, 2)) == sla::toRgba(kBlack));

    // Contrato de PixelMap::excludeFromColorStats.
    PixelMap m(3, 1);
    const RegionId a = m.createRegion();
    const RegionId b = m.createRegion();
    m.setLabel(0, 0, a, kBlack);
    m.setLabel(1, 0, a, kWhite);
    m.setLabel(2, 0, b, kWhite);
    CHECK(m.excludeFromColorStats(1, 0, kWhite) && !m.excludeFromColorStats(1, 0, kWhite));
    CHECK(m.forest().pixelCount(a) == 2 && m.forest().stats(a).samples.count() == 1);
    CHECK(m.tryMovePixel(1, 0, b, kWhite));  // a amostra não acompanha o pixel excluído
    CHECK(m.forest().stats(a).samples.count() == 1 && m.forest().stats(b).samples.count() == 1 && m.forest().pixelCount(b) == 2);
    CHECK(m.isExcludedFromColorStats(1, 0) && !m.isExcludedFromColorStats(0, 0));
    PixelMap blank(2, 2);
    CHECK_THROWS(std::logic_error, (void)blank.excludeFromColorStats(0, 0, kWhite));
    CHECK_THROWS(std::out_of_range, (void)blank.excludeFromColorStats(5, 0, kWhite));
}

CNX_TEST(stroke_parameters_are_validated) {
    const Array2D<Rgba8> img(4, 4);
    const auto bad = [&](auto&& edit) {
        cnx::SegmentParams p;
        edit(p);
        bool threw = false;
        try {
            (void)PixelSegmenter(img, p);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        return threw;
    };
    CHECK(bad([](auto& p) { p.flowSigmaPx = -1.0; }));
    CHECK(bad([](auto& p) { p.flowSigmaPx = std::numeric_limits<double>::infinity(); }));
    CHECK(bad([](auto& p) { p.strokeMinCoherence = 1.5; }));
    CHECK(bad([](auto& p) { p.strokeMinCoherence = -0.1; }));
    CHECK(bad([](auto& p) { p.strokeMaxAngle = 0.0; }));
    CHECK(bad([](auto& p) { p.strokeMaxAngle = std::numbers::pi / 4.0; }));
    CHECK(!bad([](auto& p) { p.flowSigmaPx = 0.0; p.strokeMinCoherence = 1.0; p.strokeMaxAngle = 0.78; }));
    PixelSegmenter fresh(img, cnx::SegmentParams{});
    CHECK_THROWS(std::logic_error, (void)fresh.linkStrokeSweep());
}

int main() { return cnxtest::runAll(); }
