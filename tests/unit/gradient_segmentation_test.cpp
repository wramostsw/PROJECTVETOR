// tests/unit/gradient_segmentation_test.cpp — degradês: dezenas de faixas finas em vez de poucas faixas largas (docs §1.28).
//
// Um degradê não tem tintas: a paleta de K cores o explica mal, e o teto de fusão antigo (40 níveis de cor) juntava
// as faixas vizinhas até sobrarem 7 por fita. O teto passou a ser o passo da faixa (12 níveis) e o encaixe na paleta só
// vale para quem está perto de uma tinta (SegmentParams::paletteSnapMax). Estes testes medem as duas coisas contra
// rampas de verdade conhecida, e mostram que o ajuste antigo as reprova.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "cnx/classify/Classifier.h"
#include "cnx/engine/CoreEngine.h"
#include "cnx/engine/Parameters.h"
#include "cnx/export/Exporter.h"
#include "cnx/preprocess/SignalImage.h"
#include "cnx/segment/PixelMap.h"
#include "cnx/segment/PixelSegmenter.h"
#include "logo_scenes.h"
#include "svg_support.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::PixelMap;
using cnx::PixelSegmenter;
using cnx::RegionId;
using cnx::SegmentParams;
using sla::Array2D;
using sla::Rgba;
using sla::Rgba8;

namespace {

constexpr Rgba8 kWhite{255, 255, 255, 255};

/// dist² RGBA do teto de fusão para `levels` níveis de cor (0-255) em um canal.
constexpr double ceilingOf(double levels) { return (levels / 255.0) * (levels / 255.0); }

cnx::Palette paletteOf(std::initializer_list<Rgba8> colors) {
    cnx::Palette p;
    for (const Rgba8& c : colors) p.colors.push_back(sla::toRgba(c));
    return p;
}

/// Parâmetros de segmentação como o motor os usa numa foto/degradê (perfil PhotoMed): teto de 12 níveis, encaixe só
/// perto de uma tinta.
SegmentParams gradientParams() {
    cnx::EngineParameters e;
    e.applyProfile(cnx::ImageProfile::PhotoMed);
    return e.segment;
}

/// O ajuste de antes: teto de 40 níveis e encaixe sempre.
SegmentParams oldParams() {
    SegmentParams p;
    p.mergeThreshMax = ceilingOf(40.0);
    p.paletteSnapMax = 4.0;
    return p;
}

std::uint8_t clampByte(double v) { return static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L)); }

/// Rampa horizontal de `a` a `b` nas colunas [x0, w), fundo branco à esquerda; ruído de ±1 nível (a "sujeira" de um JPEG).
Array2D<Rgba8> rampImage(int w, int h, int x0, Rgba8 a, Rgba8 b, std::uint64_t seed = 7) {
    cnxtest::SplitMix64 rng{seed};
    Array2D<Rgba8> img(w, h);
    const auto noise = [&] { return static_cast<double>(rng.below(3) - 1); };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (x < x0) {
                img(x, y) = {clampByte(255 + noise()), clampByte(255 + noise()), clampByte(255 + noise()), 255};
                continue;
            }
            const double t = static_cast<double>(x - x0) / static_cast<double>(w - 1 - x0);
            img(x, y) = {clampByte(a.r + (b.r - a.r) * t + noise()), clampByte(a.g + (b.g - a.g) * t + noise()),
                         clampByte(a.b + (b.b - a.b) * t + noise()), 255};
        }
    }
    return img;
}

struct Bands {
    int regions = 0;           ///< regiões na rampa (colunas ≥ x0)
    double rmseLevels = 0.0;   ///< √(média de ‖cor da região − pixel‖² por canal), em níveis 0-255, sobre a rampa
    double maxWidthShare = 0;  ///< a maior faixa, como fração da largura da rampa
    bool connected = true;     ///< toda região é 4-conexa
    bool ordered = true;       ///< ao longo da linha do meio, a cor das faixas avança sempre no sentido da rampa
    int snapped = 0;           ///< regiões da rampa com paletteIndex ≥ 0
};

/// Mede as faixas de uma segmentação da rampa. `a` e `b` são as pontas da rampa em [0, 1].
Bands measure(const PixelMap& map, const Array2D<Rgba8>& img, int x0, const Rgba& a, const Rgba& b) {
    Bands out;
    const Array2D<RegionId>& labels = map.compactLabels();
    const int w = img.width();
    const int h = img.height();
    std::vector<RegionId> seen;
    double sum = 0.0;
    long n = 0;
    std::vector<int> width(map.forest().capacity(), 0);
    for (int y = 0; y < h; ++y) {
        for (int x = x0; x < w; ++x) {
            const RegionId r = labels(x, y);
            if (std::find(seen.begin(), seen.end(), r) == seen.end()) seen.push_back(r);
            const Rgba c = map.forest().stats(r).color;
            const Rgba p = sla::toRgba(img(x, y));
            sum += (c.r - p.r) * (c.r - p.r) + (c.g - p.g) * (c.g - p.g) + (c.b - p.b) * (c.b - p.b);
            n += 3;
        }
    }
    out.regions = static_cast<int>(seen.size());
    out.rmseLevels = 255.0 * std::sqrt(sum / static_cast<double>(n));
    for (const RegionId r : seen) {
        if (map.forest().stats(r).paletteIndex >= 0) ++out.snapped;
        int cols = 0;
        for (int x = x0; x < w; ++x) cols += labels(x, h / 2) == r ? 1 : 0;
        out.maxWidthShare = std::max(out.maxWidthShare, static_cast<double>(cols) / static_cast<double>(w - x0));
    }
    // Conexão (flood fill) e ordem das faixas.
    std::vector<char> visited(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
    std::vector<RegionId> done;
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h && out.connected; ++y) {
        for (int x = 0; x < w; ++x) {
            if (visited[static_cast<std::size_t>(y * w + x)]) continue;
            const RegionId r = labels(x, y);
            if (std::find(done.begin(), done.end(), r) != done.end()) {
                out.connected = false;
                break;
            }
            done.push_back(r);
            stack.assign(1, {x, y});
            visited[static_cast<std::size_t>(y * w + x)] = 1;
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + cnxtest::kDx4[d];
                    const int ny = cy + cnxtest::kDy4[d];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h || visited[static_cast<std::size_t>(ny * w + nx)] || labels(nx, ny) != r) continue;
                    visited[static_cast<std::size_t>(ny * w + nx)] = 1;
                    stack.emplace_back(nx, ny);
                }
            }
        }
    }
    const double dx = b.r - a.r, dy = b.g - a.g, dz = b.b - a.b;
    const double len2 = dx * dx + dy * dy + dz * dz;
    double last = -1.0;
    for (int x = x0; x < w; ++x) {
        const Rgba c = map.forest().stats(labels(x, h / 2)).color;
        const double s = ((c.r - a.r) * dx + (c.g - a.g) * dy + (c.b - a.b) * dz) / len2;
        if (s < last - 0.01) out.ordered = false;  // avança, a menos de 1% do percurso (o ruído e a mediana por canal)
        last = std::max(last, s);
    }
    return out;
}

constexpr Rgba8 kBlue{30, 100, 230, 255};
constexpr Rgba8 kYellow{250, 200, 20, 255};

}  // namespace

CNX_TEST(a_smooth_ramp_keeps_dozens_of_thin_bands) {
    const Array2D<Rgba8> img = rampImage(260, 40, 20, kBlue, kYellow);
    // Uma paleta que não tem nada a ver com a rampa (o K de uma imagem que o classificador não entende).
    const cnx::Palette wrong = paletteOf({kWhite, Rgba8{200, 30, 30, 255}, Rgba8{20, 20, 20, 255}});
    cnx::SegmentationResult r = PixelSegmenter(img, gradientParams()).run({}, &wrong);
    const Bands b = measure(r.map, img, 20, sla::toRgba(kBlue), sla::toRgba(kYellow));
    std::printf("  rampa: %d faixas, rmse %.2f níveis, maior faixa %.1f%% da rampa, encaixadas %d\n", b.regions, b.rmseLevels, 100.0 * b.maxWidthShare, b.snapped);
    CHECK(b.regions >= 14 && b.regions <= 60);  // ~320 níveis de percurso de cor a ~20 níveis por faixa
    CHECK(b.rmseLevels < 6.0);
    CHECK(b.maxWidthShare < 0.12);  // nenhuma faixa engole mais de ~1/8 da rampa
    CHECK(b.connected && b.ordered);
    CHECK(b.snapped == 0);  // a paleta errada não vira a cor de nenhuma faixa
}

CNX_TEST(the_old_ceiling_and_palette_flatten_the_same_ramp) {
    const Array2D<Rgba8> img = rampImage(260, 40, 20, kBlue, kYellow);
    const cnx::Palette wrong = paletteOf({kWhite, Rgba8{200, 30, 30, 255}, Rgba8{20, 20, 20, 255}});
    cnx::SegmentationResult r = PixelSegmenter(img, oldParams()).run({}, &wrong);
    const Bands b = measure(r.map, img, 20, sla::toRgba(kBlue), sla::toRgba(kYellow));
    std::printf("  rampa, ajuste antigo: %d faixas, rmse %.2f níveis\n", b.regions, b.rmseLevels);
    CHECK(b.regions <= 8);
    CHECK(b.rmseLevels > 20.0);  // as faixas viram as 3 cores da paleta errada
}

CNX_TEST(the_ceiling_alone_keeps_the_bands_even_with_the_old_palette_rule) {
    // Separa as duas causas: só o teto novo (encaixe sempre) ainda dá as faixas, mas com as cores da paleta errada.
    const Array2D<Rgba8> img = rampImage(260, 40, 20, kBlue, kYellow);
    const cnx::Palette wrong = paletteOf({kWhite, Rgba8{200, 30, 30, 255}, Rgba8{20, 20, 20, 255}});
    SegmentParams p = gradientParams();
    p.paletteSnapMax = 4.0;
    cnx::SegmentationResult r = PixelSegmenter(img, p).run({}, &wrong);
    const Bands b = measure(r.map, img, 20, sla::toRgba(kBlue), sla::toRgba(kYellow));
    CHECK(b.regions >= 16);
    CHECK(b.rmseLevels > 20.0);
}

CNX_TEST(the_band_step_follows_the_merge_ceiling) {
    const Array2D<Rgba8> img = rampImage(260, 40, 20, kBlue, kYellow);
    int previous = std::numeric_limits<int>::max();
    for (const double levels : {6.0, 12.0, 24.0, 48.0}) {
        SegmentParams p = gradientParams();
        p.mergeThreshMax = ceilingOf(levels);
        p.mergeThreshStart = std::min(p.mergeThreshStart, p.mergeThreshMax);
        p.paletteSnapMax = p.mergeThreshMax / 9.0;
        cnx::SegmentationResult r = PixelSegmenter(img, p).run({}, nullptr);
        const Bands b = measure(r.map, img, 20, sla::toRgba(kBlue), sla::toRgba(kYellow));
        std::printf("  teto %4.0f níveis: %2d faixas, rmse %.2f\n", levels, b.regions, b.rmseLevels);
        CHECK_CTX(b.regions < previous, "mais teto, menos faixas");
        CHECK_CTX(b.connected, "faixas conexas");
        // Com o teto largo (48 níveis, 5 faixas) uma faixa pode se enrolar atrás da vizinha por uma ponte fina e aparecer
        // duas vezes na linha do meio (docs §1.28, limites); nos tetos realistas a ordem é exata.
        if (levels <= 24.0) CHECK_CTX(b.ordered, "faixas na ordem do degradê");
        previous = b.regions;
    }
}

CNX_TEST(a_snap_tolerance_as_wide_as_the_band_step_adds_error_to_the_ramp) {
    // Tintas sobre a própria rampa (os centros de grupo de um k-means com K = 3): uma faixa a até 12 níveis de uma
    // tinta é puxada para ela e perde a cor que tinha. Com a tolerância do ruído (4 níveis) quase nenhuma é.
    const Array2D<Rgba8> img = rampImage(260, 40, 20, kBlue, kYellow);
    const auto at = [&](double t) {
        return Rgba8{clampByte(kBlue.r + (kYellow.r - kBlue.r) * t), clampByte(kBlue.g + (kYellow.g - kBlue.g) * t),
                     clampByte(kBlue.b + (kYellow.b - kBlue.b) * t), 255};
    };
    const cnx::Palette onTheRamp = paletteOf({at(0.2), at(0.5), at(0.8)});
    const auto rmseWith = [&](double snapLevels, int* snapped) {
        SegmentParams p = gradientParams();
        p.paletteSnapMax = ceilingOf(snapLevels);
        cnx::SegmentationResult r = PixelSegmenter(img, p).run({}, &onTheRamp);
        const Bands b = measure(r.map, img, 20, sla::toRgba(kBlue), sla::toRgba(kYellow));
        *snapped = b.snapped;
        return b.rmseLevels;
    };
    int snapNoise = 0;
    int snapStep = 0;
    const double noise = rmseWith(4.0, &snapNoise);
    const double step = rmseWith(12.0, &snapStep);
    std::printf("  tolerância 4 níveis: rmse %.2f, %d faixas encaixadas; 12 níveis: rmse %.2f, %d encaixadas\n", noise, snapNoise, step, snapStep);
    CHECK(snapStep > snapNoise);
    CHECK(step > noise);
}

CNX_TEST(flat_regions_with_noise_still_snap_to_their_ink) {
    // Duas cores chapadas com ruído de ±1 nível: uma região cada, com a cor exata da paleta (a sujeira some).
    cnxtest::SplitMix64 rng{3};
    Array2D<Rgba8> img(60, 30);
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 60; ++x) {
            const Rgba8 base = x < 30 ? Rgba8{240, 240, 240, 255} : Rgba8{200, 40, 40, 255};
            img(x, y) = {clampByte(base.r + rng.below(3) - 1), clampByte(base.g + rng.below(3) - 1), clampByte(base.b + rng.below(3) - 1), 255};
        }
    }
    const cnx::Palette ink = paletteOf({Rgba8{239, 239, 239, 255}, Rgba8{201, 39, 39, 255}});  // o centro do grupo, a ~1 nível
    cnx::SegmentationResult r = PixelSegmenter(img, gradientParams()).run({}, &ink);
    CHECK(r.map.forest().rootCount() == 2);
    CHECK(r.map.forest().stats(r.map.regionAt(5, 5)).color == ink.colors[0] && r.map.forest().stats(r.map.regionAt(5, 5)).paletteIndex == 0);
    CHECK(r.map.forest().stats(r.map.regionAt(50, 5)).color == ink.colors[1] && r.map.forest().stats(r.map.regionAt(50, 5)).paletteIndex == 1);
}

CNX_TEST(a_flat_background_beside_a_ramp_stays_one_exact_ink) {
    const Array2D<Rgba8> img = rampImage(260, 40, 60, kBlue, kYellow);
    const cnx::Palette p = paletteOf({kWhite, Rgba8{200, 30, 30, 255}, Rgba8{20, 20, 20, 255}});
    cnx::SegmentationResult r = PixelSegmenter(img, gradientParams()).run({}, &p);
    const auto& bg = r.map.forest().stats(r.map.regionAt(10, 20));
    CHECK(bg.paletteIndex == 0 && bg.color == p.colors[0]);  // o fundo é o branco exato da paleta
    for (int y = 0; y < 40; ++y) CHECK(r.map.regionAt(10, y) == r.map.regionAt(10, 20));  // e é uma região só
    const Bands b = measure(r.map, img, 60, sla::toRgba(kBlue), sla::toRgba(kYellow));
    CHECK(b.regions >= 12 && b.snapped == 0 && b.connected && b.ordered);
}

CNX_TEST(a_two_dimensional_gradient_comes_out_in_connected_bands) {
    // Degradê radial: o anel de cada faixa envolve o miolo. As faixas não podem partir-se em pedaços.
    Array2D<Rgba8> img(120, 120);
    cnxtest::SplitMix64 rng{11};
    for (int y = 0; y < 120; ++y) {
        for (int x = 0; x < 120; ++x) {
            const double t = std::min(1.0, std::hypot(x - 60.0, y - 60.0) / 60.0);
            img(x, y) = {clampByte(250 - 200 * t + rng.below(3) - 1), clampByte(120 + 60 * t + rng.below(3) - 1),
                         clampByte(30 + 190 * t + rng.below(3) - 1), 255};
        }
    }
    cnx::SegmentationResult r = PixelSegmenter(img, gradientParams()).run({}, nullptr);
    const Array2D<RegionId>& labels = r.map.compactLabels();
    // Toda região é conexa (a conexão é invariante do PixelMap; aqui conferimos pelo flood fill, sem confiar nele).
    std::vector<char> seen(120 * 120, 0);
    std::vector<RegionId> done;
    std::vector<std::pair<int, int>> stack;
    bool connected = true;
    for (int y = 0; y < 120 && connected; ++y) {
        for (int x = 0; x < 120; ++x) {
            if (seen[static_cast<std::size_t>(y * 120 + x)]) continue;
            const RegionId id = labels(x, y);
            if (std::find(done.begin(), done.end(), id) != done.end()) {
                connected = false;
                break;
            }
            done.push_back(id);
            stack.assign(1, {x, y});
            seen[static_cast<std::size_t>(y * 120 + x)] = 1;
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + cnxtest::kDx4[d];
                    const int ny = cy + cnxtest::kDy4[d];
                    if (nx < 0 || ny < 0 || nx >= 120 || ny >= 120 || seen[static_cast<std::size_t>(ny * 120 + nx)] || labels(nx, ny) != id) continue;
                    seen[static_cast<std::size_t>(ny * 120 + nx)] = 1;
                    stack.emplace_back(nx, ny);
                }
            }
        }
    }
    double sum = 0.0;
    for (int y = 0; y < 120; ++y) {
        for (int x = 0; x < 120; ++x) {
            const Rgba c = r.map.forest().stats(labels(x, y)).color;
            const Rgba p = sla::toRgba(img(x, y));
            sum += (c.r - p.r) * (c.r - p.r) + (c.g - p.g) * (c.g - p.g) + (c.b - p.b) * (c.b - p.b);
        }
    }
    const double rmse = 255.0 * std::sqrt(sum / (3.0 * 120 * 120));
    std::printf("  degradê radial: %zu regiões, rmse %.2f níveis\n", r.map.forest().rootCount(), rmse);
    CHECK(connected);
    CHECK(r.map.forest().rootCount() >= 12 && r.map.forest().rootCount() <= 120);
    CHECK(rmse < 6.0);
}

// ---------------------------------------------------------------- snapColorsToPalette com limite de distância

namespace {

/// Uma linha com uma região por cor (cada cor é uma região de 3 pixels); devolve o mapa já com as cores medianas.
PixelMap regionsOfColors(const std::vector<Rgba8>& colors) {
    const int n = static_cast<int>(colors.size()) * 3;
    PixelMap map(n, 1);
    for (std::size_t k = 0; k < colors.size(); ++k) {
        const RegionId r = map.createRegion();
        for (int i = 0; i < 3; ++i) map.setLabel(static_cast<int>(k) * 3 + i, 0, r, colors[k]);
    }
    map.assignMedianColors();
    return map;
}

}  // namespace

CNX_TEST(the_palette_distance_limit_keeps_regions_without_a_nearby_ink) {
    const Rgba8 near1{250, 250, 250, 255};  // 5 níveis do branco
    const Rgba8 far1{128, 128, 128, 255};   // longe de qualquer tinta
    const cnx::Palette bw = paletteOf({kWhite, Rgba8{0, 0, 0, 255}});
    PixelMap map = regionsOfColors({near1, far1});
    const Rgba farColor = map.forest().stats(map.regionAt(4, 0)).color;
    cnx::snapColorsToPalette(map, bw, ceilingOf(12.0));
    const auto& a = map.forest().stats(map.regionAt(1, 0));
    const auto& b = map.forest().stats(map.regionAt(4, 0));
    CHECK(a.paletteIndex == 0 && a.color == bw.colors[0]);  // perto: encaixa
    CHECK(b.paletteIndex == -1 && b.color == farColor);     // longe: fica como está
    // Sem limite (o padrão) a região cinza vira a tinta mais próxima, como sempre foi.
    PixelMap all = regionsOfColors({near1, far1});
    cnx::snapColorsToPalette(all, bw);
    CHECK(all.forest().stats(all.regionAt(4, 0)).paletteIndex >= 0);
    // Limite 0: só a cor exatamente igual encaixa.
    PixelMap exact = regionsOfColors({kWhite, near1});
    cnx::snapColorsToPalette(exact, bw, 0.0);
    CHECK(exact.forest().stats(exact.regionAt(1, 0)).paletteIndex == 0);
    CHECK(exact.forest().stats(exact.regionAt(4, 0)).paletteIndex == -1);
    // O limite é inclusivo e vale para a distância² da média à tinta.
    const double d2 = sla::distance2(sla::toRgba(near1), bw.colors[0]);
    PixelMap edge = regionsOfColors({near1});
    cnx::snapColorsToPalette(edge, bw, d2 * 1.001);  // folga de arredondamento do float (a média é float)
    CHECK(edge.forest().stats(edge.regionAt(0, 0)).paletteIndex == 0);
    PixelMap below = regionsOfColors({near1});
    cnx::snapColorsToPalette(below, bw, d2 * 0.99);
    CHECK(below.forest().stats(below.regionAt(0, 0)).paletteIndex == -1);
    // Validação.
    CHECK_THROWS(std::invalid_argument, cnx::snapColorsToPalette(map, bw, -1.0));
    CHECK_THROWS(std::invalid_argument, cnx::snapColorsToPalette(map, bw, std::numeric_limits<double>::quiet_NaN()));
}

// ---------------------------------------------------------------- perfis

CNX_TEST(profiles_scale_the_merge_ceiling_by_detail) {
    const auto ceilingFor = [](cnx::ImageProfile profile) {
        cnx::EngineParameters e;
        e.applyProfile(profile);
        return e.segment;
    };
    const double med = ceilingOf(12.0);
    for (const auto m : {cnx::ImageProfile::LogoMedAA, cnx::ImageProfile::LogoMed, cnx::ImageProfile::PhotoMed}) CHECK(std::abs(ceilingFor(m).mergeThreshMax - med) < 1e-15);
    for (const auto h : {cnx::ImageProfile::LogoHighAA, cnx::ImageProfile::LogoHigh, cnx::ImageProfile::PhotoHigh}) CHECK(std::abs(ceilingFor(h).mergeThreshMax - ceilingOf(8.0)) < 1e-15);
    for (const auto l : {cnx::ImageProfile::LogoLowAA, cnx::ImageProfile::LogoLow, cnx::ImageProfile::PhotoLow}) CHECK(std::abs(ceilingFor(l).mergeThreshMax - ceilingOf(24.0)) < 1e-15);
    CHECK(ceilingFor(cnx::ImageProfile::Custom).mergeThreshMax == SegmentParams{}.mergeThreshMax);  // Custom não mexe
    // O início nunca passa do teto (o schedule exige 0 < início ≤ teto), em nenhum perfil.
    for (int i = 0; i <= 9; ++i) {
        const SegmentParams s = ceilingFor(static_cast<cnx::ImageProfile>(i));
        CHECK_CTX(s.mergeThreshStart <= s.mergeThreshMax && s.mergeThreshStart > 0.0, std::to_string(i));
    }
    // Um teto do usuário já baixo, multiplicado pelo fator do alto detalhe, cai abaixo do início: o início o acompanha.
    {
        cnx::EngineParameters e;
        e.segment.mergeThreshStart = 1.0e-4;
        e.segment.mergeThreshMax = 1.5e-4;
        e.applyProfile(cnx::ImageProfile::PhotoHigh);
        CHECK(e.segment.mergeThreshMax < 1.0e-4);
        CHECK(e.segment.mergeThreshStart == e.segment.mergeThreshMax);
        CHECK(!PixelSegmenter::thresholdSchedule(e.segment).empty());  // o cronograma exige 0 < início ≤ teto: não lança
    }
    // Só as fotos e degradês limitam o encaixe na paleta; os logos encaixam sempre.
    for (const auto p : {cnx::ImageProfile::PhotoHigh, cnx::ImageProfile::PhotoMed, cnx::ImageProfile::PhotoLow}) {
        CHECK(std::abs(ceilingFor(p).paletteSnapMax - ceilingFor(p).mergeThreshMax / 9.0) < 1e-15);  // um terço em distância
    }
    for (const auto p : {cnx::ImageProfile::LogoHighAA, cnx::ImageProfile::LogoMed, cnx::ImageProfile::LogoLow, cnx::ImageProfile::Custom}) {
        CHECK(ceilingFor(p).paletteSnapMax == 4.0);
    }
}

CNX_TEST(the_sensitivity_control_still_moves_the_ceiling_after_the_profile) {
    cnx::EngineParameters base;
    cnx::EngineParameters more = base;
    cnx::applyUserControl(more, cnx::UserControl::Sensitivity, 80);
    cnx::EngineParameters less = base;
    cnx::applyUserControl(less, cnx::UserControl::Sensitivity, 20);
    base.applyProfile(cnx::ImageProfile::PhotoMed);
    more.applyProfile(cnx::ImageProfile::PhotoMed);
    less.applyProfile(cnx::ImageProfile::PhotoMed);
    CHECK(more.segment.mergeThreshMax < base.segment.mergeThreshMax && base.segment.mergeThreshMax < less.segment.mergeThreshMax);
    CHECK(std::abs(more.segment.paletteSnapMax - more.segment.mergeThreshMax / 9.0) < 1e-15);
    // Aplicar o perfil duas vezes a partir do mesmo ponto de partida dá o mesmo resultado (cada execução parte dos parâmetros do usuário).
    cnx::EngineParameters a;
    cnx::EngineParameters b;
    a.applyProfile(cnx::ImageProfile::PhotoLow);
    b.applyProfile(cnx::ImageProfile::PhotoLow);
    CHECK(a.segment.mergeThreshMax == b.segment.mergeThreshMax && a.segment.mergeThreshStart == b.segment.mergeThreshStart);
}

// ---------------------------------------------------------------- ponta a ponta

namespace {

class SceneLoader final : public cnx::IImageLoader {
public:
    explicit SceneLoader(Array2D<Rgba8> img) : image_(std::move(img)) {}
    cnx::SourceImage load(const std::filesystem::path&, const std::stop_token&) override {
        cnx::SourceImage s;
        s.pixels = image_;
        s.formatHint = ".png";
        return s;
    }

private:
    Array2D<Rgba8> image_;
};

/// Uma fita inclinada com degradê vertical de azul a amarelo (passando por verde) em fundo branco: a cobertura da
/// borda vem de uma cena de máscara com antialiasing exato, e a cor da fita varia com y.
Array2D<Rgba8> ribbonImage(int w, int h, Array2D<std::uint8_t>* interior = nullptr) {
    cnxtest::LogoScene mask{"mascara", w, h, kWhite, {}, 2};
    mask.shapes.push_back({cnxtest::polygonShape({{30.0, 8.0}, {w - 10.0, 8.0}, {w - 30.0, h - 8.0}, {10.0, h - 8.0}}), Rgba8{0, 0, 0, 255}});
    const Array2D<Rgba8> cov = cnxtest::renderScene(mask);
    Array2D<Rgba8> img(w, h);
    if (interior != nullptr) {  // pixels cujos 8 vizinhos também estão inteiros dentro da fita (cobertura total)
        *interior = Array2D<std::uint8_t>(w, h, 0);
        for (int y = 1; y + 1 < h; ++y) {
            for (int x = 1; x + 1 < w; ++x) {
                bool all = true;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) all = all && cov(x + dx, y + dy).r == 0;
                }
                (*interior)(x, y) = all ? 1 : 0;
            }
        }
    }
    for (int y = 0; y < h; ++y) {
        const double t = std::clamp((y - 8.0) / (h - 16.0), 0.0, 1.0);
        const double r = t < 0.5 ? 30 + 2 * t * (90 - 30) : 90 + (t - 0.5) * 2 * (250 - 90);
        const double g = t < 0.5 ? 120 + 2 * t * (190 - 120) : 190 + (t - 0.5) * 2 * (210 - 190);
        const double b = t < 0.5 ? 235 - 2 * t * (235 - 90) : 90 - (t - 0.5) * 2 * (90 - 20);
        for (int x = 0; x < w; ++x) {
            const double c = 1.0 - cov(x, y).r / 255.0;  // cobertura da fita no pixel
            img(x, y) = {clampByte(255 + (r - 255) * c), clampByte(255 + (g - 255) * c), clampByte(255 + (b - 255) * c), 255};
        }
    }
    return img;
}

struct EngineResult {
    std::shared_ptr<const cnx::PipelineSnapshot> snapshot;
    cnxtest::svg::FitError fit;
    Array2D<Rgba> raster;  ///< o SVG do estágio de Bézier rasterizado
    std::size_t faces = 0;
};

EngineResult runEngine(const Array2D<Rgba8>& img, const std::function<void(cnx::EngineParameters&)>& edit = {}) {
    cnx::CoreEngine engine(std::make_shared<SceneLoader>(img));
    engine.setInput("cena.png");
    cnx::EngineParameters params = engine.parameters();
    if (edit) edit(params);
    engine.setParameters(params);
    engine.start(cnx::Stage::FitBezier);
    if (engine.wait() != cnx::RunStatus::Completed) throw std::runtime_error("o motor falhou: " + engine.lastMessage());
    EngineResult out;
    out.snapshot = engine.snapshot();
    const cnx::ExporterSettings settings;
    const std::string svg = cnxtest::svg::exportSvg(*out.snapshot->fitted, settings);
    out.raster = cnxtest::svg::rasterize(cnxtest::svg::parseSvg(svg), img.width(), img.height(), 4);
    out.fit = cnxtest::svg::compare(out.raster, cnx::SignalImage::fromStraight(img).premultiplied());
    out.faces = out.snapshot->fitted->faceCount();
    return out;
}

/// rmse (níveis 0-255) só sobre os pixels do miolo, longe da borda com antialiasing.
double interiorRmse(const EngineResult& r, const Array2D<Rgba8>& img, const Array2D<std::uint8_t>& interior) {
    double sum = 0.0;
    long n = 0;
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            if (!interior(x, y)) continue;
            const Rgba g = r.raster(x, y);
            const Rgba p = sla::toRgba(img(x, y));
            sum += (g.r - p.r) * (g.r - p.r) + (g.g - p.g) * (g.g - p.g) + (g.b - p.b) * (g.b - p.b);
            n += 3;
        }
    }
    return 255.0 * std::sqrt(sum / static_cast<double>(n));
}

}  // namespace

CNX_TEST(the_engine_vectorizes_a_gradient_ribbon_in_many_bands_with_no_help) {
    Array2D<std::uint8_t> interior;
    const Array2D<Rgba8> img = ribbonImage(140, 110, &interior);
    const EngineResult now = runEngine(img);  // tudo no automático: classificador, K da paleta, perfil
    const cnx::ImageProfile profile = now.snapshot->classification->profile;
    const double nowInterior = interiorRmse(now, img, interior);
    std::printf("  fita (automático): perfil %d, %zu tintas, %zu faces, rmse %.1f níveis (miolo %.1f)\n", static_cast<int>(profile),
                now.snapshot->palette->colors.size(), now.faces, now.fit.rmse * 255.0, nowInterior);
    CHECK(cnx::isPhoto(profile));  // muitos picos de cor: o classificador manda para o caminho de foto/degradê
    CHECK(now.faces >= 14);        // dezenas de faixas, não as 7 de antes
    CHECK(nowInterior < 6.0);      // o miolo: as faixas seguem o degradê
    CHECK(now.fit.rmse * 255.0 < 9.0);  // com a borda, que o caminho de foto não trata como antialiasing

    // O ajuste de antes (teto de 40 níveis, encaixe sempre) na mesma imagem.
    const EngineResult before = runEngine(img, [](cnx::EngineParameters& p) {
        p.segment.mergeThreshMax = ceilingOf(40.0 / 1.0);
        p.segment.paletteSnapMax = 4.0;
        p.classifier.autoClassify = false;
        p.classifier.forcedProfile = cnx::ImageProfile::Custom;  // Custom: o perfil não reescreve os dois campos
    });
    const double beforeInterior = interiorRmse(before, img, interior);
    std::printf("  fita (ajuste antigo): %zu faces, rmse %.1f níveis (miolo %.1f)\n", before.faces, before.fit.rmse * 255.0, beforeInterior);
    CHECK(before.faces < now.faces);
    CHECK(beforeInterior > nowInterior * 2.0);
}

int main() { return cnxtest::runAll(); }
