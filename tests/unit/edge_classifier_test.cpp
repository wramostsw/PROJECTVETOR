// tests/unit/edge_classifier_test.cpp — EdgeImage (Sobel + tensor de estrutura + cristas), detector de AA por
// perfis de 6 pixels, as 20 medidas e a árvore do classificador, num corpus sintético de famílias conhecidas.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

#include "cnx/classify/Classifier.h"
#include "cnx/palette/PaletteFinder.h"
#include "cnx/preprocess/AADetector.h"
#include "cnx/preprocess/EdgeImage.h"
#include "cnx/preprocess/Preprocessor.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::EdgeImage;
using cnx::EdgeProfileKind;
using cnx::ImageProfile;
using sla::Array2D;
using sla::Rgba;
using sla::Rgba8;

namespace {

Rgba8 rgb(int r, int g, int b) { return {static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b), 255}; }

// ---------------------------------------------------------------- cenas sintéticas

struct Shape {
    bool disk = false;
    double x0 = 0, y0 = 0, x1 = 0, y1 = 0;  ///< retângulo [x0, x1) × [y0, y1)
    double cx = 0, cy = 0, r = 0;           ///< disco
    Rgba8 color;
    bool contains(double x, double y) const {
        return disk ? (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r : x >= x0 && x < x1 && y >= y0 && y < y1;
    }
};

/// ss = 1: amostra no centro do pixel (borda dura, com escadinha nos discos); ss > 1: superamostragem ss×ss
/// (antialiasing de filtro-caixa). A última forma que contém o ponto pinta.
Array2D<Rgba8> render(int w, int h, Rgba8 bg, const std::vector<Shape>& shapes, int ss) {
    Array2D<Rgba8> img(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::array<double, 4> acc{};
            for (int j = 0; j < ss; ++j) {
                for (int i = 0; i < ss; ++i) {
                    const double px = x + (i + 0.5) / ss;
                    const double py = y + (j + 0.5) / ss;
                    Rgba8 c = bg;
                    for (const Shape& s : shapes) {
                        if (s.contains(px, py)) c = s.color;
                    }
                    acc[0] += c.r;
                    acc[1] += c.g;
                    acc[2] += c.b;
                    acc[3] += c.a;
                }
            }
            const double n = ss * ss;
            const auto q = [&](double v) { return static_cast<std::uint8_t>(std::clamp(std::lround(v / n), 0L, 255L)); };
            img(x, y) = {q(acc[0]), q(acc[1]), q(acc[2]), q(acc[3])};
        }
    }
    return img;
}

/// Paleta de tintas bem separadas (contraste ≥ 0,3 entre quaisquer duas).
const std::array<Rgba8, 8> kInks{rgb(255, 255, 255), rgb(20, 20, 20), rgb(220, 30, 40), rgb(30, 90, 200),
                                 rgb(250, 200, 20), rgb(30, 160, 70), rgb(140, 40, 160), rgb(255, 130, 0)};

/// Logo aleatório: fundo + 2–5 formas. `aligned`: retângulos em coordenadas inteiras (borda dura exata).
std::vector<Shape> randomLogo(cnxtest::SplitMix64& rng, int w, int h, Rgba8& bg, bool aligned) {
    std::vector<Rgba8> inks(kInks.begin(), kInks.end());
    for (std::size_t i = inks.size(); i > 1; --i) std::swap(inks[i - 1], inks[static_cast<std::size_t>(rng.below(static_cast<int>(i)))]);
    bg = inks[0];
    std::vector<Shape> shapes;
    const int count = 2 + rng.below(4);
    for (int k = 0; k < count; ++k) {
        Shape s;
        s.color = inks[static_cast<std::size_t>(1 + k % 7)];
        s.disk = rng.below(2) == 0;
        const double frac = aligned ? 0.0 : 0.2 + 0.6 * cnxtest::uniform01(rng);
        if (s.disk) {
            s.r = 7 + rng.below(12) + frac;
            s.cx = s.r + 2 + rng.below(std::max(1, w - 2 * static_cast<int>(s.r) - 4)) + frac;
            s.cy = s.r + 2 + rng.below(std::max(1, h - 2 * static_cast<int>(s.r) - 4)) + frac;
        } else {
            const int rw = 10 + rng.below(w / 2);
            const int rh = 10 + rng.below(h / 2);
            s.x0 = 2 + rng.below(std::max(1, w - rw - 4)) + frac;
            s.y0 = 2 + rng.below(std::max(1, h - rh - 4)) + frac;
            s.x1 = s.x0 + rw;
            s.y1 = s.y0 + rh;
        }
        shapes.push_back(s);
    }
    return shapes;
}

/// "Foto": gradiente entre duas cores + textura de duas frequências + ruído.
Array2D<Rgba8> randomPhoto(cnxtest::SplitMix64& rng, int w, int h) {
    std::array<double, 3> a{}, b{}, amp{}, phase{};
    for (std::size_t c = 0; c < 3; ++c) {
        a[c] = 40 + rng.below(80);
        b[c] = 140 + rng.below(100);
        amp[c] = 20 + rng.below(30);
        phase[c] = cnxtest::uniform01(rng) * 6.28;
    }
    const double fx = 0.15 + 0.3 * cnxtest::uniform01(rng);
    const double fy = 0.1 + 0.3 * cnxtest::uniform01(rng);
    Array2D<Rgba8> img(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::array<int, 3> v{};
            for (std::size_t c = 0; c < 3; ++c) {
                const double t = (x + y) / static_cast<double>(w + h);
                const double val = a[c] + (b[c] - a[c]) * t + amp[c] * std::sin(fx * x + fy * y * (1.0 + 0.3 * c) + phase[c]) +
                                   0.5 * amp[c] * std::sin(0.9 * fx * y - 1.7 * fy * x + phase[c]) + (rng.below(31) - 15);
                v[c] = std::clamp(static_cast<int>(std::lround(val)), 0, 255);
            }
            img(x, y) = rgb(v[0], v[1], v[2]);
        }
    }
    return img;
}

Array2D<Rgba8> addNoise(Array2D<Rgba8> img, cnxtest::SplitMix64& rng, int amplitude) {
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            Rgba8& p = img(x, y);
            const auto n = [&](std::uint8_t v) { return static_cast<std::uint8_t>(std::clamp(v + rng.below(2 * amplitude + 1) - amplitude, 0, 255)); };
            p = {n(p.r), n(p.g), n(p.b), p.a};
        }
    }
    return img;
}

cnx::ImageClassification classifyImage(const Array2D<Rgba8>& img, const std::string& hint = ".png") {
    cnx::SourceImage src;
    src.pixels = img;
    src.formatHint = hint;
    cnx::TaskContext ctx;
    const cnx::PreprocessedImage pre = cnx::preprocess(src, cnx::PreprocessParams{}, ctx);
    return cnx::classify(pre, src, cnx::ClassifierParams{}, cnx::PaletteParams{});
}

std::array<Rgba, 6> profile(std::array<float, 6> grays) {
    std::array<Rgba, 6> out{};
    for (std::size_t k = 0; k < 6; ++k) out[k] = {grays[k], grays[k], grays[k], 1.0f};
    return out;
}

}  // namespace

// ================================================================ EdgeImage

CNX_TEST(edge_image_of_flat_and_empty_images) {
    const EdgeImage flat = EdgeImage::compute(Array2D<Rgba8>(9, 7, rgb(90, 90, 90)));
    CHECK(flat.width() == 9 && flat.height() == 7);
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 9; ++x) CHECK(flat.magnitude(x, y) == 0.0f && flat.normal(x, y) == sla::Vec2(0, 0));
    }
    CHECK(flat.ridges(0.0f).empty());
    const Array2D<float> w = flat.sharpnessWeights(0.1);
    CHECK(w(4, 3) == 1.0f);
    const EdgeImage empty = EdgeImage::compute(Array2D<Rgba8>());
    CHECK(empty.width() == 0 && empty.ridges(0.1f).empty());
    CHECK_THROWS(std::invalid_argument, (void)flat.sharpnessWeights(0.0));
    CHECK_THROWS(std::invalid_argument, (void)flat.sharpnessWeights(-1.0));
    CHECK_THROWS(std::invalid_argument, (void)flat.sharpnessWeights(std::numeric_limits<double>::infinity()));
}

CNX_TEST(edge_image_steps_have_the_right_magnitude_normal_and_one_ridge_per_crossing) {
    // Degrau vertical preto|branco em x = 4: m = √3 nas colunas 3 e 4, normal (1, 0), crista só na coluna 3.
    Array2D<Rgba8> v(8, 6);
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 8; ++x) v(x, y) = x < 4 ? rgb(0, 0, 0) : rgb(255, 255, 255);
    }
    const EdgeImage ev = EdgeImage::compute(v);
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 8; ++x) {
            const bool near = x == 3 || x == 4;
            CHECK(std::abs(ev.magnitude(x, y) - (near ? std::sqrt(3.0f) : 0.0f)) < 1e-5f);
            if (near) CHECK(std::abs(std::abs(ev.normal(x, y).x) - 1.0) < 1e-6 && std::abs(ev.normal(x, y).y) < 1e-6);
        }
    }
    const auto rv = ev.ridges(0.5f);
    CHECK(rv.size() == 6);
    for (const cnx::EdgePoint& p : rv) CHECK(p.x == 3 && std::abs(p.magnitude - std::sqrt(3.0f)) < 1e-5f);
    // Degrau horizontal: normal (0, 1) e uma crista por coluna.
    Array2D<Rgba8> hz(6, 8);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 6; ++x) hz(x, y) = y < 5 ? rgb(200, 0, 0) : rgb(0, 0, 0);
    }
    const EdgeImage eh = EdgeImage::compute(hz);
    CHECK(std::abs(std::abs(eh.normal(2, 4).y) - 1.0) < 1e-6 && std::abs(eh.normal(2, 4).x) < 1e-6);
    CHECK(std::abs(eh.magnitude(2, 4) - 200.0f / 255.0f) < 1e-5f);  // degrau de amplitude a num canal dá a
    const auto rh = eh.ridges(0.1f);
    CHECK(rh.size() == 6 && std::all_of(rh.begin(), rh.end(), [](const cnx::EdgePoint& p) { return p.y == 4; }));
    // Limiar respeitado.
    CHECK(eh.ridges(0.8f).empty());
    // Canais com gradientes opostos (vermelho sobe, verde desce): o tensor não deixa um cancelar o outro.
    Array2D<Rgba8> opp(8, 5);
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 8; ++x) opp(x, y) = x < 4 ? rgb(0, 255, 0) : rgb(255, 0, 0);
    }
    const EdgeImage eo = EdgeImage::compute(opp);
    CHECK(std::abs(eo.magnitude(3, 2) - std::sqrt(2.0f)) < 1e-5f);
    CHECK(std::abs(std::abs(eo.normal(3, 2).x) - 1.0) < 1e-6);
    // Degrau diagonal: normal a 45°.
    Array2D<Rgba8> dg(12, 12);
    for (int y = 0; y < 12; ++y) {
        for (int x = 0; x < 12; ++x) dg(x, y) = x + y < 12 ? rgb(0, 0, 0) : rgb(255, 255, 255);
    }
    const EdgeImage ed = EdgeImage::compute(dg);
    const sla::Vec2 nd = ed.normal(6, 6);
    CHECK(std::abs(std::abs(nd.x) - std::sqrt(0.5)) < 1e-6 && std::abs(std::abs(nd.y) - std::sqrt(0.5)) < 1e-6 && nd.x * nd.y > 0);
}

CNX_TEST(edge_image_matches_the_palette_weights_and_ridges_follow_a_circle) {
    cnxtest::SplitMix64 rng{4401};
    Array2D<Rgba8> img(23, 17);
    for (int y = 0; y < 17; ++y) {
        for (int x = 0; x < 23; ++x) img(x, y) = {static_cast<std::uint8_t>(rng.below(256)), static_cast<std::uint8_t>(rng.below(256)),
                                                   static_cast<std::uint8_t>(rng.below(256)), static_cast<std::uint8_t>(rng.below(256))};
    }
    const EdgeImage e = EdgeImage::compute(img);
    const Array2D<float> legacy = cnx::PaletteFinder::edgeMagnitude(img);
    cnx::PaletteParams pp;
    const Array2D<float> weights = cnx::PaletteFinder(pp, 1).pixelWeights(img);
    const Array2D<float> mine = e.sharpnessWeights(pp.edgeScale);
    for (int y = 0; y < 17; ++y) {
        for (int x = 0; x < 23; ++x) {
            CHECK(legacy(x, y) == e.magnitude(x, y) && weights(x, y) == mine(x, y));
            CHECK(std::abs(mine(x, y) - 1.0f / (1.0f + e.magnitude(x, y) / static_cast<float>(pp.edgeScale))) < 1e-6f);
            const sla::Vec2 n = e.normal(x, y);
            if (e.magnitude(x, y) > 0.0f) CHECK(std::abs(sla::norm(n) - 1.0) < 1e-5);
        }
    }
    // Disco duro de raio 20: em torno de uma crista por pixel de perímetro (2πr ≈ 126), todas a ~r do centro.
    Rgba8 bg = rgb(255, 255, 255);
    Shape d;
    d.disk = true;
    d.cx = 32.3;
    d.cy = 31.6;
    d.r = 20.0;
    d.color = rgb(0, 0, 0);
    const Array2D<Rgba8> disk = render(64, 64, bg, {d}, 1);
    const auto ridges = EdgeImage::compute(disk).ridges(0.2f);
    const double perimeter = 2.0 * std::numbers::pi * d.r;
    std::printf("  disco duro r=20: %zu cristas (perímetro %.1f)\n", ridges.size(), perimeter);
    CHECK(ridges.size() > 0.75 * perimeter && ridges.size() < 1.25 * perimeter);
    for (const cnx::EdgePoint& p : ridges) {
        const double dist = std::hypot(p.x + 0.5 - d.cx, p.y + 0.5 - d.cy);
        CHECK(std::abs(dist - d.r) < 1.6);
        // A normal aponta (como eixo) para o centro.
        const sla::Vec2 radial{(p.x + 0.5 - d.cx) / dist, (p.y + 0.5 - d.cy) / dist};
        CHECK(std::abs(sla::dot(radial, p.normal)) > 0.8);
    }
}

// ================================================================ perfis de 6 pixels

CNX_TEST(edge_profiles_are_classified_as_step_ramp_or_other) {
    using K = EdgeProfileKind;
    const auto kind = [](std::array<float, 6> g) { return cnx::analyzeEdgeProfile(profile(g)); };
    CHECK(kind({0, 0, 0, 1, 1, 1}).kind == K::Step && kind({0, 0, 0, 1, 1, 1}).intermediate == 0);
    CHECK(kind({0, 0, 0, 0, 0, 1}).kind == K::Step);
    CHECK(kind({1, 1, 0, 0, 0, 0}).kind == K::Step);  // descendo também
    CHECK(kind({0, 0, 0.05f, 0.95f, 1, 1}).kind == K::Step);  // ruído dentro da tolerância
    const cnx::EdgeProfile one = kind({0, 0, 0.5f, 1, 1, 1});
    CHECK(one.kind == K::Ramp && one.intermediate == 1);
    const cnx::EdgeProfile two = kind({0, 0, 0.33f, 0.67f, 1, 1});
    CHECK(two.kind == K::Ramp && two.intermediate == 2);
    const cnx::EdgeProfile four = kind({0, 0.2f, 0.4f, 0.6f, 0.8f, 1});
    CHECK(four.kind == K::Ramp && four.intermediate == 4);
    // Limites da tolerância de patamar (0,12).
    CHECK(kind({0, 0, 0.11f, 1, 1, 1}).kind == K::Step);
    CHECK(kind({0, 0, 0.13f, 1, 1, 1}).kind == K::Ramp);
    CHECK(kind({0, 0, 0.87f, 1, 1, 1}).kind == K::Ramp);
    CHECK(kind({0, 0, 0.89f, 1, 1, 1}).kind == K::Step);
    // Outros: sem contraste, vai e volta, transbordo (ringing), terceira cor.
    CHECK(kind({0.5f, 0.5f, 0.52f, 0.55f, 0.56f, 0.56f}).kind == K::Other);  // contraste 0,06·√3 ≈ 0,10 < 0,15
    CHECK(kind({0.5f, 0.5f, 0.5f, 0.55f, 0.6f, 0.6f}).kind == K::Ramp);    // 0,1·√3 ≈ 0,17: já é borda (fraca)
    CHECK(kind({0, 1, 0, 1, 1, 1}).kind == K::Other);
    CHECK(kind({0, 0, 0, 1.3f, 1, 1}).kind == K::Other);
    CHECK(kind({0, 0, -0.2f, 1, 1, 1}).kind == K::Other);
    // Transbordo em passos pequenos (cada um dentro da tolerância de retrocesso): só o limite absoluto pega.
    CHECK(kind({0, -0.1f, -0.2f, 1, 1, 1}).kind == K::Other);
    CHECK(kind({0, 0, 1.3f, 1.2f, 1.1f, 1}).kind == K::Other);
    CHECK(kind({0, 0, 0.6f, 0.47f, 1, 1}).kind == K::Ramp);   // retrocesso de 0,13 < 0,15: tolerado
    CHECK(kind({0, 0, 0.7f, 0.45f, 1, 1}).kind == K::Other);  // 0,25: não
    std::array<Rgba, 6> third = profile({0, 0, 0, 1, 1, 1});
    third[2] = {1.0f, 0.0f, 0.0f, 1.0f};  // vermelho no meio de um degrau preto→branco
    CHECK(cnx::analyzeEdgeProfile(third).kind == K::Other);
    // A mistura de duas cores quaisquer é rampa, não terceira cor.
    std::array<Rgba, 6> blend{};
    const Rgba red{0.9f, 0.1f, 0.1f, 1.0f};
    const Rgba blue{0.1f, 0.2f, 0.8f, 1.0f};
    for (std::size_t k = 0; k < 6; ++k) blend[k] = k < 3 ? red : blue;
    blend[3] = red * 0.5f + blue * 0.5f;
    CHECK(cnx::analyzeEdgeProfile(blend).kind == K::Ramp);
    // Contraste mínimo configurável.
    cnx::AAParams strict;
    strict.minContrast = 2.0;
    CHECK(cnx::analyzeEdgeProfile(profile({0, 0, 0, 1, 1, 1}), strict).kind == K::Other);
}

CNX_TEST(the_detector_tells_hard_antialiased_and_blurred_edges_apart) {
    Rgba8 bg = rgb(255, 255, 255);
    Shape disk;
    disk.disk = true;
    disk.cx = 40.3;
    disk.cy = 39.7;
    disk.r = 26.0;
    disk.color = rgb(20, 20, 160);
    const auto stats = [](const Array2D<Rgba8>& img) { return cnx::detectAntialiasing(img, EdgeImage::compute(img)); };
    const cnx::AAStats hard = stats(render(80, 80, bg, {disk}, 1));
    const cnx::AAStats aa = stats(render(80, 80, bg, {disk}, 8));
    cnx::TaskContext ctx;
    const cnx::AAStats blur = stats(cnx::gaussianBlur(render(80, 80, bg, {disk}, 1), 1.5));
    std::printf("  duro: %zu degraus %zu rampas %zu outros (largura %.2f)\n", hard.steps, hard.ramps, hard.others, hard.meanTransitionWidth());
    std::printf("  AA:   %zu degraus %zu rampas %zu outros (largura %.2f)\n", aa.steps, aa.ramps, aa.others, aa.meanTransitionWidth());
    std::printf("  blur: %zu degraus %zu rampas %zu outros (largura %.2f)\n", blur.steps, blur.ramps, blur.others, blur.meanTransitionWidth());
    CHECK(hard.valid() > 80 && hard.rampShare() < 0.05 && hard.meanTransitionWidth() < 0.1);
    CHECK(aa.valid() > 80 && aa.rampShare() > 0.6 && aa.meanTransitionWidth() >= 0.5 && aa.meanTransitionWidth() <= 2.0);
    CHECK(blur.valid() > 80 && blur.rampShare() > 0.9 && blur.meanTransitionWidth() > aa.meanTransitionWidth() + 0.8);
    // Histograma de larguras coerente com as contagens.
    std::size_t sum = 0;
    for (const std::size_t c : aa.rampWidths) sum += c;
    CHECK(sum == aa.valid() && aa.rampWidths[0] == aa.steps);
    CHECK(aa.ridges == aa.steps + aa.ramps + aa.others);
    // Retângulo de bordas inteiras: só degraus (sem perfis cortados, ele fica longe da moldura).
    Shape rect;
    rect.x0 = 20;
    rect.y0 = 18;
    rect.x1 = 60;
    rect.y1 = 50;
    rect.color = rgb(0, 0, 0);
    const cnx::AAStats box = stats(render(80, 80, bg, {rect}, 4));
    CHECK(box.ramps == 0 && box.steps > 100 && box.rampShare() == 0.0);
    // Degrau diagonal com uma linha de meio-tom (x + y = 31): normal (√½, √½), amostras arredondadas ao pixel
    // mais próximo — a crista no meio-tom tem exatamente 1 amostra intermediária (truncar daria 2).
    Array2D<Rgba8> diag(32, 32);
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) diag(x, y) = x + y < 31 ? rgb(0, 0, 0) : x + y == 31 ? rgb(128, 128, 128) : rgb(255, 255, 255);
    }
    const cnx::AAStats dg = stats(diag);
    std::printf("  diagonal: %zu degraus %zu rampas (larg. 1: %zu, 2: %zu) %zu outros\n", dg.steps, dg.ramps, dg.rampWidths[1], dg.rampWidths[2], dg.others);
    CHECK(dg.ramps >= 10 && dg.rampWidths[1] == dg.ramps && dg.steps >= 10);
    // Sem cristas: tudo zero, sem divisão por zero.
    const cnx::AAStats none = stats(Array2D<Rgba8>(10, 10, bg));
    CHECK(none.ridges == 0 && none.rampShare() == 0.0 && none.meanTransitionWidth() == 0.0);
    // Perfil que sai da imagem conta como outro.
    Array2D<Rgba8> edgeAtBorder(8, 8);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) edgeAtBorder(x, y) = x < 1 ? rgb(0, 0, 0) : rgb(255, 255, 255);
    }
    const cnx::AAStats clipped = stats(edgeAtBorder);
    CHECK(clipped.ridges > 0 && clipped.others == clipped.ridges && clipped.valid() == 0);
    CHECK_THROWS(std::invalid_argument, (void)cnx::detectAntialiasing(Array2D<Rgba8>(4, 4), EdgeImage::compute(Array2D<Rgba8>(5, 4))));
}

// ================================================================ medidas

CNX_TEST(measures_of_a_flat_logo_an_antialiased_logo_and_a_photo) {
    Rgba8 bg = rgb(255, 255, 255);
    Shape a;
    a.x0 = 8;
    a.y0 = 10;
    a.x1 = 40;
    a.y1 = 34;
    a.color = rgb(220, 30, 40);
    Shape b;
    b.disk = true;
    b.cx = 50.4;
    b.cy = 44.6;
    b.r = 14;
    b.color = rgb(30, 90, 200);
    const auto measure = [](const Array2D<Rgba8>& img, const std::string& hint = ".png") { return classifyImage(img, hint).measures; };

    const cnx::ImageMeasures flat = measure(render(72, 64, bg, {a, b}, 1));
    CHECK(flat.significantColors == 3 && flat.flatShare == 1.0 && flat.mainPeaks == 3);
    CHECK(flat.peakShare == 1.0 && flat.betweenShare == 0.0);
    CHECK(flat.quantizationRmse < 1e-9 && flat.edgeErrorShare == 0.0 && flat.interiorNoise == 0.0);
    CHECK(flat.rampShare < 0.05 && flat.edgeProfiles > 50);
    CHECK(flat.distinctColorRatio == 3.0 / (72 * 64));
    CHECK(flat.quadTreeLeavesPerBlock >= 1.0);
    CHECK(!flat.lossyFormat && measure(render(72, 64, bg, {a, b}, 1), ".JPEG").lossyFormat);

    Shape a2 = a;  // bordas fracionárias: o retângulo também ganha AA
    a2.x0 += 0.5;
    a2.y0 += 0.4;
    a2.x1 += 0.3;
    a2.y1 += 0.6;
    const cnx::ImageMeasures aa = measure(render(72, 64, bg, {a2, b}, 8));
    std::printf("  AA: picos %d cobertura %.3f+%.3f rmse %.3f erro-borda %.3f bordas %.3f rampas %.2f (%zu perfis)\n", aa.mainPeaks, aa.peakShare,
                aa.betweenShare, aa.quantizationRmse, aa.edgeErrorShare, aa.edgeFraction, aa.rampShare, aa.edgeProfiles);
    CHECK(aa.betweenShare > 0.01 && aa.peakShare + aa.betweenShare > 0.97);
    CHECK(aa.edgeErrorShare > 0.95 && aa.rampShare > 0.5);
    CHECK(aa.significantColors >= 3 && aa.flatShare < 1.0);

    cnxtest::SplitMix64 rng{4402};
    const cnx::ImageMeasures photo = measure(randomPhoto(rng, 72, 64));
    std::printf("  foto: picos %d cobertura %.3f+%.3f rmse %.3f erro-borda %.3f bordas %.3f ruído %.3f folhas/bloco %.1f\n", photo.mainPeaks,
                photo.peakShare, photo.betweenShare, photo.quantizationRmse, photo.edgeErrorShare, photo.edgeFraction, photo.interiorNoise,
                photo.quadTreeLeavesPerBlock);
    CHECK(photo.peakShare + photo.betweenShare < 0.8 && photo.quantizationRmse > 0.05);
    CHECK(photo.interiorNoise > 0.02 && photo.distinctColorRatio > 0.5);
    CHECK(photo.quadTreeLeavesPerBlock > 4.0 * flat.quadTreeLeavesPerBlock);

    // Validação e cancelamento.
    cnx::PreprocessedImage emptyPre;
    cnx::TaskContext ctx;
    CHECK_THROWS(std::invalid_argument, (void)cnx::measureImage(emptyPre, cnx::SourceImage{}, cnx::PaletteParams{}, ctx));
    cnx::PreprocessedImage pre;
    pre.image = render(16, 16, bg, {a}, 1);
    std::stop_source stop;
    stop.request_stop();
    cnx::TaskContext cancelled(stop.get_token(), nullptr, cnx::ProgressRange{});
    CHECK_THROWS(cnx::CancelTaskException, (void)cnx::measureImage(pre, cnx::SourceImage{}, cnx::PaletteParams{}, cancelled));
}

CNX_TEST(exact_values_of_the_simpler_measures) {
    cnx::TaskContext ctx;
    const auto measureRaw = [&](const Array2D<Rgba8>& img, bool alpha = false) {
        cnx::PreprocessedImage pre;
        pre.image = img;
        cnx::SourceImage src;
        src.hasAlpha = alpha;
        return cnx::measureImage(pre, src, cnx::PaletteParams{}, ctx);
    };
    // Degrau vertical 8×6: 6 cristas (uma por linha), 12 pixels com m = √3.
    Array2D<Rgba8> step(8, 6);
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 8; ++x) step(x, y) = x < 4 ? rgb(0, 0, 0) : rgb(255, 255, 255);
    }
    const cnx::ImageMeasures s = measureRaw(step);
    CHECK(s.strongEdgeFraction == 6.0 / 48.0);
    CHECK(std::abs(s.meanEdgeMagnitude - 12.0 * std::sqrt(3.0) / 48.0) < 1e-6);
    CHECK(s.edgeFraction == 12.0 / 48.0);
    CHECK(s.edgeProfiles == 6 && s.stepShare == 1.0 && s.rampShare == 0.0 && s.otherShare == 0.0 && s.meanTransitionWidth == 0.0);
    // Estreita demais para o perfil de 6 pixels: todas as cristas viram "outro".
    Array2D<Rgba8> narrow(5, 20);
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 5; ++x) narrow(x, y) = x < 2 ? rgb(0, 0, 0) : rgb(255, 255, 255);
    }
    const cnx::ImageMeasures n = measureRaw(narrow);
    CHECK(n.otherShare == 1.0 && n.edgeProfiles == 0 && n.stepShare == 0.0 && n.strongEdgeFraction > 0.0);
    // Um degrau inteiro (x = 3|4) e uma borda cortada pela moldura (x = 10|11): stepShare é sobre os válidos,
    // otherShare sobre as cristas.
    Array2D<Rgba8> mixed(12, 6);
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 12; ++x) mixed(x, y) = x < 4 ? rgb(0, 0, 0) : x == 11 ? rgb(255, 0, 0) : rgb(255, 255, 255);
    }
    const cnx::ImageMeasures mx = measureRaw(mixed);
    CHECK(mx.edgeProfiles == 6 && mx.stepShare == 1.0 && mx.otherShare == 0.5 && mx.strongEdgeFraction == 12.0 / 72.0);
    // QuadTree: imagem uniforme 40×40 = 3×3 blocos iniciais, cada um uma folha.
    const cnx::ImageMeasures u = measureRaw(Array2D<Rgba8>(40, 40, rgb(9, 9, 9)), true);
    CHECK(u.quadTreeLeavesPerBlock == 1.0 && u.hasAlpha && !measureRaw(step).hasAlpha);
    CHECK(u.mainPeaks == 1 && u.significantColors == 1 && u.edgeProfiles == 0 && u.otherShare == 0.0);
    // Borda só no alfa também é borda (os 4 canais contam).
    Array2D<Rgba8> alpha(8, 6, rgb(100, 100, 100));
    for (int y = 0; y < 6; ++y) {
        for (int x = 4; x < 8; ++x) alpha(x, y).a = 0;
    }
    const EdgeImage ea = EdgeImage::compute(alpha);
    CHECK(std::abs(ea.magnitude(3, 2) - 1.0f) < 1e-6f && ea.ridges(0.5f).size() == 6);
    // Tinta pequena (0,5 % da área, abaixo de peakMinShare) num logo sem perdas continua sendo tinta.
    Array2D<Rgba8> small(60, 60, rgb(255, 255, 255));
    for (int y = 0; y < 60; ++y) {
        for (int x = 0; x < 30; ++x) small(x, y) = rgb(30, 90, 200);
    }
    for (int y = 40; y < 44; ++y) {
        for (int x = 40; x < 44; ++x) small(x, y) = rgb(220, 30, 40);  // 16 px = 0,44 %
    }
    const cnx::ImageMeasures sm = measureRaw(small);
    CHECK(sm.mainPeaks == 2 && sm.significantColors == 3);
    CHECK(sm.peakShare == 1.0 && sm.quantizationRmse < 1e-9);
    // Nenhuma célula chega a 1 %: o único "pico" é a célula mais cheia (15 px pretos); as outras 1585 cores
    // são distintas (1 px cada, não significativas). O erro é então exatamente a distância de cada pixel ao preto.
    Array2D<Rgba8> sparse(40, 40);
    double err = 0.0;
    for (int i = 0; i < 1600; ++i) {
        Rgba8 c = rgb(0, 0, 0);
        if (i >= 15) {
            const int j = i - 15;
            c = rgb(j % 256, 20 + (j / 256) * 37, (j * 7) % 256);
        }
        sparse(i % 40, i / 40) = c;
        err += (c.r * c.r + c.g * c.g + c.b * c.b) / (255.0 * 255.0);
    }
    const cnx::ImageMeasures sp = measureRaw(sparse);
    CHECK(sp.mainPeaks == 1 && sp.significantColors == 1);
    CHECK(std::abs(sp.quantizationRmse - std::sqrt(err / 1600.0)) < 1e-12);
}

CNX_TEST(between_peaks_means_a_real_mixture_of_two_distinct_inks) {
    // Duas tintas em metades de 60×60 e 30 pixels "extras" (10 cores × 3 px: nem pico, < 1 %, nem cor
    // significativa, < 0,2 %), todos numa mesma célula do histograma.
    cnx::TaskContext ctx;
    const auto between = [&](Rgba8 a, Rgba8 b, const std::function<Rgba8(int)>& extra) {
        Array2D<Rgba8> img(60, 60);
        for (int y = 0; y < 60; ++y) {
            for (int x = 0; x < 60; ++x) img(x, y) = x < 30 ? a : b;
        }
        for (int k = 0; k < 30; ++k) img(k, 30) = extra(k % 10);
        cnx::PreprocessedImage pre;
        pre.image = img;
        return cnx::measureImage(pre, cnx::SourceImage{}, cnx::PaletteParams{}, ctx);
    };
    const Rgba8 black = rgb(0, 0, 0);
    const Rgba8 white = rgb(255, 255, 255);
    // Cinza médio (média 119,5, t ≈ 0,47) sobre o segmento preto–branco: mistura.
    const cnx::ImageMeasures mid = between(black, white, [](int k) { return rgb(115 + k, 115 + k, 115 + k); });
    CHECK(mid.mainPeaks == 2 && mid.betweenShare == 30.0 / 3600.0 && mid.peakShare == 3570.0 / 3600.0);
    // Cinza escuro (média 21,5, t ≈ 0,08): fora do raio do preto, mas dentro da margem da ponta — não é mistura.
    const cnx::ImageMeasures dark = between(black, white, [](int k) { return rgb(17 + k, 17 + k, 17 + k); });
    CHECK(dark.betweenShare == 0.0 && dark.peakShare == 3570.0 / 3600.0);
    // Cor a 0,058 do segmento (azulada: (117,5; 117,5; 135,5)) — fora do tubo de 0,04.
    const cnx::ImageMeasures off = between(black, white, [](int k) { return rgb(113 + k, 113 + k, 131 + k); });
    CHECK(off.betweenShare == 0.0 && off.peakShare == 3570.0 / 3600.0);
    // Picos a 0,19 (< minPeakSeparation): o cinza do meio (105,5) não é mistura de duas tintas, é a mesma tinta.
    const cnx::ImageMeasures close = between(rgb(92, 92, 92), rgb(120, 120, 120), [](int k) { return rgb(101 + k, 101 + k, 101 + k); });
    CHECK(close.mainPeaks == 2 && close.betweenShare == 0.0 && close.peakShare == 3570.0 / 3600.0);
}

CNX_TEST(peaks_absorb_light_noise_but_not_distinct_inks) {
    // Duas tintas a 10 níveis de distância num canal são a mesma (≤ peakRadius); a 40 níveis, duas.
    Array2D<Rgba8> close(40, 40);
    Array2D<Rgba8> apart(40, 40);
    for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 40; ++x) {
            close(x, y) = x < 20 ? rgb(120, 60, 60) : rgb(130, 60, 60);
            apart(x, y) = x < 20 ? rgb(120, 60, 60) : rgb(160, 60, 60);
        }
    }
    CHECK(classifyImage(close).measures.mainPeaks == 1);
    CHECK(classifyImage(apart).measures.mainPeaks == 2);
    // Uma tinta com ruído de ±6 níveis em cima de um limite de faixa (128) continua um pico só.
    cnxtest::SplitMix64 rng{4403};
    const Array2D<Rgba8> noisy = addNoise(Array2D<Rgba8>(40, 40, rgb(128, 128, 64)), rng, 6);
    const cnx::ImageMeasures m = classifyImage(noisy).measures;
    CHECK(m.mainPeaks == 1 && m.peakShare > 0.99);
}

// ================================================================ a árvore

CNX_TEST(the_decision_tree_follows_its_rules_branch_by_branch) {
    cnx::ImageMeasures logo;  // um logo chapado típico
    logo.mainPeaks = 3;
    logo.peakShare = 1.0;
    logo.edgeFraction = 0.08;
    logo.edgeProfiles = 100;
    // Os padrões documentados…
    const cnx::ClassificationRules defaults;
    CHECK(defaults.minLogoCoverage == 0.90 && defaults.minLogoCoverageLossy == 0.95 && defaults.maxFlatRmse == 0.05 &&
          defaults.minEdgeErrorLift == 1.5 && defaults.minBetweenShare == 0.01 && defaults.minAAProfiles == 16 && defaults.minRampShare == 0.25);
    // …e, para testar as fronteiras sem arredondamento, limiares exatos em binário.
    cnx::ClassificationRules rules;
    rules.minLogoCoverage = 0.875;
    rules.minLogoCoverageLossy = 0.9375;
    rules.maxFlatRmse = 0.0625;
    rules.minEdgeErrorLift = 1.5;
    rules.minBetweenShare = 0.015625;
    rules.minAAProfiles = 16;
    rules.minRampShare = 0.25;
    logo.edgeFraction = 0.0625;
    const auto decide = [&](const std::function<void(cnx::ImageMeasures&)>& edit) {
        cnx::ImageMeasures m = logo;
        edit(m);
        return cnx::decideProfile(m, 12, rules);
    };
    CHECK(decide([](auto&) {}) == ImageProfile::LogoMed);
    CHECK(cnx::decideProfile(logo, 12) == ImageProfile::LogoMed);
    // 1. cores demais.
    CHECK(decide([](auto& m) { m.mainPeaks = 12; }) == ImageProfile::LogoMed);
    CHECK(decide([](auto& m) { m.mainPeaks = 13; }) == ImageProfile::PhotoMed);
    // 2. cobertura (tinta + mistura), com limiar mais alto para JPEG.
    CHECK(decide([](auto& m) { m.peakShare = 0.75; m.betweenShare = 0.125; }) == ImageProfile::LogoMedAA);  // no limite
    CHECK(decide([](auto& m) { m.peakShare = 0.75; m.betweenShare = 0.120; }) == ImageProfile::PhotoMed);
    CHECK(decide([](auto& m) { m.peakShare = 0.875; }) == ImageProfile::LogoMed);
    CHECK(decide([](auto& m) { m.peakShare = 0.8; }) == ImageProfile::PhotoMed);
    CHECK(decide([](auto& m) { m.peakShare = 0.9; m.lossyFormat = true; }) == ImageProfile::PhotoMed);
    CHECK(decide([](auto& m) { m.peakShare = 0.9375; m.lossyFormat = true; }) == ImageProfile::LogoMed);
    CHECK(decide([](auto& m) { m.peakShare = 0.9; }) == ImageProfile::LogoMed);
    // 3. erro que sobra: tolerado se pequeno, ou se concentrado nas bordas (lift = erro na borda / pixels de borda).
    CHECK(decide([](auto& m) { m.quantizationRmse = 0.0625; m.edgeErrorShare = 0.0; }) == ImageProfile::LogoMed);
    CHECK(decide([](auto& m) { m.quantizationRmse = 0.07; m.edgeErrorShare = 0.0859375; }) == ImageProfile::PhotoMed);  // lift 1,375
    CHECK(decide([](auto& m) { m.quantizationRmse = 0.07; m.edgeErrorShare = 0.09375; }) == ImageProfile::LogoMed);     // lift 1,5
    CHECK(decide([](auto& m) { m.quantizationRmse = 0.2; m.edgeErrorShare = 1.0; m.edgeFraction = 0.0; }) == ImageProfile::PhotoMed);
    // 4. AA: misturas entre tintas, ou rampas com perfis suficientes.
    CHECK(decide([](auto& m) { m.betweenShare = 0.015625; }) == ImageProfile::LogoMedAA);
    CHECK(decide([](auto& m) { m.betweenShare = 0.015; }) == ImageProfile::LogoMed);
    CHECK(decide([](auto& m) { m.rampShare = 0.25; }) == ImageProfile::LogoMedAA);
    CHECK(decide([](auto& m) { m.rampShare = 0.24; }) == ImageProfile::LogoMed);
    CHECK(decide([](auto& m) { m.rampShare = 0.9; m.edgeProfiles = 15; }) == ImageProfile::LogoMed);
    CHECK(decide([](auto& m) { m.rampShare = 0.9; m.edgeProfiles = 16; }) == ImageProfile::LogoMedAA);    // maxColors é o do chamador; as regras são substituíveis.
    cnx::ImageMeasures five = logo;
    five.mainPeaks = 5;
    CHECK(cnx::decideProfile(five, 4) == ImageProfile::PhotoMed && cnx::decideProfile(five, 5) == ImageProfile::LogoMed);
    cnx::ClassificationRules loose;
    loose.minLogoCoverage = 0.5;
    cnx::ImageMeasures half = logo;
    half.peakShare = 0.6;
    CHECK(cnx::decideProfile(half, 12) == ImageProfile::PhotoMed && cnx::decideProfile(half, 12, loose) == ImageProfile::LogoMed);
}

CNX_TEST(the_classifier_gets_every_image_of_a_synthetic_corpus_right) {
    struct Family {
        const char* name;
        ImageProfile expected;
        std::function<Array2D<Rgba8>(cnxtest::SplitMix64&)> make;
        const char* hint = ".png";
    };
    const int w = 80;
    const int h = 72;
    const std::vector<Family> families{
        {"logo chapado", ImageProfile::LogoMed, [&](cnxtest::SplitMix64& r) { Rgba8 bg; auto s = randomLogo(r, w, h, bg, true); return render(w, h, bg, s, 1); }},
        {"logo chapado (JPEG sem ruído)", ImageProfile::LogoMed, [&](cnxtest::SplitMix64& r) { Rgba8 bg; auto s = randomLogo(r, w, h, bg, true); return render(w, h, bg, s, 1); }, ".jpg"},
        {"logo com ruído ±5", ImageProfile::LogoMed, [&](cnxtest::SplitMix64& r) { Rgba8 bg; auto s = randomLogo(r, w, h, bg, true); return addNoise(render(w, h, bg, s, 1), r, 5); }},
        {"logo com AA", ImageProfile::LogoMedAA, [&](cnxtest::SplitMix64& r) { Rgba8 bg; auto s = randomLogo(r, w, h, bg, false); return render(w, h, bg, s, 6); }},
        {"logo com AA (JPEG)", ImageProfile::LogoMedAA, [&](cnxtest::SplitMix64& r) { Rgba8 bg; auto s = randomLogo(r, w, h, bg, false); return render(w, h, bg, s, 6); }, ".jpeg"},
        {"logo desfocado", ImageProfile::LogoMedAA, [&](cnxtest::SplitMix64& r) {
             Rgba8 bg;
             auto s = randomLogo(r, w, h, bg, true);
             return cnx::gaussianBlur(render(w, h, bg, s, 1), 0.8 + 0.7 * cnxtest::uniform01(r));
         }},
        {"foto", ImageProfile::PhotoMed, [&](cnxtest::SplitMix64& r) { return randomPhoto(r, w, h); }},
        {"foto (JPEG)", ImageProfile::PhotoMed, [&](cnxtest::SplitMix64& r) { return randomPhoto(r, w, h); }, ".jpg"},
        {"logo com ruído forte ±40", ImageProfile::PhotoMed, [&](cnxtest::SplitMix64& r) { Rgba8 bg; auto s = randomLogo(r, w, h, bg, true); return addNoise(render(w, h, bg, s, 1), r, 40); }},
        {"gradiente", ImageProfile::PhotoMed, [&](cnxtest::SplitMix64& r) {
             Array2D<Rgba8> g(w, h);
             const int c0 = r.below(30);
             for (int y = 0; y < h; ++y) {
                 for (int x = 0; x < w; ++x) g(x, y) = rgb(c0 + (225 - c0) * x / (w - 1), 40 + 150 * y / (h - 1), 200 - 150 * x / (w - 1));
             }
             return g;
         }},
    };
    cnxtest::SplitMix64 rng{20261021};
    int wrong = 0;
    for (const Family& f : families) {
        int ok = 0;
        constexpr int kSamples = 12;
        double minCov = 1e9, maxCov = -1e9, minRamp = 1e9, maxRamp = -1e9, minLift = 1e9, maxRmse = -1e9;
        for (int i = 0; i < kSamples; ++i) {
            const cnx::ImageClassification c = classifyImage(f.make(rng), f.hint);
            const cnx::ImageMeasures& m = c.measures;
            const double cov = m.peakShare + m.betweenShare;
            const double lift = m.edgeFraction > 0 ? m.edgeErrorShare / m.edgeFraction : 0.0;
            minCov = std::min(minCov, cov);
            maxCov = std::max(maxCov, cov);
            minRamp = std::min(minRamp, m.rampShare);
            maxRamp = std::max(maxRamp, m.rampShare);
            minLift = std::min(minLift, lift);
            maxRmse = std::max(maxRmse, m.quantizationRmse);
            if (c.profile == f.expected) {
                ++ok;
            } else {
                std::printf("    ERRO %s #%d: picos %d cob %.3f+%.3f rmse %.3f lift %.2f rampas %.2f (%zu perfis)\n", f.name, i, m.mainPeaks,
                            m.peakShare, m.betweenShare, m.quantizationRmse, lift, m.rampShare, m.edgeProfiles);
            }
        }
        std::printf("  %-30s %2d/%d  cobertura [%.3f, %.3f] rampas [%.2f, %.2f] lift min %.2f rmse max %.3f\n", f.name, ok, kSamples, minCov,
                    maxCov, minRamp, maxRamp, minLift, maxRmse);
        CHECK_CTX(ok == kSamples, f.name);
        wrong += kSamples - ok;
    }
    CHECK(wrong == 0);
}

CNX_TEST(classify_honors_the_forced_profile_without_measuring) {
    cnx::SourceImage src;
    src.pixels = Array2D<Rgba8>(8, 8, rgb(1, 2, 3));
    cnx::TaskContext ctx;
    const cnx::PreprocessedImage pre = cnx::preprocess(src, cnx::PreprocessParams{}, ctx);
    cnx::ClassifierParams forced;
    forced.autoClassify = false;
    forced.forcedProfile = ImageProfile::PhotoHigh;
    std::stop_source stop;
    stop.request_stop();
    cnx::TaskContext cancelled(stop.get_token(), nullptr, cnx::ProgressRange{});
    // Forçado: nem mede (o contexto cancelado não é consultado).
    const cnx::ImageClassification c = cnx::classify(pre, src, forced, cnx::PaletteParams{}, cancelled);
    CHECK(c.profile == ImageProfile::PhotoHigh && !c.automatic && c.measures.mainPeaks == 0);
    CHECK_THROWS(cnx::CancelTaskException, (void)cnx::classify(pre, src, cnx::ClassifierParams{}, cnx::PaletteParams{}, cancelled));
}

int main() { return cnxtest::runAll(); }
