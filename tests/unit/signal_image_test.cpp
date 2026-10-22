// tests/unit/signal_image_test.cpp — SignalImage (sinal pré-multiplicado em float): ida e volta de 8 bits,
// média de área, bilinear e gaussiana sem vazamento do RGB escondido, o pré-processamento e o alvo do
// GenerativeModel.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include "cnx/engine/CoreEngine.h"
#include "cnx/preprocess/Preprocessor.h"
#include "cnx/preprocess/SignalImage.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/vector/GenerativeModel.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::SignalImage;
using sla::Array2D;
using sla::Rgba;
using sla::Rgba8;

namespace {

Rgba8 rgba(int r, int g, int b, int a) {
    return {static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b), static_cast<std::uint8_t>(a)};
}

/// Soma do sinal em cada canal, ponderada pela área de um pixel (conservação de massa).
std::array<double, 4> mass(const SignalImage& s, double pixelArea) {
    std::array<double, 4> m{};
    for (int y = 0; y < s.height(); ++y) {
        for (int x = 0; x < s.width(); ++x) {
            const Rgba& p = s.premultiplied()(x, y);
            m[0] += p.r * pixelArea;
            m[1] += p.g * pixelArea;
            m[2] += p.b * pixelArea;
            m[3] += p.a * pixelArea;
        }
    }
    return m;
}

/// Média straight ingênua (a regra antiga), para comparar: mistura o RGB escondido sob alfa 0.
Rgba8 naiveAverage(const Rgba8& a, const Rgba8& b) {
    const auto m = [](int u, int v) { return static_cast<int>(std::lround((u + v) / 2.0)); };
    return rgba(m(a.r, b.r), m(a.g, b.g), m(a.b, b.b), m(a.a, b.a));
}

}  // namespace

CNX_TEST(premultiplication_round_trips_every_8_bit_value_and_hides_invisible_rgb) {
    // Todas as 256 × 256 combinações (valor, alfa) num canal: ida e volta exata se a > 0.
    Array2D<Rgba8> all(256, 256);
    for (int a = 0; a < 256; ++a) {
        for (int v = 0; v < 256; ++v) all(v, a) = rgba(v, 255 - v, v / 2, a);
    }
    const SignalImage s = SignalImage::fromStraight(all);
    const Array2D<Rgba8> back = s.toStraight8();
    int wrong = 0;
    for (int a = 0; a < 256; ++a) {
        for (int v = 0; v < 256; ++v) {
            const Rgba8 expected = a == 0 ? rgba(0, 0, 0, 0) : all(v, a);
            wrong += back(v, a) == expected ? 0 : 1;
            const Rgba& p = s.premultiplied()(v, a);
            CHECK(p.a == static_cast<float>(a / 255.0) && p.r == static_cast<float>(v / 255.0 * (a / 255.0)));
        }
    }
    CHECK(wrong == 0);
    // straight(): a cor sem o alfa; invisível → (0, 0, 0, 0).
    CHECK(s.straight(200, 0) == Rgba(0, 0, 0, 0));
    const Rgba c = s.straight(51, 128);
    CHECK(std::abs(c.r - 51 / 255.0f) < 1e-6f && std::abs(c.g - 204 / 255.0f) < 1e-6f && std::abs(c.a - 128 / 255.0f) < 1e-6f);
    // Imagem vazia.
    const SignalImage empty = SignalImage::fromStraight(Array2D<Rgba8>());
    CHECK(empty.width() == 0 && empty.toStraight8().empty() && empty.blurred(2.0).width() == 0);
}

CNX_TEST(area_reduction_does_not_leak_the_rgb_hidden_under_zero_alpha) {
    // Vermelho opaco | transparente com RGB branco (ou preto) escondido; 2 → 1 px.
    for (const Rgba8 hidden : {rgba(255, 255, 255, 0), rgba(0, 0, 0, 0), rgba(17, 200, 90, 0)}) {
        Array2D<Rgba8> two(2, 1);
        two(0, 0) = rgba(255, 0, 0, 255);
        two(1, 0) = hidden;
        const Rgba8 got = cnx::resampleArea(two, 1, 1)(0, 0);
        const Rgba8 naive = naiveAverage(two(0, 0), hidden);
        std::printf("  vermelho + transparente (%d,%d,%d): sinal (%d,%d,%d,%d), média straight (%d,%d,%d,%d)\n", hidden.r, hidden.g, hidden.b, got.r,
                    got.g, got.b, got.a, naive.r, naive.g, naive.b, naive.a);
        CHECK(got == rgba(255, 0, 0, 128));  // meia cobertura de vermelho puro
        CHECK(!(naive == got));              // a média straight inventa rosa, vermelho-escuro ou verde
    }
    // Duas cores semitransparentes: a média pondera a cor pelo alfa.
    Array2D<Rgba8> semi(2, 1);
    semi(0, 0) = rgba(255, 0, 0, 200);
    semi(1, 0) = rgba(0, 0, 255, 50);
    const Rgba8 m = cnx::resampleArea(semi, 1, 1)(0, 0);
    CHECK(m.a == 125 && m.r == 204 && m.b == 51 && m.g == 0);  // (255·200, 255·50)/250 = 204 e 51
}

CNX_TEST(area_reduction_is_exact_and_conserves_the_signal) {
    cnxtest::SplitMix64 rng{6101};
    Array2D<Rgba8> img(7, 5);
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 7; ++x) img(x, y) = rgba(rng.below(256), rng.below(256), rng.below(256), rng.below(256));
    }
    const SignalImage s = SignalImage::fromStraight(img);
    const SignalImage r = s.resampledArea(3, 2);
    CHECK(r.width() == 3 && r.height() == 2);
    const auto before = mass(s, 1.0);
    const auto after = mass(r, (7.0 / 3.0) * (5.0 / 2.0));
    for (std::size_t k = 0; k < 4; ++k) CHECK(std::abs(before[k] - after[k]) < 1e-5 * (1.0 + before[k]));
    // O pixel (0, 0) de saída cobre [0, 7/3) × [0, 5/2): colunas 0, 1 e 1/3 da 2; linhas 0, 1 e 1/2 da 2.
    const std::array<double, 3> wx{3.0 / 7.0, 3.0 / 7.0, 1.0 / 7.0};
    const std::array<double, 3> wy{2.0 / 5.0, 2.0 / 5.0, 1.0 / 5.0};
    double expected = 0.0;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) expected += wx[static_cast<std::size_t>(x)] * wy[static_cast<std::size_t>(y)] * s.premultiplied()(x, y).g;
    }
    CHECK(std::abs(r.premultiplied()(0, 0).g - expected) < 1e-6);
    // Mesmo tamanho = cópia; só reduz; tamanho ≥ 1.
    const SignalImage same = s.resampledArea(7, 5);
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 7; ++x) CHECK(same.premultiplied()(x, y) == s.premultiplied()(x, y));
    }
    CHECK_THROWS(std::invalid_argument, (void)s.resampledArea(8, 5));
    CHECK_THROWS(std::invalid_argument, (void)s.resampledArea(7, 6));
    CHECK_THROWS(std::invalid_argument, (void)s.resampledArea(7, 0));
    // Em float, sem quantizar: a média de 0 e 1 níveis é 0,5 nível, que o 8 bits arredondaria.
    Array2D<Rgba8> fine(2, 1);
    fine(0, 0) = rgba(100, 100, 100, 255);
    fine(1, 0) = rgba(101, 101, 101, 255);
    const float half = SignalImage::fromStraight(fine).resampledArea(1, 1).premultiplied()(0, 0).r;
    CHECK(std::abs(half - 100.5f / 255.0f) < 1e-7f);
}

CNX_TEST(blur_and_bilinear_work_on_the_signal) {
    // Quadrado azul opaco sobre transparente com RGB branco escondido: o desfoque não deixa o branco vazar.
    Array2D<Rgba8> img(16, 16, rgba(255, 255, 255, 0));
    for (int y = 5; y < 11; ++y) {
        for (int x = 5; x < 11; ++x) img(x, y) = rgba(0, 0, 255, 255);
    }
    const Array2D<Rgba8> b = cnx::gaussianBlur(img, 1.5);
    int halo = 0;
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            const Rgba8 p = b(x, y);
            if (p.a == 0) {
                CHECK(p == rgba(0, 0, 0, 0));
                continue;
            }
            ++halo;
            CHECK(p.r == 0 && p.g == 0 && p.b == 255);  // azul puro com alfa parcial
        }
    }
    CHECK(halo > 100);
    // A cor straight fica em [0, 1] mesmo depois de filtrar (a soma em float pode passar o alfa por um ulp).
    {
        cnxtest::SplitMix64 rng{6104};
        Array2D<Rgba8> noisy(24, 24);
        for (int y = 0; y < 24; ++y) {
            for (int x = 0; x < 24; ++x) noisy(x, y) = rgba(255, rng.below(256), 255, 1 + rng.below(255));
        }
        float worst = 0.0f;
        for (const SignalImage& f : {SignalImage::fromStraight(noisy), SignalImage::fromStraight(noisy).blurred(1.3),
                                     SignalImage::fromStraight(noisy).resampledArea(7, 5), SignalImage::fromStraight(noisy).resampledBilinear(31, 29)}) {
            for (int y = 0; y < f.height(); ++y) {
                for (int x = 0; x < f.width(); ++x) {
                    const Rgba c = f.straight(x, y);
                    worst = std::max({worst, c.r, c.g, c.b});
                    CHECK(c.r <= 1.0f && c.b <= 1.0f && c.r >= 0.0f);
                }
            }
        }
        CHECK(worst == 1.0f);
    }
    // Conservação: um impulso longe das bordas mantém a massa; σ ≤ 0 e NaN copiam; σ infinito é erro.
    Array2D<Rgba8> imp(21, 21, rgba(0, 0, 0, 0));
    imp(10, 10) = rgba(255, 255, 255, 255);
    const SignalImage s = SignalImage::fromStraight(imp);
    const auto m0 = mass(s, 1.0);
    const auto m1 = mass(s.blurred(1.7), 1.0);
    for (std::size_t k = 0; k < 4; ++k) CHECK(std::abs(m0[k] - m1[k]) < 1e-5);
    CHECK(s.blurred(0.0).premultiplied()(10, 10) == s.premultiplied()(10, 10));
    CHECK(s.blurred(std::numeric_limits<double>::quiet_NaN()).premultiplied()(10, 10) == s.premultiplied()(10, 10));
    CHECK_THROWS(std::invalid_argument, (void)s.blurred(std::numeric_limits<double>::infinity()));
    // Perfil exato: separável, g(k) = exp(−k²/2σ²)/Σ, raio ⌈3σ⌉ = 6 para σ = 1,7 (zero a partir de 7).
    {
        const double sigma = 1.7;
        double norm = 0.0;
        for (int k = -6; k <= 6; ++k) norm += std::exp(-0.5 * k * k / (sigma * sigma));
        const auto g = [&](int k) { return std::exp(-0.5 * k * k / (sigma * sigma)) / norm; };
        const SignalImage bl = s.blurred(sigma);
        for (int k = 0; k <= 6; ++k) {
            CHECK_CTX(std::abs(bl.premultiplied()(10 + k, 10).a - g(k) * g(0)) < 1e-7, "k = " + std::to_string(k));
            CHECK(std::abs(bl.premultiplied()(10 + k, 10 + k).a - g(k) * g(k)) < 1e-7);
        }
        CHECK(bl.premultiplied()(17, 10).a == 0.0f && bl.premultiplied()(3, 10).a == 0.0f);
    }
    // Raio ⌈3σ⌉: com σ = 0,2 é 1, e o vizinho imediato ainda recebe peso (exp(−12,5) > 0).
    CHECK(s.blurred(0.2).premultiplied()(11, 10).a > 0.0f && s.blurred(0.2).premultiplied()(12, 10).a == 0.0f);
    // Bilinear 2 → 4 colunas entre vermelho opaco e transparente (RGB branco escondido): alfa 1, ¾, ¼, 0 e o
    // vermelho puro onde é visível.
    Array2D<Rgba8> two(2, 1);
    two(0, 0) = rgba(255, 0, 0, 255);
    two(1, 0) = rgba(255, 255, 255, 0);
    const Array2D<Rgba8> up = SignalImage::fromStraight(two).resampledBilinear(4, 1).toStraight8();
    CHECK(up(0, 0) == rgba(255, 0, 0, 255) && up(1, 0) == rgba(255, 0, 0, 191) && up(2, 0) == rgba(255, 0, 0, 64) && up(3, 0) == rgba(0, 0, 0, 0));
    // Nas bordas não extrapola: 100 | 200 → 100, 125, 175, 200 (sem o t limitado a [0, 1] daria 75 e 225).
    Array2D<Rgba8> ramp(2, 1);
    ramp(0, 0) = rgba(100, 100, 100, 255);
    ramp(1, 0) = rgba(200, 200, 200, 255);
    const Array2D<Rgba8> ru = SignalImage::fromStraight(ramp).resampledBilinear(4, 1).toStraight8();
    CHECK(ru(0, 0).r == 100 && ru(1, 0).r == 125 && ru(2, 0).r == 175 && ru(3, 0).r == 200);
    CHECK_THROWS(std::invalid_argument, (void)SignalImage().resampledBilinear(2, 2));
    CHECK_THROWS(std::invalid_argument, (void)SignalImage::fromStraight(two).resampledBilinear(0, 1));
}

CNX_TEST(preprocess_keeps_the_signal_and_derives_the_8_bit_images_from_it) {
    // Transparente com RGB escondido aleatório + um retângulo vermelho opaco; reduz 40×30 → 20×15.
    cnxtest::SplitMix64 rng{6102};
    Array2D<Rgba8> px(40, 30);
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 40; ++x) px(x, y) = rgba(rng.below(256), rng.below(256), rng.below(256), 0);
    }
    for (int y = 6; y < 22; ++y) {
        for (int x = 9; x < 31; ++x) px(x, y) = rgba(220, 30, 30, 255);
    }
    cnx::SourceImage src;
    src.pixels = px;
    src.hasAlpha = true;
    cnx::PreprocessParams params;
    params.targetAreaPx = 300.0;
    params.blurSigmaPx = 1.0;
    cnx::TaskContext ctx;
    const cnx::PreprocessedImage pre = cnx::preprocess(src, params, ctx);
    CHECK(pre.signal.width() == 20 && pre.signal.height() == 15 && !pre.alphaFlattened);
    const Array2D<Rgba8> derived = pre.signal.toStraight8();
    int transparent = 0;
    for (int y = 0; y < 15; ++y) {
        for (int x = 0; x < 20; ++x) {
            CHECK(pre.image(x, y) == derived(x, y));
            const Rgba8 p = pre.image(x, y);
            if (p.a == 0) {
                ++transparent;
                CHECK(p == rgba(0, 0, 0, 0));  // um único "transparente", qualquer que fosse o RGB escondido
            } else {
                CHECK(p.r == 220 && p.g == 30 && p.b == 30);  // borda do retângulo: vermelho com alfa parcial, sem franja
            }
            const Rgba8 q = pre.smooth(x, y);
            if (q.a > 0) CHECK(std::abs(q.r - 220) <= 1 && std::abs(q.g - 30) <= 1 && std::abs(q.b - 30) <= 1);
        }
    }
    CHECK(transparent > 100);
    // Sem reamostragem: os pixels visíveis voltam idênticos; os invisíveis viram (0, 0, 0, 0).
    params.targetAreaPx = 1.0e6;
    params.blurSigmaPx = 0.0;
    const cnx::PreprocessedImage same = cnx::preprocess(src, params, ctx);
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 40; ++x) CHECK(same.image(x, y) == (px(x, y).a == 0 ? rgba(0, 0, 0, 0) : px(x, y)));
    }
    // A redução fica no sinal, sem passar por 8 bits: 100 e 101 níveis dão 100,5 exatos.
    {
        Array2D<Rgba8> two(2, 1);
        two(0, 0) = rgba(100, 100, 100, 255);
        two(1, 0) = rgba(101, 101, 101, 255);
        cnx::SourceImage s2;
        s2.pixels = two;
        cnx::PreprocessParams p2;
        p2.targetAreaPx = 0.5;
        const cnx::PreprocessedImage half = cnx::preprocess(s2, p2, ctx);
        CHECK(half.signal.width() == 1 && std::abs(half.signal.premultiplied()(0, 0).r - 100.5f / 255.0f) < 1e-7f);
    }
    // σ infinito é recusado.
    params.blurSigmaPx = std::numeric_limits<double>::infinity();
    CHECK_THROWS(std::invalid_argument, (void)cnx::preprocess(src, params, ctx));
}

CNX_TEST(the_generative_target_is_premultiplied_like_the_model) {
    // Borda vertical exata em x = 10,5 entre vermelho opaco e transparente (RGB escondido aleatório): a coluna
    // 10 tem cobertura ½, em straight (220, 30, 30, 128). O modelo sintetiza ½ × vermelho pré-multiplicado.
    cnxtest::SplitMix64 rng{6103};
    const int w = 20;
    const int h = 8;
    Array2D<Rgba8> img(w, h);
    cnx::PixelMap map(w, h);
    const cnx::RegionId fg = map.createRegion();
    const cnx::RegionId bg = map.createRegion();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            img(x, y) = x < 10 ? rgba(220, 30, 30, 255) : x == 10 ? rgba(220, 30, 30, 128) : rgba(rng.below(256), rng.below(256), rng.below(256), 0);
            map.setLabel(x, y, x <= 10 ? fg : bg, img(x, y));
        }
    }
    map.assignMedianColors();
    (void)map.compact();
    cnx::TraceResult t = cnx::BoundaryTracer().trace(map);
    cnx::VectorImage& v = t.image;
    int moved = 0;
    for (cnx::NodeId n = 0; n < static_cast<cnx::NodeId>(v.nodeCount()); ++n) {
        if (v.position(n).x == 11.0) {
            v.setPosition(n, {10.5, v.position(n).y});
            ++moved;
        }
    }
    CHECK(moved >= 2);
    for (cnx::FaceId f = 1; f < static_cast<cnx::FaceId>(v.faceCount()); ++f) {
        v.face(f).color = v.face(f).color.a > 0.5f ? Rgba{220 / 255.0f, 30 / 255.0f, 30 / 255.0f, 1.0f} : Rgba{0.0f, 0.0f, 0.0f, 0.0f};
    }
    const cnx::GenerativeModel model(SignalImage::fromStraight(img).premultiplied());
    // Alvo straight (a regra antiga), com o mesmo modelo: a coluna de borda não bate com cobertura × cor.
    Array2D<Rgba> straightTarget(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) straightTarget(x, y) = sla::toRgba(img(x, y));
    }
    const double e = model.energy(v);
    const double es = cnx::GenerativeModel(straightTarget).energy(v);
    std::printf("  energia da borda exata: alvo pré-multiplicado %.6f, alvo straight %.4f\n", e, es);
    CHECK(e < 1e-4 && es > 1.0);  // só o arredondamento de 128/255 contra ½
    // targetFrom pré-multiplica igual ao SignalImage.
    const Array2D<Rgba> tf = cnx::GenerativeModel::targetFrom(img);
    const SignalImage si = SignalImage::fromStraight(img);
    bool equal = true;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) equal = equal && tf(x, y) == si.premultiplied()(x, y);
    }
    CHECK(equal);
    // Uma face semitransparente uniforme é sintetizada exatamente: energia zero com o alvo pré-multiplicado.
    Array2D<Rgba8> semi(6, 4, rgba(40, 160, 200, 100));
    cnx::PixelMap one(6, 4);
    const cnx::RegionId r = one.createRegion();
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 6; ++x) one.setLabel(x, y, r, semi(x, y));
    }
    one.assignMedianColors();
    (void)one.compact();
    cnx::TraceResult ts = cnx::BoundaryTracer().trace(one);
    CHECK(cnx::GenerativeModel(SignalImage::fromStraight(semi).premultiplied()).energy(ts.image) < 1e-12);
}

int main() { return cnxtest::runAll(); }
