// tests/unit/preprocess_classify_test.cpp — pré-processamento (alfa, média de área, gaussiana) e casos básicos do
// classificador (o corpus e as medidas estão em edge_classifier_test).
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <utility>

#include "cnx/classify/Classifier.h"
#include "cnx/preprocess/Preprocessor.h"
#include "test_harness.h"
#include "test_support.h"

using sla::Array2D;
using sla::Rgba8;

namespace {

Rgba8 gray(int v, int a = 255) { return {static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(a)}; }

cnx::SourceImage source(Array2D<Rgba8> px, std::string hint = ".png") {
    cnx::SourceImage s;
    s.pixels = std::move(px);
    s.formatHint = std::move(hint);
    return s;
}

double meanOf(const Array2D<Rgba8>& img, int channel) {
    double s = 0.0;
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            const Rgba8 c = img(x, y);
            s += channel == 0 ? c.r : channel == 1 ? c.g : channel == 2 ? c.b : c.a;
        }
    }
    return s / (static_cast<double>(img.width()) * img.height());
}

}  // namespace

CNX_TEST(area_resampling_is_an_exact_box_average) {
    // 4×4 → 2×2: cada pixel de saída é a média de um bloco 2×2.
    Array2D<Rgba8> in(4, 4);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) in(x, y) = gray(10 * (x + 4 * y));
    }
    const Array2D<Rgba8> out = cnx::resampleArea(in, 2, 2);
    CHECK(out.width() == 2 && out.height() == 2);
    CHECK(out(0, 0).r == 25 && out(1, 0).r == 45 && out(0, 1).r == 105 && out(1, 1).r == 125);  // (0+10+40+50)/4 = 25, …
    // Razão não inteira (5 → 2): a massa se conserva e uma imagem constante continua constante.
    Array2D<Rgba8> flat(5, 7, gray(77));
    const Array2D<Rgba8> small = cnx::resampleArea(flat, 2, 3);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 2; ++x) CHECK(small(x, y) == gray(77));
    }
    Array2D<Rgba8> ramp(7, 5);
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 7; ++x) ramp(x, y) = gray(30 * x);
    }
    CHECK(std::abs(meanOf(cnx::resampleArea(ramp, 3, 2), 0) - meanOf(ramp, 0)) < 1.0);
    CHECK_THROWS(std::invalid_argument, (void)cnx::resampleArea(in, 5, 2));  // só reduz
    CHECK_THROWS(std::invalid_argument, (void)cnx::resampleArea(in, 0, 2));
    // Mesmo tamanho = cópia.
    const Array2D<Rgba8> same = cnx::resampleArea(in, 4, 4);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) CHECK(same(x, y) == in(x, y));
    }
}

CNX_TEST(gaussian_blur_is_normalized_symmetric_and_edge_replicating) {
    Array2D<Rgba8> flat(9, 9, gray(100));
    const Array2D<Rgba8> b = cnx::gaussianBlur(flat, 1.5);
    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 9; ++x) CHECK(b(x, y) == gray(100));  // bordas replicadas: nada escurece nas bordas
    }
    // Impulso: simétrico e espalhado, com a massa (quase) conservada.
    Array2D<Rgba8> impulse(15, 15, gray(0));
    impulse(7, 7) = gray(255);
    const Array2D<Rgba8> r = cnx::gaussianBlur(impulse, 1.2);
    CHECK(r(7, 7).r < 255 && r(7, 7).r > r(8, 7).r && r(8, 7).r > r(9, 7).r);
    for (int d = 1; d <= 5; ++d) {
        CHECK(r(7 + d, 7) == r(7 - d, 7) && r(7, 7 + d) == r(7, 7 - d) && r(7 + d, 7) == r(7, 7 + d));
    }
    CHECK(std::abs(meanOf(r, 0) * 225.0 - 255.0) < 15.0);  // arredondamento de 8 bits
    CHECK(cnx::gaussianBlur(impulse, 0.0)(7, 7) == gray(255));  // σ = 0: cópia
    // O raio cobre 3σ: a 3 pixels do impulso (2,5σ) o peso ainda é visível.
    CHECK(r(10, 7).r > 0 && r(4, 7).r > 0);
    // Borda replicada (não circular): numa rampa, a ponta esquerda continua baixa e a direita alta.
    Array2D<Rgba8> ramp(20, 3);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 20; ++x) ramp(x, y) = gray(10 * x);
    }
    const Array2D<Rgba8> rb = cnx::gaussianBlur(ramp, 1.5);
    CHECK(rb(0, 1).r < 30 && rb(19, 1).r > 160);
}

CNX_TEST(preprocess_flattens_constant_alpha_resamples_and_blurs_a_copy) {
    cnx::PreprocessParams params;
    params.targetAreaPx = 1.0e6;
    cnxtest::SplitMix64 rng{1101};
    Array2D<Rgba8> px(40, 30);
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 40; ++x) px(x, y) = {static_cast<std::uint8_t>(rng.below(256)), 50, 60, 128};
    }
    cnx::TaskContext ctx;
    // Alfa constante (128) → opaco; área pequena → sem reamostragem, cópia igual.
    cnx::PreprocessedImage a = cnx::preprocess(source(px), params, ctx);
    CHECK(a.alphaFlattened && a.scale == 1.0 && a.image.width() == 40 && a.image.height() == 30);
    CHECK(a.image(3, 3).a == 255 && a.image(3, 3).r == px(3, 3).r);
    CHECK(a.smooth(5, 5) == a.image(5, 5));  // σ = 0: a "desfocada" é a mesma imagem
    // Alfa variável fica como está.
    px(0, 0).a = 10;
    cnx::PreprocessedImage b = cnx::preprocess(source(px), params, ctx);
    CHECK(!b.alphaFlattened && b.image(0, 0).a == 10 && b.image(1, 0).a == 128);
    // flattenAlpha desligado: alfa constante permanece.
    px(0, 0).a = 128;
    params.flattenAlpha = false;
    CHECK(cnx::preprocess(source(px), params, ctx).image(3, 3).a == 128);
    params.flattenAlpha = true;

    // Área-alvo: 40×30 = 1200 px → alvo 300 px → s = 0,5 → 20×15.
    params.targetAreaPx = 300.0;
    const cnx::PreprocessedImage c = cnx::preprocess(source(px), params, ctx);
    CHECK(c.image.width() == 20 && c.image.height() == 15 && std::abs(c.scale - 0.5) < 1e-12);
    // Ampliar só com allowUpscale (bilinear).
    params.targetAreaPx = 4800.0;
    CHECK(cnx::preprocess(source(px), params, ctx).image.width() == 40);
    params.allowUpscale = true;
    const cnx::PreprocessedImage d = cnx::preprocess(source(px), params, ctx);
    CHECK(d.image.width() == 80 && d.image.height() == 60 && std::abs(d.scale - 2.0) < 1e-12);
    // Ampliação bilinear com centros de pixel: 2 → 4 colunas dá 0, 25, 75, 100.
    {
        Array2D<Rgba8> two(2, 1);
        two(0, 0) = gray(0);
        two(1, 0) = gray(100);
        cnx::PreprocessParams up;
        up.targetAreaPx = 8.0;
        up.allowUpscale = true;
        const cnx::PreprocessedImage u = cnx::preprocess(source(two), up, ctx);
        CHECK(u.image.width() == 4 && u.image.height() == 2);
        CHECK(u.image(0, 0).r == 0 && u.image(1, 0).r == 25 && u.image(2, 0).r == 75 && u.image(3, 0).r == 100);
    }
    // Desfoque: a imagem original fica intacta; a cópia muda.
    params.allowUpscale = false;
    params.targetAreaPx = 1.0e6;
    params.blurSigmaPx = 1.0;
    const cnx::PreprocessedImage e = cnx::preprocess(source(px), params, ctx);
    int differ = 0;
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 40; ++x) differ += e.smooth(x, y) == e.image(x, y) ? 0 : 1;
    }
    CHECK(differ > 500 && e.image(3, 3).r == px(3, 3).r);
}

CNX_TEST(preprocess_validates_its_inputs_and_honors_cancellation) {
    cnx::TaskContext ctx;
    CHECK_THROWS(std::invalid_argument, (void)cnx::preprocess(source(Array2D<Rgba8>()), cnx::PreprocessParams{}, ctx));
    cnx::PreprocessParams bad;
    bad.targetAreaPx = 0.0;
    CHECK_THROWS(std::invalid_argument, (void)cnx::preprocess(source(Array2D<Rgba8>(4, 4)), bad, ctx));
    bad = cnx::PreprocessParams{};
    bad.blurSigmaPx = -1.0;
    CHECK_THROWS(std::invalid_argument, (void)cnx::preprocess(source(Array2D<Rgba8>(4, 4)), bad, ctx));
    std::stop_source stop;
    stop.request_stop();
    cnx::TaskContext cancelled(stop.get_token(), nullptr, cnx::ProgressRange{});
    CHECK_THROWS(cnx::CancelTaskException, (void)cnx::preprocess(source(Array2D<Rgba8>(4, 4)), cnx::PreprocessParams{}, cancelled));
}

CNX_TEST(the_classifier_separates_flat_logos_aa_logos_and_photos) {
    const cnx::PaletteParams palette;
    cnx::TaskContext ctx;
    const cnx::PreprocessParams pp;
    const auto classifyImage = [&](const Array2D<Rgba8>& img, const std::string& hint, const cnx::ClassifierParams& cp = {}) {
        const cnx::SourceImage src = source(img, hint);
        const cnx::PreprocessedImage pre = cnx::preprocess(src, pp, ctx);
        return cnx::classify(pre, src, cp, palette);
    };
    // Logo chapado: 4 cores exatas em quadrantes.
    Array2D<Rgba8> flat(64, 64);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) flat(x, y) = {static_cast<std::uint8_t>(x < 32 ? 255 : 0), static_cast<std::uint8_t>(y < 32 ? 255 : 0), 40, 255};
    }
    const cnx::ImageClassification logo = classifyImage(flat, ".png");
    CHECK(logo.profile == cnx::ImageProfile::LogoMed && logo.automatic);
    CHECK(logo.measures.significantColors == 4 && logo.measures.flatShare >= 0.99);
    CHECK(logo.measures.edgeFraction > 0.0 && logo.measures.edgeFraction < 0.2);  // as duas linhas de quadrante
    // O mesmo conteúdo marcado como JPEG continua logo: decide o conteúdo (sem ruído de compressão aqui);
    // a extensão só endurece o limiar de cobertura (edge_classifier_test cobre o JPEG com ruído).
    CHECK(classifyImage(flat, ".JPG").profile == cnx::ImageProfile::LogoMed);
    CHECK(classifyImage(flat, ".jpeg").measures.lossyFormat);

    // Logo com antialiasing: disco com anel de transição (~4 %).
    const auto scene = cnxtest::aaScene(64, 64, [](double x, double y) { return (x - 32.2) * (x - 32.2) + (y - 31.7) * (y - 31.7) <= 20.0 * 20.0; }, 8);
    Array2D<Rgba8> aa(64, 64);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) aa(x, y) = sla::toRgba8(scene.target(x, y));
    }
    const cnx::ImageClassification disk = classifyImage(aa, ".png");
    CHECK(disk.profile == cnx::ImageProfile::LogoMedAA);
    CHECK(disk.measures.flatShare >= 0.80 && disk.measures.flatShare < 0.99);

    // Foto: gradiente com ruído, muitas cores.
    cnxtest::SplitMix64 rng{1102};
    Array2D<Rgba8> photo(64, 64);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            photo(x, y) = {static_cast<std::uint8_t>(std::clamp(4 * x + rng.below(20), 0, 255)), static_cast<std::uint8_t>(std::clamp(4 * y + rng.below(20), 0, 255)),
                           static_cast<std::uint8_t>(rng.below(256)), 255};
        }
    }
    const cnx::ImageClassification ph = classifyImage(photo, ".png");
    CHECK(ph.profile == cnx::ImageProfile::PhotoMed && ph.measures.flatShare < 0.80);

    // Limiar de foto: 70 % de pixels em 4 cores chapadas e 30 % de ruído único → foto (< 80 %).
    {
        cnxtest::SplitMix64 noise{1103};
        Array2D<Rgba8> mixed(64, 64);
        for (int y = 0; y < 64; ++y) {
            for (int x = 0; x < 64; ++x) {
                const bool flatPart = (y * 64 + x) % 10 < 7;
                mixed(x, y) = flatPart ? gray(60 * (x % 4), 255)
                                       : Rgba8{static_cast<std::uint8_t>(noise.below(256)), static_cast<std::uint8_t>(noise.below(256)), static_cast<std::uint8_t>(noise.below(256)), 255};
            }
        }
        const cnx::ImageClassification mx = classifyImage(mixed, ".png");
        CHECK(mx.measures.flatShare > 0.6 && mx.measures.flatShare < 0.8 && mx.measures.significantColors <= 4);
        CHECK(mx.profile == cnx::ImageProfile::PhotoMed);
    }
    // Limite de cores: exatamente maxColors cores chapadas ainda é logo; uma a mais vira foto.
    for (const int extra : {0, 1}) {
        const int n = palette.maxColors + extra;
        Array2D<Rgba8> stripes(8 * n, 16);
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 8 * n; ++x) stripes(x, y) = gray(20 * (x / 8), 255);  // 20 níveis: um pico por cor
        }
        const cnx::ImageClassification st = classifyImage(stripes, ".png");
        CHECK(st.measures.significantColors == n);
        CHECK(st.profile == (extra == 0 ? cnx::ImageProfile::LogoMed : cnx::ImageProfile::PhotoMed));
    }

    // Perfil forçado: devolve o pedido, sem medir.
    cnx::ClassifierParams forced;
    forced.autoClassify = false;
    forced.forcedProfile = cnx::ImageProfile::LogoLowAA;
    const cnx::ImageClassification f = classifyImage(photo, ".png", forced);
    CHECK(f.profile == cnx::ImageProfile::LogoLowAA && !f.automatic && f.measures.significantColors == 0);
}

int main() { return cnxtest::runAll(); }
