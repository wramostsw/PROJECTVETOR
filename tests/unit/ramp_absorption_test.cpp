// tests/unit/ramp_absorption_test.cpp — rampas de antialiasing no caminho de degradê, sem paleta (docs §1.30).
//
// Com o teto de fusão de degradê (12 níveis) a rampa de 2 a 3 px entre duas cores vira uma região de 1 px por degrau, e
// essas faixas coladas nunca se suavizam: dentes de 2 a 3 px nas bordas das letras, milhares de faces, suavização lenta.
// `absorbRamps` funde às vizinhas sólidas as regiões finas explicadas por elas (ruído perto de uma sólida, ou mistura de
// duas) e deixa o que é traço de verdade. Estes testes medem cada regra contra cenas de verdade conhecida.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/segment/PixelMap.h"
#include "cnx/segment/PixelSegmenter.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/vector/ContourSmoother.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::PixelMap;
using cnx::PixelSegmenter;
using cnx::RegionId;
using cnx::SegmentParams;
using sla::Array2D;
using sla::Rgba8;

namespace {

constexpr Rgba8 kBlue{30, 70, 160, 255};
constexpr Rgba8 kWhite{255, 255, 255, 255};

/// Parâmetros de segmentação do perfil de foto/degradê (teto de 12 níveis, absorvedor ligado).
SegmentParams photoParams() {
    cnx::EngineParameters e;
    e.applyProfile(cnx::ImageProfile::PhotoMed);
    return e.segment;
}

SegmentParams withoutAbsorber() {
    SegmentParams p = photoParams();
    p.absorbRamps = false;
    return p;
}

std::uint8_t byteOf(double v) { return static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L)); }

Rgba8 mix(Rgba8 a, Rgba8 b, double t) {
    return {byteOf(a.r + (b.r - a.r) * t), byteOf(a.g + (b.g - a.g) * t), byteOf(a.b + (b.b - a.b) * t), 255};
}

/// Azul à esquerda, branco à direita e, entre eles, `ramp.size()` colunas de 1 px com a mistura `t` de cada uma.
Array2D<Rgba8> rampScene(const std::vector<double>& ramp, int leftCols = 25, int rightCols = 25, int h = 30) {
    const int w = leftCols + static_cast<int>(ramp.size()) + rightCols;
    Array2D<Rgba8> img(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (x < leftCols) img(x, y) = kBlue;
            else if (x < leftCols + static_cast<int>(ramp.size())) img(x, y) = mix(kBlue, kWhite, ramp[static_cast<std::size_t>(x - leftCols)]);
            else img(x, y) = kWhite;
        }
    }
    return img;
}

cnx::SegmentationResult run(const Array2D<Rgba8>& img, const SegmentParams& p) { return PixelSegmenter(img, p).run(); }

std::size_t regionCount(cnx::SegmentationResult& r) { return r.map.forest().rootCount(); }

/// Toda região é 4-conexa e todo pixel tem região (flood fill por força bruta).
bool allConnected(const PixelMap& map) {
    const Array2D<RegionId>& labels = map.compactLabels();
    const int w = labels.width();
    const int h = labels.height();
    std::vector<char> seen(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
    std::vector<RegionId> done;
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (seen[static_cast<std::size_t>(y * w + x)]) continue;
            const RegionId r = labels(x, y);
            if (r == cnx::kNoRegion || std::find(done.begin(), done.end(), r) != done.end()) return false;
            done.push_back(r);
            stack.assign(1, {x, y});
            seen[static_cast<std::size_t>(y * w + x)] = 1;
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + cnxtest::kDx4[d];
                    const int ny = cy + cnxtest::kDy4[d];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h || seen[static_cast<std::size_t>(ny * w + nx)] || labels(nx, ny) != r) continue;
                    seen[static_cast<std::size_t>(ny * w + nx)] = 1;
                    stack.emplace_back(nx, ny);
                }
            }
        }
    }
    return true;
}

}  // namespace

CNX_TEST(a_three_pixel_ramp_is_split_between_its_two_sides_by_color) {
    const Array2D<Rgba8> img = rampScene({0.20, 0.55, 0.85});
    cnx::SegmentationResult off = run(img, withoutAbsorber());
    CHECK(regionCount(off) == 5);  // azul, as 3 colunas da rampa e branco: o que o teto de 12 níveis faz com uma rampa
    cnx::SegmentationResult on = run(img, photoParams());
    CHECK(regionCount(on) == 2 && on.stats.rampsAbsorbed == 3);
    const Array2D<RegionId>& l = on.map.compactLabels();
    for (int y = 0; y < 30; ++y) {
        CHECK(l(24, y) == l(25, y) && l(25, y) != l(26, y));  // t = 0,20 fica com o azul
        CHECK(l(26, y) == l(27, y) && l(27, y) == l(28, y));  // t = 0,55 e 0,85 ficam com o branco
        CHECK(l(24, y) == l(0, y) && l(28, y) == l(52, y));
    }
    CHECK(allConnected(on.map));
    // O estágio desligado não faz nada (e diz que não fez).
    CHECK(off.stats.rampsAbsorbed == 0);
}

CNX_TEST(the_ramp_goes_to_the_side_its_color_is_nearer_to) {
    // Rampa espelhada: as duas colunas do meio trocam de lado com o t.
    for (const double tMid : {0.30, 0.45, 0.65, 0.80}) {
        const Array2D<Rgba8> img = rampScene({0.10, tMid, 0.90});
        cnx::SegmentationResult r = run(img, photoParams());
        CHECK_CTX(regionCount(r) == 2, "rampa de 3 px vira 2 regiões");
        const Array2D<RegionId>& l = r.map.compactLabels();
        const bool toBlue = l(26, 10) == l(0, 10);
        CHECK_CTX(toBlue == (tMid < 0.5), "a coluna do meio vai para o lado de cor mais próxima");
    }
}

CNX_TEST(the_depth_is_the_thickest_ramp_absorbed_and_a_soft_gradient_is_left_alone) {
    // `rampMaxDepth` é a espessura máxima, em px, da rampa absorvida: a fina de fora de uma rampa de n px só é explicada se
    // enxerga a sólida do outro lado, a n camadas. Rampas mais grossas são degradês suaves e ficam como faixas.
    const auto absorbed = [](const std::vector<double>& ramp, int depth) {
        SegmentParams p = photoParams();
        p.rampMaxDepth = depth;
        return run(rampScene(ramp), p).stats.rampsAbsorbed;
    };
    const std::vector<double> two{0.30, 0.70};
    const std::vector<double> three{0.20, 0.55, 0.85};
    const std::vector<double> five{0.10, 0.30, 0.50, 0.70, 0.90};
    const std::vector<double> eight{0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90};
    CHECK(absorbed(two, 1) == 0 && absorbed(two, 2) == 2);
    CHECK(absorbed(three, 2) == 0 && absorbed(three, 3) == 3 && absorbed(three, 4) == 3);
    CHECK(absorbed(five, 4) == 0 && absorbed(five, 5) == 5);
    // 8 colunas dão 7 regiões finas (duas colunas vizinhas de cor parecida já se fundem na segmentação).
    CHECK(absorbed(eight, 6) == 0 && absorbed(eight, 7) == 7 && absorbed(eight, 8) == 7);
    CHECK(photoParams().rampMaxDepth == 4);  // o padrão cobre a borda borrada de um JPEG (3 a 4 px) e deixa o degradê suave
    CHECK(absorbed(five, photoParams().rampMaxDepth) == 0 && absorbed(three, photoParams().rampMaxDepth) == 3);
    cnx::SegmentationResult r5 = run(rampScene(five), [] { SegmentParams p = photoParams(); p.rampMaxDepth = 5; return p; }());
    CHECK(regionCount(r5) == 2 && allConnected(r5.map));
    const Array2D<RegionId>& l = r5.map.compactLabels();
    for (int k = 0; k < 5; ++k) CHECK_CTX((l(25 + k, 5) == l(0, 5)) == (k < 2), "a rampa de 5 px é repartida pela cor (t < 0,5 com o azul)");
    SegmentParams zero = photoParams();
    zero.rampMaxDepth = 0;  // 0 vale como 1 (o mínimo): não trava nem lança
    CHECK(run(rampScene(three), zero).stats.rampsAbsorbed == 0);
}
CNX_TEST(a_narrow_solid_keeps_its_own_color_after_absorbing_its_ramps) {
    // Branco | rampa de 3 | azul de 6 colunas | rampa de 3 | branco. Metade dos pixels do azul, depois de absorver as
    // rampas, seriam de rampa: se entrassem na mediana, o azul viraria um azul claro (docs §1.30).
    constexpr int kH = 30;
    Array2D<Rgba8> img(60, kH);
    const std::vector<double> up{0.85, 0.40, 0.15};  // do branco para o azul (parte de branco de cada coluna)
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < 60; ++x) {
            if (x < 20 || x >= 40) img(x, y) = kWhite;
            else if (x < 23) img(x, y) = mix(kBlue, kWhite, up[static_cast<std::size_t>(x - 20)]);
            else if (x < 29) img(x, y) = kBlue;
            else if (x < 32) img(x, y) = mix(kBlue, kWhite, 1.0 - up[static_cast<std::size_t>(x - 29)]);
            else img(x, y) = kWhite;
        }
    }
    cnx::SegmentationResult r = run(img, photoParams());
    CHECK(regionCount(r) == 3 && r.stats.rampsAbsorbed == 6);  // branco da esquerda, azul, branco da direita
    const RegionId blue = r.map.regionAt(25, 10);
    const sla::Rgba c = r.map.forest().stats(blue).color;
    const sla::Rgba want = sla::toRgba(kBlue);
    CHECK(std::abs(c.r - want.r) * 255.0 < 1.0 && std::abs(c.g - want.g) * 255.0 < 1.0 && std::abs(c.b - want.b) * 255.0 < 1.0);
    const RegionId white = r.map.regionAt(2, 10);
    CHECK(sla::distance2(r.map.forest().stats(white).color, sla::toRgba(kWhite)) < 1e-9);
    // As rampas ficaram dentro das regiões (topologia): o azul fica com as 6 colunas + as 3 colunas de rampa de t < 0,5 de
    // branco (2 na rampa da esquerda: 0,40 e 0,15; 1 na da direita: 0,15; a de 0,60 vai para o branco).
    CHECK(r.map.forest().pixelCount(blue) == static_cast<std::uint32_t>(kH * (6 + 2 + 1)));
    CHECK(allConnected(r.map));

    // Agora a faixa azul tem 3 colunas e as duas rampas, inteiras do lado do azul, trazem 6: os pixels absorvidos são
    // o dobro dos próprios. A mediana, se os contasse, viraria um azul esmaecido.
    Array2D<Rgba8> thin(60, kH);
    const std::vector<double> side{0.40, 0.30, 0.20};
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < 60; ++x) {
            if (x < 20 || x >= 29) thin(x, y) = kWhite;
            else if (x < 23) thin(x, y) = mix(kBlue, kWhite, side[static_cast<std::size_t>(x - 20)]);
            else if (x < 26) thin(x, y) = kBlue;
            else thin(x, y) = mix(kBlue, kWhite, side[static_cast<std::size_t>(28 - x)]);
        }
    }
    cnx::SegmentationResult t = run(thin, photoParams());
    CHECK(regionCount(t) == 3 && t.stats.rampsAbsorbed == 6);
    const sla::Rgba cb = t.map.forest().stats(t.map.regionAt(24, 10)).color;
    CHECK(sla::distance2(cb, sla::toRgba(kBlue)) < 1e-9);  // exatamente o azul, sem a rampa que ele absorveu
    CHECK(t.map.forest().pixelCount(t.map.regionAt(24, 10)) == static_cast<std::uint32_t>(kH * 9));  // as 3 + as 6 colunas de rampa
}

CNX_TEST(only_the_photo_profiles_turn_the_ramp_absorber_on) {
    for (int i = 0; i <= 9; ++i) {
        cnx::EngineParameters e;
        e.applyProfile(static_cast<cnx::ImageProfile>(i));
        const bool photo = i >= static_cast<int>(cnx::ImageProfile::PhotoHigh) && i <= static_cast<int>(cnx::ImageProfile::PhotoLow);
        CHECK_CTX(e.segment.absorbRamps == photo, std::to_string(i));
        CHECK_CTX(e.segment.rampMaxDepth == 4 && e.segment.rampNearFactor == 4.0, std::to_string(i));  // perfil nenhum mexe nos outros dois
    }
    // Os perfis de logo com antialiasing (SubPixelSegmenter) ficam como estavam: nada de absorvedor antes das praias.
    cnx::EngineParameters logo;
    logo.applyProfile(cnx::ImageProfile::LogoMedAA);
    CHECK(!logo.segment.absorbRamps);
    // Absorvedor desligado: o estágio devolve 0 e não toca o mapa, mesmo chamado direto.
    const Array2D<Rgba8> img = rampScene({0.20, 0.55, 0.85});
    PixelSegmenter seg(img, withoutAbsorber());
    seg.initializeFromQuadTree();
    CHECK(seg.absorbRamps() == 0);
}

CNX_TEST(a_real_thin_stroke_of_another_color_is_kept) {
    // Traço de 2 px escuro em fundo branco: só tem o branco como vizinha, não é mistura de nada.
    Array2D<Rgba8> wire(60, 30, kWhite);
    for (int x = 5; x < 55; ++x) {
        wire(x, 14) = Rgba8{40, 40, 40, 255};
        wire(x, 15) = Rgba8{40, 40, 40, 255};
    }
    cnx::SegmentationResult w = run(wire, photoParams());
    CHECK(regionCount(w) == 2 && w.stats.rampsAbsorbed == 0);
    // Fio verde de 1 px entre o azul e o branco: o verde não está na reta azul-branco, então não é rampa.
    Array2D<Rgba8> green = rampScene({0.5});
    for (int y = 0; y < 30; ++y) green(25, y) = Rgba8{0, 200, 0, 255};
    cnx::SegmentationResult g = run(green, photoParams());
    CHECK(regionCount(g) == 3 && g.stats.rampsAbsorbed == 0);
    // O mesmo fio, com o absorvedor desligado, dá o mesmo mapa.
    CHECK(regionCount(*std::make_unique<cnx::SegmentationResult>(run(green, withoutAbsorber()))) == 3);
    // Um traço escuro ALÉM do azul (sobre a reta branco-azul, mas com t > 1): é um contorno mais escuro, não uma mistura.
    Array2D<Rgba8> dark(60, 30);
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 60; ++x) dark(x, y) = x < 25 ? kWhite : x == 25 ? Rgba8{0, 42, 146, 255} : kBlue;
    }
    cnx::SegmentationResult dk = run(dark, photoParams());
    CHECK(regionCount(dk) == 3 && dk.stats.rampsAbsorbed == 0);
}

CNX_TEST(the_blend_rule_needs_a_real_contrast_between_the_two_sides) {
    // Duas cores a 28 níveis uma da outra (abaixo do contraste mínimo das praias, 36) e uma coluna entre elas, a 14 níveis
    // de cada: com a regra de proximidade desligada nada a explica, porque misturar duas cores quase iguais não é rampa.
    Array2D<Rgba8> img(60, 30);
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 60; ++x) img(x, y) = x < 25 ? Rgba8{100, 100, 100, 255} : x == 25 ? Rgba8{108, 108, 108, 255} : Rgba8{116, 116, 116, 255};
    }
    SegmentParams p = photoParams();
    p.quadTreeVariance = 0.0;
    p.rampNearFactor = 0.0;
    cnx::SegmentationResult r = run(img, p);
    CHECK(regionCount(r) == 3 && r.stats.rampsAbsorbed == 0);
    p.rampNearFactor = 4.0;  // com a proximidade ligada a mesma coluna é ruído de uma das duas e some
    CHECK(regionCount(*std::make_unique<cnx::SegmentationResult>(run(img, p))) == 2);
}

CNX_TEST(a_strip_against_the_image_border_is_thin_because_the_border_is_not_a_neighbor) {
    // Faixa de 2 px colada na borda esquerda, a 17 níveis do branco que a segue: o que está fora da imagem não é vizinho
    // da mesma região, então nenhum pixel dela é interior e ela é fina (e, perto da sólida, ruído).
    Array2D<Rgba8> img(40, 30, kWhite);
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 2; ++x) img(x, y) = Rgba8{245, 245, 245, 255};
    }
    SegmentParams p = photoParams();
    p.quadTreeVariance = 0.0;
    SegmentParams off = p;
    off.absorbRamps = false;
    CHECK(regionCount(*std::make_unique<cnx::SegmentationResult>(run(img, off))) == 2);  // sem o absorvedor a faixa é uma região
    cnx::SegmentationResult r = run(img, p);
    CHECK(regionCount(r) == 1 && r.stats.rampsAbsorbed == 1);
}

CNX_TEST(a_thin_bar_near_a_solid_color_is_noise_but_only_inside_the_nearness_factor) {
    // quadTreeVariance = 0: sem isso a QuadTree trata o traço de baixo contraste como parte do bloco branco e ele nem chega a ser região.
    const auto exact = [](SegmentParams p) {
        p.quadTreeVariance = 0.0;
        return p;
    };
    Array2D<Rgba8> img(40, 30, kWhite);
    // Um traço de 2 x 20 px (grande demais para a QuadTree o deixar passar como homogêneo, fino demais para ter pixel
    // interior) a 17 níveis do branco: acima do teto de fusão (12), abaixo do de proximidade (24).
    for (int y = 15; y < 17; ++y) {
        for (int x = 10; x < 30; ++x) img(x, y) = Rgba8{245, 245, 245, 255};
    }
    cnx::SegmentationResult off = run(img, exact(withoutAbsorber()));
    CHECK(regionCount(off) == 2);
    cnx::SegmentationResult on = run(img, exact(photoParams()));
    CHECK(regionCount(on) == 1 && on.stats.rampsAbsorbed == 1);
    SegmentParams strict = exact(photoParams());
    strict.rampNearFactor = 1.0;  // "perto" = 12 níveis: o traço de 17 fica
    cnx::SegmentationResult kept = run(img, strict);
    CHECK(regionCount(kept) == 2 && kept.stats.rampsAbsorbed == 0);
    // Um traço a 95 níveis não é ruído: é um detalhe (e tampouco é mistura de nada).
    Array2D<Rgba8> detail(40, 30, kWhite);
    for (int y = 15; y < 17; ++y) {
        for (int x = 10; x < 30; ++x) detail(x, y) = Rgba8{200, 200, 200, 255};
    }
    cnx::SegmentationResult d = run(detail, exact(photoParams()));
    CHECK(regionCount(d) == 2);
}

CNX_TEST(wide_gradient_bands_are_solid_and_untouched) {
    // Degradê de 200 colunas: cada faixa tem dezenas de colunas e pixels interiores. O absorvedor não mexe em nada.
    Array2D<Rgba8> img(200, 30);
    for (int x = 0; x < 200; ++x) {
        const double t = x / 199.0;
        for (int y = 0; y < 30; ++y) img(x, y) = mix(Rgba8{30, 100, 230, 255}, Rgba8{250, 200, 20, 255}, t);
    }
    cnx::SegmentationResult off = run(img, withoutAbsorber());
    cnx::SegmentationResult on = run(img, photoParams());
    CHECK(regionCount(off) > 10);
    CHECK(regionCount(on) == regionCount(off) && on.stats.rampsAbsorbed == 0);
    const Array2D<RegionId>& a = off.map.compactLabels();
    const Array2D<RegionId>& b = on.map.compactLabels();
    bool same = true;
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 200; ++x) same = same && a(x, y) == b(x, y);
    }
    CHECK(same);
}

CNX_TEST(the_absorber_keeps_every_region_connected_on_random_scenes) {
    for (int seed = 1; seed <= 14; ++seed) {
        cnxtest::SplitMix64 rng{static_cast<std::uint64_t>(seed) * 977};
        const int w = 44 + rng.below(12);
        const int h = 36 + rng.below(10);
        Array2D<Rgba8> img(w, h);
        const std::vector<Rgba8> ink{kBlue, kWhite, Rgba8{220, 40, 40, 255}, Rgba8{30, 150, 70, 255}, Rgba8{250, 190, 20, 255}};
        // Blocos de tinta, com rampas de 1 a 3 px nas fronteiras verticais e traços finos soltos.
        std::vector<int> cut{0};
        while (cut.back() < w - 8) cut.push_back(std::min(w, cut.back() + 6 + rng.below(8)));  // nunca passa da largura
        if (cut.back() != w) cut.push_back(w);
        for (std::size_t b = 0; b + 1 < cut.size(); ++b) {
            const Rgba8 a = ink[rng.below(5)];
            const Rgba8 c = ink[rng.below(5)];
            const int ramp = 1 + rng.below(3);
            for (int y = 0; y < h; ++y) {
                for (int x = cut[b]; x < cut[b + 1]; ++x) {
                    const int d = x - (cut[b + 1] - ramp);
                    img(x, y) = d < 0 ? a : mix(a, c, (d + 1.0) / (ramp + 1.0));
                }
            }
        }
        for (int k = 0; k < 4; ++k) {
            const int y = 3 + rng.below(h - 6);
            const Rgba8 s = ink[rng.below(5)];
            for (int x = 2; x < w - 2; ++x) img(x, y) = s;
        }
        cnx::SegmentationResult r = run(img, photoParams());
        CHECK_CTX(allConnected(r.map), "seed " + std::to_string(seed));
        CHECK_CTX(regionCount(r) <= run(img, withoutAbsorber()).map.forest().rootCount(), "o absorvedor nunca cria regiões");
    }
}

CNX_TEST(a_blurred_edge_becomes_one_crisp_boundary_in_the_middle_of_the_blur) {
    // Degrau vertical em x = 30,4, borrado por uma rampa linear de 3 px (o que um JPEG reamostrado faz com uma borda).
    constexpr int kW = 80;
    constexpr int kH = 40;
    constexpr double kEdge = 30.4;
    Array2D<Rgba8> img(kW, kH);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const double alpha = std::clamp((x + 0.5 - (kEdge - 1.5)) / 3.0, 0.0, 1.0);
            img(x, y) = mix(kBlue, kWhite, alpha);
        }
    }
    const auto boundaryX = [&](const SegmentParams& sp, std::size_t* faces) {
        cnx::SegmentationResult seg = PixelSegmenter(img, sp).run();
        cnx::TraceResult traced = cnx::BoundaryTracer().trace(seg.map);
        cnx::EngineParameters e;
        e.applyProfile(cnx::ImageProfile::PhotoMed);
        cnx::ContourSmoother(e.smooth, nullptr).run(traced.image);
        *faces = traced.image.faceCount() - 1;  // sem o exterior
        double sum = 0.0;
        int n = 0;
        for (cnx::EdgeId ed = 0; ed < static_cast<cnx::EdgeId>(traced.image.edgeCount()); ++ed) {
            const cnx::Edge& edge = traced.image.edge(ed);
            if (edge.left == cnx::kExteriorFace || edge.right == cnx::kExteriorFace) continue;
            for (const cnx::NodeId node : traced.image.edgeNodes(ed)) {
                const sla::Vec2 p = traced.image.position(node);
                sum += p.x;
                ++n;
            }
        }
        return n > 0 ? sum / n : -1.0;
    };
    std::size_t facesOff = 0;
    std::size_t facesOn = 0;
    const double xOff = boundaryX(withoutAbsorber(), &facesOff);
    const double xOn = boundaryX(photoParams(), &facesOn);
    std::printf("  borda borrada em x = %.1f: sem absorver %zu faces, fronteira em %.2f; absorvendo %zu faces, fronteira em %.2f\n", kEdge, facesOff, xOff, facesOn, xOn);
    CHECK(facesOff >= 5);  // as lascas da rampa
    CHECK(facesOn == 2);   // azul e branco
    CHECK(std::abs(xOn - kEdge) <= 0.75);  // a fronteira nítida cai no meio do borrão, a menos de um pixel inteiro
}

int main() { return cnxtest::runAll(); }
