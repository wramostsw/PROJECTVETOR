// tests/unit/sub_pixel_segmenter_test.cpp — SubPixelSegmenter: micro-regiões, praias, sombras, paleta
// e o pipeline sobre um disco com antialiasing; mais PixelMap::reassignPixel contra força bruta.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <numbers>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

#include "cnx/engine/Parameters.h"
#include "cnx/segment/PixelSegmenter.h"
#include "cnx/segment/SubPixelSegmenter.h"
#include "cnx/vector/BoundaryTracer.h"
#include "logo_scenes.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::PixelMap;
using cnx::RegionId;
using cnx::SegmentParams;
using cnx::SubPixelSegmenter;
using cnxtest::SplitMix64;
using sla::Array2D;
using sla::Rgba8;

namespace {

constexpr Rgba8 kBlack{0, 0, 0, 255};
constexpr Rgba8 kWhite{255, 255, 255, 255};

Rgba8 gray(int v) { return {static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v), 255}; }

cnx::Palette blackAndWhite() {
    cnx::Palette p;
    p.colors = {sla::toRgba(kWhite), sla::toRgba(kBlack)};
    return p;
}

struct Picture {
    Array2D<Rgba8> image;
    PixelMap map;
};

/// Mapa e imagem a partir de desenhos ASCII: `labels` define as regiões (um caractere = uma região) e
/// `colors` a cor de cada pixel (via `table`).
Picture picture(const std::vector<std::string>& labels, const std::vector<std::string>& colors, const std::map<char, Rgba8>& table) {
    const int h = static_cast<int>(labels.size());
    const int w = static_cast<int>(labels[0].size());
    Picture p{Array2D<Rgba8>(w, h), PixelMap(w, h)};
    std::map<char, RegionId> ids;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const char l = labels[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
            if (!ids.count(l)) ids[l] = p.map.createRegion();
            p.image(x, y) = table.at(colors[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)]);
            p.map.setLabel(x, y, ids[l], p.image(x, y));
        }
    }
    return p;
}

bool allRegionsConnected(PixelMap& map) {
    const int w = map.width();
    const int h = map.height();
    std::vector<char> seen(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
    std::map<RegionId, int> components;
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (seen[static_cast<std::size_t>(y * w + x)]) continue;
            const RegionId r = map.regionAt(x, y);
            if (++components[r] > 1) return false;
            stack.assign(1, {x, y});
            seen[static_cast<std::size_t>(y * w + x)] = 1;
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + cnxtest::kDx4[d];
                    const int ny = cy + cnxtest::kDy4[d];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    if (seen[static_cast<std::size_t>(ny * w + nx)] || map.regionAt(nx, ny) != r) continue;
                    seen[static_cast<std::size_t>(ny * w + nx)] = 1;
                    stack.emplace_back(nx, ny);
                }
            }
        }
    }
    return true;
}

}  // namespace

// ================================================================ PixelMap::reassignPixel

CNX_TEST(reassign_pixel_keeps_sizes_and_contacts_exact) {
    SplitMix64 rng{801};
    for (int scene = 0; scene < 4; ++scene) {
        cnxtest::Scene s = cnxtest::blockScene(14 + scene, 11, 3, rng, 4);
        cnxtest::randomMerges(s.map, 8, rng);
        const int w = s.map.width();
        const int h = s.map.height();
        std::vector<RegionId> roots;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const RegionId r = s.map.regionAt(x, y);
                if (std::find(roots.begin(), roots.end(), r) == roots.end()) roots.push_back(r);
            }
        }
        int moved = 0;
        for (int i = 0; i < 300; ++i) {
            const int x = rng.below(w);
            const int y = rng.below(h);
            const RegionId to = roots[static_cast<std::size_t>(rng.below(static_cast<int>(roots.size())))];  // qualquer região, vizinha ou não
            const bool changed = s.map.reassignPixel(x, y, to, s.image(x, y));
            moved += changed ? 1 : 0;
            CHECK(changed == (s.map.regionAt(x, y) == to) || !changed);
        }
        CHECK(moved > 100);

        // Tamanhos, medianas e contatos contra força bruta.
        std::map<RegionId, std::uint32_t> sizes;
        std::map<std::pair<RegionId, RegionId>, std::uint32_t> contacts;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                ++sizes[s.map.regionAt(x, y)];
                for (const auto& [dx, dy] : {std::pair{1, 0}, std::pair{0, 1}}) {
                    if (x + dx >= w || y + dy >= h) continue;
                    const RegionId a = s.map.regionAt(x, y);
                    const RegionId b = s.map.regionAt(x + dx, y + dy);
                    if (a != b) ++contacts[{std::min(a, b), std::max(a, b)}];
                }
            }
        }
        const std::string tag = "cena " + std::to_string(scene);
        for (const RegionId a : roots) {
            if (!s.map.forest().isRoot(a)) continue;
            CHECK_CTX(s.map.forest().pixelCount(a) == sizes[a], tag);
            CHECK_CTX(s.map.forest().stats(a).samples.count() == sizes[a], tag);
            for (const RegionId b : roots) {
                if (b <= a || !s.map.forest().isRoot(b)) continue;
                const auto it = contacts.find({a, b});
                CHECK_CTX(s.map.forest().contactCount(a, b) == (it == contacts.end() ? 0u : it->second), tag);
            }
        }
        // O mapa continua compactável, e o compact() reconstrói os mesmos contatos do zero.
        const RegionId regions = s.map.compact();
        CHECK_CTX(regions > 0, tag);
    }
}

CNX_TEST(reassign_pixel_validates_arguments_and_reports_no_ops) {
    cnxtest::SplitMix64 rng{802};
    cnxtest::Scene s = cnxtest::blockScene(6, 6, 3, rng, 3);
    const RegionId r = s.map.regionAt(0, 0);
    CHECK(!s.map.reassignPixel(0, 0, r, s.image(0, 0)));  // já pertence: nada muda
    CHECK_THROWS(std::out_of_range, (void)s.map.reassignPixel(-1, 0, r, kBlack));
    CHECK_THROWS(std::out_of_range, (void)s.map.reassignPixel(0, 0, 999, kBlack));
    PixelMap empty(2, 2);
    const RegionId e = empty.createRegion();
    CHECK_THROWS(std::logic_error, (void)empty.reassignPixel(0, 0, e, kBlack));  // pixel sem região
}

// ================================================================ micro-regiões

CNX_TEST(micro_regions_are_absorbed_smallest_first) {
    // A (cinza 0) é grande; y (2 px, cinza 50) e x (1 px, cinza 110) se tocam; z (2 px, cinza 20) só toca A.
    // Distâncias: y→A = 50 < y→x = 60 < x→A = 110. Em ordem crescente de tamanho, x (1 px) vai primeiro
    // para y (60 < 110); y fica com 3 px e sobrevive. Uma varredura única por id absorveria y em A.
    const std::vector<std::string> labels{
        "AAAAAAAAAAAA",
        "AAAAyyxAAAAA",
        "AAAAAAAAzzAA"};
    const std::vector<std::string> colors{
        "kkkkkkkkkkkk",
        "kkkkgghkkkkk",
        "kkkkkkkkjjkk"};
    const std::map<char, Rgba8> table{{'k', gray(0)}, {'g', gray(50)}, {'h', gray(110)}, {'j', gray(20)}};
    Picture p = picture(labels, colors, table);
    SegmentParams params;
    const SubPixelSegmenter seg(p.image, params, blackAndWhite());
    CHECK(seg.mergeMicroRegions(p.map) == 2);  // x → y e z → A
    CHECK(p.map.regionAt(4, 1) == p.map.regionAt(5, 1) && p.map.regionAt(5, 1) == p.map.regionAt(6, 1));
    CHECK(p.map.regionAt(4, 1) != p.map.regionAt(0, 0));  // y (agora com 3 px) sobreviveu
    CHECK(p.map.regionAt(8, 2) == p.map.regionAt(0, 0));  // z foi para A
    CHECK(p.map.forest().rootCount() == 2);
    CHECK(allRegionsConnected(p.map));

    // Com o limite em 1 px, só x é limpo.
    Picture q = picture(labels, colors, table);
    SegmentParams one;
    one.microRegionMaxPixels = 1;
    CHECK(SubPixelSegmenter(q.image, one, blackAndWhite()).mergeMicroRegions(q.map) == 1);
    CHECK(q.map.regionAt(8, 2) != q.map.regionAt(0, 0));  // z (2 px) fica
}

CNX_TEST(a_lone_region_is_never_absorbed) {
    const std::vector<std::string> labels{"AA"};
    const std::vector<std::string> colors{"kk"};
    Picture p = picture(labels, colors, {{'k', kBlack}});
    CHECK(SubPixelSegmenter(p.image, SegmentParams{}, blackAndWhite()).mergeMicroRegions(p.map) == 0);
    CHECK(p.map.forest().rootCount() == 1);
}

// ================================================================ praias

CNX_TEST(a_one_pixel_beach_is_split_between_its_two_dominant_colors) {
    // Branco | faixa de 1 px com tons de transição | preto. Cada pixel da faixa vai para o lado da
    // sua própria cor (t = fração de preto < ½ → branco). A faixa tem 6 pixels, cores 200, 60, 100, 230, 140, 30.
    const std::vector<std::string> labels{
        "AAAAbBBBBB", "AAAAbBBBBB", "AAAAbBBBBB", "AAAAbBBBBB", "AAAAbBBBBB", "AAAAbBBBBB"};
    const std::array<int, 6> strip{200, 60, 100, 230, 140, 30};
    std::vector<std::string> colors;
    std::map<char, Rgba8> table{{'w', kWhite}, {'k', kBlack}};
    for (int y = 0; y < 6; ++y) {
        const char c = static_cast<char>('0' + y);
        table[c] = gray(strip[static_cast<std::size_t>(y)]);
        colors.push_back(std::string("wwww") + c + "kkkkk");
    }
    Picture p = picture(labels, colors, table);
    const SubPixelSegmenter seg(p.image, SegmentParams{}, blackAndWhite());
    const auto result = seg.splitAndMergeBeaches(p.map);
    CHECK(result.beaches == 1);
    // O último pixel da faixa fica isolado (não é ponto simples) e é fundido inteiro, ao lado do seu tom.
    CHECK(result.moved == 5 && result.merged == 1);
    const RegionId white = p.map.regionAt(0, 0);
    const RegionId black = p.map.regionAt(9, 0);
    CHECK(white != black);
    for (int y = 0; y < 6; ++y) {
        const bool towardWhite = strip[static_cast<std::size_t>(y)] > 127;  // mais claro que o meio → lado branco
        CHECK_CTX(p.map.regionAt(4, y) == (towardWhite ? white : black), "linha " + std::to_string(y));
        for (int x = 0; x < 4; ++x) CHECK(p.map.regionAt(x, y) == white);
        for (int x = 5; x < 10; ++x) CHECK(p.map.regionAt(x, y) == black);
    }
    CHECK(p.map.forest().rootCount() == 2 + 0);  // só sobram as duas dominantes (a faixa esvaziou)
    CHECK(allRegionsConnected(p.map));
}

CNX_TEST(a_two_pixel_beach_puts_the_boundary_in_the_middle_of_the_transition) {
    // Faixa de 2 colunas: a da esquerda é 25 % preta, a da direita 75 %. A fronteira cai entre elas.
    const std::vector<std::string> labels{"AAAbbBBB", "AAAbbBBB", "AAAbbBBB", "AAAbbBBB", "AAAbbBBB"};
    const std::vector<std::string> colors(5, "wwwlrkkk");
    Picture p = picture(labels, colors, {{'w', kWhite}, {'k', kBlack}, {'l', gray(192)}, {'r', gray(64)}});
    const SubPixelSegmenter seg(p.image, SegmentParams{}, blackAndWhite());
    const auto result = seg.splitAndMergeBeaches(p.map);
    CHECK(result.beaches == 1);
    const RegionId white = p.map.regionAt(0, 0);
    const RegionId black = p.map.regionAt(7, 0);
    for (int y = 0; y < 5; ++y) {
        CHECK(p.map.regionAt(3, y) == white);  // coluna clara → branco
        CHECK(p.map.regionAt(4, y) == black);  // coluna escura → preto
    }
    CHECK(p.map.forest().rootCount() == 2);
    CHECK(allRegionsConnected(p.map));
}

CNX_TEST(regions_that_are_not_beaches_are_left_alone) {
    const auto untouched = [](const std::vector<std::string>& labels, const std::vector<std::string>& colors,
                              const std::map<char, Rgba8>& table, const char* what) {
        Picture p = picture(labels, colors, table);
        const std::size_t before = p.map.forest().rootCount();
        const auto result = SubPixelSegmenter(p.image, SegmentParams{}, blackAndWhite()).splitAndMergeBeaches(p.map);
        CHECK_CTX(result.beaches == 0 && result.moved == 0, what);
        CHECK_CTX(p.map.forest().rootCount() == before, what);
    };
    // (a) faixa de uma terceira cor (vermelho entre branco e preto): fora da reta A–B.
    untouched({"AAAbBBB", "AAAbBBB", "AAAbBBB", "AAAbBBB"}, {"wwwrkkk", "wwwrkkk", "wwwrkkk", "wwwrkkk"},
              {{'w', kWhite}, {'k', kBlack}, {'r', Rgba8{255, 0, 0, 255}}}, "terceira cor");
    // (b) banda larga de cinza (3 colunas, com pixels interiores): é uma região de verdade.
    untouched({"AAbbbBBB", "AAbbbBBB", "AAbbbBBB", "AAbbbBBB", "AAbbbBBB"}, std::vector<std::string>(5, "wwgggkkk"),
              {{'w', kWhite}, {'k', kBlack}, {'g', gray(128)}}, "banda larga");
    // (d) a faixa toca três regiões e as duas dominantes cobrem só 75 % do contorno (< 80 %): não é praia.
    untouched({"AAAbBBB", "AAAbBBB", "AAAbBBB", "CCCbCCC", "CCCbCCC", "CCCbCCC"},
              {"wwwgkkk", "wwwgkkk", "wwwgkkk", "kkkgkkk", "kkkgkkk", "kkkgkkk"},
              {{'w', kWhite}, {'k', kBlack}, {'g', gray(128)}}, "contorno repartido em três");
    // (c) contraste baixo entre as dominantes (2 níveis): ambíguo, não se mexe.
    untouched({"AAAbBBB", "AAAbBBB", "AAAbBBB"}, std::vector<std::string>(3, "aaabccc"),
              {{'a', gray(100)}, {'b', gray(101)}, {'c', gray(102)}}, "contraste baixo");
}

// ================================================================ sombras

namespace {

/// 32×24: faixa superior = gradiente horizontal (colunas 0–23) + preto chapado (24–31); meio = cinza chapado;
/// base = cinza com ruído de ±12.
Picture shadowScene(SplitMix64& rng, bool noisyGradient = false) {
    const int w = 32;
    const int h = 24;
    Picture p{Array2D<Rgba8>(w, h), PixelMap(w, h)};
    const RegionId s = p.map.createRegion();   // gradiente
    const RegionId k = p.map.createRegion();   // preto chapado
    const RegionId f = p.map.createRegion();   // cinza chapado
    const RegionId n = p.map.createRegion();   // cinza com ruído
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            RegionId id;
            Rgba8 c;
            if (y < 8) {
                if (x < 24) {
                    id = s;
                    c = gray(std::clamp(255 - static_cast<int>(std::lround(255.0 * x / 23.0)) + (noisyGradient ? rng.below(25) - 12 : 0), 0, 255));
                } else {
                    id = k;
                    c = kBlack;
                }
            } else if (y < 16) {
                id = f;
                c = gray(128);
            } else {
                id = n;
                c = gray(128 + rng.below(25) - 12);
            }
            p.image(x, y) = c;
            p.map.setLabel(x, y, id, c);
        }
    }
    return p;
}

}  // namespace

CNX_TEST(shadow_segments_are_flagged_only_for_smooth_gradients) {
    SplitMix64 rng{803};
    Picture p = shadowScene(rng);
    const SubPixelSegmenter seg(p.image, SegmentParams{}, blackAndWhite());
    CHECK(seg.flagShadowSegments(p.map) == 1);
    const auto flagged = [&](int x, int y) { return (p.map.forest().stats(p.map.regionAt(x, y)).flags & cnx::RegionFlag::Shadow) != 0; };
    CHECK(flagged(5, 3));     // gradiente suave
    CHECK(!flagged(28, 3));   // preto chapado
    CHECK(!flagged(10, 12));  // cinza chapado
    CHECK(!flagged(10, 20));  // ruído: variância alta, mas o desfoque a elimina (não é suave)
}

CNX_TEST(shadow_segments_are_rebuilt_in_bands_from_the_blurred_image) {
    SplitMix64 rng{804};
    Picture p = shadowScene(rng);
    const SubPixelSegmenter seg(p.image, SegmentParams{}, blackAndWhite());
    CHECK(seg.flagShadowSegments(p.map) == 1);
    CHECK(seg.rebuildShadowSegments(p.map) == 1);  // o gradiente vira duas faixas: branca e preta
    const RegionId bright = p.map.regionAt(0, 3);
    const RegionId dark = p.map.regionAt(23, 3);
    CHECK(bright != dark);
    // A fronteira cai no meio do gradiente (entre as colunas 11 e 12), em todas as linhas do centro.
    for (int y = 1; y < 6; ++y) {
        for (int x = 0; x < 24; ++x) CHECK_CTX(p.map.regionAt(x, y) == (x <= 11 ? bright : dark), std::to_string(x) + "," + std::to_string(y));
    }
    CHECK(allRegionsConnected(p.map));
    // As outras regiões não mudaram.
    CHECK(p.map.forest().pixelCount(p.map.regionAt(10, 12)) == 32u * 8u);
}

CNX_TEST(shadow_rebuild_uses_the_blurred_image_so_noise_does_not_fragment_the_bands) {
    // Gradiente com ruído de ±12 por pixel: classificar a imagem bruta fragmentaria a fronteira em
    // ilhas; a imagem desfocada (5×5) tem ruído ~1,4 e dá exatamente duas faixas.
    SplitMix64 rng{805};
    Picture p = shadowScene(rng, true);
    const SubPixelSegmenter seg(p.image, SegmentParams{}, blackAndWhite());
    CHECK(seg.flagShadowSegments(p.map) >= 1);
    CHECK(seg.rebuildShadowSegments(p.map) == 1);
    const RegionId bright = p.map.regionAt(0, 3);
    const RegionId dark = p.map.regionAt(23, 3);
    CHECK(bright != dark);
    for (int y = 1; y < 6; ++y) {
        for (int x = 0; x < 24; ++x) CHECK_CTX(p.map.regionAt(x, y) == (x <= 11 ? bright : dark), std::to_string(x) + "," + std::to_string(y));
    }
}

// ================================================================ sombras ao redor de formas com AA (docs §1.26)

namespace {

/// Cenas de logo → imagem, paleta com as tintas exatas da cena (índice = classe de cor) e a classe de cada forma.
struct Logo {
    Array2D<Rgba8> image;
    cnx::Palette palette;
    std::vector<int> classOfShape;
    cnxtest::LogoScene scene;
};

Logo makeLogo(const cnxtest::LogoScene& scene) {
    Logo l{cnxtest::renderScene(scene), {}, {}, scene};
    std::vector<Rgba8> classes{scene.background};
    for (const cnxtest::Shape& sh : scene.shapes) {
        std::size_t c = classes.size();
        for (std::size_t i = 0; i < classes.size(); ++i) {
            if (classes[i].r == sh.color.r && classes[i].g == sh.color.g && classes[i].b == sh.color.b) c = i;
        }
        if (c == classes.size()) classes.push_back(sh.color);
        l.classOfShape.push_back(static_cast<int>(c));
    }
    for (const Rgba8 c : classes) l.palette.colors.push_back(sla::toRgba(c));
    return l;
}

cnx::SegmentParams logoSegmentParams() {
    cnx::EngineParameters p;
    p.applyProfile(cnx::ImageProfile::LogoMedAA);
    return p.segment;
}

/// Mapa depois do PixelSegmenter, com os parâmetros do perfil de logo com AA.
PixelMap pixelSegment(const Logo& l) {
    cnx::PixelSegmenter seg(l.image, logoSegmentParams());
    return seg.run({}, &l.palette).map;
}

/// Pixels cuja tinta final (índice da paleta) não existe em nenhum ponto do pixel: a franja de uma terceira cor.
int fringePixels(const Logo& l, PixelMap& map) {
    int fringe = 0;
    constexpr int kSamples = 8;
    for (int y = 0; y < l.scene.height; ++y) {
        for (int x = 0; x < l.scene.width; ++x) {
            std::vector<char> present(l.palette.colors.size(), 0);
            for (int j = 0; j < kSamples; ++j) {
                for (int i = 0; i < kSamples; ++i) {
                    int cls = 0;
                    for (std::size_t k = 0; k < l.scene.shapes.size(); ++k) {
                        if (l.scene.shapes[k].inside(x + (i + 0.5) / kSamples, y + (j + 0.5) / kSamples)) cls = l.classOfShape[k];
                    }
                    present[static_cast<std::size_t>(cls)] = 1;
                }
            }
            const std::int32_t ink = map.forest().stats(map.regionAt(x, y)).paletteIndex;
            if (ink < 0 || !present[static_cast<std::size_t>(ink)]) ++fringe;
        }
    }
    return fringe;
}

}  // namespace

CNX_TEST(a_flat_background_with_beach_halo_is_not_flagged_as_a_shadow) {
    // Retângulo vermelho com AA sobre branco, paleta com um amarelo a mais. Depois das praias, o fundo carrega os pixels
    // de AA do lado branco do retângulo; vistos como um todo eles davam ao fundo chapado a variância de um gradiente,
    // e a reconstrução refazia o fundo pela cor de paleta mais próxima da imagem desfocada: o amarelo.
    cnxtest::LogoScene scene{"retangulo", 40, 30, Rgba8{255, 255, 255, 255}, {}, 3};
    scene.shapes.push_back({cnxtest::rectShape(8.5, 6.5, 30.5, 20.5), Rgba8{220, 40, 40, 255}});
    scene.shapes.push_back({cnxtest::rectShape(-9, -9, -8, -8), Rgba8{250, 190, 20, 255}});  // só para a tinta existir na paleta
    Logo l = makeLogo(scene);
    PixelMap map = pixelSegment(l);
    const SubPixelSegmenter seg(l.image, logoSegmentParams(), l.palette);
    seg.mergeMicroRegions(map);
    CHECK(seg.splitAndMergeBeaches(map).beaches > 0);  // o halo existe: sem praias o teste não prova nada
    CHECK(seg.flagShadowSegments(map) == 0);
    CHECK(seg.rebuildShadowSegments(map) == 0);
}

CNX_TEST(antialiased_logos_come_out_without_a_third_ink_fringe) {
    for (const cnxtest::LogoScene& scene : {cnxtest::shapesScene(), cnxtest::lettersScene()}) {
        Logo l = makeLogo(scene);
        const SubPixelSegmenter seg(l.image, logoSegmentParams(), l.palette);
        cnx::SubPixelResult r = seg.run(pixelSegment(l));
        const int fringe = fringePixels(l, r.map);
        std::printf("  %s: %d regiões, %d pixels com tinta que não existe neles (sombras marcadas %d, faixas novas %d)\n", scene.name.c_str(),
                    r.stats.finalRegions, fringe, r.stats.shadowFlagged, r.stats.shadowNewRegions);
        CHECK_CTX(fringe == 0, scene.name);
        // Uma região por forma conectada (e o fundo): nada de faixas de terceira cor entre elas.
        CHECK_CTX(r.stats.finalRegions == (scene.name == "formas" ? 7 : 5), scene.name);
        CHECK_CTX(allRegionsConnected(r.map), scene.name);
    }
}

namespace {

/// 32×8: coluna 0–3 branca, 4–27 gradiente linear de `top` a 0, 28–31 preta. `red` acrescenta duas linhas vermelhas
/// no pé (linhas 6–7) ou no topo (linhas 0–1) do gradiente, que ocupa as outras seis; `redFirst` cria a região
/// vermelha antes das demais (a ordem em que a floresta lista as vizinhas muda com isso).
struct FadeOptions {
    enum class Red { None, Below, Above } red = Red::None;
    bool redFirst = false;
    int top = 255;
    bool mirror = false;  // branco à direita e preto à esquerda
};

struct Fade {
    Array2D<Rgba8> image;
    PixelMap map;
    RegionId fade = 0, white = 0, black = 0, red = 0;
    int firstRow = 0, lastRow = 7;  ///< linhas do gradiente
};

Fade makeFade(FadeOptions o = {}) {
    const int w = 32;
    const int h = 8;
    const bool hasRed = o.red != FadeOptions::Red::None;
    Fade f{Array2D<Rgba8>(w, h), PixelMap(w, h)};
    if (hasRed && o.redFirst) f.red = f.map.createRegion();
    f.fade = f.map.createRegion();
    f.white = f.map.createRegion();
    f.black = f.map.createRegion();
    if (hasRed && !o.redFirst) f.red = f.map.createRegion();
    if (o.red == FadeOptions::Red::Below) f.lastRow = 5;
    if (o.red == FadeOptions::Red::Above) f.firstRow = 2;
    for (int y = 0; y < h; ++y) {
        for (int x0 = 0; x0 < w; ++x0) {
            const int x = o.mirror ? w - 1 - x0 : x0;  // coluna na disposição não espelhada
            RegionId id = f.fade;
            Rgba8 c = gray(std::clamp(o.top - static_cast<int>(std::lround(o.top * (x - 4) / 23.0)), 0, 255));
            if (hasRed && (y < f.firstRow || y > f.lastRow)) {
                id = f.red;
                c = Rgba8{220, 40, 40, 255};
            } else if (x < 4) {
                id = f.white;
                c = kWhite;
            } else if (x >= 28) {
                id = f.black;
                c = kBlack;
            }
            f.image(x0, y) = c;
            f.map.setLabel(x0, y, id, c);
        }
    }
    return f;
}
cnx::Palette withInk(Rgba8 extra) {
    cnx::Palette p = blackAndWhite();
    p.colors.push_back(sla::toRgba(extra));
    return p;
}

/// Gradiente de branco a preto com uma faixa vermelha na vizinhança: a reconstrução tem de dar exatamente duas faixas,
/// com a fronteira depois da coluna `lastBright` (contada a partir da ponta branca).
void expectTwoBands(FadeOptions o, int lastBright) {
    cnx::Palette p = blackAndWhite();
    p.colors.push_back(sla::toRgba(Rgba8{220, 40, 40, 255}));
    const std::string tag = std::string(o.red == FadeOptions::Red::Below ? "embaixo" : "em cima") + (o.redFirst ? ", criado antes" : ", criado depois") + (o.mirror ? ", espelhado" : "") + ", topo " + std::to_string(o.top);
    Fade f = makeFade(o);
    const SubPixelSegmenter seg(f.image, SegmentParams{}, p);
    CHECK_CTX(seg.flagShadowSegments(f.map) == 1, tag);
    CHECK_CTX(seg.rebuildShadowSegments(f.map) == 1, tag);
    const RegionId bright = f.map.regionAt(o.mirror ? 26 : 5, 3);
    const RegionId dark = f.map.regionAt(o.mirror ? 5 : 26, 3);
    CHECK_CTX(bright != dark, tag);
    for (int y = f.firstRow; y <= f.lastRow; ++y) {
        for (int x = 4; x < 28; ++x) {
            const bool isBright = o.mirror ? (31 - x) <= lastBright : x <= lastBright;
            CHECK_CTX(f.map.regionAt(x, y) == (isBright ? bright : dark), tag + ": " + std::to_string(x) + "," + std::to_string(y));
        }
    }
}
}  // namespace

CNX_TEST(a_fade_between_white_and_black_never_gets_a_band_in_a_third_ink) {
    // O cinza do meio do gradiente está mais perto do amarelo (distância² 0,43) que do branco ou do preto (0,75): pela
    // regra antiga de "cor de paleta mais próxima" ele virava uma faixa amarela entre as duas.
    Fade f = makeFade();
    const SubPixelSegmenter seg(f.image, SegmentParams{}, withInk(Rgba8{250, 188, 26, 255}));
    CHECK(seg.flagShadowSegments(f.map) == 1);
    CHECK(seg.rebuildShadowSegments(f.map) == 1);  // duas faixas, não três
    const RegionId bright = f.map.regionAt(5, 3);
    const RegionId dark = f.map.regionAt(26, 3);
    CHECK(bright != dark);
    for (int y = 0; y < 8; ++y) {
        for (int x = 4; x < 28; ++x) CHECK_CTX(f.map.regionAt(x, y) == (x <= 15 ? bright : dark), std::to_string(x) + "," + std::to_string(y));
    }
}

CNX_TEST(a_palette_gray_on_the_line_between_the_neighbor_inks_still_gets_its_band) {
    // O cinza 128 da paleta está sobre a reta branco–preto: é uma mistura legítima, e a sombra ganha as três faixas.
    Fade f = makeFade();
    const SubPixelSegmenter seg(f.image, SegmentParams{}, withInk(gray(128)));
    CHECK(seg.flagShadowSegments(f.map) == 1);
    CHECK(seg.rebuildShadowSegments(f.map) == 2);
    const RegionId bright = f.map.regionAt(5, 3);
    const RegionId mid = f.map.regionAt(15, 3);
    const RegionId dark = f.map.regionAt(26, 3);
    CHECK(bright != mid && mid != dark && bright != dark);
}

CNX_TEST(the_blur_does_not_bleed_a_neighbor_ink_into_the_fade) {
    // Uma faixa vermelha toca o pé (ou o topo) do gradiente. Um desfoque comum mistura o vermelho nas duas linhas de cima
    // dele e o cinza do meio vira vermelho; o desfoque restrito à região deixa o gradiente em duas faixas. E a vizinha
    // vermelha também tem de ficar de fora do par de extremos, onde quer que ela apareça na lista de vizinhas (por isso
    // as oito disposições: vermelho em cima ou embaixo, criado antes ou depois das outras regiões, espelhado ou não).
    for (const FadeOptions::Red where : {FadeOptions::Red::Below, FadeOptions::Red::Above}) {
        for (const bool redFirst : {false, true}) {
            for (const bool mirror : {false, true}) expectTwoBands({where, redFirst, 255, mirror}, 15);
        }
    }
}

CNX_TEST(the_pair_of_extremes_is_chosen_from_all_the_colors_of_the_region) {
    // O gradiente começa com um platô de branco puro (top = 300 satura em 255 nas 4 primeiras colunas), e o último pixel
    // da região, em ordem de varredura, cai nele (espelhado ou não). O branco é extremo de dois pares (branco–preto e
    // branco–vermelho), então esse pixel sozinho empata os dois e a ordem das vizinhas decide; as demais cores é que
    // desempatam: elas só estão na reta branco–preto. Com o par errado a faixa escura viraria vermelha, e a fronteira
    // sairia em torno do cinza 158 (coluna 15) em vez do cinza 128 (entre as colunas 17 e 18).
    for (const FadeOptions::Red where : {FadeOptions::Red::Below, FadeOptions::Red::Above}) {
        for (const bool redFirst : {false, true}) {
            for (const bool mirror : {false, true}) expectTwoBands({where, redFirst, 300, mirror}, 17);
        }
    }
}

CNX_TEST(the_blur_radius_parameter_controls_the_reconstruction) {
    // Gradiente de 150 a 0 entre o branco e o preto: com raio 2 a ponta esquerda (150) fica clara e vira uma faixa branca;
    // com raio 8 a janela, cortada pela borda da região, tem média 124 (< 127,5) e a faixa clara não existe.
    for (const auto& [radius, bands] : {std::pair{2, 1}, std::pair{8, 0}}) {
        Fade f = makeFade({FadeOptions::Red::None, false, 150});
        SegmentParams params;
        params.shadowBlurRadius = radius;
        const SubPixelSegmenter seg(f.image, params, blackAndWhite());
        CHECK_CTX(seg.flagShadowSegments(f.map) == 1, "raio " + std::to_string(radius));
        CHECK_CTX(seg.rebuildShadowSegments(f.map) == bands, "raio " + std::to_string(radius));
    }
}

CNX_TEST(a_lone_fade_region_is_rebuilt_with_the_whole_palette) {
    // Sem vizinhas não há tintas vizinhas para consultar: a região única do mapa usa a paleta inteira.
    const int w = 32;
    const int h = 8;
    Picture p{Array2D<Rgba8>(w, h), PixelMap(w, h)};
    const RegionId only = p.map.createRegion();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            p.image(x, y) = gray(255 - static_cast<int>(std::lround(255.0 * x / (w - 1))));
            p.map.setLabel(x, y, only, p.image(x, y));
        }
    }
    const SubPixelSegmenter seg(p.image, SegmentParams{}, blackAndWhite());
    CHECK(seg.flagShadowSegments(p.map) == 1);
    CHECK(seg.rebuildShadowSegments(p.map) == 1);
    CHECK(p.map.regionAt(3, 3) != p.map.regionAt(28, 3));
}
CNX_TEST(shadow_detection_counts_only_interior_pixels) {
    // Região de 20×20 chapada branca, com 1 px de borda em cinza-claro (o halo de AA que as praias deixam): os 76 pixels
    // de borda sozinhos têm variância e o desfoque comum a preserva, os 324 do interior não têm nenhuma. Marcada pelos
    // pixels de borda seria sombra; pelos do interior, não.
    const int n = 20;
    Picture p{Array2D<Rgba8>(n + 4, n + 4), PixelMap(n + 4, n + 4)};
    const RegionId dark = p.map.createRegion();
    const RegionId box = p.map.createRegion();
    for (int y = 0; y < n + 4; ++y) {
        for (int x = 0; x < n + 4; ++x) {
            const bool inside = x >= 2 && x < n + 2 && y >= 2 && y < n + 2;
            const bool rim = inside && (x == 2 || x == n + 1 || y == 2 || y == n + 1);
            const Rgba8 c = !inside ? kBlack : (rim ? gray(180) : kWhite);
            p.image(x, y) = c;
            p.map.setLabel(x, y, inside ? box : dark, c);
        }
    }
    SegmentParams params;
    params.shadowMinPixels = 16;
    const SubPixelSegmenter seg(p.image, params, blackAndWhite());
    CHECK(seg.flagShadowSegments(p.map) == 0);
    // O pixel da borda da imagem também não é interior (falta um vizinho): uma região que ocupa a imagem inteira, com o
    // anel externo em cinza, é chapada por dentro.
    Picture whole{Array2D<Rgba8>(24, 24), PixelMap(24, 24)};
    const RegionId only = whole.map.createRegion();
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 24; ++x) {
            const bool edge = x == 0 || y == 0 || x == 23 || y == 23;
            whole.image(x, y) = edge ? gray(150) : kWhite;
            whole.map.setLabel(x, y, only, whole.image(x, y));
        }
    }
    SegmentParams one;
    one.shadowBlurRadius = 0;  // sem suavização a variância do anel, se contada, seria inteira
    CHECK(SubPixelSegmenter(whole.image, one, blackAndWhite()).flagShadowSegments(whole.map) == 0);
}

CNX_TEST(shadow_detection_still_flags_a_gradient_that_only_its_interior_carries) {
    // Uma caixa de 12×12 com um gradiente no interior (100 pixels) e a borda chapada: continua sendo sombra, então a
    // regra por pixels interiores não deixou de enxergar o gradiente. E o mínimo de pixels conta só os interiores.
    const int n = 12;
    Picture p{Array2D<Rgba8>(n + 4, n + 4), PixelMap(n + 4, n + 4)};
    const RegionId outer = p.map.createRegion();
    const RegionId box = p.map.createRegion();
    for (int y = 0; y < n + 4; ++y) {
        for (int x = 0; x < n + 4; ++x) {
            const bool inside = x >= 2 && x < n + 2 && y >= 2 && y < n + 2;
            const bool rim = inside && (x == 2 || x == n + 1 || y == 2 || y == n + 1);
            const Rgba8 c = !inside ? kBlack : (rim ? gray(128) : gray(30 + 20 * (x - 3)));
            p.image(x, y) = c;
            p.map.setLabel(x, y, inside ? box : outer, c);
        }
    }
    SegmentParams params;
    params.shadowMinPixels = 100;  // exatamente os interiores
    params.shadowBlurRadius = 1;
    CHECK(SubPixelSegmenter(p.image, params, blackAndWhite()).flagShadowSegments(p.map) == 1);
    params.shadowMinPixels = 101;  // a região tem 144 pixels, mas só 100 interiores
    PixelMap again = p.map;
    CHECK(SubPixelSegmenter(p.image, params, blackAndWhite()).flagShadowSegments(again) == 0);
}

// ================================================================ paleta

CNX_TEST(palette_stage_merges_neighbors_with_the_same_palette_color_and_sweeps_small_fragments) {
    // Branco quase igual em duas regiões vizinhas (P e Q) e preto quase igual em outras duas (B1, B2);
    // um fragmento T de 3 px de cor clara cercado de preto só toca a região preta.
    const std::vector<std::string> labels{
        "PPPPQQQQBBBBCCCC",
        "PUUPQQQQBBBBCCCC",
        "PUUPQQQQBTTTCCCC",
        "PPPPQQQQBBBBCCCC"};
    const std::vector<std::string> colors{
        "wwwwvvvvkkkkjjjj",
        "wuuwvvvvkkkkjjjj",
        "wuuwvvvvkgggjjjj",
        "wwwwvvvvkkkkjjjj"};
    const std::map<char, Rgba8> table{{'u', gray(4)}, {'w', gray(252)}, {'v', gray(250)}, {'k', gray(3)}, {'j', gray(5)}, {'g', gray(230)}};
    Picture p = picture(labels, colors, table);
    SegmentParams params;
    params.minRegionPixels = 4;
    const SubPixelSegmenter seg(p.image, params, blackAndWhite());
    cnx::snapColorsToPalette(p.map, blackAndWhite());
    // T é clara: snap → branco; as vizinhas mais próximas (pretas) são as únicas que ela toca.
    CHECK(p.map.forest().stats(p.map.regionAt(10, 2)).paletteIndex == 0);
    CHECK(seg.mergeSamePaletteNeighbors(p.map) == 2);  // P+Q e B+C
    CHECK(p.map.regionAt(0, 0) == p.map.regionAt(7, 3));
    CHECK(p.map.regionAt(8, 0) == p.map.regionAt(15, 3));
    CHECK(p.map.regionAt(0, 0) != p.map.regionAt(8, 0));
    CHECK(seg.enforceMinNumPixels(p.map) == 1);  // T (3 px < 4) vai para o preto; U (exatamente 4 px) fica
    const RegionId black = p.map.regionAt(8, 0);
    CHECK(p.map.regionAt(10, 2) == black);
    CHECK(p.map.forest().stats(black).paletteIndex == 1);
    CHECK(p.map.forest().stats(black).color == sla::toRgba(kBlack));
    CHECK(p.map.regionAt(1, 1) != p.map.regionAt(0, 0) && p.map.forest().pixelCount(p.map.regionAt(1, 1)) == 4);  // U, com 4 px
    CHECK(p.map.forest().stats(p.map.regionAt(1, 1)).paletteIndex == 1);
    CHECK(p.map.forest().rootCount() == 3);
    CHECK(allRegionsConnected(p.map));
}

CNX_TEST(palette_merge_ignores_regions_without_a_palette_index_and_the_neighbor_color_prevails) {
    // Regiões ainda sem índice de paleta (−1) nunca se fundem entre si, mesmo vizinhas e de cor igual.
    {
        Picture p = picture({"LLLRRR"}, {"kkkkkk"}, {{'k', kBlack}});
        const SubPixelSegmenter seg(p.image, SegmentParams{}, blackAndWhite());
        CHECK(p.map.forest().stats(p.map.regionAt(0, 0)).paletteIndex == -1);
        CHECK(seg.mergeSamePaletteNeighbors(p.map) == 0);
        CHECK(p.map.forest().rootCount() == 2);
    }
    // Duas regiões vizinhas de 3 px, ambas abaixo do mínimo: a de menor id (clara) é processada primeiro,
    // funde-se à escura e a cor da vizinha prevalece, embora a união escolha a raiz da região clara
    // (empate de tamanho → menor id).
    {
        Picture p = picture({"LLLRRR"}, {"wwwkkk"}, {{'w', gray(250)}, {'k', gray(3)}});
        SegmentParams params;
        params.minRegionPixels = 4;
        const SubPixelSegmenter seg(p.image, params, blackAndWhite());
        cnx::snapColorsToPalette(p.map, blackAndWhite());
        CHECK(p.map.forest().stats(p.map.regionAt(0, 0)).paletteIndex == 0);
        CHECK(p.map.forest().stats(p.map.regionAt(5, 0)).paletteIndex == 1);
        CHECK(seg.enforceMinNumPixels(p.map) == 1);
        CHECK(p.map.forest().rootCount() == 1);
        const RegionId only = p.map.regionAt(0, 0);
        CHECK(p.map.forest().stats(only).paletteIndex == 1);
        CHECK(p.map.forest().stats(only).color == sla::toRgba(kBlack));
    }
}

CNX_TEST(split_disconnected_separates_the_components_of_a_region) {
    // Região A de 5 pixels em linha; o pixel do meio vai para uma região nova B, e A fica em duas ilhas de 2 px.
    Picture p = picture({"AAAAA"}, {"kkkkk"}, {{'k', kBlack}});
    const RegionId a = p.map.regionAt(0, 0);
    const RegionId b = p.map.createRegion();
    CHECK(p.map.reassignPixel(2, 0, b, p.image(2, 0)));
    CHECK(!allRegionsConnected(p.map));
    const std::size_t before = p.map.forest().rootCount();
    CHECK(SubPixelSegmenter::splitDisconnected(p.map, p.image) == 1);
    CHECK(p.map.forest().rootCount() == before + 1);
    CHECK(allRegionsConnected(p.map));
    CHECK(p.map.regionAt(0, 0) == a && p.map.regionAt(1, 0) == a);   // a ilha encontrada primeiro fica com o id (empate)
    CHECK(p.map.regionAt(3, 0) != a && p.map.regionAt(3, 0) == p.map.regionAt(4, 0));
    CHECK(p.map.regionAt(2, 0) == b);
    // Contagens e medianas acompanham: 2 + 1 + 2 pixels.
    CHECK(p.map.forest().pixelCount(a) == 2 && p.map.forest().pixelCount(b) == 1);
    CHECK(p.map.forest().pixelCount(p.map.regionAt(3, 0)) == 2);
    CHECK(SubPixelSegmenter::splitDisconnected(p.map, p.image) == 0);  // idempotente
}
// ================================================================ pipeline

namespace {

struct AaDisk {
    Array2D<Rgba8> image;
    Array2D<double> coverage;
    PixelMap segmented;
    int plainRegions = 0;
};

AaDisk makeAaDisk(sla::Vec2 center, double radius) {
    const auto scene = cnxtest::aaScene(40, 40, [&](double x, double y) {
        return (x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) <= radius * radius;
    }, 16);
    AaDisk d{Array2D<Rgba8>(40, 40), scene.coverage, PixelMap(), 0};
    for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 40; ++x) d.image(x, y) = sla::toRgba8(scene.target(x, y));
    }
    cnx::PixelSegmenter seg(d.image, SegmentParams{});
    cnx::SegmentationResult r = seg.run();
    d.plainRegions = r.stats.finalRegions;
    d.segmented = std::move(r.map);
    return d;
}

}  // namespace

CNX_TEST(an_antialiased_disk_becomes_exactly_two_regions_with_the_boundary_at_half_coverage) {
    const sla::Vec2 center{20.3, 19.8};
    const double radius = 11.4;
    AaDisk d = makeAaDisk(center, radius);
    const SubPixelSegmenter seg(d.image, SegmentParams{}, blackAndWhite());
    const cnx::SubPixelResult r = seg.run(std::move(d.segmented));
    std::printf("  disco AA: PixelSegmenter %d regiões -> SubPixelSegmenter %d (micro %d, praias %d [%d px, %d fundidas], sombras %d, paleta %d, pequenas %d)\n",
                d.plainRegions, r.stats.finalRegions, r.stats.microMerges, r.stats.beaches, r.stats.beachPixelsMoved,
                r.stats.beachesMerged, r.stats.shadowFlagged, r.stats.paletteMerges, r.stats.smallMerges);
    CHECK(r.stats.finalRegions == 2);
    CHECK(r.stats.splitRegions == 0);
    const auto& labels = r.map.compactLabels();
    // Cada região tem cor e índice da paleta, exatos.
    for (RegionId id = 0; id < r.stats.finalRegions; ++id) {
        const std::int32_t k = r.map.forest().stats(id).paletteIndex;
        CHECK(k == 0 || k == 1);
        CHECK(r.map.regionColor(id) == blackAndWhite().colors[static_cast<std::size_t>(k)]);
    }
    // Pixel escuro ⇔ cobertura ≥ ½ (±1 nível de quantização de 8 bits).
    int mismatches = 0;
    int dark = 0;
    for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 40; ++x) {
            const bool isDark = r.map.forest().stats(labels(x, y)).paletteIndex == 1;
            dark += isDark ? 1 : 0;
            if (isDark != (d.coverage(x, y) >= 0.5)) {
                ++mismatches;
                CHECK_CTX(std::abs(d.coverage(x, y) - 0.5) < 0.02, "pixel " + std::to_string(x) + "," + std::to_string(y));
            }
        }
    }
    std::printf("  disco AA: %d pixels escuros (verdade %.1f), %d divergências em casos de cobertura ≈ ½\n", dark,
                std::numbers::pi * radius * radius, mismatches);
    CHECK(mismatches <= 4);
    CHECK(std::abs(dark - std::numbers::pi * radius * radius) < 0.02 * std::numbers::pi * radius * radius);
    // O mapa alimenta o traçador, e o resultado passa em I1–I4 com uma face por região.
    PixelMap map = r.map;
    const cnx::TraceResult t = cnx::BoundaryTracer().trace(map);
    const auto report = t.image.validate();
    CHECK_CTX(report.ok, report.message);
    CHECK(t.image.faceCount() == 3);  // exterior + fundo + disco
}

CNX_TEST(the_pipeline_is_deterministic_cancellable_and_validates_its_inputs) {
    const sla::Vec2 center{20.3, 19.8};
    AaDisk a = makeAaDisk(center, 11.4);
    AaDisk b = makeAaDisk(center, 11.4);
    const SubPixelSegmenter segA(a.image, SegmentParams{}, blackAndWhite());
    const SubPixelSegmenter segB(b.image, SegmentParams{}, blackAndWhite());
    PixelMap copy = a.segmented;  // uma cópia intacta para o teste de cancelamento
    const cnx::SubPixelResult ra = segA.run(std::move(a.segmented));
    const cnx::SubPixelResult rb = segB.run(std::move(b.segmented));
    bool same = true;
    for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 40; ++x) same = same && ra.map.compactLabels()(x, y) == rb.map.compactLabels()(x, y);
    }
    CHECK(same);
    CHECK(ra.stats.beachPixelsMoved == rb.stats.beachPixelsMoved && ra.stats.finalRegions == rb.stats.finalRegions);

    std::stop_source source;
    source.request_stop();
    CHECK_THROWS(cnx::CancelTaskException,
                 (void)segA.run(copy, cnx::TaskContext(source.get_token(), nullptr, cnx::ProgressRange{})));

    cnx::Palette empty;
    CHECK_THROWS(std::invalid_argument, (void)SubPixelSegmenter(a.image, SegmentParams{}, empty));
    CHECK_THROWS(std::invalid_argument, (void)segA.run(PixelMap(10, 10)));  // dimensões diferentes
    PixelMap incomplete(40, 40);
    const RegionId only = incomplete.createRegion();
    incomplete.setLabel(0, 0, only, kBlack);
    CHECK_THROWS(std::invalid_argument, (void)segA.run(std::move(incomplete)));  // pixels sem região
}

int main() { return cnxtest::runAll(); }
