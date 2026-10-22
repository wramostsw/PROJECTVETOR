// tests/unit/svg_export_test.cpp â€” Exporter + SvgWriter. O SVG Ã© lido de volta por um leitor e um rasterizador
// prÃ³prios deste teste (independentes do exportador: caminhos M/L/C/Z, cÃºbicas discretizadas, regra nonzero
// pelo nÃºmero de voltas), e a geometria Ã© conferida: cada ponto numa forma sÃ³ (sem frestas nem sobreposiÃ§Ã£o),
// a face certa em cada pixel (compostos e bolo em camadas), Ã¡reas iguais Ã s de Green das curvas.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stop_token>
#include <string>
#include <vector>

#include "cnx/classify/Classifier.h"
#include "cnx/engine/CoreEngine.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/export/ExportOutputStage.h"
#include "cnx/export/Exporter.h"
#include "cnx/export/SvgWriter.h"
#include "cnx/fit/BezierFitter.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/vector/ContourSmoother.h"
#include "cnx/segment/PixelMap.h"
#include "cnx/vector/VectorImage.h"
#include "svg_support.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::CakeStackMode;
using cnx::Exporter;
using cnx::ExporterSettings;
using cnx::FaceId;
using sla::Vec2;
namespace fs = std::filesystem;

using cnxtest::svg::Path;
using cnxtest::svg::Svg;
using cnxtest::svg::exportSvg;
using cnxtest::svg::parseSvg;
using cnxtest::svg::pathArea;
using cnxtest::svg::winding;

namespace {


/// Cena de blocos fundidos ao acaso (junÃ§Ãµes, ilhas e furos), com todas as cores opacas.
cnxtest::VectorScene scene(std::uint64_t seed, int w = 30, int h = 22) {
    cnxtest::SplitMix64 rng{seed};
    cnxtest::VectorScene s = cnxtest::vectorScene(w, h, 3, 120, rng);
    for (FaceId f = 1; f < static_cast<FaceId>(s.image.faceCount()); ++f) s.image.face(f).color.a = 1.0f;
    return s;
}

/// Cena com furos aninhados (3 nÃ­veis), ilhas de mesma cor em faces diferentes e uma sela diagonal: cada
/// componente 4-conexa de um rÃ³tulo vira uma regiÃ£o. RÃ³tulos â†’ cores pela tabela.
cnxtest::VectorScene labeledScene(const sla::Array2D<int>& label);

cnxtest::VectorScene nestedScene() {
    const int w = 40;
    const int h = 30;
    sla::Array2D<int> label(w, h, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (x >= 3 && x < 30 && y >= 3 && y < 27) label(x, y) = 1;
            if (x >= 7 && x < 26 && y >= 7 && y < 23) label(x, y) = 2;
            if (x >= 11 && x < 22 && y >= 11 && y < 19) label(x, y) = 0;  // furo do furo: o fundo de novo (outra face)
            if (x >= 14 && x < 19 && y >= 13 && y < 17) label(x, y) = 3;
            if (x >= 32 && x < 36 && y >= 4 && y < 8) label(x, y) = 2;   // sela: dois quadrados pela quina
            if (x >= 36 && x < 39 && y >= 8 && y < 11) label(x, y) = 2;
        }
    }
    return labeledScene(label);
}

/// Discos e anÃ©is que se tocam (junÃ§Ãµes), com furos: depois da suavizaÃ§Ã£o, o ajuste dÃ¡ cÃºbicas de verdade.
cnxtest::VectorScene roundScene(int variant) {
    const int w = 44;
    const int h = 34;
    sla::Array2D<int> label(w, h, 0);
    const double cx = 15.3 + variant;
    const double cy = 16.7;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double r1 = std::hypot(x + 0.5 - cx, y + 0.5 - cy);
            const double r2 = std::hypot(x + 0.5 - 31.4, y + 0.5 - 15.2 - variant);
            if (r1 < 12.2) label(x, y) = 1;
            if (r1 < 7.6) label(x, y) = 2;
            if (r1 < 3.4) label(x, y) = 0;
            if (r2 < 9.1) label(x, y) = 3;
            if (r2 < 4.3) label(x, y) = 1;
        }
    }
    return labeledScene(label);
}

cnxtest::VectorScene labeledScene(const sla::Array2D<int>& label) {
    const int w = label.width();
    const int h = label.height();
    const std::array<sla::Rgba8, 4> colors{sla::Rgba8{255, 255, 255, 255}, sla::Rgba8{220, 30, 30, 255}, sla::Rgba8{30, 30, 220, 255}, sla::Rgba8{250, 200, 20, 255}};
    cnxtest::VectorScene s{cnx::VectorImage(), sla::Array2D<sla::Rgba8>(w, h), {}};
    cnx::PixelMap map(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) s.source(x, y) = colors[static_cast<std::size_t>(label(x, y))];
    }
    std::vector<std::array<int, 2>> stack;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (map.regionAtConst(x, y) != cnx::kNoRegion) continue;
            const cnx::RegionId id = map.createRegion();
            map.setLabel(x, y, id, s.source(x, y));
            stack.assign(1, {x, y});
            while (!stack.empty()) {
                const auto [cx, cy] = stack.back();
                stack.pop_back();
                for (std::size_t d = 0; d < 4; ++d) {
                    const int nx = cx + cnxtest::kDx4[d];
                    const int ny = cy + cnxtest::kDy4[d];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h || map.regionAtConst(nx, ny) != cnx::kNoRegion || label(nx, ny) != label(x, y)) continue;
                    map.setLabel(nx, ny, id, s.source(nx, ny));
                    stack.push_back({nx, ny});
                }
            }
        }
    }
    map.assignMedianColors();
    (void)map.compact();
    cnx::TraceResult t = cnx::BoundaryTracer().trace(map);
    s.image = std::move(t.image);
    s.pixelFaces = std::move(t.pixelFaces);
    return s;
}

/// Pontos de amostra de um pixel: o centro e 4 pontos com deslocamentos "irracionais" (longe das fronteiras
/// em coordenadas inteiras ou meias da escadinha).
std::array<Vec2, 5> samples(int x, int y) {
    return {Vec2{x + 0.5, y + 0.5}, Vec2{x + 0.137, y + 0.271}, Vec2{x + 0.829, y + 0.313}, Vec2{x + 0.241, y + 0.877}, Vec2{x + 0.763, y + 0.659}};
}

}  // namespace

// ================================================================ o escritor

CNX_TEST(the_svg_writer_emits_exact_fixed_point_text) {
    ExporterSettings s;
    s.pointsPerPixel = 0.75;
    std::ostringstream out;
    cnx::SvgWriter w(out, s);
    cnx::DocumentInfo info;
    info.widthPx = 40;
    info.heightPx = 30;
    info.title = "a<b & \"c\"";
    w.beginFile(info);
    w.beginGroup(sla::Rgba{1.0f, 0.5f, 0.0f, 1.0f});
    const cnx::StrokeStyle stroke{sla::Rgba{1.0f, 0.5f, 0.0f, 1.0f}, 0.09375};
    w.beginShape(sla::Rgba{1.0f, 0.5f, 0.0f, 0.5f}, &stroke);
    w.moveTo({1.0, 2.0});
    w.lineTo({-0.001, 3.14159});
    w.curveTo({1.005, 2.5}, {10.0, -1.25}, {1.0, 2.0});
    w.closeSubpath();
    w.moveTo({5, 5});
    w.lineTo({6, 5});
    w.lineTo({5, 6});
    w.closeSubpath();
    w.endShape();
    w.endGroup();
    w.beginShape(sla::Rgba{0.0f, 0.0f, 1.0f, 1.0f}, nullptr);
    w.moveTo({0, 0});
    w.lineTo({1, 0});
    w.lineTo({0, 1});
    w.closeSubpath();
    w.endShape();
    w.endFile();
    const std::string expected =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
        "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"30.00pt\" height=\"22.50pt\" viewBox=\"0 0 40.00 30.00\">\n"
        "  <title>a&lt;b &amp; &quot;c&quot;</title>\n"
        "  <desc>cnx</desc>\n"
        "  <g id=\"c_ff8000ff\">\n"
        "    <path fill=\"#ff8000\" opacity=\"0.50\" stroke=\"#ff8000\" stroke-width=\"0.125\" "
        "d=\"M 1.00 2.00 L 0.00 3.14 C 1.00 2.50 10.00 -1.25 1.00 2.00 Z M 5.00 5.00 L 6.00 5.00 L 5.00 6.00 Z\"/>\n"
        "  </g>\n"
        "  <path fill=\"#0000ff\" opacity=\"1.00\" d=\"M 0.00 0.00 L 1.00 0.00 L 0.00 1.00 Z\"/>\n"
        "</svg>\n";
    std::printf("%s", out.str() == expected ? "" : ("  obtido:\n" + out.str()).c_str());
    CHECK(out.str() == expected);
    // NÃºmeros: casas fixas, sem "-0", arredondamento do double exato, 0 casas.
    CHECK(cnx::SvgWriter::number(-0.004, 2) == "0.00" && cnx::SvgWriter::number(-0.006, 2) == "-0.01");
    CHECK(cnx::SvgWriter::number(2.4, 0) == "2" && cnx::SvgWriter::number(2.6, 0) == "3" && cnx::SvgWriter::number(-1.5, 3) == "-1.500");
    CHECK(cnx::SvgWriter::number(123456789.126, 2) == "123456789.13");
    CHECK_THROWS(cnx::ExportError, (void)cnx::SvgWriter::number(std::nan(""), 2));
    CHECK(cnx::SvgWriter::hexColor(sla::Rgba{0.0f, 1.0f, 0.5f, 1.0f}) == "#00ff80" && cnx::SvgWriter::hexColor(sla::Rgba{2.0f, -1.0f, 0.1f, 1.0f}) == "#ff001a");
}

// ================================================================ geometria

CNX_TEST(compound_paths_cover_every_point_exactly_once_and_with_the_right_face) {
    int holesInNested = 0;
    for (std::uint64_t seed = 7000; seed < 7007; ++seed) {
        cnxtest::VectorScene sc = seed == 7000 ? nestedScene() : scene(seed);  // 7000: a cena com furos aninhados
        ExporterSettings s;
        s.cakeStackMode = CakeStackMode::Compound;
        const Svg svg = parseSvg(exportSvg(sc.image, s));
        const std::vector<FaceId> order = Exporter(sc.image, s).drawOrder();
        CHECK(svg.wellFormed && svg.paths.size() == order.size());
        int holes = 0;
        for (const Path& p : svg.paths) {
            holes += static_cast<int>(p.rings.size()) - 1;
            CHECK(p.cubics == 0 && p.lines > 0);  // sem ajuste: a poligonal dos nós, só com "L"
        }
        int wrong = 0;
        for (int y = 0; y < sc.source.height(); ++y) {
            for (int x = 0; x < sc.source.width(); ++x) {
                for (const Vec2 q : samples(x, y)) {
                    int count = 0;
                    std::size_t which = 0;
                    for (std::size_t i = 0; i < svg.paths.size(); ++i) {
                        if (winding(svg.paths[i], q) != 0) {
                            ++count;
                            which = i;
                        }
                    }
                    wrong += count == 1 && order[which] == sc.pixelFaces(x, y) ? 0 : 1;
                }
            }
        }
        CHECK_CTX(wrong == 0, "semente " + std::to_string(seed) + ": " + std::to_string(wrong) + " amostras erradas");
        if (seed == 7000) holesInNested = holes;
    }
    // Furos: fundo (o quadro vermelho e os quadrados da sela), vermelho (o azul), azul (o miolo branco), miolo
    // branco (o amarelo).
    std::printf("  cena aninhada: %d furos\n", holesInNested);
    CHECK(holesInNested >= 4);
}

CNX_TEST(loops_starting_on_a_reversed_open_edge_are_written_from_the_right_point) {
    // O tracer só começa laços em arestas diretas ou fechadas. Procuro uma cena em que um laço (externo ou
    // furo) comece numa meia-aresta INVERSA de aresta aberta: o primeiro moveTo tem de ser o último nó dela.
    const auto startsReversedOpen = [](const cnx::VectorImage& v) {
        const auto bad = [&](cnx::HalfEdgeId h) {
            const auto nodes = v.edgeNodes(cnx::edgeOf(h));
            return cnx::isReversed(h) && nodes.front() != nodes.back();
        };
        for (FaceId f = 1; f < static_cast<FaceId>(v.faceCount()); ++f) {
            if (v.face(f).outerLoop != cnx::kInvalidId && bad(v.face(f).outerLoop)) return true;
            for (const cnx::HalfEdgeId h : v.holeLoops(f)) {
                if (bad(h)) return true;
            }
        }
        return false;
    };
    int found = 0;
    for (std::uint64_t seed = 8000; seed < 8300 && found < 2; ++seed) {
        cnxtest::VectorScene sc = scene(seed, 24, 18);
        if (!startsReversedOpen(sc.image)) continue;
        ++found;
        ExporterSettings s;
        s.cakeStackMode = CakeStackMode::Compound;
        const Svg svg = parseSvg(exportSvg(sc.image, s));
        const std::vector<FaceId> order = Exporter(sc.image, s).drawOrder();
        int wrong = 0;
        for (int y = 0; y < sc.source.height(); ++y) {
            for (int x = 0; x < sc.source.width(); ++x) {
                for (const Vec2 q : samples(x, y)) {
                    int count = 0;
                    std::size_t which = 0;
                    for (std::size_t i = 0; i < svg.paths.size(); ++i) {
                        if (winding(svg.paths[i], q) != 0) {
                            ++count;
                            which = i;
                        }
                    }
                    wrong += count == 1 && order[which] == sc.pixelFaces(x, y) ? 0 : 1;
                }
            }
        }
        CHECK_CTX(wrong == 0, "semente " + std::to_string(seed));
    }
    CHECK(found >= 1);
}

CNX_TEST(parents_are_drawn_before_children_whatever_the_face_ids) {
    // Ilha (id 1) dentro de um quadrado (id 2): o filho tem id MENOR que o pai. A ordem tem de ser pai, filho.
    cnx::VectorImage v(10.0, 10.0);
    const FaceId island = v.addFace(sla::Rgba{1.0f, 0.0f, 0.0f, 1.0f}, 0);
    const FaceId square = v.addFace(sla::Rgba{0.0f, 0.0f, 1.0f, 1.0f}, 1);
    const cnx::NodeId f0 = v.addNode({0, 0}, cnx::NodeFlag::LoopAnchor);
    const std::array<cnx::NodeId, 5> frame{f0, v.addNode({10, 0}), v.addNode({10, 10}), v.addNode({0, 10}), f0};
    (void)v.addEdge(frame, square, cnx::kExteriorFace);
    const cnx::NodeId i0 = v.addNode({3, 3}, cnx::NodeFlag::LoopAnchor);
    const std::array<cnx::NodeId, 5> inner{i0, v.addNode({7, 3}), v.addNode({7, 7}), v.addNode({3, 7}), i0};
    (void)v.addEdge(inner, island, square);
    v.linkFaceLoops();
    CHECK(v.face(island).parent == square);
    const std::vector<FaceId> order = Exporter(v, ExporterSettings{}).drawOrder();
    CHECK((order == std::vector<FaceId>{square, island}));
    // No SVG em modo bolo, o quadrado azul é pintado antes e a ilha vermelha fica por cima.
    ExporterSettings stacked;
    stacked.cakeStackMode = CakeStackMode::Stacked;
    const Svg svg = parseSvg(exportSvg(v, stacked));
    CHECK(svg.paths.size() == 2 && svg.paths[0].fill == "#0000ff" && svg.paths[1].fill == "#ff0000");
}

CNX_TEST(stacked_layers_paint_the_right_face_and_grouping_keeps_it) {
    for (std::uint64_t seed = 7100; seed < 7105; ++seed) {
        cnxtest::VectorScene sc = seed == 7100 ? nestedScene() : scene(seed);
        for (const CakeStackMode mode : {CakeStackMode::Stacked, CakeStackMode::CompoundByColor}) {
            ExporterSettings s;
            s.cakeStackMode = mode;
            const Svg svg = parseSvg(exportSvg(sc.image, s));
            // No modo bolo sÃ³ hÃ¡ laÃ§os externos; a cor de cada pixel Ã© a do Ãºltimo caminho que o cobre.
            if (mode == CakeStackMode::Stacked) {
                for (const Path& p : svg.paths) CHECK(p.rings.size() == 1);
            }
            int wrong = 0;
            for (int y = 0; y < sc.source.height(); ++y) {
                for (int x = 0; x < sc.source.width(); ++x) {
                    for (const Vec2 q : samples(x, y)) {
                        std::string painted;
                        for (const Path& p : svg.paths) {
                            if (winding(p, q) != 0) painted = p.fill;
                        }
                        wrong += painted == cnx::SvgWriter::hexColor(sc.image.face(sc.pixelFaces(x, y)).color) ? 0 : 1;
                    }
                }
            }
            CHECK_CTX(wrong == 0, std::string(mode == CakeStackMode::Stacked ? "bolo" : "agrupado") + ", semente " + std::to_string(seed));
            if (mode == CakeStackMode::CompoundByColor) {
                // Um grupo por cor, ids Ãºnicos e vÃ¡lidos, e cada caminho dentro do grupo da sua cor.
                std::set<std::string> ids(svg.groups.begin(), svg.groups.end());
                std::set<std::string> colors;
                for (const Path& p : svg.paths) colors.insert(p.fill);
                CHECK(ids.size() == svg.groups.size() && svg.groups.size() == colors.size());
                for (const Path& p : svg.paths) CHECK(p.group.rfind("c_", 0) == 0 && p.group.substr(2, 6) == p.fill.substr(1));
            }
        }
    }
}

CNX_TEST(fitted_curves_share_boundaries_exactly_and_keep_the_green_areas) {
    for (std::uint64_t seed = 7201; seed < 7205; ++seed) {
        // Como no motor: suavizaÃ§Ã£o (a escadinha crua daria sÃ³ retas) e depois o ajuste de BÃ©zier.
        cnxtest::VectorScene sc = seed == 7204 ? nestedScene() : roundScene(static_cast<int>(seed - 7201));
        cnx::SmoothParams sp;
        sp.maxCgIterations = 60;
        cnx::ContourSmoother(sp).run(sc.image);
        cnx::BezierFitter(cnx::FitParams{}).run(sc.image);
        sc.image.updateFaceAreas();
        CHECK(sc.image.hasCurves());
        for (const int decimals : {2, 6}) {
            ExporterSettings s;
            s.coordinateDecimals = decimals;
            const Svg svg = parseSvg(exportSvg(sc.image, s));
            const std::vector<FaceId> order = Exporter(sc.image, s).drawOrder();
            std::size_t cubics = 0;
            for (const Path& p : svg.paths) cubics += p.cubics;
            // Mesmo arredondado a 2 casas: a fronteira compartilhada tem os mesmos nÃºmeros nas duas faces.
            cnxtest::SplitMix64 rng{seed};
            int gaps = 0;
            for (int i = 0; i < 4000; ++i) {
                const Vec2 q{cnxtest::uniform01(rng) * sc.image.width(), cnxtest::uniform01(rng) * sc.image.height()};
                int count = 0;
                for (const Path& p : svg.paths) count += winding(p, q) != 0 ? 1 : 0;
                gaps += count == 1 ? 0 : 1;
            }
            // A cena dos retângulos (7204) só tem retas, e as retas ajustadas saem como "L" (não como cúbicas).
            CHECK_CTX(gaps == 0 && (cubics > 0 || seed == 7204), "semente " + std::to_string(seed) + ", " + std::to_string(decimals) + " casas: " + std::to_string(gaps));
            if (seed == 7204) {
                // A cena dos retângulos: o ajuste devolve retas (SegmentKind::Line), escritas como "L", e curvas.
                std::size_t lines = 0;
                for (const Path& p : svg.paths) lines += p.lines;
                CHECK(lines > 10);
            }
            if (decimals == 6) {
                // Ãrea de cada forma (discretizada) = Ã¡rea de Green das cÃºbicas da face (VectorImage::loopArea).
                double worst = 0.0;
                for (std::size_t i = 0; i < svg.paths.size(); ++i) {
                    const double expected = sc.image.face(order[i]).area;
                    worst = std::max(worst, std::abs(pathArea(svg.paths[i]) - expected) / std::max(1.0, std::abs(expected)));
                }
                std::printf("  semente %llu: %zu cÃºbicas, maior erro relativo de Ã¡rea %.2e\n", static_cast<unsigned long long>(seed), cubics, worst);
                CHECK(worst < 1e-4);
            }
        }
    }
}

// ================================================================ opÃ§Ãµes

CNX_TEST(stroke_opacity_size_and_background_removal) {
    cnxtest::VectorScene sc = scene(7301);
    ExporterSettings s;
    s.pointsPerPixel = 0.75;
    {
        const Svg svg = parseSvg(exportSvg(sc.image, s));
        CHECK(svg.width == "22.50pt" && svg.height == "16.50pt" && svg.viewBox == "0 0 30.00 22.00");
        for (const Path& p : svg.paths) CHECK(p.stroke == p.fill && p.strokeWidth == "0.125" && p.opacity == "1.00");
    }
    s.shapeStroking = false;
    for (const Path& p : parseSvg(exportSvg(sc.image, s)).paths) CHECK(p.stroke.empty() && p.strokeWidth.empty());
    // Opacidade com 2 casas; alfa 0 nÃ£o Ã© escrito.
    sc.image.face(1).color.a = 0.5f;
    sc.image.face(2).color.a = 0.0f;
    const Svg semi = parseSvg(exportSvg(sc.image, s));
    {
        const std::vector<FaceId> o = Exporter(sc.image, s).drawOrder();
        const auto at = static_cast<std::size_t>(std::find(o.begin(), o.end(), 1) - o.begin());
        CHECK(semi.paths.size() == sc.image.faceCount() - 2 && semi.paths[at].opacity == "0.50");
    }
    sc.image.face(1).color.a = 1.0f;
    sc.image.face(2).color.a = 1.0f;
    // Fundo removido: a face some. Nos compostos fica um buraco; no bolo, aparece o pai (ou nada).
    const FaceId hidden = sc.pixelFaces(0, 0);
    s.background.setHidden(hidden, true);
    for (const CakeStackMode mode : {CakeStackMode::Compound, CakeStackMode::Stacked}) {
        s.cakeStackMode = mode;
        const Svg svg = parseSvg(exportSvg(sc.image, s));
        CHECK(svg.paths.size() == sc.image.faceCount() - 2);
        const FaceId parent = sc.image.face(hidden).parent;
        int wrong = 0;
        for (int y = 0; y < sc.source.height(); ++y) {
            for (int x = 0; x < sc.source.width(); ++x) {
                if (sc.pixelFaces(x, y) != hidden) continue;
                std::string painted;
                for (const Path& p : svg.paths) {
                    if (winding(p, {x + 0.5, y + 0.5}) != 0) painted = p.fill;
                }
                const std::string expected =
                    mode == CakeStackMode::Stacked && parent != cnx::kExteriorFace && parent != cnx::kInvalidId ? cnx::SvgWriter::hexColor(sc.image.face(parent).color) : "";
                wrong += painted == expected ? 0 : 1;
            }
        }
        CHECK(wrong == 0);
    }
    // Ordem de desenho: pai antes dos filhos.
    s.background.clear();
    const std::vector<FaceId> order = Exporter(sc.image, s).drawOrder();
    for (std::size_t i = 0; i < order.size(); ++i) {
        const FaceId parent = sc.image.face(order[i]).parent;
        if (parent == cnx::kExteriorFace || parent == cnx::kInvalidId) continue;
        CHECK(std::find(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(i), parent) != order.begin() + static_cast<std::ptrdiff_t>(i));
    }
    // ValidaÃ§Ã£o das configuraÃ§Ãµes.
    for (const auto& edit : std::vector<void (*)(ExporterSettings&)>{[](ExporterSettings& e) { e.coordinateDecimals = 10; }, [](ExporterSettings& e) { e.coordinateDecimals = -1; },
                                                                       [](ExporterSettings& e) { e.strokeWidthPt = -0.1; }, [](ExporterSettings& e) { e.pointsPerPixel = 0.0; },
                                                                       [](ExporterSettings& e) { e.pointsPerPixel = std::nan(""); }}) {
        ExporterSettings bad;
        edit(bad);
        CHECK_THROWS(cnx::ExportError, (void)Exporter(sc.image, bad));
    }
}

// ================================================================ arquivos e motor

CNX_TEST(files_are_written_atomically_and_honor_cancellation) {
    cnxtest::VectorScene sc = scene(7401);
    const fs::path dir = fs::temp_directory_path() / "cnx_svg_export_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path file = dir / "logo.svg";
    const Exporter ex(sc.image, ExporterSettings{});
    ex.writeFile(cnx::ExportFormat::Svg, file);
    std::ifstream in(file, std::ios::binary);
    const std::string first((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const Svg svg = parseSvg(first);
    CHECK(svg.wellFormed && svg.title == "logo" && svg.paths.size() == ex.drawOrder().size());
    // Formato nÃ£o implementado: erro, o destino fica intacto e nenhum temporÃ¡rio sobra.
    CHECK_THROWS(cnx::ExportError, (ex.writeFile(cnx::ExportFormat::Pdf, file)));
    std::ifstream again(file, std::ios::binary);
    CHECK(std::string((std::istreambuf_iterator<char>(again)), std::istreambuf_iterator<char>()) == first);
    again.close();
    // Cancelado: nada Ã© criado.
    std::stop_source stop;
    stop.request_stop();
    CHECK_THROWS(cnx::CancelTaskException, (ex.writeFile(cnx::ExportFormat::Svg, dir / "cancelado.svg", stop.get_token())));
    CHECK(!fs::exists(dir / "cancelado.svg"));
    // DiretÃ³rio inexistente.
    CHECK_THROWS(cnx::ExportError, (ex.writeFile(cnx::ExportFormat::Svg, dir / "nao" / "existe.svg")));
    int leftovers = 0;
    for (const auto& e : fs::directory_iterator(dir)) leftovers += e.path().extension() == ".cnxtmp" ? 1 : 0;
    CHECK(leftovers == 0);
    // SubstituiÃ§Ã£o de um arquivo existente com outro conteÃºdo.
    ExporterSettings grouped;
    grouped.cakeStackMode = CakeStackMode::CompoundByColor;
    Exporter(sc.image, grouped).writeFile(cnx::ExportFormat::Svg, file);
    std::ifstream third(file, std::ios::binary);
    CHECK(!parseSvg(std::string((std::istreambuf_iterator<char>(third)), std::istreambuf_iterator<char>())).groups.empty());
    third.close();
    // Depois de todas as gravações com sucesso, só o destino existe (o temporário foi renomeado).
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(dir)) names.push_back(e.path().filename().string());
    CHECK((names == std::vector<std::string>{"logo.svg"}));
    fs::remove_all(dir);
}

namespace {

class LogoLoader final : public cnx::IImageLoader {
public:
    cnx::SourceImage load(const fs::path&, const std::stop_token&) override {
        cnx::SourceImage s;
        s.pixels = sla::Array2D<sla::Rgba8>(48, 36, sla::Rgba8{255, 255, 255, 255});
        for (int y = 0; y < 36; ++y) {
            for (int x = 0; x < 48; ++x) {
                if (x >= 5 && x < 24 && y >= 4 && y < 18) s.pixels(x, y) = {220, 30, 30, 255};
                if ((x - 33.5) * (x - 33.5) + (y - 24.5) * (y - 24.5) < 64.0) s.pixels(x, y) = {30, 30, 220, 255};
                if ((x - 33.5) * (x - 33.5) + (y - 24.5) * (y - 24.5) < 9.0) s.pixels(x, y) = {255, 255, 255, 255};  // furo
            }
        }
        s.formatHint = ".png";
        return s;
    }
};

}  // namespace

CNX_TEST(the_core_engine_writes_an_svg_that_redraws_the_image) {
    const fs::path file = fs::temp_directory_path() / "cnx_engine_logo.svg";
    fs::remove(file);
    cnx::CoreEngine engine(std::make_shared<LogoLoader>());
    engine.setInput("logo.png");
    engine.setOutputStage(std::make_shared<cnx::ExportOutputStage>());
    cnx::ExportRequest req;
    req.file = file;
    req.settings.cakeStackMode = CakeStackMode::CompoundByColor;  // as configurações do pedido valem
    engine.setExportRequests({req});
    engine.start(cnx::Stage::Output);
    CHECK(engine.wait() == cnx::RunStatus::Completed);
    std::ifstream in(file, std::ios::binary);
    const Svg svg = parseSvg(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
    in.close();
    const auto snap = engine.snapshot();
    const sla::Array2D<sla::Rgba8>& img = snap->source->pixels;
    CHECK(svg.wellFormed && svg.paths.size() == 4 && svg.width == "48.00pt");
    CHECK(svg.groups.size() == 3);  // branco (fundo e miolo), vermelho, azul
    // Sem o resultado do ajuste, o estágio recusa.
    cnx::TaskContext ctx;
    CHECK_THROWS(cnx::ExportError, cnx::ExportOutputStage().write(cnx::PipelineSnapshot{}, {req}, ctx));
    std::size_t cubics = 0;
    std::size_t rings = 0;
    for (const Path& p : svg.paths) {
        cubics += p.cubics;
        rings += p.rings.size();
    }
    CHECK(cubics > 0 && rings == 7);  // fundo (2 furos), retÃ¢ngulo, anel azul (1 furo) e o miolo branco
    // O SVG redesenha a imagem: fora da borda (pixels cujos 8 vizinhos tÃªm a mesma cor), a cor da forma no
    // centro do pixel Ã© exatamente a do pixel. Na borda, o cÃ­rculo vetorial e a escadinha de pixels diferem por
    // natureza (meio pixel), e conta sÃ³ a proporÃ§Ã£o total.
    int same = 0;
    int interior = 0;
    int interiorWrong = 0;
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            std::string painted;
            for (const Path& p : svg.paths) {
                if (winding(p, {x + 0.5, y + 0.5}) != 0) painted = p.fill;
            }
            const bool ok = painted == cnx::SvgWriter::hexColor(sla::toRgba(img(x, y)));
            same += ok ? 1 : 0;
            bool edge = false;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (img.inBounds(x + dx, y + dy) && !(img(x + dx, y + dy) == img(x, y))) edge = true;
                }
            }
            if (!edge) {
                ++interior;
                interiorWrong += ok ? 0 : 1;
            }
        }
    }
    const double share = same / static_cast<double>(img.width() * img.height());
    std::printf("  motor â†’ SVG: %zu caminhos, %zu cÃºbicas; %d pixels interiores errados de %d; %.2f %% de todos certos\n", svg.paths.size(), cubics,
                interiorWrong, interior, 100.0 * share);
    CHECK(interiorWrong == 0 && interior > 1400 && share > 0.95);
    fs::remove(file);
}

int main() { return cnxtest::runAll(); }
