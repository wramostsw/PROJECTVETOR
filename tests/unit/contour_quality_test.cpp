// tests/unit/contour_quality_test.cpp — qualidade da vetorização ponta a ponta (docs §1.25): o SVG de cada estágio
// (escadinha, suavizado, Bézier) é rasterizado e comparado com a imagem de origem de logos sintéticos de verdade
// conhecida. Mede o que os testes de energia não veem: letras, cantos e fios deformados pela suavização.
//
// Números de ANTES da correção, medidos com este mesmo código (paleta com o número certo de tintas):
//   letras  suavizado rmse 0,0201, Bézier 0,0224, cantos das letras a 1,2–1,9 px;
//   formas  suavizado rmse 0,1064, Bézier 0,1085, o fio de 34 × 3 px achatado num quadrado de ~10 px.
// CNX_QUALITY_DIR=<pasta> grava as cenas em PNG e os SVG de cada estágio para olhar; CNX_WORST=1 lista os piores pixels.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>

#include "cnx/classify/Classifier.h"
#include "cnx/engine/CoreEngine.h"
#include "cnx/export/Exporter.h"
#include "cnx/fit/BezierFitter.h"
#include "cnx/preprocess/SignalImage.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/vector/ContourSmoother.h"
#include "logo_scenes.h"
#include "svg_support.h"
#include "test_harness.h"

namespace fs = std::filesystem;
using cnxtest::svg::FitError;
using sla::Vec2;

namespace {

class SceneLoader final : public cnx::IImageLoader {
public:
    explicit SceneLoader(sla::Array2D<sla::Rgba8> img) : image_(std::move(img)) {}
    cnx::SourceImage load(const fs::path&, const std::stop_token&) override {
        cnx::SourceImage s;
        s.pixels = image_;
        s.formatHint = ".png";
        return s;
    }

private:
    sla::Array2D<sla::Rgba8> image_;
};

std::string env(const char* name) {
    char* value = nullptr;
    std::size_t size = 0;
    std::string out;
    if (_dupenv_s(&value, &size, name) == 0 && value != nullptr) {
        out = value;
        std::free(value);
    }
    return out;
}

struct Stages {
    FitError staircase;
    FitError smoothed;
    FitError fitted;
    std::string svgStaircase;
    std::string svgSmoothed;
    std::string svgFitted;
    std::shared_ptr<const cnx::PipelineSnapshot> snapshot;
    double seconds = 0.0;
};

/// Roda o motor completo sobre a cena (com o nº certo de tintas, para isolar a suavização do critério de K da
/// paleta, que é outro assunto) e mede o SVG de cada estágio contra a imagem de origem.
Stages run(const cnxtest::LogoScene& scene, const std::function<void(cnx::EngineParameters&)>& edit = {}) {
    const sla::Array2D<sla::Rgba8> img = cnxtest::renderScene(scene);
    cnx::CoreEngine engine(std::make_shared<SceneLoader>(img));
    engine.setInput("cena.png");
    cnx::EngineParameters params = engine.parameters();
    params.palette.forcedNumColors = scene.inks;
    if (edit) edit(params);
    engine.setParameters(params);
    const auto t0 = std::chrono::steady_clock::now();
    engine.start(cnx::Stage::Output);
    if (engine.wait() != cnx::RunStatus::Completed) throw std::runtime_error("o motor falhou: " + engine.lastMessage());
    Stages out;
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    out.snapshot = engine.snapshot();
    const sla::Array2D<sla::Rgba> target = cnx::SignalImage::fromStraight(img).premultiplied();
    const cnx::ExporterSettings settings;
    cnx::TraceResult traced = cnx::BoundaryTracer().trace(*out.snapshot->segmentation);
    out.svgStaircase = cnxtest::svg::exportSvg(traced.image, settings);
    out.svgSmoothed = cnxtest::svg::exportSvg(*out.snapshot->smoothed, settings);
    out.svgFitted = cnxtest::svg::exportSvg(*out.snapshot->fitted, settings);
    const bool worst = !env("CNX_WORST").empty();
    const auto measure = [&](const std::string& svg) {
        const sla::Array2D<sla::Rgba> got = cnxtest::svg::rasterize(cnxtest::svg::parseSvg(svg), scene.width, scene.height, 4);
        if (worst) {
            std::vector<std::tuple<double, int, int>> list;
            for (int y = 0; y < scene.height; ++y) {
                for (int x = 0; x < scene.width; ++x) {
                    const sla::Rgba a = got(x, y);
                    const sla::Rgba b = target(x, y);
                    list.emplace_back(std::max({std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b), std::abs(a.a - b.a)}), x, y);
                }
            }
            std::sort(list.rbegin(), list.rend());
            std::printf("      piores:");
            for (std::size_t i = 0; i < 6; ++i) std::printf(" (%d,%d)=%.2f", std::get<1>(list[i]), std::get<2>(list[i]), std::get<0>(list[i]));
            std::printf("\n");
        }
        return cnxtest::svg::compare(got, target);
    };
    out.staircase = measure(out.svgStaircase);
    out.smoothed = measure(out.svgSmoothed);
    out.fitted = measure(out.svgFitted);
    return out;
}

void print(const char* name, const Stages& s) {
    const auto row = [](const char* stage, const FitError& e) {
        std::printf("    %-10s rmse %.4f  maior erro %.3f  pixels ruins %4d de %d\n", stage, e.rmse, e.maxError, e.bad, e.total);
    };
    std::printf("  %s (%.1f s): %zu regiões\n", name, s.seconds, s.snapshot->segmentation->forest().rootCount());
    row("escadinha", s.staircase);
    row("suavizado", s.smoothed);
    row("Bezier", s.fitted);
}

void save(const std::string& dir, const cnxtest::LogoScene& scene, const Stages& s, const std::string& tag) {
    if (dir.empty()) return;
    fs::create_directories(dir);
    const auto w = [&](const std::string& suffix, const std::string& text) { std::ofstream(fs::path(dir) / (scene.name + tag + suffix), std::ios::binary) << text; };
    w("_escadinha.svg", s.svgStaircase);
    w("_suavizado.svg", s.svgSmoothed);
    w("_bezier.svg", s.svgFitted);
    std::ofstream(fs::path(dir) / (scene.name + ".png"), std::ios::binary) << cnxtest::encodePng(cnxtest::renderScene(scene));
}

/// Distância de cada canto verdadeiro ao ponto mais próximo do contorno do SVG (as cúbicas discretizadas).
std::vector<double> cornerErrors(const std::string& svgText, const std::vector<Vec2>& corners) {
    const cnxtest::svg::Svg svg = cnxtest::svg::parseSvg(svgText);
    std::vector<double> out;
    for (const Vec2 c : corners) {
        double best = 1e9;
        for (const cnxtest::svg::Path& p : svg.paths) {
            for (const auto& ring : p.rings) {
                for (const Vec2 q : ring) best = std::min(best, sla::norm(q - c));
            }
        }
        out.push_back(best);
    }
    return out;
}

/// Os 16 cantos vivos das letras: as pontas das 2 diagonais do "x", os 4 do corte do "c" e os 4 pés do "n".
std::vector<Vec2> letterCorners() {
    std::vector<Vec2> out;
    for (const auto& [a, b] : {std::pair<Vec2, Vec2>{{72.4, 22.1}, {90.6, 44.3}}, {{90.6, 22.1}, {72.4, 44.3}}}) {
        const Vec2 d = b - a;
        const Vec2 n = Vec2{d.y, -d.x} / sla::norm(d) * 2.9;
        for (const Vec2 e : {a, b}) {
            out.push_back(e + n);
            out.push_back(e - n);
        }
    }
    const double a40 = std::atan(0.8391);
    for (const double r : {7.4, 13.2}) {
        out.push_back({22.3 + r * std::cos(a40), 31.8 + r * std::sin(a40)});
        out.push_back({22.3 + r * std::cos(a40), 31.8 - r * std::sin(a40)});
    }
    for (const double x : {42.3, 48.1, 56.9, 62.7}) out.push_back({x, 44.2});
    return out;
}

double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (const double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double maxOf(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

/// Caixa da face de uma cor (laço externo, nós).
sla::Box2 faceBox(const cnx::VectorImage& v, const sla::Rgba& color) {
    sla::Box2 b;
    for (cnx::FaceId f = 1; f < static_cast<cnx::FaceId>(v.faceCount()); ++f) {
        if (std::abs(v.face(f).color.r - color.r) > 0.02f || std::abs(v.face(f).color.g - color.g) > 0.02f || std::abs(v.face(f).color.b - color.b) > 0.02f) continue;
        cnx::HalfEdgeId h = v.face(f).outerLoop;
        const cnx::HalfEdgeId first = h;
        do {
            for (const cnx::NodeId n : v.edgeNodes(cnx::edgeOf(h))) b.expand(v.position(n));
            h = v.next(h);
        } while (h != first);
    }
    return b;
}

}  // namespace

CNX_TEST(letters_keep_their_corners_and_the_vector_matches_the_image) {
    const cnxtest::LogoScene scene = cnxtest::lettersScene();
    const Stages s = run(scene);
    print("letras", s);
    save(env("CNX_QUALITY_DIR"), scene, s, "");
    // Antes: smoothed 0,0201 e Bézier 0,0224. Agora os cantos ficam vivos e o ajuste vê a reta inteira.
    CHECK(s.smoothed.rmse < s.staircase.rmse * 0.6);
    CHECK(s.smoothed.rmse < 0.016 && s.fitted.rmse < 0.022);
    CHECK(s.smoothed.maxError < 0.55 && s.fitted.maxError < 0.55);
    CHECK(s.fitted.rmse < 1.6 * s.smoothed.rmse);
    // Cantos: distância do vértice verdadeiro ao contorno. Rodar com a perfuração desligada (giro 0) dá o "antes".
    const std::vector<Vec2> corners = letterCorners();
    const std::vector<double> fitted = cornerErrors(s.svgFitted, corners);
    const std::vector<double> smoothed = cornerErrors(s.svgSmoothed, corners);
    const Stages off = run(scene, [](cnx::EngineParameters& p) { p.smooth.punctureCornerThresh = 0.0; });
    const std::vector<double> rounded = cornerErrors(off.svgFitted, corners);
    std::printf("  cantos (16): distância média ao vértice verdadeiro: suavizado %.2f px, Bézier %.2f px (máx %.2f); sem perfuração %.2f px (máx %.2f)\n", mean(smoothed),
                mean(fitted), maxOf(fitted), mean(rounded), maxOf(rounded));
    CHECK(mean(fitted) < 0.75 && maxOf(fitted) < 1.5);
    CHECK(mean(rounded) > 1.4 * mean(fitted));  // a perfuração é o que mantém os cantos
}

CNX_TEST(thin_lines_triangles_and_disks_are_not_deformed) {
    const cnxtest::LogoScene scene = cnxtest::shapesScene();
    const Stages s = run(scene);
    print("formas", s);
    save(env("CNX_QUALITY_DIR"), scene, s, "");
    // Antes: smoothed 0,1064 (o dobro da escadinha, 0,0507) e Bézier 0,1085. A segmentação deixa uma franja amarela em
    // volta do retângulo vermelho (a cor da paleta mais próxima do rosa do antialiasing), que é o piso do erro aqui.
    CHECK(s.smoothed.rmse < s.staircase.rmse);
    CHECK(s.smoothed.rmse < 0.040 && s.fitted.rmse < 0.040);
    CHECK(s.smoothed.maxError < 0.55 && s.fitted.maxError < 0.55);
    CHECK(s.fitted.rmse < 1.1 * s.smoothed.rmse);
    // O fio preto de 34 × 2,4 px (antes: um quadrado de ~10 × 10 px): continua comprido e fino.
    const sla::Box2 line = faceBox(*s.snapshot->smoothed, sla::Rgba{0.08f, 0.08f, 0.08f, 1.0f});
    std::printf("  fio preto: caixa %.1f × %.1f px (verdade 34 × 2,4; a escadinha tem 34 × 3)\n", line.max.x - line.min.x, line.max.y - line.min.y);
    CHECK(line.max.x - line.min.x > 32.0 && line.max.y - line.min.y < 4.5);
    // O disco amarelo (raio 9,3): o contorno ajustado fica a ~0,5 px do círculo verdadeiro.
    // (os recortes amarelos em volta do retângulo vermelho também são amarelos: o disco é o anel com mais pontos.)
    const cnxtest::svg::Svg svg = cnxtest::svg::parseSvg(s.svgFitted);
    const cnxtest::svg::Path* disk = nullptr;
    for (const cnxtest::svg::Path& p : svg.paths) {
        const auto channel = [&](std::size_t at) { return std::stoi(p.fill.substr(at, 2), nullptr, 16); };
        const bool yellow = std::abs(channel(1) - 250) < 12 && std::abs(channel(3) - 190) < 12 && std::abs(channel(5) - 20) < 25;
        if (yellow && (disk == nullptr || p.rings[0].size() > disk->rings[0].size())) disk = &p;
    }
    CHECK(disk != nullptr);
    double worstRadial = 1e9;
    if (disk != nullptr) {
        worstRadial = 0.0;
        for (const Vec2 q : disk->rings[0]) worstRadial = std::max(worstRadial, std::abs(sla::norm(q - Vec2{74.0, 54.0}) - 9.3));
    }
    std::printf("  disco amarelo: desvio radial máximo %.2f px\n", worstRadial);
    CHECK(worstRadial < 0.45);
}

CNX_TEST(smoothing_is_at_least_as_good_as_the_staircase_it_started_from) {
    // Para qualquer cena: o suavizado nunca é pior que a escadinha, e o ajuste de Bézier não piora o suavizado em mais
    // de 60 %. Cenas de lados longos e curtos, discos pequenos e grandes e traços de 1,5 a 6 px.
    cnxtest::LogoScene thin{"fios", 80, 56, sla::Rgba8{255, 255, 255, 255}, {}, 3};
    thin.shapes.push_back({cnxtest::rectShape(6.0, 8.0, 70.0, 9.6), sla::Rgba8{20, 20, 20, 255}});         // 1,6 px
    thin.shapes.push_back({cnxtest::rectShape(6.0, 16.3, 70.0, 19.5), sla::Rgba8{20, 20, 20, 255}});       // 3,2 px
    thin.shapes.push_back({cnxtest::rectShape(10.0, 26.0, 16.0, 50.0), sla::Rgba8{220, 40, 40, 255}});     // 6 × 24
    thin.shapes.push_back({cnxtest::diskShape(40.0, 38.0, 6.2), sla::Rgba8{220, 40, 40, 255}});
    thin.shapes.push_back({cnxtest::strokeShape(52.0, 50.0, 74.0, 26.0, 1.4), sla::Rgba8{20, 20, 20, 255}});  // diagonal de 2,8 px
    const Stages s = run(thin);
    print("fios", s);
    save(env("CNX_QUALITY_DIR"), thin, s, "");
    CHECK(s.smoothed.rmse < s.staircase.rmse);
    CHECK(s.fitted.rmse < 1.6 * s.smoothed.rmse);
}

int main() { return cnxtest::runAll(); }
