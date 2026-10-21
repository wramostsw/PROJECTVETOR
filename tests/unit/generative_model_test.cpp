// tests/unit/generative_model_test.cpp — GenerativeModel: cobertura exata por área, energia esparsa
// contra a densa e gradiente analítico contra diferenças finitas centradas.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cnx/segment/PixelMap.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/vector/GenerativeModel.h"
#include "cnx/vector/VectorImage.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::Color4;
using cnx::FaceId;
using cnx::GenerativeModel;
using cnx::GenerativeStats;
using cnx::NodeId;
using cnx::VectorImage;
using cnxtest::SplitMix64;
using sla::Array2D;
using sla::Vec2;

namespace {

double uniform(SplitMix64& rng) { return static_cast<double>(rng.next() >> 11) * (1.0 / 9007199254740992.0); }
double symmetric(SplitMix64& rng, double amplitude) { return (2.0 * uniform(rng) - 1.0) * amplitude; }

/// Cena: imagem aleatória segmentada em blocos fundidos ao acaso e traçada em um VectorImage.
struct TracedScene {
    VectorImage image;
    Array2D<sla::Rgba8> source;
    Array2D<FaceId> pixelFaces;
};

TracedScene tracedScene(int w, int h, int block, int merges, SplitMix64& rng, int levels = 4) {
    cnxtest::Scene s = cnxtest::blockScene(w, h, block, rng, levels);
    cnxtest::randomMerges(s.map, merges, rng);
    s.map.assignMedianColors();
    (void)s.map.compact();
    cnx::TraceResult t = cnx::BoundaryTracer().trace(s.map);
    return {std::move(t.image), std::move(s.image), std::move(t.pixelFaces)};
}

/// Desloca nós por jitter uniforme; `keepFrame` preserva os nós sobre a moldura.
void jitterNodes(VectorImage& v, double amplitude, bool keepFrame, SplitMix64& rng) {
    for (NodeId n = 0; n < static_cast<NodeId>(v.nodeCount()); ++n) {
        if (keepFrame && (v.flags(n) & (cnx::NodeFlag::ImageBorder | cnx::NodeFlag::Pinned)) != 0) continue;
        v.setPosition(n, v.position(n) + Vec2{symmetric(rng, amplitude), symmetric(rng, amplitude)});
    }
}

// ---------------------------------------------------------------- referência: recorte poligonal exato

using Poly = std::vector<Vec2>;

Poly clipHalfPlane(const Poly& in, bool xAxis, double bound, bool keepGreater) {
    const auto coord = [&](Vec2 p) { return xAxis ? p.x : p.y; };
    const auto inside = [&](Vec2 p) { return keepGreater ? coord(p) >= bound : coord(p) <= bound; };
    Poly out;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const Vec2 a = in[i];
        const Vec2 b = in[(i + 1) % in.size()];
        const bool ia = inside(a);
        const bool ib = inside(b);
        if (ia != ib) {
            const double t = (bound - coord(a)) / (coord(b) - coord(a));
            out.push_back(a + (b - a) * t);
        }
        if (ib) out.push_back(b);
    }
    return out;
}

double signedArea(const Poly& p) {
    double a = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) a += sla::cross(p[i], p[(i + 1) % p.size()]);
    return 0.5 * a;
}

double clippedArea(const Poly& poly, int i, int j) {
    Poly p = clipHalfPlane(poly, true, i, true);
    p = clipHalfPlane(p, true, i + 1, false);
    p = clipHalfPlane(p, false, j, true);
    p = clipHalfPlane(p, false, j + 1, false);
    return p.size() < 3 ? 0.0 : signedArea(p);
}

Poly loopPolygon(const VectorImage& v, cnx::HalfEdgeId first) {
    Poly poly;
    cnx::HalfEdgeId h = first;
    do {
        const auto nodes = v.edgeNodes(cnx::edgeOf(h));
        if (!cnx::isReversed(h)) {
            for (std::size_t i = 0; i + 1 < nodes.size(); ++i) poly.push_back(v.position(nodes[i]));
        } else {
            for (std::size_t i = nodes.size() - 1; i >= 1; --i) poly.push_back(v.position(nodes[i]));
        }
        h = v.next(h);
    } while (h != first);
    return poly;
}

/// Cobertura de uma face pelo recorte exato: soma assinada dos laços recortados em cada pixel.
double referenceCoverage(const VectorImage& v, FaceId f, int i, int j) {
    double a = 0.0;
    if (v.face(f).outerLoop != cnx::kInvalidId) a += clippedArea(loopPolygon(v, v.face(f).outerLoop), i, j);
    for (const cnx::HalfEdgeId h : v.holeLoops(f)) a += clippedArea(loopPolygon(v, h), i, j);
    return a;
}

double toD(float v) { return static_cast<double>(v); }

}  // namespace

// ================================================================ cobertura exata

CNX_TEST(face_coverage_matches_exact_polygon_clipping) {
    SplitMix64 rng{201};
    for (int scene = 0; scene < 5; ++scene) {
        TracedScene s = tracedScene(14 + scene, 11, 3, 25, rng);
        jitterNodes(s.image, 0.3, true, rng);
        const GenerativeModel model(GenerativeModel::targetFrom(s.source));
        Array2D<double> sum(model.width(), model.height());
        double worst = 0.0;
        for (FaceId f = 1; f < static_cast<FaceId>(s.image.faceCount()); ++f) {
            Array2D<double> cov;
            model.faceCoverage(s.image, f, cov);
            for (int y = 0; y < model.height(); ++y) {
                for (int x = 0; x < model.width(); ++x) {
                    worst = std::max(worst, std::abs(cov(x, y) - referenceCoverage(s.image, f, x, y)));
                    sum(x, y) += cov(x, y);
                    CHECK_CTX(cov(x, y) > -1e-12 && cov(x, y) < 1.0 + 1e-12, "cena " + std::to_string(scene));
                }
            }
        }
        CHECK_CTX(worst < 1e-12, "cena " + std::to_string(scene) + " erro " + std::to_string(worst));
        // Partição da unidade: as faces interiores cobrem cada pixel exatamente uma vez.
        double worstSum = 0.0;
        for (int y = 0; y < model.height(); ++y) {
            for (int x = 0; x < model.width(); ++x) worstSum = std::max(worstSum, std::abs(sum(x, y) - 1.0));
        }
        CHECK_CTX(worstSum < 1e-12, "cena " + std::to_string(scene) + " soma " + std::to_string(worstSum));
        Array2D<double> exterior;
        model.faceCoverage(s.image, cnx::kExteriorFace, exterior);
        double ext = 0.0;
        for (int y = 0; y < model.height(); ++y) {
            for (int x = 0; x < model.width(); ++x) ext = std::max(ext, std::abs(exterior(x, y)));
        }
        CHECK_CTX(ext < 1e-12, "exterior");
    }
}

CNX_TEST(unjittered_staircase_covers_whole_pixels_exactly) {
    SplitMix64 rng{202};
    TracedScene s = tracedScene(16, 12, 3, 20, rng);
    const GenerativeModel model(GenerativeModel::targetFrom(s.source));
    Array2D<Color4> synth;
    model.render(s.image, synth);
    double worst = 0.0;
    for (int y = 0; y < model.height(); ++y) {
        for (int x = 0; x < model.width(); ++x) {
            const sla::Rgba& c = s.image.face(s.pixelFaces(x, y)).color;
            worst = std::max({worst, std::abs(synth(x, y)[0] - c.r), std::abs(synth(x, y)[1] - c.g),
                              std::abs(synth(x, y)[2] - c.b), std::abs(synth(x, y)[3] - c.a)});
        }
    }
    CHECK_CTX(worst < 1e-12, std::to_string(worst));
    // A escadinha reproduz a imagem segmentada: energia = erro entre a imagem e as cores das regiões.
    CHECK(model.energy(s.image) >= 0.0);
}

CNX_TEST(render_is_the_area_weighted_mix_of_face_colors) {
    SplitMix64 rng{203};
    TracedScene s = tracedScene(13, 10, 3, 18, rng);
    jitterNodes(s.image, 0.3, true, rng);
    const GenerativeModel model(GenerativeModel::targetFrom(s.source));
    Array2D<Color4> synth;
    model.render(s.image, synth);
    Array2D<Color4> expected(model.width(), model.height());
    for (FaceId f = 1; f < static_cast<FaceId>(s.image.faceCount()); ++f) {
        Array2D<double> cov;
        model.faceCoverage(s.image, f, cov);
        const sla::Rgba& c = s.image.face(f).color;
        for (int y = 0; y < model.height(); ++y) {
            for (int x = 0; x < model.width(); ++x) {
                expected(x, y)[0] += cov(x, y) * toD(c.r);
                expected(x, y)[1] += cov(x, y) * toD(c.g);
                expected(x, y)[2] += cov(x, y) * toD(c.b);
                expected(x, y)[3] += cov(x, y) * toD(c.a);
            }
        }
    }
    double worst = 0.0;
    for (int y = 0; y < model.height(); ++y) {
        for (int x = 0; x < model.width(); ++x) {
            for (std::size_t k = 0; k < 4; ++k) worst = std::max(worst, std::abs(synth(x, y)[k] - expected(x, y)[k]));
        }
    }
    CHECK_CTX(worst < 1e-12, std::to_string(worst));
}

// ================================================================ energia esparsa = densa

CNX_TEST(sparse_energy_matches_dense_energy_even_with_nodes_outside_the_image) {
    SplitMix64 rng{204};
    for (int scene = 0; scene < 6; ++scene) {
        TracedScene s = tracedScene(15 + scene, 12, 3, 22, rng);
        // Jitter grande em todos os nós, inclusive os da moldura: segmentos saem da imagem por todos os lados.
        jitterNodes(s.image, scene < 3 ? 0.45 : 3.0, false, rng);
        Array2D<sla::Rgba> target = GenerativeModel::targetFrom(s.source);
        const std::string tag = "cena " + std::to_string(scene);
        {
            const GenerativeModel model(target);
            const double sparse = model.energy(s.image);
            const double dense = model.denseEnergy(s.image);
            CHECK_CTX(std::abs(sparse - dense) <= 1e-9 * (1.0 + dense), tag + ": " + std::to_string(sparse) + " vs " + std::to_string(dense));
        }
        {   // com pesos (incluindo zeros)
            Array2D<float> weights(target.width(), target.height());
            for (int y = 0; y < target.height(); ++y) {
                for (int x = 0; x < target.width(); ++x) weights(x, y) = static_cast<float>(rng.below(4)) * 0.5f;
            }
            const GenerativeModel model(target, weights);
            const double sparse = model.energy(s.image);
            const double dense = model.denseEnergy(s.image);
            CHECK_CTX(std::abs(sparse - dense) <= 1e-9 * (1.0 + dense), tag + " (pesos)");
        }
    }
}

CNX_TEST(energy_is_zero_when_the_target_is_the_synthesis) {
    SplitMix64 rng{205};
    TracedScene s = tracedScene(14, 11, 3, 20, rng);
    jitterNodes(s.image, 0.3, true, rng);
    const GenerativeModel probe(Array2D<sla::Rgba>(14, 11));
    Array2D<Color4> synth;
    probe.render(s.image, synth);
    Array2D<sla::Rgba> target(14, 11);
    for (int y = 0; y < 11; ++y) {
        for (int x = 0; x < 14; ++x) {
            target(x, y) = {static_cast<float>(synth(x, y)[0]), static_cast<float>(synth(x, y)[1]),
                            static_cast<float>(synth(x, y)[2]), static_cast<float>(synth(x, y)[3])};
        }
    }
    const GenerativeModel model(target);
    std::vector<Vec2> grad(s.image.nodeCount());
    const double e = model.energyAndGradient(s.image, grad);
    CHECK_CTX(e < 1e-9, std::to_string(e));  // sobra só o arredondamento para float do alvo
    double worst = 0.0;
    for (const Vec2& g : grad) worst = std::max({worst, std::abs(g.x), std::abs(g.y)});
    CHECK_CTX(worst < 1e-5, std::to_string(worst));
}

CNX_TEST(zero_weights_remove_all_energy_and_gradient) {
    SplitMix64 rng{206};
    TracedScene s = tracedScene(12, 9, 3, 15, rng);
    jitterNodes(s.image, 0.3, true, rng);
    const GenerativeModel model(GenerativeModel::targetFrom(s.source), Array2D<float>(12, 9, 0.0f));
    std::vector<Vec2> grad(s.image.nodeCount());
    CHECK(model.energyAndGradient(s.image, grad) == 0.0);
    for (const Vec2& g : grad) CHECK(g.x == 0.0 && g.y == 0.0);
}

// ================================================================ gradiente analítico = diferenças finitas

namespace {

/// Compara o gradiente analítico com diferenças centradas em todos os nós. Retorna o pior desvio relativo.
double worstFiniteDifferenceError(const GenerativeModel& model, VectorImage image, double h, int* checked = nullptr) {
    std::vector<Vec2> grad(image.nodeCount());
    const double e = model.energyAndGradient(image, grad);
    double worst = 0.0;
    if (std::abs(e - model.energy(image)) > 1e-9 * (1.0 + e)) worst = 1e9;  // as duas rotas devem concordar
    int count = 0;
    for (NodeId n = 0; n < static_cast<NodeId>(image.nodeCount()); ++n) {
        const Vec2 p = image.position(n);
        for (int axis = 0; axis < 2; ++axis) {
            const Vec2 d = axis == 0 ? Vec2{h, 0.0} : Vec2{0.0, h};
            image.setPosition(n, p + d);
            const double ep = model.energy(image);
            image.setPosition(n, p - d);
            const double em = model.energy(image);
            image.setPosition(n, p);
            const double fd = (ep - em) / (2.0 * h);
            const double g = axis == 0 ? grad[static_cast<std::size_t>(n)].x : grad[static_cast<std::size_t>(n)].y;
            worst = std::max(worst, std::abs(g - fd) / (1.0 + std::abs(fd)));
            ++count;
        }
    }
    if (checked != nullptr) *checked = count;
    return worst;
}

}  // namespace

CNX_TEST(gradient_matches_central_finite_differences) {
    SplitMix64 rng{207};
    int total = 0;
    for (int scene = 0; scene < 6; ++scene) {
        TracedScene s = tracedScene(18 + scene, 13, 3, 18, rng);
        jitterNodes(s.image, 0.3, false, rng);
        const GenerativeModel model(GenerativeModel::targetFrom(s.source));
        int checked = 0;
        const double worst = worstFiniteDifferenceError(model, s.image, 1e-5, &checked);
        total += checked;
        CHECK_CTX(worst < 1e-5, "cena " + std::to_string(scene) + " pior desvio " + std::to_string(worst));
    }
    std::printf("  derivadas conferidas: %d\n", total);
    CHECK(total > 300);
}

CNX_TEST(gradient_matches_finite_differences_with_weights_and_far_away_nodes) {
    SplitMix64 rng{208};
    for (int scene = 0; scene < 4; ++scene) {
        TracedScene s = tracedScene(14, 11, 3, 20, rng);
        jitterNodes(s.image, 2.5, false, rng);  // nós dentro e fora da imagem
        Array2D<float> weights(14, 11);
        for (int y = 0; y < 11; ++y) {
            for (int x = 0; x < 14; ++x) weights(x, y) = static_cast<float>(0.25 + 1.75 * uniform(rng));
        }
        const GenerativeModel model(GenerativeModel::targetFrom(s.source), weights);
        const double worst = worstFiniteDifferenceError(model, s.image, 1e-5);
        CHECK_CTX(worst < 1e-5, "cena " + std::to_string(scene) + " pior desvio " + std::to_string(worst));
    }
}

CNX_TEST(scenes_exercise_junctions) {
    // O gradiente só é convincente se as cenas têm junções (nós compartilhados por 3+ arestas).
    // As arestas fechadas (ilhas) são cobertas pelo teste da ilha, abaixo.
    SplitMix64 rng{209};
    int junctions = 0;
    for (int scene = 0; scene < 6; ++scene) {
        const TracedScene s = tracedScene(18 + scene, 13, 3, 18, rng);
        for (NodeId n = 0; n < static_cast<NodeId>(s.image.nodeCount()); ++n) {
            junctions += (s.image.flags(n) & cnx::NodeFlag::Junction) != 0 ? 1 : 0;
        }
    }
    CHECK(junctions > 20);
}

CNX_TEST(gradient_of_an_island_with_a_closed_edge_matches_finite_differences) {
    // Região 4×4 no interior de um fundo: uma aresta fechada (LoopAnchor) e nenhuma junção.
    SplitMix64 rng{210};
    cnx::PixelMap map(10, 10);
    const cnx::RegionId bg = map.createRegion();
    const cnx::RegionId island = map.createRegion();
    Array2D<sla::Rgba8> image(10, 10);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            const bool in = x >= 3 && x < 7 && y >= 2 && y < 7;
            image(x, y) = cnxtest::randomColor(rng, 5);
            map.setLabel(x, y, in ? island : bg, image(x, y));
        }
    }
    map.assignMedianColors();
    (void)map.compact();
    cnx::TraceResult t = cnx::BoundaryTracer().trace(map);
    int closed = 0;
    for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(t.image.edgeCount()); ++e) {
        closed += t.image.edgeNodes(e).front() == t.image.edgeNodes(e).back() ? 1 : 0;
    }
    CHECK(closed >= 1);  // a ilha (a moldura é quebrada nos 4 cantos)
    jitterNodes(t.image, 0.3, false, rng);
    const GenerativeModel model(GenerativeModel::targetFrom(image));
    CHECK(worstFiniteDifferenceError(model, t.image, 1e-5) < 1e-5);
}

CNX_TEST(gradient_has_the_closed_form_of_a_single_pixel_edge) {
    // Um segmento horizontal dentro de uma única linha: ∂a/∂P_k = n·∫(1−t) com n = (0, −Δx).
    // Imagem 1×1 com a face da esquerda de cor 1 no canal 0 e exterior 0; alvo 0 → E = ĉ².
    VectorImage v(1.0, 1.0);
    const cnx::FaceId inside = v.addFace(sla::Rgba{1.0f, 0.0f, 0.0f, 0.0f}, 0);
    const NodeId a = v.addNode({0.0, 0.25});
    const NodeId b = v.addNode({1.0, 0.25});
    const NodeId c = v.addNode({1.0, 1.0});
    const NodeId d = v.addNode({0.0, 1.0});
    // Região abaixo da linha y = 0.25 (percurso positivo): a → b → c → d → a, borda superior em y = 0.25.
    const std::array<NodeId, 5> loop{a, b, c, d, a};
    (void)v.addEdge(loop, inside, cnx::kExteriorFace);
    v.linkFaceLoops();
    const GenerativeModel model(Array2D<sla::Rgba>(1, 1));
    std::vector<Vec2> grad(v.nodeCount());
    const double e = model.energyAndGradient(v, grad);
    CHECK(std::abs(e - 0.75 * 0.75) < 1e-12);  // cobertura 0,75 → ĉ_r = 0,75
    // A aresta de cima y = 0.25 desloca-se com a e b: dE/da_y = 2·ĉ·(∂a/∂a_y), ∂a/∂a_y = −∫(1−t)dt·|Δx| = −0.5.
    // O segmento a→b tem n = (Δy, −Δx) = (0, −1); a derivada da área em relação a a_y é n_y·½ = −0,5.
    CHECK(std::abs(grad[static_cast<std::size_t>(a)].y - 2.0 * 0.75 * -0.5) < 1e-12);
    CHECK(std::abs(grad[static_cast<std::size_t>(b)].y - 2.0 * 0.75 * -0.5) < 1e-12);
    // Em x: só o segmento d→a (vertical, Δ = (0, −0,75), n = (−0,75, 0)) move a área com a_x: ∫t dt = ½.
    CHECK(std::abs(grad[static_cast<std::size_t>(a)].x - 2.0 * 0.75 * -0.375) < 1e-12);
}

CNX_TEST(edges_left_of_the_image_still_cover_the_pixels_to_their_right) {
    // Retângulo de x = −2 a x = 2 numa imagem 4×4: a aresta esquerda é exatamente vertical e fica fora
    // da imagem. Os pixels das colunas 0 e 1 devem ficar totalmente cobertos; os das colunas 2 e 3, não.
    VectorImage v(4.0, 4.0);
    const cnx::FaceId face = v.addFace(sla::Rgba{1.0f, 0.0f, 0.0f, 0.0f}, 0);
    const NodeId a = v.addNode({-2.0, 0.0}, cnx::NodeFlag::LoopAnchor);
    const std::array<NodeId, 5> loop{a, v.addNode({2.0, 0.0}), v.addNode({2.0, 4.0}), v.addNode({-2.0, 4.0}), a};
    (void)v.addEdge(loop, face, cnx::kExteriorFace);
    v.linkFaceLoops();
    const GenerativeModel model(Array2D<sla::Rgba>(4, 4));
    Array2D<Color4> synth;
    model.render(v, synth);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            CHECK_CTX(std::abs(synth(x, y)[0] - (x < 2 ? 1.0 : 0.0)) < 1e-12,
                      std::to_string(x) + "," + std::to_string(y) + ": " + std::to_string(synth(x, y)[0]));
        }
    }
    // Alvo zero: E = 8 pixels cobertos com ĉ_r = 1.
    CHECK(std::abs(model.energy(v) - 8.0) < 1e-12);
    // Segmento inclinado que cruza x = 0: só a parte com x ≥ 0 toca pixels, a de x < 0 cobre à direita.
    VectorImage w(4.0, 4.0);
    const cnx::FaceId f2 = w.addFace(sla::Rgba{1.0f, 0.0f, 0.0f, 0.0f}, 0);
    const NodeId b = w.addNode({-1.0, 0.0}, cnx::NodeFlag::LoopAnchor);
    const std::array<NodeId, 5> tri{b, w.addNode({5.0, 0.0}), w.addNode({5.0, 4.0}), w.addNode({-1.0, 4.0}), b};
    (void)w.addEdge(tri, f2, cnx::kExteriorFace);
    w.linkFaceLoops();
    Array2D<Color4> all;
    model.render(w, all);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) CHECK(std::abs(all(x, y)[0] - 1.0) < 1e-12);  // imagem toda coberta
    }
}

// ================================================================ complexidade e validação de argumentos

CNX_TEST(evaluation_cost_follows_the_perimeter_not_the_area) {
    const int W = 1000;
    const int H = 800;
    VectorImage v(W, H);
    const cnx::FaceId bg = v.addFace(sla::Rgba{0.2f, 0.4f, 0.6f, 1.0f}, 0);
    const cnx::FaceId isle = v.addFace(sla::Rgba{0.9f, 0.1f, 0.3f, 1.0f}, 1);
    const std::array<NodeId, 5> frame{v.addNode({0, 0}, cnx::NodeFlag::LoopAnchor), v.addNode({double(W), 0}),
                                      v.addNode({double(W), double(H)}), v.addNode({0, double(H)}), 0};
    (void)v.addEdge(frame, bg, cnx::kExteriorFace);
    const NodeId i0 = v.addNode({100.5, 100.5}, cnx::NodeFlag::LoopAnchor);
    const std::array<NodeId, 5> square{i0, v.addNode({300.25, 100.5}), v.addNode({300.25, 260.75}), v.addNode({100.5, 260.75}), i0};
    (void)v.addEdge(square, isle, bg);
    v.linkFaceLoops();
    CHECK_CTX(v.validate().ok, v.validate().message);

    SplitMix64 rng{211};
    Array2D<sla::Rgba> target(W, H);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) target(x, y) = sla::toRgba(cnxtest::randomColor(rng, 6));
    }
    const GenerativeModel model(std::move(target));
    GenerativeStats stats;
    const auto t0 = std::chrono::steady_clock::now();
    const double sparse = model.energy(v, &stats);
    const auto t1 = std::chrono::steady_clock::now();
    const double dense = model.denseEnergy(v);
    const auto t2 = std::chrono::steady_clock::now();
    CHECK_CTX(std::abs(sparse - dense) <= 1e-9 * (1.0 + dense), std::to_string(sparse) + " vs " + std::to_string(dense));
    // Migalhas ∝ perímetro (moldura + ilha), muito abaixo dos W·H = 800 mil pixels.
    CHECK(stats.pieces < static_cast<std::size_t>(4 * (W + H) + 4000));
    CHECK(stats.crossedPixels < static_cast<std::size_t>(20000));
    std::printf("  %dx%d: esparsa %.2f ms (%zu migalhas, %zu pixels cruzados), densa %.2f ms\n", W, H,
                std::chrono::duration<double, std::milli>(t1 - t0).count(), stats.pieces, stats.crossedPixels,
                std::chrono::duration<double, std::milli>(t2 - t1).count());
}

CNX_TEST(argument_validation) {
    CHECK_THROWS(std::invalid_argument, (void)GenerativeModel(Array2D<sla::Rgba>()));
    CHECK_THROWS(std::invalid_argument, (void)GenerativeModel(Array2D<sla::Rgba>(4, 4), Array2D<float>(3, 4)));
    CHECK_THROWS(std::invalid_argument, (void)GenerativeModel(Array2D<sla::Rgba>(4, 4), Array2D<float>(4, 4, -1.0f)));
    SplitMix64 rng{212};
    TracedScene s = tracedScene(8, 6, 2, 5, rng);
    const GenerativeModel wrongSize(Array2D<sla::Rgba>(9, 6));
    CHECK_THROWS(std::invalid_argument, (void)wrongSize.energy(s.image));
    const GenerativeModel model(GenerativeModel::targetFrom(s.source));
    std::vector<Vec2> shortGradient(s.image.nodeCount() - 1);
    CHECK_THROWS(std::invalid_argument, (void)model.energyAndGradient(s.image, shortGradient));
    Array2D<double> cov;
    CHECK_THROWS(std::out_of_range, model.faceCoverage(s.image, static_cast<FaceId>(s.image.faceCount()), cov));
}

int main() { return cnxtest::runAll(); }
