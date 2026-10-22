// tests/unit/contour_smoother_test.cpp — ContourSmoother de ponta a ponta: precisão subpixel com o
// modelo gerador, preservação de cantos, invariantes topológicas e o fiscal anti-cruzamento.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stop_token>
#include <string>
#include <vector>

#include "cnx/vector/ContourSmoother.h"
#include "cnx/vector/CrossingGuard.h"
#include "test_harness.h"
#include "test_support.h"

using cnx::ContourSmoother;
using cnx::GenerativeModel;
using cnx::MeasurementType;
using cnx::NodeId;
using cnx::SmoothParams;
using cnx::VectorImage;
using cnxtest::SplitMix64;
using sla::Vec2;

namespace {

std::vector<NodeId> closedEdgeNodes(const VectorImage& v) {
    for (cnx::EdgeId e = 0; e < static_cast<cnx::EdgeId>(v.edgeCount()); ++e) {
        const auto nodes = v.edgeNodes(e);
        if (nodes.size() >= 4 && nodes.front() == nodes.back()) return {nodes.begin(), nodes.end() - 1};
    }
    return {};
}

SmoothParams allPhases(MeasurementType m) {
    SmoothParams p;
    p.measurementTypes = {m, m, m};
    return p;
}

double meanAbsRadialError(const VectorImage& v, const std::vector<NodeId>& nodes, Vec2 center, double radius) {
    double s = 0.0;
    for (const NodeId n : nodes) s += std::abs(sla::norm(v.position(n) - center) - radius);
    return s / static_cast<double>(nodes.size());
}

/// Cena do disco com antialiasing: círculo de raio 10,6 centrado em (18,3; 17,8) numa imagem 36×36.
constexpr Vec2 kDiskCenter{18.3, 17.8};
constexpr double kDiskRadius = 10.6;

cnxtest::AaScene diskScene() {
    return cnxtest::aaScene(36, 36, [](double x, double y) {
        const double dx = x - kDiskCenter.x;
        const double dy = y - kDiskCenter.y;
        return dx * dx + dy * dy <= kDiskRadius * kDiskRadius;
    }, 16);
}

/// Invariantes que toda execução deve manter.
void checkInvariants(const VectorImage& before, const VectorImage& after, const std::string& tag) {
    const auto report = after.validate();
    CHECK_CTX(report.ok, tag + ": " + report.message);
    CHECK_CTX(cnx::CrossingGuard::find(after).empty(), tag + ": cruzamentos");
    double area = 0.0;
    for (cnx::FaceId f = 1; f < static_cast<cnx::FaceId>(after.faceCount()); ++f) area += after.face(f).area;
    CHECK_CTX(std::abs(area - after.width() * after.height()) < 1e-6 * after.width() * after.height(), tag + ": área total");
    for (NodeId n = 0; n < static_cast<NodeId>(after.nodeCount()); ++n) {
        CHECK_CTX(!after.isQuarantined(n), tag + ": quarentena vazou");
        if ((after.flags(n) & cnx::NodeFlag::Pinned) != 0) CHECK_CTX(after.position(n) == before.position(n), tag + ": nó fixo moveu");
    }
}

}  // namespace

// ================================================================ precisão subpixel (modelo gerador)

CNX_TEST(generative_fidelity_places_a_disk_boundary_with_subpixel_accuracy) {
    cnxtest::AaScene s = diskScene();
    const GenerativeModel model(s.target);
    const std::vector<NodeId> ring = closedEdgeNodes(s.image);
    CHECK(ring.size() > 20);
    const double errBefore = meanAbsRadialError(s.image, ring, kDiskCenter, kDiskRadius);
    const double eBefore = model.energy(s.image);

    VectorImage before = s.image;
    const ContourSmoother smoother(allPhases(MeasurementType::Generative), &model);
    const auto t0 = std::chrono::steady_clock::now();
    const cnx::SmoothResult r = smoother.run(s.image);
    const auto t1 = std::chrono::steady_clock::now();

    const double errAfter = meanAbsRadialError(s.image, ring, kDiskCenter, kDiskRadius);
    const double eAfter = model.energy(s.image);
    std::printf("  disco: erro radial médio %.3f -> %.3f px; E_gen %.3f -> %.3f; %.0f ms\n", errBefore, errAfter, eBefore,
                eAfter, std::chrono::duration<double, std::milli>(t1 - t0).count());
    CHECK(errBefore > 0.2);   // a escadinha erra de verdade
    CHECK(errAfter < 0.1);    // erro subpixel depois da suavização
    CHECK(eAfter < 0.25 * eBefore);
    for (const cnx::PhaseStats& ph : r.phases) {
        CHECK(ph.energyEnd <= ph.energyBegin + 1e-9);
        CHECK(ph.iterations > 0);
    }
    checkInvariants(before, s.image, "disco");
}

CNX_TEST(generative_fidelity_straightens_a_slanted_edge_onto_the_true_line) {
    // Região escura abaixo da reta y = 0,55 x + 4,3 numa imagem 32×32.
    const auto inside = [](double x, double y) { return y > 0.55 * x + 4.3; };
    cnxtest::AaScene s = cnxtest::aaScene(32, 32, inside, 16);
    const GenerativeModel model(s.target);
    VectorImage before = s.image;
    const ContourSmoother smoother(allPhases(MeasurementType::Generative), &model);
    (void)smoother.run(s.image);

    // Nós interiores da fronteira (nem junções nem moldura): distância à reta verdadeira.
    const double norm = std::sqrt(1.0 + 0.55 * 0.55);
    double worst = 0.0;
    double worstBefore = 0.0;
    int count = 0;
    for (NodeId n = 0; n < static_cast<NodeId>(s.image.nodeCount()); ++n) {
        const std::uint8_t fl = s.image.flags(n);
        if ((fl & (cnx::NodeFlag::Junction | cnx::NodeFlag::ImageBorder)) != 0) continue;
        const Vec2 p = s.image.position(n);
        const Vec2 q = before.position(n);
        worst = std::max(worst, std::abs(p.y - (0.55 * p.x + 4.3)) / norm);
        worstBefore = std::max(worstBefore, std::abs(q.y - (0.55 * q.x + 4.3)) / norm);
        ++count;
    }
    std::printf("  reta: %d nós interiores; desvio máximo %.3f -> %.3f px\n", count, worstBefore, worst);
    CHECK(count > 10);
    CHECK(worstBefore > 0.3);
    CHECK(worst < 0.12);
    checkInvariants(before, s.image, "reta");
}

// ================================================================ mola e priors

CNX_TEST(spring_smoothing_reduces_the_staircase_roughness_of_a_disk) {
    cnxtest::AaScene s = diskScene();
    const std::vector<NodeId> ring = closedEdgeNodes(s.image);
    const auto roughness = [&](const VectorImage& v) {  // desvio-padrão do raio
        double m = 0.0;
        for (const NodeId n : ring) m += sla::norm(v.position(n) - kDiskCenter);
        m /= static_cast<double>(ring.size());
        double var = 0.0;
        for (const NodeId n : ring) {
            const double d = sla::norm(v.position(n) - kDiskCenter) - m;
            var += d * d;
        }
        return std::sqrt(var / static_cast<double>(ring.size()));
    };
    const double before = roughness(s.image);
    VectorImage original = s.image;
    const ContourSmoother smoother(allPhases(MeasurementType::Spring));
    const cnx::SmoothResult r = smoother.run(s.image);
    const double after = roughness(s.image);
    std::printf("  mola: rugosidade radial %.3f -> %.3f px\n", before, after);
    CHECK(after < 0.5 * before);
    for (const cnx::PhaseStats& ph : r.phases) CHECK(ph.energyEnd <= ph.energyBegin + 1e-9);
    checkInvariants(original, s.image, "mola");
}

/// Região em L (6 cantos, lados ≥ 6 px) numa imagem 40×30: um polígono cujo prior angular tem
/// onde mexer (num retângulo todos os giros já são iguais e o prior não desloca nada).
cnxtest::AaScene lShapeScene() {
    return cnxtest::aaScene(40, 30, [](double x, double y) {
        return (x >= 8 && x <= 28 && y >= 6 && y <= 12) || (x >= 8 && x <= 14 && y >= 6 && y <= 22);
    }, 4);
}

CNX_TEST(puncturing_preserves_the_right_angle_of_a_triangle) {
    // Triângulo retângulo de catetos ~20 px, com o vértice reto em (6,3; 6,2) (fora da grade): a
    // hipotenusa é uma escadinha, que a suavização endireita. Com o modelo gerador, a fidelidade
    // coloca os lados na posição certa; o prior angular, porém, arredonda o ângulo reto. Com a
    // perfuração (fase do meio) o prior some no canto e o nó chega ao vértice verdadeiro.
    const Vec2 trueCorner{6.3, 6.2};
    const auto build = [] {
        return cnxtest::aaScene(36, 36, [](double x, double y) { return x >= 6.3 && y >= 6.2 && x + y <= 33.1; }, 8);
    };
    cnxtest::AaScene withPuncture = build();
    cnxtest::AaScene without = build();
    const GenerativeModel model(withPuncture.target);
    SmoothParams base = allPhases(MeasurementType::Generative);
    base.airPressureWeight = 0.0;
    base.punctureOnPhase = 1;  // a fase explícita do meio (o padrão agora é a 1ª: ver puncturing_on_the_staircase_beats_puncturing_after_smoothing)

    const VectorImage original = withPuncture.image;
    NodeId corner = cnx::kInvalidId;  // o nó do ângulo reto, na escadinha em (6, 6)
    for (NodeId n = 0; n < static_cast<NodeId>(original.nodeCount()); ++n) {
        if (original.position(n) == Vec2{6.0, 6.0}) corner = n;
    }
    CHECK(corner != cnx::kInvalidId);

    const cnx::SmoothResult r1 = ContourSmoother(base, &model).run(withPuncture.image);
    SmoothParams off = base;
    off.punctureCornerThresh = 0.0;
    const cnx::SmoothResult r2 = ContourSmoother(off, &model).run(without.image);

    std::printf("  cantos perfurados na fase do meio: %d (outras fases: %d, %d)\n", r1.phases[1].puncturedNodes,
                r1.phases[0].puncturedNodes, r1.phases[2].puncturedNodes);
    for (const cnx::PhaseStats& ph : r1.phases) {
        std::printf("    fase: %d iterações, estado %d, E %.6f -> %.6f (ang %.6f)\n", ph.iterations, static_cast<int>(ph.status),
                    ph.energyBegin, ph.energyEnd, ph.termsEnd.angular);
    }
    CHECK(r1.phases[1].puncturedNodes >= 1);
    CHECK(r1.phases[0].puncturedNodes == 0 && r1.phases[2].puncturedNodes == 0);
    CHECK(r2.phases[1].puncturedNodes == 0);
    CHECK((withPuncture.image.flags(corner) & cnx::NodeFlag::Corner) != 0);
    CHECK((without.image.flags(corner) & cnx::NodeFlag::Corner) == 0);
    const double dWith = sla::norm(withPuncture.image.position(corner) - trueCorner);
    const double dWithout = sla::norm(without.image.position(corner) - trueCorner);
    std::printf("  ângulo reto: distância ao vértice verdadeiro com perfuração %.3f px, sem %.3f px\n", dWith, dWithout);
    CHECK(dWith < dWithout);
    checkInvariants(original, withPuncture.image, "triângulo com perfuração");
    checkInvariants(original, without.image, "triângulo sem perfuração");
}

CNX_TEST(corner_detection_on_staircases_finds_real_corners_and_skips_arcs) {
    // Formas de verdade conhecida, rasterizadas e traçadas (a escadinha é o que o detector vê na fase 0).
    const auto corners = [](const cnxtest::AaScene& s, const SmoothParams& p = SmoothParams{}) {
        VectorImage image = s.image;
        const cnx::SmoothTopology topo = cnx::SmoothTopology::build(image, p.airPressureMaxNodes);
        return cnx::ContourSmoother::puncture(image, topo, p);
    };
    const auto disk = [](double r) {
        return cnxtest::aaScene(44, 44, [r](double x, double y) { return (x - 22.3) * (x - 22.3) + (y - 21.7) * (y - 21.7) <= r * r; }, 4);
    };
    // Arcos: nenhum canto, de r = 5 a 12 (os degraus de 1 px de um círculo não são cantos).
    for (const double r : {5.5, 7.4, 9.3, 12.0}) CHECK_CTX(corners(disk(r)) == 0, "disco r = " + std::to_string(r));
    // Anel: dois arcos.
    const cnxtest::AaScene ring = cnxtest::aaScene(44, 44, [](double x, double y) {
        const double d = std::hypot(x - 22.3, y - 21.7);
        return d <= 14.0 && d >= 7.5;
    }, 4);
    CHECK(corners(ring) == 0);
    // Retângulos: 4 cantos, de 34 × 3 (mais estreito que o braço de 4 px: só a meia escala o vê) a 12 × 12.
    const auto rect = [](double x0, double y0, double x1, double y1) {
        return cnxtest::aaScene(44, 30, [=](double x, double y) { return x >= x0 && x <= x1 && y >= y0 && y <= y1; }, 4);
    };
    CHECK(corners(rect(3, 8.2, 37, 10.6)) == 4);  // fio de 34 × 3
    CHECK(corners(rect(10, 6, 22, 18)) == 4);
    CHECK(corners(rect(5, 5, 8, 24)) == 4);  // 3 × 19
    // Sem a meia escala o fio de 3 px não teria nenhum canto: o braço de 4 px passa do canto seguinte.
    SmoothParams wide;
    wide.punctureArmPx = 4.0;
    const cnxtest::AaScene thin = rect(3, 8.2, 37, 10.6);
    CHECK(corners(thin, wide) == 4);
    // Triângulo retângulo (hipotenusa em escada): 3 cantos. Losango (escadinha nos 4 lados): 4 pontas.
    CHECK(corners(cnxtest::aaScene(36, 36, [](double x, double y) { return x >= 6.3 && y >= 6.2 && x + y <= 33.1; }, 4)) == 3);
    const int diamond = corners(cnxtest::aaScene(44, 44, [](double x, double y) { return std::abs(x - 22.3) + std::abs(y - 21.7) <= 14.0; }, 4));
    // Cada ponta do losango é cega (2–3 px na escadinha) e sai como 2 nós vizinhos, no máximo.
    std::printf("  cantos na escadinha: losango %d (pontas verdadeiras: 4)\n", diamond);
    CHECK(diamond >= 4 && diamond <= 8);
    // Limiar ≤ 0 desliga tudo; já marcados não contam de novo.
    SmoothParams off;
    off.punctureCornerThresh = 0.0;
    CHECK(corners(thin, off) == 0);
    VectorImage image = thin.image;
    const cnx::SmoothTopology topo = cnx::SmoothTopology::build(image, 8);
    CHECK(cnx::ContourSmoother::puncture(image, topo, SmoothParams{}) == 4 && cnx::ContourSmoother::puncture(image, topo, SmoothParams{}) == 0);
}

CNX_TEST(puncturing_works_on_any_phase_and_defaults_to_the_first) {
    // O mesmo triângulo, com a perfuração na 1ª fase (padrão: sobre a escadinha, antes de qualquer suavização) e na
    // 2ª. Em lados retos as duas chegam ao vértice; a diferença entre elas está nas formas curvas (contour_quality_test).
    const Vec2 trueCorner{6.3, 6.2};
    const auto build = [] {
        return cnxtest::aaScene(36, 36, [](double x, double y) { return x >= 6.3 && y >= 6.2 && x + y <= 33.1; }, 8);
    };
    double distance[2] = {0.0, 0.0};
    int punctured[2] = {0, 0};
    for (int phase = 0; phase < 2; ++phase) {
        cnxtest::AaScene s = build();
        const GenerativeModel model(s.target);
        SmoothParams p = allPhases(MeasurementType::Generative);
        p.airPressureWeight = 0.0;
        p.punctureOnPhase = phase;
        NodeId corner = cnx::kInvalidId;
        for (NodeId n = 0; n < static_cast<NodeId>(s.image.nodeCount()); ++n) {
            if (s.image.position(n) == Vec2{6.0, 6.0}) corner = n;
        }
        const cnx::SmoothResult r = ContourSmoother(p, &model).run(s.image);
        punctured[phase] = r.phases[static_cast<std::size_t>(phase)].puncturedNodes;
        distance[phase] = sla::norm(s.image.position(corner) - trueCorner);
        CHECK_CTX((s.image.flags(corner) & cnx::NodeFlag::Corner) != 0, "fase " + std::to_string(phase));
    }
    std::printf("  perfuração na fase 0: %d cantos, vértice a %.3f px; na fase 1: %d cantos, a %.3f px\n", punctured[0], distance[0], punctured[1], distance[1]);
    CHECK(punctured[0] >= 1 && punctured[1] >= 1);
    CHECK(distance[0] < 0.15 && distance[1] < 0.15);
    // O padrão é a 1ª fase.
    CHECK(SmoothParams{}.punctureOnPhase == 0);
}

CNX_TEST(air_pressure_keeps_small_shapes_from_changing_area) {
    // A região em L tem 6 nós, abaixo do limite de 8: com pressão do ar a área fica quase intacta.
    const auto build = lShapeScene;
    const auto area = [](const VectorImage& v) {
        const std::vector<NodeId> ring = closedEdgeNodes(v);
        double a = 0.0;
        for (std::size_t i = 0; i < ring.size(); ++i) a += sla::cross(v.position(ring[i]), v.position(ring[(i + 1) % ring.size()]));
        return 0.5 * a;
    };
    SmoothParams withAir = allPhases(MeasurementType::Spring);
    withAir.spacingWeight = {0.0, 0.0, 0.0};
    withAir.punctureCornerThresh = 0.0;  // sem cantos perfurados o prior angular arredonda o L e a área muda (é o que a pressão do ar segura)
    withAir.airPressureWeight = 200.0;
    SmoothParams noAir = withAir;
    noAir.airPressureWeight = 0.0;
    cnxtest::AaScene a = build();
    cnxtest::AaScene b = build();
    const double a0 = area(a.image);
    (void)ContourSmoother(withAir).run(a.image);
    (void)ContourSmoother(noAir).run(b.image);
    std::printf("  pressão do ar: área %.2f -> %.2f (com) e %.2f (sem)\n", a0, area(a.image), area(b.image));
    CHECK(std::abs(area(b.image) - a0) > 1.0);                               // sem pressão a área muda de verdade
    CHECK(std::abs(area(a.image) - a0) < 0.25 * std::abs(area(b.image) - a0));  // com pressão, bem menos
}

// ================================================================ invariantes em cenas aleatórias

CNX_TEST(random_scenes_keep_topology_pinned_nodes_and_frame_anchors) {
    SplitMix64 rng{501};
    for (int scene = 0; scene < 5; ++scene) {
        cnxtest::VectorScene s = cnxtest::vectorScene(22 + scene, 16, 3, 30, rng);
        const VectorImage before = s.image;
        SmoothParams p;
        p.maxCgIterations = 80;
        const cnx::SmoothResult r = ContourSmoother(p).run(s.image);
        const std::string tag = "cena " + std::to_string(scene);
        checkInvariants(before, s.image, tag);
        int punctured = 0;
        for (const cnx::PhaseStats& ph : r.phases) {
            CHECK_CTX(ph.energyEnd <= ph.energyBegin + 1e-9, tag);
            punctured += ph.puncturedNodes;
        }
        int corners = 0;
        for (NodeId n = 0; n < static_cast<NodeId>(s.image.nodeCount()); ++n) {
            const std::uint8_t fl = s.image.flags(n);
            if ((fl & cnx::NodeFlag::Corner) != 0 && (fl & cnx::NodeFlag::Pinned) == 0) ++corners;
            // Nós da moldura deslizam ao longo dela, sem sair (âncora de margem).
            if ((fl & cnx::NodeFlag::ImageBorder) != 0 && (fl & cnx::NodeFlag::Pinned) == 0) {
                const Vec2 o = before.position(n);
                const Vec2 q = s.image.position(n);
                const double off = (o.x == 0.0 || o.x == s.image.width()) ? std::min(std::abs(q.x), std::abs(q.x - s.image.width()))
                                                                          : std::min(std::abs(q.y), std::abs(q.y - s.image.height()));
                CHECK_CTX(off == 0.0, tag + ": nó da moldura saiu dela");  // trava exata
            }
        }
        CHECK_CTX(corners == punctured, tag);
    }
}

CNX_TEST(crossing_guard_rolls_back_steps_that_would_cross_and_quarantines_the_nodes) {
    // Duas barras escuras de 2 px separadas por um vão de 2 px, mas o alvo (antialiasing) mostra uma
    // única barra sólida: o modelo gerador puxa as bordas do vão uma contra a outra, até se tocarem
    // e, com passos longos, até tentarem se atravessar.
    cnxtest::AaScene s = cnxtest::aaScene(24, 24, [](double x, double y) {
        return x >= 3 && x <= 21 && ((y >= 8 && y <= 10) || (y >= 12 && y <= 14));
    }, 4);
    const cnxtest::AaScene solid = cnxtest::aaScene(24, 24, [](double x, double y) { return x >= 3 && x <= 21 && y >= 8 && y <= 14; }, 4);
    const GenerativeModel model(solid.target);
    const VectorImage before = s.image;
    SmoothParams p = allPhases(MeasurementType::Generative);
    p.maxCgIterations = 150;
    p.maxNodeStepPx = 1.0;
    p.airPressureWeight = 0.0;  // a pressão do ar segura a área do vão (4 nós < 8) e ele nunca fecharia
    const cnx::SmoothResult r = ContourSmoother(p, &model).run(s.image);
    int rejects = 0;
    for (const cnx::PhaseStats& ph : r.phases) {
        rejects += ph.crossingRejects;
        CHECK(ph.energyEnd <= ph.energyBegin + 1e-9);
    }
    std::printf("  barras: iterações desfeitas pelo fiscal: %d (por fase %d, %d, %d)\n", rejects, r.phases[0].crossingRejects,
                r.phases[1].crossingRejects, r.phases[2].crossingRejects);
    for (const cnx::PhaseStats& ph : r.phases) {
        std::printf("    fase: %d iterações, estado %d, nós livres %d, E %.4f -> %.4f\n", ph.iterations, static_cast<int>(ph.status),
                    ph.freeNodes, ph.energyBegin, ph.energyEnd);
    }
    CHECK(rejects > 0);
    // Com a quarentena o CG segue adiante depois de cada rejeição; sem ela repetiria o mesmo passo
    // inválido até esgotar o orçamento de iterações (150 por fase).
    CHECK(rejects < 10);
    checkInvariants(before, s.image, "barras");
}

// ================================================================ argumentos, cancelamento, determinismo

CNX_TEST(argument_validation_and_cancellation) {
    SplitMix64 rng{503};
    cnxtest::VectorScene s = cnxtest::vectorScene(14, 10, 3, 10, rng);
    // Fase generativa sem modelo.
    CHECK_THROWS(std::invalid_argument, (void)ContourSmoother(allPhases(MeasurementType::Generative)).run(s.image));
    // Prior não implementado e parâmetros inválidos.
    SmoothParams bad;
    bad.priorType = 2;
    CHECK_THROWS(std::invalid_argument, ContourSmoother(bad, nullptr));
    bad = SmoothParams{};
    bad.maxNodeStepPx = 0.0;
    CHECK_THROWS(std::invalid_argument, ContourSmoother(bad, nullptr));
    // Contornos que já se cruzam na entrada.
    cnxtest::VectorScene crossed = cnxtest::vectorScene(14, 10, 3, 10, rng);
    cnxtest::jitterNodes(crossed.image, 3.0, false, rng);
    CHECK(!cnx::CrossingGuard::find(crossed.image).empty());
    CHECK_THROWS(std::invalid_argument, (void)ContourSmoother(SmoothParams{}).run(crossed.image));
    // Segmento já colapsado na entrada (nós quase coincidentes).
    cnxtest::VectorScene collapsed = cnxtest::vectorScene(14, 10, 3, 10, rng);
    const auto nodes = collapsed.image.edgeNodes(0);
    collapsed.image.setPosition(nodes[1], collapsed.image.position(nodes[0]) + Vec2{0.01, 0.0});
    CHECK_THROWS(std::invalid_argument, (void)ContourSmoother(SmoothParams{}).run(collapsed.image));
    // Cancelamento.
    std::stop_source source;
    source.request_stop();
    CHECK_THROWS(cnx::CancelTaskException,
                 (void)ContourSmoother(SmoothParams{}).run(s.image, cnx::TaskContext(source.get_token(), nullptr, cnx::ProgressRange{})));
}

CNX_TEST(runs_are_deterministic_and_leave_no_flags_behind_when_puncturing_is_off) {
    SplitMix64 rng{504};
    const cnxtest::VectorScene s = cnxtest::vectorScene(20, 14, 3, 25, rng);
    SmoothParams p;
    p.maxCgIterations = 60;
    p.punctureCornerThresh = 0.0;
    VectorImage a = s.image;
    VectorImage b = s.image;
    (void)ContourSmoother(p).run(a);
    (void)ContourSmoother(p).run(b);
    bool same = true;
    for (NodeId n = 0; n < static_cast<NodeId>(a.nodeCount()); ++n) {
        same = same && a.position(n) == b.position(n) && a.flags(n) == b.flags(n);
        CHECK((a.flags(n) & cnx::NodeFlag::Corner) == (s.image.flags(n) & cnx::NodeFlag::Corner));
    }
    CHECK(same);
}

int main() { return cnxtest::runAll(); }
