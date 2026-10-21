// tests/unit/header_smoke_test.cpp — verifica os trechos inline dos cabeçalhos públicos e garante
// que todos compilam juntos. Sem framework: código de saída = número de falhas.
// Substituir por GoogleTest/Catch2 quando o CMake entrar.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string_view>

#include "cnx/engine/CoreEngine.h"
#include "cnx/engine/Fingerprint.h"
#include "cnx/engine/Parameters.h"
#include "cnx/engine/TaskContext.h"
#include "cnx/export/Exporter.h"
#include "cnx/segment/PixelMap.h"
#include "cnx/vector/VectorImage.h"
#include "sla/Array.h"
#include "sla/BezierCurve.h"
#include "sla/Vec.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FALHOU: %s\n", what);
        ++failures;
    }
}

// ------------------------------------------------------------------ tempo de compilação
static_assert(cnx::twin(cnx::halfEdge(3, false)) == cnx::halfEdge(3, true));
static_assert(cnx::edgeOf(cnx::halfEdge(41, true)) == 41);
static_assert(cnx::isReversed(cnx::twin(cnx::halfEdge(0, false))));
static_assert(sla::BezierCurve::fromLine({0, 0}, {3, 0}).eval(0.5) == sla::Vec2{1.5, 0});
static_assert(sla::outwardNormal({1, 0}) == sla::Vec2{0, -1});
static_assert(cnx::kStageProgress[cnx::stageIndex(cnx::Stage::SmoothContours)].begin == 0.20);
static_assert(cnx::kStageProgress.back().end == 1.0);
static_assert(cnx::isAntialiasedLogo(cnx::ImageProfile::LogoLowAA));
static_assert(!cnx::isAntialiasedLogo(cnx::ImageProfile::LogoHigh));
static_assert(cnx::isPhoto(cnx::ImageProfile::PhotoMed) && !cnx::isPhoto(cnx::ImageProfile::Custom));
static_assert(sla::toRgba8(sla::toRgba(sla::Rgba8{12, 200, 255, 0})) == sla::Rgba8{12, 200, 255, 0});

// ------------------------------------------------------------------ execução
void testFingerprint() {
    cnx::Fingerprint a;
    cnx::Fingerprint b;
    a.add(1.0).add(std::string_view("ContourSmoother::prior_strengths")).add(42);
    b.add(1.0).add(std::string_view("ContourSmoother::prior_strengths")).add(42);
    check(a == b, "Fingerprint determinístico");

    cnx::Fingerprint negZero;
    cnx::Fingerprint posZero;
    negZero.add(-0.0);
    posZero.add(0.0);
    check(negZero == posZero, "Fingerprint: -0.0 == +0.0");

    cnx::Fingerprint up1;
    cnx::Fingerprint up2;
    up1.add(1);
    up2.add(2);
    cnx::Fingerprint down1;
    cnx::Fingerprint down2;
    down1.add(up1).add(5.0);
    down2.add(up2).add(5.0);
    check(!(down1 == down2), "Fingerprint: mudança a montante propaga");

    cnx::Fingerprint s1;
    cnx::Fingerprint s2;
    s1.add(std::string_view("ab")).add(std::string_view("c"));
    s2.add(std::string_view("a")).add(std::string_view("bc"));
    check(!(s1 == s2), "Fingerprint: fronteira entre strings");

    check(cnx::Fingerprint{}.value() == 14695981039346656037ull, "Fingerprint: base FNV-1a");
}

void testMultiplicativeScale() {
    auto f = [](double v) { return cnx::multiplicativeScale(v, 0, 10, 1e-4, 1e-1); };
    check(std::abs(f(0) - 1e-4) < 1e-15, "escala: mínimo");
    check(std::abs(f(10) - 1e-1) < 1e-12, "escala: máximo");
    check(std::abs(f(5) - std::sqrt(1e-4 * 1e-1)) < 1e-12, "escala: ponto médio = média geométrica");
    check(std::abs(f(3) / f(2) - f(8) / f(7)) < 1e-12, "escala: passos iguais, mesmo fator");
    check(f(-5) == f(0) && f(15) == f(10), "escala: saturação");

    bool threw = false;
    try {
        (void)cnx::multiplicativeScale(1, 0, 10, 0.0, 1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "escala: rejeita pMin = 0");
}

void testBezier() {
    const sla::BezierCurve c{{0, 0}, {1, 3}, {4, 3}, {5, 0}};
    const auto [left, right] = c.split(0.3);
    check(left.p[3] == right.p[0], "split: continuidade");
    for (double t : {0.0, 0.25, 0.5, 0.9, 1.0}) {
        check(sla::norm(left.eval(t) - c.eval(0.3 * t)) < 1e-12, "split: metade esquerda");
        check(sla::norm(right.eval(t) - c.eval(0.3 + 0.7 * t)) < 1e-12, "split: metade direita");
        check(sla::norm(c.reversed().eval(t) - c.eval(1.0 - t)) < 1e-12, "reversed: B(1 - t)");
    }

    const double h = 1e-6;
    for (double t : {0.1, 0.5, 0.8}) {
        const sla::Vec2 fd1 = (c.eval(t + h) - c.eval(t - h)) / (2 * h);
        const sla::Vec2 fd2 = (c.d1(t + h) - c.d1(t - h)) / (2 * h);
        check(sla::norm(fd1 - c.d1(t)) < 1e-6, "d1 vs. diferenças finitas");
        check(sla::norm(fd2 - c.d2(t)) < 1e-6, "d2 vs. diferenças finitas");
    }

    for (double t : {0.0, 0.37, 1.0}) {
        const auto b = sla::BezierCurve::basis(t);
        const auto b1 = sla::BezierCurve::basisD1(t);
        const auto b2 = sla::BezierCurve::basisD2(t);
        check(std::abs(b[0] + b[1] + b[2] + b[3] - 1.0) < 1e-15, "Bernstein: partição da unidade");
        check(std::abs(b1[0] + b1[1] + b1[2] + b1[3]) < 1e-14, "Bernstein: soma de b' = 0");
        check(std::abs(b2[0] + b2[1] + b2[2] + b2[3]) < 1e-13, "Bernstein: soma de b'' = 0");
    }

    const sla::CurveSegment seg{sla::SegmentKind::Line, sla::BezierCurve::fromLine({1, 1}, {4, 5})};
    check(seg.reversed().start() == seg.end() && seg.reversed().kind == sla::SegmentKind::Line,
          "CurveSegment::reversed");

    sla::Box2 box;
    check(box.empty(), "Box2 vazia por padrão");
    box.expand({1, 2});
    check(!box.empty(), "Box2 após expand");
}

void testArrays() {
    sla::Array2D<int> img(7, 5, -1);
    check(img.width() == 7 && img.height() == 5 && img.size() == 35, "Array2D: dimensões");
    img(6, 4) = 9;
    check(img.at(6, 4) == 9 && img.row(4)[6] == 9, "Array2D: indexação por linhas");

    bool threw = false;
    try {
        (void)img.at(7, 0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    check(threw, "Array2D::at verifica limites");

    sla::Array<double> v(100);
    check(reinterpret_cast<std::uintptr_t>(v.data()) % 64 == 0, "Array: alinhamento de 64 B");
}

void testTaskContext() {
    std::stop_source source;
    double last = -1.0;
    std::shared_ptr<const cnx::TaskContext::ProgressFn> fn =
        std::make_shared<cnx::TaskContext::ProgressFn>([&last](double g) { last = g; });
    cnx::TaskContext ctx(source.get_token(), fn, cnx::kStageProgress[cnx::stageIndex(cnx::Stage::SmoothContours)]);

    ctx.checkpoint(0.5);
    check(std::abs(last - 0.5) < 1e-12, "progresso: 50 % da suavização = 50 % global");
    cnx::TaskContext sub = ctx.sub(0.5, 1.0);
    sub.report(1.0);
    check(std::abs(last - 0.8) < 1e-12, "progresso: fim da subfaixa = 80 % global");

    source.request_stop();
    bool cancelled = false;
    try {
        ctx.checkpoint(0.9);
    } catch (const cnx::CancelTaskException&) {
        cancelled = true;
    }
    check(cancelled, "cancelamento cooperativo");
}

void testDefaults() {
    const cnx::ExporterSettings s;
    check(s.strokeWidthPt == 0.09375 && s.cakeStackMode == cnx::CakeStackMode::Compound, "ExporterSettings padrão");
    check(!s.background.canUndo() && !s.background.canRedo() && s.background.hiddenFaces().empty(),
          "BackgroundSelection inicial");

    const cnx::EngineParameters p;
    check(p.smooth.spacingWeight[0] == 5.0 && p.smooth.spacingWeight[2] == 0.5, "SmoothParams: 5.0 → 0.5 → 0.5");
    check(p.fit.statThreshInitial == 1e-8 && p.fit.maxRounds == 20, "FitParams padrão");

    const cnx::VectorImage empty;
    check(empty.nodeCount() == 0 && empty.edgeCount() == 0 && !empty.hasCurves(), "VectorImage vazia");

    const cnx::RegionForest forest;
    check(forest.capacity() == 0 && forest.rootCount() == 0, "RegionForest vazia");
}

}  // namespace

int main() {
    testFingerprint();
    testMultiplicativeScale();
    testBezier();
    testArrays();
    testTaskContext();
    testDefaults();
    if (failures == 0) std::printf("OK: todos os testes passaram\n");
    return failures;
}
