// src/export/Exporter.cpp — ordem de desenho, laços com as curvas compartilhadas, camadas e arquivo atômico.
#include "cnx/export/Exporter.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <system_error>

#include "cnx/engine/TaskContext.h"
#include "cnx/export/SvgWriter.h"

namespace cnx {

std::unique_ptr<IShapeWriter> makeWriter(ExportFormat format, std::ostream& out, const ExporterSettings& settings) {
    switch (format) {
        case ExportFormat::Svg: return std::make_unique<SvgWriter>(out, settings);
        case ExportFormat::Eps: throw ExportError("exportação EPS ainda não implementada");
        case ExportFormat::Ai: throw ExportError("exportação AI ainda não implementada");
        case ExportFormat::Pdf: throw ExportError("exportação PDF ainda não implementada");
        case ExportFormat::Dxf: throw ExportError("exportação DXF ainda não implementada");
    }
    throw ExportError("formato de exportação desconhecido");
}

Exporter::Exporter(const VectorImage& image, ExporterSettings settings) : image_(image), settings_(std::move(settings)) {
    if (settings_.coordinateDecimals < 0 || settings_.coordinateDecimals > 9) throw ExportError("Exporter: coordinateDecimals fora de [0, 9]");
    if (!(settings_.strokeWidthPt >= 0.0) || !std::isfinite(settings_.strokeWidthPt)) throw ExportError("Exporter: strokeWidthPt < 0 ou não finito");
    if (!(settings_.pointsPerPixel > 0.0) || !std::isfinite(settings_.pointsPerPixel)) throw ExportError("Exporter: pointsPerPixel <= 0 ou não finito");
}

std::vector<FaceId> Exporter::drawOrder() const {
    const auto n = static_cast<FaceId>(image_.faceCount());
    std::vector<int> depth(static_cast<std::size_t>(n), -1);
    // Profundidade pela cadeia de Face::parent (sem ciclos: VectorImage::validate garante).
    const auto depthOf = [&](FaceId f) {
        int d = 0;
        FaceId p = image_.face(f).parent;
        while (p != kInvalidId && p != kExteriorFace) {
            ++d;
            p = image_.face(p).parent;
        }
        return d;
    };
    std::vector<FaceId> order;
    for (FaceId f = 1; f < n; ++f) {
        const Face& fc = image_.face(f);
        if (fc.outerLoop == kInvalidId || !(fc.color.a > 0.0f) || !settings_.background.isShapeVisible(f)) continue;
        depth[static_cast<std::size_t>(f)] = depthOf(f);
        order.push_back(f);
    }
    std::stable_sort(order.begin(), order.end(), [&](FaceId a, FaceId b) { return depth[static_cast<std::size_t>(a)] < depth[static_cast<std::size_t>(b)]; });
    return order;
}

void Exporter::emitLoop(IShapeWriter& writer, HalfEdgeId first) const {
    bool started = false;
    HalfEdgeId h = first;
    do {
        const EdgeId e = edgeOf(h);
        const bool rev = isReversed(h);
        const auto curves = image_.edgeCurves(e);
        if (!curves.empty()) {
            const std::size_t m = curves.size();
            for (std::size_t i = 0; i < m; ++i) {
                const sla::CurveSegment s = rev ? curves[m - 1 - i].reversed() : curves[i];
                if (!started) {
                    writer.moveTo(s.start());
                    started = true;
                }
                if (s.kind == sla::SegmentKind::Line) {
                    writer.lineTo(s.end());
                } else {
                    writer.curveTo(s.curve.p[1], s.curve.p[2], s.end());
                }
            }
        } else {
            // Sem curvas (antes do ajuste): a poligonal dos nós.
            const auto nodes = image_.edgeNodes(e);
            const std::size_t m = nodes.size();
            for (std::size_t i = 0; i + 1 < m; ++i) {
                const NodeId a = rev ? nodes[m - 1 - i] : nodes[i];
                const NodeId b = rev ? nodes[m - 2 - i] : nodes[i + 1];
                if (!started) {
                    writer.moveTo(image_.position(a));
                    started = true;
                }
                writer.lineTo(image_.position(b));
            }
        }
        h = image_.next(h);
    } while (h != first);
    writer.closeSubpath();
}

void Exporter::emitFace(IShapeWriter& writer, FaceId f) const {
    const Face& fc = image_.face(f);
    StrokeStyle stroke{fc.color, settings_.strokeWidthPt};
    writer.beginShape(fc.color, settings_.shapeStroking ? &stroke : nullptr);
    emitLoop(writer, fc.outerLoop);
    if (settings_.cakeStackMode != CakeStackMode::Stacked) {
        for (const HalfEdgeId hole : image_.holeLoops(f)) emitLoop(writer, hole);
    }
    writer.endShape();
}

void Exporter::write(IShapeWriter& writer, const std::stop_token& stop) const { writeDocument(writer, stop, {}); }

void Exporter::writeDocument(IShapeWriter& writer, const std::stop_token& stop, std::string title) const {
    DocumentInfo info;
    info.widthPx = image_.width();
    info.heightPx = image_.height();
    info.pointsPerPixel = settings_.pointsPerPixel;
    info.title = std::move(title);
    writer.beginFile(info);
    const std::vector<FaceId> order = drawOrder();
    const auto check = [&] {
        if (stop.stop_requested()) throw CancelTaskException{};
    };
    if (settings_.cakeStackMode == CakeStackMode::CompoundByColor) {
        // Grupos por cor exata, na ordem da 1ª aparição; dentro do grupo, a ordem de desenho.
        std::vector<sla::Rgba> colors;
        for (const FaceId f : order) {
            const sla::Rgba& c = image_.face(f).color;
            if (std::find(colors.begin(), colors.end(), c) == colors.end()) colors.push_back(c);
        }
        for (const sla::Rgba& c : colors) {
            writer.beginGroup(c);
            for (const FaceId f : order) {
                if (!(image_.face(f).color == c)) continue;
                check();
                emitFace(writer, f);
            }
            writer.endGroup();
        }
    } else {
        for (const FaceId f : order) {
            check();
            emitFace(writer, f);
        }
    }
    writer.endFile();
}

void Exporter::writeFile(ExportFormat format, const std::filesystem::path& file, const std::stop_token& stop) const {
    std::filesystem::path tmp = file;
    tmp += ".cnxtmp";
    try {
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) throw ExportError("não foi possível criar " + tmp.string());
            const std::unique_ptr<IShapeWriter> writer = makeWriter(format, out, settings_);
            writeDocument(*writer, stop, file.stem().string());
            out.flush();
            if (!out) throw ExportError("falha ao escrever " + tmp.string());
        }
        std::error_code ec;
        std::filesystem::rename(tmp, file, ec);
        if (ec) throw ExportError("não foi possível substituir " + file.string() + ": " + ec.message());
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        throw;
    }
}

}  // namespace cnx
