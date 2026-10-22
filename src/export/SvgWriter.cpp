// src/export/SvgWriter.cpp — SVG 1.1 em texto, com números de casas fixas independentes de locale.
#include "cnx/export/SvgWriter.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <ostream>

namespace cnx {
namespace {

int channel(float v) { return static_cast<int>(std::lround(std::clamp(static_cast<double>(v), 0.0, 1.0) * 255.0)); }

std::string escapeXml(const std::string& s) {
    std::string out;
    for (const char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out += c;
        }
    }
    return out;
}

/// Menor representação que volta ao mesmo double (larguras como 0.09375, que 2 casas destruiriam).
std::string shortest(double v) {
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v);
    return std::string(buf, r.ptr);
}

}  // namespace

SvgWriter::SvgWriter(std::ostream& out, const ExporterSettings& settings)
    : out_(out), decimals_(settings.coordinateDecimals), pointsPerPixel_(settings.pointsPerPixel) {}

std::string SvgWriter::hexColor(const sla::Rgba& c) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string s = "#";
    for (const float v : {c.r, c.g, c.b}) {
        const int x = channel(v);
        s += kHex[x >> 4];
        s += kHex[x & 15];
    }
    return s;
}

std::string SvgWriter::number(double v, int decimals) {
    if (!std::isfinite(v)) throw ExportError("SVG: coordenada não finita");
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::fixed, decimals);
    std::string s(buf, r.ptr);
    if (s[0] == '-' && s.find_first_not_of("-0.") == std::string::npos) s.erase(0, 1);  // "-0.00" → "0.00"
    return s;
}

void SvgWriter::point(sla::Vec2 p) { out_ << number(p.x, decimals_) << ' ' << number(p.y, decimals_); }

void SvgWriter::beginFile(const DocumentInfo& info) {
    out_ << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    out_ << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"" << number(info.widthPx * pointsPerPixel_, decimals_) << "pt\" height=\""
         << number(info.heightPx * pointsPerPixel_, decimals_) << "pt\" viewBox=\"0 0 " << number(info.widthPx, decimals_) << ' '
         << number(info.heightPx, decimals_) << "\">\n";
    if (!info.title.empty()) out_ << indent_ << "<title>" << escapeXml(info.title) << "</title>\n";
    if (!info.creator.empty()) out_ << indent_ << "<desc>" << escapeXml(info.creator) << "</desc>\n";
}

void SvgWriter::beginGroup(const sla::Rgba& color) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string id = "c_";
    for (const float v : {color.r, color.g, color.b, color.a}) {
        const int x = channel(v);
        id += kHex[x >> 4];
        id += kHex[x & 15];
    }
    out_ << indent_ << "<g id=\"" << id << "\">\n";
    indent_ += "  ";
}

void SvgWriter::endGroup() {
    indent_.resize(indent_.size() - 2);
    out_ << indent_ << "</g>\n";
}

void SvgWriter::beginShape(const sla::Rgba& fill, const StrokeStyle* stroke) {
    out_ << indent_ << "<path fill=\"" << hexColor(fill) << "\" opacity=\"" << number(std::clamp(static_cast<double>(fill.a), 0.0, 1.0), 2) << '"';
    if (stroke != nullptr) {
        out_ << " stroke=\"" << hexColor(stroke->color) << "\" stroke-width=\"" << shortest(stroke->widthPt / pointsPerPixel_) << '"';
    }
    out_ << " d=\"";
    first_ = true;
}

void SvgWriter::moveTo(sla::Vec2 p) {
    out_ << (first_ ? "M " : " M ");
    first_ = false;
    point(p);
}

void SvgWriter::lineTo(sla::Vec2 p) {
    out_ << " L ";
    point(p);
}

void SvgWriter::curveTo(sla::Vec2 c1, sla::Vec2 c2, sla::Vec2 p) {
    out_ << " C ";
    point(c1);
    out_ << ' ';
    point(c2);
    out_ << ' ';
    point(p);
}

void SvgWriter::closeSubpath() { out_ << " Z"; }

void SvgWriter::endShape() { out_ << "\"/>\n"; }

void SvgWriter::endFile() { out_ << "</svg>\n"; }

}  // namespace cnx
