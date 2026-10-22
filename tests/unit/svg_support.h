// tests/unit/svg_support.h — leitor e rasterizador de SVG para os testes (independentes do exportador): caminhos
// M/L/C/Z, cúbicas discretizadas em 256 segmentos, regra nonzero, e um rasterizador com superamostragem para
// comparar o vetor com a imagem de origem (docs §1.24, §1.25).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnx/export/Exporter.h"
#include "cnx/export/SvgWriter.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnxtest::svg {

using cnx::Exporter;
using cnx::ExporterSettings;
using sla::Vec2;

struct Path {
    std::string fill;
    std::string opacity;
    std::string stroke;
    std::string strokeWidth;
    std::string group;
    std::vector<std::vector<Vec2>> rings;  ///< subcaminhos discretizados (fechados)
    std::size_t cubics = 0;
    std::size_t lines = 0;
};

struct Svg {
    std::string width, height, viewBox, title;
    std::vector<Path> paths;
    std::vector<std::string> groups;
    bool wellFormed = false;
};

std::string attribute(const std::string& tag, const std::string& name) {
    const std::string key = " " + name + "=\"";
    const std::size_t a = tag.find(key);
    if (a == std::string::npos) return {};
    const std::size_t b = tag.find('"', a + key.size());
    return tag.substr(a + key.size(), b - a - key.size());
}

/// Caminho "M x y L x y C x1 y1 x2 y2 x y Z ...": cada cÃºbica vira 256 segmentos (desvio â‰ª 0,001 px aqui).
void parsePathData(const std::string& d, Path& p) {
    std::istringstream in(d);
    in.imbue(std::locale::classic());
    std::string cmd;
    Vec2 cur;
    while (in >> cmd) {
        if (cmd == "M") {
            p.rings.emplace_back();
            in >> cur.x >> cur.y;
            p.rings.back().push_back(cur);
        } else if (cmd == "L") {
            in >> cur.x >> cur.y;
            p.rings.back().push_back(cur);
            ++p.lines;
        } else if (cmd == "C") {
            std::array<Vec2, 4> c{cur};
            for (std::size_t k = 1; k < 4; ++k) in >> c[k].x >> c[k].y;
            for (int i = 1; i <= 256; ++i) {
                const double t = i / 256.0;
                const double u = 1.0 - t;
                p.rings.back().push_back(c[0] * (u * u * u) + c[1] * (3 * u * u * t) + c[2] * (3 * u * t * t) + c[3] * (t * t * t));
            }
            cur = c[3];
            ++p.cubics;
        } else if (cmd == "Z") {
            // o anel fecha no primeiro ponto (o Ãºltimo ponto escrito jÃ¡ Ã© o primeiro)
        } else {
            throw std::runtime_error("comando de caminho inesperado: " + cmd);
        }
    }
}

Svg parseSvg(const std::string& text) {
    Svg s;
    std::string group;
    int depth = 0;
    bool svgOpen = false;
    bool svgClosed = false;
    for (std::size_t pos = text.find('<'); pos != std::string::npos; pos = text.find('<', pos + 1)) {
        const std::size_t end = text.find('>', pos);
        const std::string tag = text.substr(pos, end - pos + 1);
        if (tag.rfind("<svg", 0) == 0) {
            svgOpen = true;
            s.width = attribute(tag, "width");
            s.height = attribute(tag, "height");
            s.viewBox = attribute(tag, "viewBox");
        } else if (tag == "</svg>") {
            svgClosed = true;
        } else if (tag.rfind("<title>", 0) == 0) {
            s.title = text.substr(end + 1, text.find("</title>", end) - end - 1);
        } else if (tag.rfind("<g ", 0) == 0) {
            group = attribute(tag, "id");
            s.groups.push_back(group);
            ++depth;
        } else if (tag == "</g>") {
            group.clear();
            --depth;
        } else if (tag.rfind("<path", 0) == 0) {
            Path p;
            p.fill = attribute(tag, "fill");
            p.opacity = attribute(tag, "opacity");
            p.stroke = attribute(tag, "stroke");
            p.strokeWidth = attribute(tag, "stroke-width");
            p.group = group;
            parsePathData(attribute(tag, "d"), p);
            s.paths.push_back(std::move(p));
        }
    }
    s.wellFormed = svgOpen && svgClosed && depth == 0 && text.rfind("<?xml", 0) == 0;
    return s;
}

/// NÃºmero de voltas (regra nonzero) do ponto em relaÃ§Ã£o aos anÃ©is do caminho.
int winding(const Path& p, Vec2 q) {
    int w = 0;
    for (const auto& ring : p.rings) {
        for (std::size_t i = 0; i + 1 < ring.size(); ++i) {
            const Vec2 a = ring[i];
            const Vec2 b = ring[i + 1];
            if (a.y <= q.y) {
                if (b.y > q.y && sla::cross(b - a, q - a) > 0) ++w;
            } else if (b.y <= q.y && sla::cross(b - a, q - a) < 0) {
                --w;
            }
        }
    }
    return w;
}

double pathArea(const Path& p) {
    double a = 0.0;
    for (const auto& ring : p.rings) {
        for (std::size_t i = 0; i + 1 < ring.size(); ++i) a += sla::cross(ring[i], ring[i + 1]);
    }
    return 0.5 * a;
}

std::string exportSvg(const cnx::VectorImage& v, const ExporterSettings& s) {
    std::ostringstream out;
    cnx::SvgWriter w(out, s);
    Exporter(v, s).write(w);
    return out.str();
}

/// Rasteriza com ss × ss amostras por pixel (a última forma que cobre a amostra ganha, como num visualizador):
/// devolve o sinal PRÉ-MULTIPLICADO médio (r·a, g·a, b·a, a); amostra sem forma = (0, 0, 0, 0). Só opacidade 1.00.
inline sla::Array2D<sla::Rgba> rasterize(const Svg& svg, int w, int h, int ss) {
    const int sw = w * ss;
    const int sh = h * ss;
    std::vector<int> owner(static_cast<std::size_t>(sw) * static_cast<std::size_t>(sh), -1);
    std::vector<std::pair<double, int>> hits;
    for (std::size_t pi = 0; pi < svg.paths.size(); ++pi) {
        const Path& p = svg.paths[pi];
        for (int sy = 0; sy < sh; ++sy) {
            const double y = (sy + 0.5) / ss;
            hits.clear();
            for (const auto& ring : p.rings) {
                for (std::size_t i = 0; i + 1 < ring.size(); ++i) {
                    const Vec2 a = ring[i];
                    const Vec2 b = ring[i + 1];
                    if ((a.y <= y) == (b.y <= y)) continue;
                    hits.emplace_back(a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x), b.y > a.y ? 1 : -1);
                }
            }
            std::sort(hits.begin(), hits.end());
            int wind = 0;
            for (std::size_t k = 0; k + 1 < hits.size(); ++k) {
                wind += hits[k].second;
                if (wind == 0) continue;
                const int x0 = std::max(0, static_cast<int>(std::ceil(hits[k].first * ss - 0.5)));
                const int x1 = std::min(sw, static_cast<int>(std::ceil(hits[k + 1].first * ss - 0.5)));
                for (int sx = x0; sx < x1; ++sx) owner[static_cast<std::size_t>(sy) * static_cast<std::size_t>(sw) + static_cast<std::size_t>(sx)] = static_cast<int>(pi);
            }
        }
    }
    std::vector<sla::Rgba> colors;
    for (const Path& p : svg.paths) {
        const auto hex = [&](std::size_t at) { return static_cast<float>(std::stoi(p.fill.substr(at, 2), nullptr, 16)) / 255.0f; };
        colors.push_back({hex(1), hex(3), hex(5), 1.0f});
    }
    sla::Array2D<sla::Rgba> out(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double r = 0, g = 0, b = 0, a = 0;
            for (int j = 0; j < ss; ++j) {
                for (int i = 0; i < ss; ++i) {
                    const int o = owner[static_cast<std::size_t>(y * ss + j) * static_cast<std::size_t>(sw) + static_cast<std::size_t>(x * ss + i)];
                    if (o < 0) continue;
                    r += colors[static_cast<std::size_t>(o)].r;
                    g += colors[static_cast<std::size_t>(o)].g;
                    b += colors[static_cast<std::size_t>(o)].b;
                    a += 1.0;
                }
            }
            const double n = static_cast<double>(ss) * ss;
            out(x, y) = {static_cast<float>(r / n), static_cast<float>(g / n), static_cast<float>(b / n), static_cast<float>(a / n)};
        }
    }
    return out;
}

/// Erro de uma rasterização contra o alvo (ambos pré-multiplicados, canais em [0, 1]).
struct FitError {
    double rmse = 0.0;      ///< √(média de ‖erro‖² / 4 canais) sobre todos os pixels
    double maxError = 0.0;  ///< maior erro de canal
    int bad = 0;            ///< pixels com algum canal errado em mais de 0,2
    int total = 0;
};

inline FitError compare(const sla::Array2D<sla::Rgba>& got, const sla::Array2D<sla::Rgba>& target) {
    FitError e;
    double sum = 0.0;
    for (int y = 0; y < target.height(); ++y) {
        for (int x = 0; x < target.width(); ++x) {
            const sla::Rgba a = got(x, y);
            const sla::Rgba b = target(x, y);
            const std::array<double, 4> d{double(a.r) - b.r, double(a.g) - b.g, double(a.b) - b.b, double(a.a) - b.a};
            double worst = 0.0;
            for (const double v : d) {
                sum += v * v;
                worst = std::max(worst, std::abs(v));
            }
            e.maxError = std::max(e.maxError, worst);
            e.bad += worst > 0.2 ? 1 : 0;
            ++e.total;
        }
    }
    e.rmse = std::sqrt(sum / (4.0 * e.total));
    return e;
}

}  // namespace cnxtest::svg