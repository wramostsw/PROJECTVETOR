// tests/unit/logo_scenes.h — logos sintéticos com a verdade conhecida: formas analíticas desenhadas com
// antialiasing de filtro-caixa exato por superamostragem (o mesmo antialiasing que o GenerativeModel supõe),
// letras de traço fino, cantos vivos, anel, triângulo agudo e fio de 2,4 px (docs §1.25).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnxtest {

struct Shape {
    std::function<bool(double, double)> inside;
    sla::Rgba8 color;
};

struct LogoScene {
    std::string name;
    int width = 0;
    int height = 0;
    sla::Rgba8 background;
    std::vector<Shape> shapes;  ///< a última que contém o ponto pinta
    int inks = 0;               ///< cores distintas da cena (fundo incluído)
};

inline std::function<bool(double, double)> rectShape(double x0, double y0, double x1, double y1) {
    return [=](double x, double y) { return x >= x0 && x < x1 && y >= y0 && y < y1; };
}
inline std::function<bool(double, double)> diskShape(double cx, double cy, double r) {
    return [=](double x, double y) { return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r; };
}
/// Coroa [r0, r1] com um filtro extra (meio anel, abertura de um "c"…).
inline std::function<bool(double, double)> ringShape(double cx, double cy, double r0, double r1, std::function<bool(double, double)> keep = {}) {
    return [=](double x, double y) {
        const double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        return d2 >= r0 * r0 && d2 <= r1 * r1 && (!keep || keep(x - cx, y - cy));
    };
}
/// Traço reto de meia largura hw (extremidades retas, como um pincel).
inline std::function<bool(double, double)> strokeShape(double ax, double ay, double bx, double by, double hw) {
    return [=](double x, double y) {
        const double dx = bx - ax;
        const double dy = by - ay;
        const double len = std::hypot(dx, dy);
        const double t = ((x - ax) * dx + (y - ay) * dy) / len;
        const double n = ((x - ax) * dy - (y - ay) * dx) / len;
        return t >= 0.0 && t <= len && std::abs(n) <= hw;
    };
}
inline std::function<bool(double, double)> polygonShape(std::vector<sla::Vec2> pts) {
    return [pts = std::move(pts)](double x, double y) {
        bool in = false;
        for (std::size_t i = 0, j = pts.size() - 1; i < pts.size(); j = i++) {
            if (((pts[i].y > y) != (pts[j].y > y)) && x < (pts[j].x - pts[i].x) * (y - pts[i].y) / (pts[j].y - pts[i].y) + pts[i].x) in = !in;
        }
        return in;
    };
}

/// Renderiza com ss × ss amostras (a média das cores: todas opacas).
inline sla::Array2D<sla::Rgba8> renderScene(const LogoScene& s, int ss = 8) {
    sla::Array2D<sla::Rgba8> img(s.width, s.height);
    for (int y = 0; y < s.height; ++y) {
        for (int x = 0; x < s.width; ++x) {
            double r = 0, g = 0, b = 0;
            for (int j = 0; j < ss; ++j) {
                for (int i = 0; i < ss; ++i) {
                    const double px = x + (i + 0.5) / ss;
                    const double py = y + (j + 0.5) / ss;
                    sla::Rgba8 c = s.background;
                    for (const Shape& sh : s.shapes) {
                        if (sh.inside(px, py)) c = sh.color;
                    }
                    r += c.r;
                    g += c.g;
                    b += c.b;
                }
            }
            const double n = static_cast<double>(ss) * ss;
            img(x, y) = {static_cast<std::uint8_t>(std::lround(r / n)), static_cast<std::uint8_t>(std::lround(g / n)), static_cast<std::uint8_t>(std::lround(b / n)), 255};
        }
    }
    return img;
}

/// PNG RGB de 8 bits com blocos armazenados (sem compressão), só para olhar a cena num visualizador.
inline std::string encodePng(const sla::Array2D<sla::Rgba8>& img) {
    const auto be32 = [](std::string& s, std::uint32_t v) {
        for (int k = 3; k >= 0; --k) s.push_back(static_cast<char>(v >> (8 * k)));
    };
    const auto chunk = [&](const char* type, const std::string& body) {
        std::string c;
        be32(c, static_cast<std::uint32_t>(body.size()));
        std::string typed(type, 4);
        typed += body;
        c += typed;
        std::uint32_t crc = 0xFFFFFFFFu;
        for (const char ch : typed) {
            crc ^= static_cast<std::uint8_t>(ch);
            for (int k = 0; k < 8; ++k) crc = (crc & 1u) ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
        }
        be32(c, ~crc);
        return c;
    };
    std::string raw;
    for (int y = 0; y < img.height(); ++y) {
        raw.push_back(0);
        for (int x = 0; x < img.width(); ++x) {
            raw.push_back(static_cast<char>(img(x, y).r));
            raw.push_back(static_cast<char>(img(x, y).g));
            raw.push_back(static_cast<char>(img(x, y).b));
        }
    }
    std::string z = "\x78\x01";
    for (std::size_t pos = 0; pos < raw.size() || pos == 0;) {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - pos);
        z.push_back(pos + n >= raw.size() ? 1 : 0);
        z.push_back(static_cast<char>(n));
        z.push_back(static_cast<char>(n >> 8));
        z.push_back(static_cast<char>(~n));
        z.push_back(static_cast<char>(~n >> 8));
        z += raw.substr(pos, n);
        pos += n;
        if (n == 0) break;
    }
    std::uint32_t a = 1;
    std::uint32_t b = 0;
    for (const char ch : raw) {
        a = (a + static_cast<std::uint8_t>(ch)) % 65521u;
        b = (b + a) % 65521u;
    }
    be32(z, (b << 16) | a);
    std::string ihdr;
    be32(ihdr, static_cast<std::uint32_t>(img.width()));
    be32(ihdr, static_cast<std::uint32_t>(img.height()));
    ihdr += std::string("\x08\x02\x00\x00\x00", 5);
    return std::string("\x89PNG\r\n\x1a\n", 8) + chunk("IHDR", ihdr) + chunk("IDAT", z) + chunk("IEND", "");
}

/// "cnx" branco sobre azul, com um ponto vermelho: traços de 5,8 px, arcos, diagonais.
inline LogoScene lettersScene() {
    const sla::Rgba8 blue{30, 70, 160, 255};
    const sla::Rgba8 white{255, 255, 255, 255};
    LogoScene s{"letras", 120, 64, blue, {}, 3};
    // c: coroa de r 7,4..13,2 com a abertura (±40°) à direita.
    s.shapes.push_back({ringShape(22.3, 31.8, 7.4, 13.2, [](double dx, double dy) { return !(dx > 0 && std::abs(dy) < dx * 0.8391); }), white});
    // n: duas hastes e um arco (meia coroa de r 4,4..10,2).
    s.shapes.push_back({rectShape(42.3, 26.5, 48.1, 44.2), white});
    s.shapes.push_back({rectShape(56.9, 26.5, 62.7, 44.2), white});
    s.shapes.push_back({ringShape(52.5, 26.5, 4.4, 10.2, [](double, double dy) { return dy < 0; }), white});
    // x: duas diagonais de 5,8 px.
    s.shapes.push_back({strokeShape(72.4, 22.1, 90.6, 44.3, 2.9), white});
    s.shapes.push_back({strokeShape(90.6, 22.1, 72.4, 44.3, 2.9), white});
    s.shapes.push_back({diskShape(104.6, 24.3, 4.7), sla::Rgba8{220, 40, 40, 255}});
    return s;
}

/// Retângulo de cantos vivos, anel, triângulo agudo, fio de 2,4 px e disco, em fundo branco.
inline LogoScene shapesScene() {
    LogoScene s{"formas", 96, 72, sla::Rgba8{255, 255, 255, 255}, {}, 6};
    s.shapes.push_back({rectShape(8.5, 6.5, 40.5, 30.5), sla::Rgba8{220, 40, 40, 255}});
    s.shapes.push_back({ringShape(66.0, 20.0, 7.5, 14.0), sla::Rgba8{30, 70, 160, 255}});
    s.shapes.push_back({polygonShape({{10.5, 66.0}, {34.5, 40.5}, {50.5, 66.0}}), sla::Rgba8{30, 150, 70, 255}});
    s.shapes.push_back({rectShape(56.0, 38.2, 90.0, 40.6), sla::Rgba8{20, 20, 20, 255}});
    s.shapes.push_back({diskShape(74.0, 54.0, 9.3), sla::Rgba8{250, 190, 20, 255}});
    return s;
}

}  // namespace cnxtest
