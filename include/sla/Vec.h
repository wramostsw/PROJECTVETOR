// sla/Vec.h — vetores 2D e cores RGBA do núcleo numérico.
//
// Convenções válidas em todo o motor (docs/ARQUITETURA.md §2.1):
//   * x cresce para a direita, y para baixo; o pixel (i, j) ocupa [i, i+1) × [j, j+1).
//   * Área assinada (shoelace) A = ½ Σ (x_k y_{k+1} − x_{k+1} y_k): laços externos têm A > 0,
//     furos têm A < 0.
#pragma once

#include <cmath>
#include <cstdint>

namespace sla {

/// Vetor/ponto 2D em precisão dupla. Agregado trivial (16 bytes), passado por valor.
struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    constexpr Vec2() noexcept = default;
    constexpr Vec2(double x_, double y_) noexcept : x(x_), y(y_) {}

    constexpr Vec2 operator+(Vec2 o) const noexcept { return {x + o.x, y + o.y}; }
    constexpr Vec2 operator-(Vec2 o) const noexcept { return {x - o.x, y - o.y}; }
    constexpr Vec2 operator-() const noexcept { return {-x, -y}; }
    constexpr Vec2 operator*(double s) const noexcept { return {x * s, y * s}; }
    constexpr Vec2 operator/(double s) const noexcept { return {x / s, y / s}; }
    constexpr Vec2& operator+=(Vec2 o) noexcept { x += o.x; y += o.y; return *this; }
    constexpr Vec2& operator-=(Vec2 o) noexcept { x -= o.x; y -= o.y; return *this; }
    constexpr Vec2& operator*=(double s) noexcept { x *= s; y *= s; return *this; }

    friend constexpr bool operator==(const Vec2&, const Vec2&) noexcept = default;
};

constexpr Vec2 operator*(double s, Vec2 v) noexcept { return v * s; }
constexpr double dot(Vec2 a, Vec2 b) noexcept { return a.x * b.x + a.y * b.y; }
/// Componente z do produto vetorial; sinal consistente com a área de shoelace.
constexpr double cross(Vec2 a, Vec2 b) noexcept { return a.x * b.y - a.y * b.x; }
constexpr double norm2(Vec2 a) noexcept { return dot(a, a); }
inline double norm(Vec2 a) noexcept { return std::sqrt(norm2(a)); }
constexpr Vec2 lerp(Vec2 a, Vec2 b, double t) noexcept { return a + (b - a) * t; }

/// Normal externa não normalizada de um segmento de direção d, para a face de área positiva:
/// n = (Δy, −Δx), ‖n‖ = ‖d‖. Base da derivada de forma do GenerativeModel (§2.3).
constexpr Vec2 outwardNormal(Vec2 d) noexcept { return {d.y, -d.x}; }

/// Cor RGBA em ponto flutuante, canais em [0, 1], alfa não pré-multiplicado.
/// O espaço de mistura (sRGB codificado ou linear) é decidido pelo estágio que a usa.
struct Rgba {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;

    constexpr Rgba() noexcept = default;
    constexpr Rgba(float r_, float g_, float b_, float a_ = 1.0f) noexcept : r(r_), g(g_), b(b_), a(a_) {}

    constexpr Rgba operator+(const Rgba& o) const noexcept { return {r + o.r, g + o.g, b + o.b, a + o.a}; }
    constexpr Rgba operator-(const Rgba& o) const noexcept { return {r - o.r, g - o.g, b - o.b, a - o.a}; }
    constexpr Rgba operator*(float s) const noexcept { return {r * s, g * s, b * s, a * s}; }
    constexpr Rgba& operator+=(const Rgba& o) noexcept { r += o.r; g += o.g; b += o.b; a += o.a; return *this; }

    friend constexpr bool operator==(const Rgba&, const Rgba&) noexcept = default;
};

/// Produto interno nos 4 canais, acumulado em double (métrica RGB + alfa da paleta e do modelo gerador).
constexpr double dot(const Rgba& p, const Rgba& q) noexcept {
    return double(p.r) * q.r + double(p.g) * q.g + double(p.b) * q.b + double(p.a) * q.a;
}
constexpr double distance2(const Rgba& p, const Rgba& q) noexcept {
    const Rgba d = p - q;
    return dot(d, d);
}

/// Pixel de 8 bits por canal (entrada decodificada e estatísticas de mediana).
struct Rgba8 {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 255;

    friend constexpr bool operator==(const Rgba8&, const Rgba8&) noexcept = default;
};

constexpr Rgba toRgba(Rgba8 c) noexcept {
    constexpr float k = 1.0f / 255.0f;
    return {c.r * k, c.g * k, c.b * k, c.a * k};
}

constexpr Rgba8 toRgba8(const Rgba& c) noexcept {
    auto q = [](float v) noexcept {
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return static_cast<std::uint8_t>(v * 255.0f + 0.5f);
    };
    return {q(c.r), q(c.g), q(c.b), q(c.a)};
}

}  // namespace sla
