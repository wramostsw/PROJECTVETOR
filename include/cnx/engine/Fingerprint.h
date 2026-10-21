// cnx/engine/Fingerprint.h — impressão digital encadeada dos estágios (recálculo inteligente).
//
//   fp_k = H(parâmetros_k, fp_{k−1}, semente);   estágio k sujo ⇔ fp_k ≠ fp_k(cache)
//
// H = FNV-1a 64 bits sobre uma serialização canônica (bytes em little-endian explícito, −0.0 → +0.0,
// NaN canônico), portanto estável entre plataformas e execuções. Não é criptográfico: a colisão só
// causaria reaproveitar um cache indevido, e com ~7 estágios a probabilidade é desprezível (~2⁻⁶⁴).
#pragma once

#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

namespace cnx {

class Fingerprint {
public:
    constexpr Fingerprint() noexcept = default;
    constexpr explicit Fingerprint(std::uint64_t raw) noexcept : h_(raw) {}

    template <std::integral T>
    constexpr Fingerprint& add(T v) noexcept {
        if constexpr (std::is_signed_v<T>) {
            return addBits(static_cast<std::uint64_t>(static_cast<std::int64_t>(v)));
        } else {
            return addBits(static_cast<std::uint64_t>(v));
        }
    }

    Fingerprint& add(double v) noexcept {
        if (v == 0.0) v = 0.0;  // −0.0 e +0.0 têm o mesmo efeito em todos os estágios
        if (std::isnan(v)) v = std::numeric_limits<double>::quiet_NaN();
        return addBits(std::bit_cast<std::uint64_t>(v));
    }

    constexpr Fingerprint& add(std::string_view s) noexcept {
        for (const char c : s) mix(static_cast<unsigned char>(c));
        return addBits(static_cast<std::uint64_t>(s.size()));  // separa "ab"+"c" de "a"+"bc"
    }

    /// Encadeia a impressão digital do estágio anterior.
    constexpr Fingerprint& add(Fingerprint upstream) noexcept { return addBits(upstream.h_); }

    constexpr std::uint64_t value() const noexcept { return h_; }

    friend constexpr bool operator==(const Fingerprint&, const Fingerprint&) noexcept = default;

private:
    static constexpr std::uint64_t kOffsetBasis = 14695981039346656037ull;
    static constexpr std::uint64_t kPrime = 1099511628211ull;

    constexpr void mix(unsigned char byte) noexcept {
        h_ ^= byte;
        h_ *= kPrime;
    }

    constexpr Fingerprint& addBits(std::uint64_t v) noexcept {
        for (int i = 0; i < 8; ++i) mix(static_cast<unsigned char>((v >> (8 * i)) & 0xFFu));
        return *this;
    }

    std::uint64_t h_ = kOffsetBasis;
};

}  // namespace cnx
