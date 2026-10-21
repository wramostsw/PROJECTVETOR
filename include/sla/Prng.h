// sla/Prng.h — myPrng: gerador pseudoaleatório determinístico (SplitMix64).
//
// Mesma semente, mesma sequência, em qualquer plataforma. A semente entra em todas as impressões
// digitais (docs/ARQUITETURA.md §0, item 15). Não é criptográfico.
#pragma once

#include <cstdint>

namespace sla {

class myPrng {
public:
    explicit constexpr myPrng(std::uint64_t seed = 0) noexcept : state_(seed) {}

    constexpr std::uint64_t next() noexcept {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    /// Uniforme em [0, 1) com 53 bits de mantissa.
    double uniform() noexcept { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

    /// Inteiro em [0, n) sem viés (rejeição de Lemire). Pré-condição: n > 0.
    std::uint32_t below(std::uint32_t n) noexcept {
        std::uint64_t m = (next() >> 32) * n;
        std::uint32_t low = static_cast<std::uint32_t>(m);
        if (low < n) {
            const std::uint32_t threshold = (0u - n) % n;
            while (low < threshold) {
                m = (next() >> 32) * n;
                low = static_cast<std::uint32_t>(m);
            }
        }
        return static_cast<std::uint32_t>(m >> 32);
    }

    constexpr std::uint64_t state() const noexcept { return state_; }

private:
    std::uint64_t state_;
};

}  // namespace sla
