// tests/unit/test_support.h — geradores determinísticos e cenas de segmentação compartilhados pelos testes.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cnx/segment/PixelMap.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnxtest {

/// PRNG determinístico (splitmix64): mesmas sequências em qualquer plataforma.
struct SplitMix64 {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int below(int n) { return static_cast<int>(next() % static_cast<std::uint64_t>(n)); }
};

/// Cor com `levels` valores por canal (poucos níveis geram empates na mediana).
inline sla::Rgba8 randomColor(SplitMix64& rng, int levels) {
    const int step = 255 / std::max(1, levels - 1);
    auto v = [&] { return static_cast<std::uint8_t>(rng.below(levels) * step); };
    return {v(), v(), v(), v()};
}

inline constexpr std::array<int, 4> kDx4{1, 0, -1, 0};
inline constexpr std::array<int, 4> kDy4{0, 1, 0, -1};

/// Imagem aleatória + PixelMap com uma região por bloco block×block (superpixels iniciais).
struct Scene {
    cnx::PixelMap map;
    sla::Array2D<sla::Rgba8> image;
};

inline Scene blockScene(int w, int h, int block, SplitMix64& rng, int levels) {
    Scene s{cnx::PixelMap(w, h), sla::Array2D<sla::Rgba8>(w, h)};
    const int bw = (w + block - 1) / block;
    const int bh = (h + block - 1) / block;
    std::vector<cnx::RegionId> ids;
    for (int i = 0; i < bw * bh; ++i) ids.push_back(s.map.createRegion());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            s.image(x, y) = randomColor(rng, levels);
            s.map.setLabel(x, y, ids[static_cast<std::size_t>((y / block) * bw + x / block)], s.image(x, y));
        }
    }
    return s;
}

/// Funde pares de regiões 4-vizinhas escolhidos ao acaso ("juntar").
inline void randomMerges(cnx::PixelMap& map, int count, SplitMix64& rng) {
    for (int i = 0; i < count; ++i) {
        const int x = rng.below(map.width());
        const int y = rng.below(map.height());
        const auto d = static_cast<std::size_t>(rng.below(4));
        const int qx = x + kDx4[d];
        const int qy = y + kDy4[d];
        if (qx < 0 || qy < 0 || qx >= map.width() || qy >= map.height()) continue;
        const cnx::RegionId a = map.regionAt(x, y);
        const cnx::RegionId b = map.regionAt(qx, qy);
        if (a != b) map.forest().unite(a, b);
    }
}

struct MoveStats {
    int accepted = 0;
    int refused = 0;
    int inconsistent = 0;  ///< resultado de tryMovePixel diferente de isSimplePoint
};

/// "Trocar": tenta mover pixels de fronteira para uma região 4-vizinha escolhida ao acaso.
inline MoveStats randomMoves(cnx::PixelMap& map, const sla::Array2D<sla::Rgba8>& image, int count, SplitMix64& rng) {
    MoveStats stats;
    for (int i = 0; i < count; ++i) {
        const int x = rng.below(map.width());
        const int y = rng.below(map.height());
        const auto d = static_cast<std::size_t>(rng.below(4));
        const int qx = x + kDx4[d];
        const int qy = y + kDy4[d];
        if (qx < 0 || qy < 0 || qx >= map.width() || qy >= map.height()) continue;
        const cnx::RegionId target = map.regionAt(qx, qy);
        if (target == map.regionAt(x, y)) continue;
        const bool simple = map.isSimplePoint(x, y);
        const bool moved = map.tryMovePixel(x, y, target, image(x, y));
        stats.inconsistent += moved != simple ? 1 : 0;
        (moved ? stats.accepted : stats.refused) += 1;
    }
    return stats;
}

}  // namespace cnxtest
