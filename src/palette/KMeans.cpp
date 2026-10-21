// src/palette/KMeans.cpp — sementes por picos + k-means++ e Lloyd ponderado.
#include "cnx/palette/KMeans.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cnx {

std::vector<Color4d> chooseSeeds(std::span<const KMeansItem> items, std::size_t k, double minSeparation, sla::myPrng& rng) {
    std::vector<Color4d> seeds;
    if (items.empty() || k == 0) return seeds;
    const double sep2 = minSeparation * minSeparation;

    // 1. Picos do histograma (itens já em ordem de peso), suprimindo os vizinhos de um pico escolhido.
    for (const KMeansItem& it : items) {
        if (seeds.size() >= k) break;
        if (!(it.weight > 0.0)) continue;
        bool far = true;
        for (const Color4d& s : seeds) {
            if (distance2(it.pos, s) < sep2) {
                far = false;
                break;
            }
        }
        if (far) seeds.push_back(it.pos);
    }

    // 2. Faltantes: k-means++ (D² ponderado pelo peso do item).
    std::vector<double> d2(items.size());
    while (seeds.size() < k) {
        double total = 0.0;
        for (std::size_t i = 0; i < items.size(); ++i) {
            double best = std::numeric_limits<double>::infinity();
            for (const Color4d& s : seeds) best = std::min(best, distance2(items[i].pos, s));
            d2[i] = items[i].weight * best;
            total += d2[i];
        }
        if (!(total > 0.0)) break;  // todos os itens já coincidem com alguma semente
        double target = rng.uniform() * total;
        std::size_t pick = items.size() - 1;
        for (std::size_t i = 0; i < items.size(); ++i) {
            target -= d2[i];
            if (target < 0.0) {
                pick = i;
                break;
            }
        }
        while (d2[pick] <= 0.0 && pick > 0) --pick;  // arredondamento: nunca escolhe um item de D² nulo
        seeds.push_back(items[pick].pos);
    }
    return seeds;
}

KMeansResult kmeans(std::span<const KMeansItem> items, std::vector<Color4d> seeds, int maxRounds) {
    if (items.empty()) throw std::invalid_argument("kmeans: sem itens");
    if (seeds.empty()) throw std::invalid_argument("kmeans: sem sementes");
    double totalWeight = 0.0;
    for (const KMeansItem& it : items) totalWeight += it.weight;
    if (!(totalWeight > 0.0)) throw std::invalid_argument("kmeans: peso total nulo");

    const std::size_t k = seeds.size();
    std::vector<Color4d> centers = std::move(seeds);
    std::vector<int> assign(items.size(), -1);
    KMeansResult out;
    for (int round = 0; round < std::max(1, maxRounds); ++round) {
        // Atribuição ao centro mais próximo (empate → menor índice).
        bool changed = false;
        for (std::size_t i = 0; i < items.size(); ++i) {
            int best = 0;
            double bestD = distance2(items[i].pos, centers[0]);
            for (std::size_t c = 1; c < k; ++c) {
                const double d = distance2(items[i].pos, centers[c]);
                if (d < bestD) {
                    bestD = d;
                    best = static_cast<int>(c);
                }
            }
            if (assign[i] != best) {
                assign[i] = best;
                changed = true;
            }
        }
        // Atualização: média ponderada de cada grupo; grupo vazio mantém o centro (será descartado).
        std::vector<Color4d> sum(k, Color4d{});
        std::vector<double> w(k, 0.0);
        for (std::size_t i = 0; i < items.size(); ++i) {
            const auto c = static_cast<std::size_t>(assign[i]);
            w[c] += items[i].weight;
            for (std::size_t d = 0; d < 4; ++d) sum[c][d] += items[i].weight * items[i].pos[d];
        }
        for (std::size_t c = 0; c < k; ++c) {
            if (w[c] > 0.0) {
                for (std::size_t d = 0; d < 4; ++d) centers[c][d] = sum[c][d] / w[c];
            }
        }
        out.rounds = round + 1;
        if (!changed) {
            out.converged = true;
            break;
        }
    }

    // Descarta os centros vazios, ordena por peso e remapeia as atribuições.
    std::vector<double> weight(k, 0.0);
    for (std::size_t i = 0; i < items.size(); ++i) weight[static_cast<std::size_t>(assign[i])] += items[i].weight;
    std::vector<std::size_t> order;
    for (std::size_t c = 0; c < k; ++c) {
        if (weight[c] > 0.0) order.push_back(c);
    }
    out.discarded = static_cast<int>(k - order.size());
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return weight[a] > weight[b]; });
    std::vector<int> remap(k, -1);
    for (std::size_t n = 0; n < order.size(); ++n) {
        remap[order[n]] = static_cast<int>(n);
        out.centers.push_back(centers[order[n]]);
        out.clusterWeight.push_back(weight[order[n]]);
    }
    out.assignment.resize(items.size());
    double err = 0.0;
    for (std::size_t i = 0; i < items.size(); ++i) {
        out.assignment[i] = remap[static_cast<std::size_t>(assign[i])];
        err += items[i].weight * distance2(items[i].pos, out.centers[static_cast<std::size_t>(out.assignment[i])]) + items[i].scatter;
    }
    out.error = err / totalWeight;
    return out;
}

}  // namespace cnx
