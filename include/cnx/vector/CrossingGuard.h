// cnx/vector/CrossingGuard.h — fiscal anti-cruzamento (§5): detecta auto-interseções das poligonais.
//
// Um cruzamento é a interseção própria de dois segmentos que não compartilham nó (os que
// compartilham um nó se tocam por construção). Índice espacial: grade uniforme de segmentos
// (em vez da kd-tree de pontos da ANN, que não indexa segmentos); cada par candidato é testado
// uma só vez, na célula que contém o canto inferior-esquerdo da interseção das caixas.
// Custo O(S + candidatos) em S segmentos, contra O(S²) da força bruta.
#pragma once

#include <vector>

#include "cnx/vector/VectorImage.h"

namespace cnx {

/// Dois segmentos (a0→a1 e b0→b1, em ids de nós) que se cruzam.
struct Crossing {
    NodeId a0 = kInvalidId;
    NodeId a1 = kInvalidId;
    NodeId b0 = kInvalidId;
    NodeId b1 = kInvalidId;

    friend bool operator==(const Crossing&, const Crossing&) noexcept = default;
};

class CrossingGuard {
public:
    /// Cruzamentos de todas as arestas, ordenados (resultado determinístico).
    static std::vector<Crossing> find(const VectorImage& image);
    /// Referência O(S²) para testes.
    static std::vector<Crossing> findBruteForce(const VectorImage& image);
    /// Nós dos segmentos envolvidos, sem repetição, em ordem crescente de id.
    static std::vector<NodeId> offendingNodes(const std::vector<Crossing>& crossings);
    /// Extremidades (sem repetição, em ordem crescente) dos segmentos de comprimento < minLength.
    /// Nós quase coincidentes tornam o prior angular (∝ 1/‖Δ‖²) rígido demais para a busca em linha;
    /// o suavizador trata isso como um passo inválido, como um cruzamento.
    static std::vector<NodeId> shortSegmentNodes(const VectorImage& image, double minLength);
};

}  // namespace cnx
