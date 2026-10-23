// cnx/vector/EdgeAnchors.h — posição sub-pixel das fronteiras nítidas (docs §1.31).
//
// A segmentação decide a que região cada PIXEL pertence, então a fronteira sai sobre a grade de pixels (a escadinha do
// BoundaryTracer) e o erro de posição é de até meio pixel. Um pixel de borda de um logo é uma mistura das duas cores na
// proporção da área que cada uma cobre; somar essa cobertura ao longo da normal dá a posição da borda com fração de pixel,
// sem modelo generativo nem otimização:
//
//     perfil t(s) = fração da cor B na amostra a s px do nó, ao longo da normal (A em s < 0, B em s > 0);
//     ∫_{−R}^{R} t ds = R − s₀      ⇒      s₀ = R − ∫ t ds        (a borda fica em s₀; vale para bordas desfocadas, simétricas)
//
// `refineEdgeAnchors` só desloca as ÂNCORAS da mola de fidelidade (VectorImage::originalPosition): o suavizador continua
// partindo da escadinha e a mola passa a puxar para a borda medida. Num canto a normal não existe: ele vai para a
// interseção das retas dos seus dois braços (de 2 a 6 px do nó), cada uma com a sua posição sub-pixel, o que desloca
// também os lados retos entre dois cantos (uma borda horizontal ou vertical não tem nó nenhum entre eles). Fica de fora
// quem não tem medida confiável: nós de junção, a moldura, fronteiras de pouco contraste, curvas fechadas demais e perfis
// que não são a transição limpa entre as duas faces da aresta (uma terceira cor, JPEG com anel forte).
#pragma once

#include <optional>

#include "cnx/engine/Parameters.h"
#include "cnx/vector/VectorImage.h"
#include "sla/Array.h"
#include "sla/Vec.h"

namespace cnx {

struct AnchorStats {
    int candidates = 0;        ///< nós de arestas entre duas faces visíveis e de contraste suficiente
    int moved = 0;             ///< âncoras deslocadas (inclui os cantos)
    int cornersMoved = 0;      ///< cantos levados à interseção das retas dos dois braços
    int skippedCorner = 0;     ///< canto ou curva fechada sem dois braços retos e medidos: fica na escadinha
    int skippedProfile = 0;    ///< perfil que não é a transição entre as duas faces
    int lowContrast = 0;       ///< nós ignorados por contraste abaixo de anchorMinContrast
    double meanShiftPx = 0.0;  ///< média de |deslocamento| entre os movidos
};

/// Posição da borda ao longo de `normal` (unitária) a partir de `p`, em px: positiva se a borda está do lado de B.
/// `a` é a cor do lado negativo e `b` a do positivo (0..1 por canal). O perfil é amostrado em passos de `step` px, com
/// interpolação bilinear (centro do pixel em x + ½), de −reach a +reach. Vazio se o perfil não for uma transição limpa
/// de a para b: as pontas têm de estar a menos de 0,2 e a mais de 0,8 de b, e nenhuma amostra pode se afastar da reta a–b
/// mais de 40 % do contraste (uma terceira cor, mesmo de 1 px, invalida o perfil). Lança std::invalid_argument se reach ≤ 0, step ≤ 0, step > reach ou a imagem for vazia.
std::optional<double> edgeOffset(const sla::Array2D<sla::Rgba8>& source, sla::Vec2 p, sla::Vec2 normal, const sla::Rgba& a,
                                 const sla::Rgba& b, double reach, double step);

/// Desloca as âncoras de `image` pela posição sub-pixel das bordas medida em `source` (a imagem em que a segmentação
/// foi feita: mesmas dimensões que `image`; senão lança std::invalid_argument). Não mexe em position().
AnchorStats refineEdgeAnchors(VectorImage& image, const sla::Array2D<sla::Rgba8>& source, const SmoothParams& params);

}  // namespace cnx
