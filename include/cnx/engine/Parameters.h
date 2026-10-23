// cnx/engine/Parameters.h — parâmetros por estágio, perfis embutidos, controles de usuário e .apf.
//
// Valores marcados "a calibrar" são provisórios: devem sair do treinamento no conjunto de dados de
// referência (SegmentationTrainer + busca de hiperparâmetros), nunca de constantes copiadas.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnx/engine/Fingerprint.h"

namespace cnx {

/// Perfis embutidos (§2). A especificação fala em 10 perfis e enumera 9; `Custom` (parâmetros de
/// um .apf ou da interface) completa a lista até a confirmação.
enum class ImageProfile : std::uint8_t {
    LogoHighAA,  // EP_LOGO_HIG_AA
    LogoMedAA,   // EP_LOGO_MED_AA
    LogoLowAA,   // EP_LOGO_LOW_AA
    LogoHigh,    // EP_LOGO_HIG
    LogoMed,     // EP_LOGO_MED
    LogoLow,     // EP_LOGO_LOW
    PhotoHigh,   // EP_PHOTO_HIG
    PhotoMed,    // EP_PHOTO_MED
    PhotoLow,    // EP_PHOTO_LOW
    Custom,
};

constexpr bool isAntialiasedLogo(ImageProfile p) noexcept { return p <= ImageProfile::LogoLowAA; }
constexpr bool isPhoto(ImageProfile p) noexcept {
    return p >= ImageProfile::PhotoHigh && p <= ImageProfile::PhotoLow;
}

/// Termo de fidelidade de cada fase da suavização (`measurement_types`).
enum class MeasurementType : std::uint8_t {
    Spring = 0,      ///< mola até a escadinha original (com zona livre opcional)
    Generative = 1,  ///< análise por síntese (§6) — logos com AA, nas 3 fases
    None = 2,        ///< sem termo de fidelidade — fotos, fases 2–3 (interpretação da spec, a confirmar)
};

/// Escala multiplicativa dos controles de usuário:
///   v ∈ [vMin, vMax]  ↦  p = pMin · (pMax / pMin)^((v − vMin) / (vMax − vMin)),   pMin, pMax > 0.
/// Passos iguais do controle multiplicam o parâmetro pelo mesmo fator (efeito suave no início,
/// forte no fim, quando pMax ≫ pMin). v fora da faixa é saturado.
inline double multiplicativeScale(double v, double vMin, double vMax, double pMin, double pMax) {
    if (!(vMax > vMin) || !(pMin > 0.0) || !(pMax > 0.0)) {
        throw std::invalid_argument("multiplicativeScale: requer vMax > vMin e pMin, pMax > 0");
    }
    const double s = std::clamp((v - vMin) / (vMax - vMin), 0.0, 1.0);
    return pMin * std::pow(pMax / pMin, s);
}

struct PreprocessParams {
    double targetAreaPx = 1.0e6;  ///< área-alvo do redimensionamento (px²) — a calibrar
    double blurSigmaPx = 0.0;     ///< σ do desfoque gaussiano (paleta/segmentação); 0 desativa
    bool flattenAlpha = true;     ///< ensureFlattened quando o alfa é constante
    bool allowUpscale = false;    ///< também amplia imagens menores que a área-alvo (por padrão só reduz)

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct ClassifierParams {
    bool autoClassify = true;
    ImageProfile forcedProfile = ImageProfile::Custom;  ///< usado quando autoClassify = false

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct PaletteParams {
    int minColors = 2;
    int maxColors = 12;
    int kmeansMaxRounds = 10;
    int histogramBins = 16;              ///< por canal; 8 no modo rápido
    double colorPenalty = 0.05;          ///< nota = log(erro + 1) + (K − 2)·penalidade — a calibrar
    std::optional<int> forcedNumColors;  ///< ignora a seleção automática
    double edgeScale = 0.1;              ///< peso do pixel = 1 / (1 + borda/edgeScale), borda = gradiente em [0, ~2] — a calibrar
    double exactColorMinShare = 0.002;   ///< cor exata só conta se tem ≥ essa fração dos pixels (imagem chapada) — a calibrar
    int refineRounds = 2;                ///< rodadas de k-means sobre os pixels, só para o K escolhido

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct SegmentParams {
    int quadTreeBlock = 16;
    double quadTreeVariance = 1.0e-3;  ///< — a calibrar
    double mergeThreshStart = 1.0e-4;  ///< limiar inicial de "juntar" (distância² RGBA) — a calibrar
    double mergeThreshGrowth = 1.5;    ///< fator multiplicativo por rodada — a calibrar
    /// Teto de "juntar": duas vizinhas só se fundem se as cores medianas estão a dist² ≤ teto. Em degradês o teto é o
    /// passo da faixa (docs §1.28): com ele em 40 níveis de cor (2,5e-2) um logo em degradê virava 7 faixas por fita;
    /// 12 níveis (padrão) dão dezenas de faixas finas. applyProfile o multiplica por perfil (alto/médio/baixo detalhe).
    double mergeThreshMax = 12.0 * 12.0 / (255.0 * 255.0);
    /// Encaixe na paleta (snapColorsToPalette): uma região só toma a cor da paleta mais próxima se a sua cor está a
    /// dist² ≤ este valor; senão mantém a própria cor mediana. Uma região chapada com ruído fica a poucos níveis da sua
    /// tinta e é encaixada (cor exata); uma faixa de degradê não tem tinta por perto e fica como está. 4,0 (o máximo
    /// possível em RGBA) = sempre encaixa. applyProfile o põe em 1/9 do teto de fusão (um terço em distância: 4 níveis
    /// no perfil médio) nos perfis de foto; os de logo mantêm 4,0.
    double paletteSnapMax = 4.0;
    /// Rampas de antialiasing sem paleta (PixelSegmenter::absorbRamps, docs §1.30): liga o estágio que funde às
    /// vizinhas sólidas as regiões finas (sem pixel interior) que são ruído ou mistura de duas cores. Desligado por padrão;
    /// applyProfile o liga nos perfis de foto/degradê.
    bool absorbRamps = false;
    /// Uma fina está "perto" de uma sólida se dist² ≤ rampNearFactor · mergeThreshMax (4: o dobro em distância).
    double rampNearFactor = 4.0;
    /// Espessura máxima, em px, da rampa de antialiasing que o absorvedor desfaz: é a profundidade da busca pelas finas
    /// vizinhas até as sólidas que a explicam (uma rampa de n px só é explicada se a fina de fora enxerga a sólida do outro
    /// lado, a n camadas). Rampas mais grossas são degradês suaves e ficam como faixas. 4 cobre a borda borrada de um
    /// JPEG; mais profundidade = bordas mais nítidas, menos faces e mais erro de pixel nas bordas borradas (docs §1.30).
    int rampMaxDepth = 4;
    int minRegionPixels = 4;           ///< enforceMinNumPixels — a calibrar
    /// Orçamento da simulação "e se" de uma fusão: o erro quadrático da união pode superar a soma
    /// dos erros das partes em no máximo `whatIfFactor · limiar · min(nₐ, n_b)`. Para regiões
    /// chapadas isso equivale a dist²(medianas) ≤ limiar (fator 1); para regiões heterogêneas é mais severo.
    double whatIfFactor = 1.0;
    int finalSweepCap = 8;       ///< varreduras extras de juntar/trocar/diagonal no limiar máximo (convergência)

    // ---- traços finos (FlowField, docs §1.20) — todos a calibrar ----
    /// Liga pares diagonais de mesma cor ao longo do campo de orientação, antes de enforceMinNumPixels: um traço
    /// de 1 px em diagonal é uma cadeia de pixels que só se tocam pela quina e, em 4-conectividade, sumiria.
    bool linkStrokes = true;
    double flowSigmaPx = 1.0;         ///< σ da integração do tensor de estrutura (px), ≥ 0
    double strokeMinCoherence = 0.5;  ///< coerência mínima do campo nos dois pixels do par, ∈ [0, 1]
    double strokeMaxAngle = 0.7;      ///< ângulo máximo (rad) entre a tangente do campo e a diagonal do par, ∈ (0, π/4)

    // ---- SubPixelSegmenter (§4, regiões com antialiasing) — todos a calibrar ----
    int microRegionMaxPixels = 2;      ///< limpa regiões de 1 px, depois 2 px… até este tamanho
    double beachLineTolerance = 0.25;  ///< praia: distância da cor à reta A–B ≤ tolerância · ‖cor_B − cor_A‖
    double beachMinContrast = 0.02;    ///< praia: ‖cor_B − cor_A‖² mínimo entre as cores dominantes
    double beachNeighborShare = 0.8;   ///< praia: fração do contato que as duas dominantes devem somar
    double shadowVariance = 2.0e-3;    ///< sombra: variância de cor mínima da região (Σ canais, [0, 1])
    double shadowSmoothRatio = 0.5;    ///< sombra: variância da imagem desfocada / variância bruta mínima
    int shadowMinPixels = 16;
    int shadowBlurRadius = 2;          ///< raio do desfoque de caixa usado na detecção e na reconstrução

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct SmoothParams {
    static constexpr int kPhases = 3;

    std::array<MeasurementType, kPhases> measurementTypes{MeasurementType::Spring, MeasurementType::Spring,
                                                          MeasurementType::Spring};
    std::array<double, kPhases> spacingWeight{5.0, 0.5, 0.5};
    double fidelityWeight = 1.0;          ///< λ_fid: peso da mola (ou do modelo gerador) frente aos priors — a calibrar
    int priorType = 3;                    ///< 3 = suavidade angular (único implementado; o "raiz quadrada" fica desativado)
    double priorStrength = 10.0;          ///< prior_strengths
    double airPressureWeight = 5.0;       ///< air_pressure_weight
    int airPressureMaxNodes = 8;          ///< aplica-se a contornos com menos de 8 nós
    double borderAnchorWeight = 10.0;     ///< peso da mola de margem — a calibrar
    /// Além da mola, trava exatamente a coordenada perpendicular dos nós da moldura: eles deslizam
    /// ao longo dela e nunca saem. Sem isso a moldura deriva e a soma das áreas deixa de ser W·H (I4).
    bool borderHardConstraint = true;
    double fidelityFreeRadiusPx = 0.0;    ///< raio da zona livre da mola; 0 = mola simples
    int quarantineRounds = 2;
    /// Segmentos mais curtos que isto são tratados como colapso (nós coincidentes): a iteração é
    /// desfeita e os nós vão para a quarentena, como num cruzamento.
    double minSegmentPx = 0.05;
    /// Canto perfurado: um nó é canto se o giro entre os braços de comprimento `punctureArmPx` for
    /// ≥ `punctureCornerThresh` (rad) e os dois braços forem retos, isto é, nenhum nó intermediário
    /// se afasta da corda em mais de `punctureFlatnessThresh` (px). Limiar ≤ 0 desativa. A regra
    /// substitui o vetor de 21 medidas da especificação, que não está definido — a calibrar.
    double punctureCornerThresh = 0.8;    ///< puncture_corner_thresh (rad)
    double punctureFlatnessThresh = 0.3;  ///< puncture_flatness_thresh (px)
    double punctureArmPx = 4.0;
    /// Contraste (níveis de cor 0-255, distância entre as cores das duas faces da fronteira) abaixo do qual um nó nunca
    /// é canto perfurado: entre faixas parecidas de um degradê os degraus de 8 px do JPEG parecem cantos vivos e seriam
    /// preservados (docs §1.29). 0 desliga. Os cantos de um logo (letra contra fundo) têm contraste de dezenas de níveis ou mais.
    double punctureMinContrast = 24.0;
    /// Dois cantos seguidos na mesma aresta, de giros opostos e a no máximo este comprimento de arco um do outro, são um
    /// degrau de bloco 8×8 do JPEG (a borda sai e logo volta) e não cantos do desenho: nenhum é perfurado (docs §1.32).
    /// 0 desliga.
    double jogMaxLedgePx = 2.5;
    /// Peso da mola de fidelidade pelo contraste da fronteira: escala ∝ contraste² (a informação de posição de uma
    /// fronteira cresce com o quadrado do contraste), do piso `contrastFloor` até 1 em `contrastFullLevels` níveis de cor.
    /// 0 desliga, e é o padrão: no logo em degradê medido (docs §1.29) enfraquecer a mola não tirou a escada (quem a
    /// prende é o canto perfurado, `punctureMinContrast`) e, somado ao portão do canto, deixou a fronteira um pouco pior.
    /// Só a mola é pesada: a medição generativa já usa os pixels.
    double contrastFullLevels = 0.0;
    double contrastFloor = 0.05;
    /// Posição sub-pixel das bordas (docs §1.31): as âncoras da mola de fidelidade passam da escadinha para a borda medida
    /// pela cobertura dos pixels de transição, ao longo da normal (±`anchorReachPx`). Só nós de fronteiras com contraste
    /// ≥ `anchorMinContrast` níveis, fora de cantos e junções; o deslocamento é limitado a `anchorMaxShiftPx`. Desligado por
    /// padrão; os perfis de foto/degradê ligam (os de logo com AA já medem a imagem pelo modelo gerador).
    bool subPixelAnchors = false;
    double anchorReachPx = 2.5;
    double anchorMaxShiftPx = 1.0;
    double anchorMinContrast = 48.0;  ///< medido no degradê (docs §1.31): com 24 a classe de contraste suave piora 8 %, com 48 melhora 9 %
    /// Quando o fiscal acha cruzamentos num passo do CG, só os nós infratores voltam à posição anterior (e vão para a
    /// quarentena) e os demais ficam com o passo; até `holdRounds` rodadas, se a retenção cria novo cruzamento. Só se não
    /// resolver o passo inteiro é descartado (docs §1.33). 0 desliga e restaura o descarte direto.
    int holdRounds = 0;
    /// Limiar do reinício de Powell do CG (`CgOptions::restartRatio`). Padrão clássico 0,2; 0,5 conserva a direção conjugada.
    double cgRestartRatio = 0.2;
    int punctureOnPhase = 0;           ///< índice 0-based: na 1ª fase, sobre a escadinha (depois dela os cantos já se arredondaram)
    int maxCgIterations = 200;            ///< por fase
    double maxNodeStepPx = 0.5;           ///< região de confiança da busca em linha (§2.3)
    double gradientTolerance = 1.0e-5;    ///< norma do infinito do gradiente nos nós livres
    double energyTolerance = 1.0e-10;     ///< decréscimo relativo de energia por iteração

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct FitParams {
    double statThreshInitial = 1.0e-8;
    double statThreshFinal = 0.2;  ///< stat_thresh_final (0.2 alta fidelidade/AA; 0.5…10.0 compactos)
    int maxRounds = 20;
    int initialFragmentNodes = 4;  ///< — a calibrar
    int newtonMaxIterations = 10;
    int finalSweeps = 8;           ///< varreduras extras no limiar final, até nenhuma fusão/deslocamento restar
    bool enforceG1 = true;         ///< tangente compartilhada nas quebras que não são cantos/junções

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct EngineParameters {
    std::uint64_t seed = 0x9E3779B97F4A7C15ull;  ///< semente do myPrng; entra em todas as impressões digitais

    PreprocessParams preprocess;
    ClassifierParams classifier;
    PaletteParams palette;
    SegmentParams segment;
    SmoothParams smooth;
    FitParams fit;

    /// Sobrescreve os campos que o perfil define (measurement_types, stat_thresh_final, etc.).
    void applyProfile(ImageProfile profile);
};

/// Os ajustes de usuário. A especificação fala em 11 e nomeia 5; os demais entram quando definidos.
enum class UserControl : std::uint8_t { Sensitivity, Complexity, Smoothing, Curves, Antialiasing };

/// Aplica um controle (valor inteiro da interface) via multiplicativeScale sobre o(s) parâmetro(s) associado(s).
void applyUserControl(EngineParameters& params, UserControl control, int value);

/// Erro de leitura/escrita de .apf com o número da linha (1-based). 0 = erro de E/S, ou incoerência que não
/// vem de nenhuma linha do arquivo (os valores que já estavam em `params` violam uma regra).
class ParameterFileError : public std::runtime_error {
public:
    ParameterFileError(const std::string& message, int line) : std::runtime_error(message), line_(line) {}
    int line() const noexcept { return line_; }

private:
    int line_;
};

/// Perfis .apf em texto plano, uma entrada por linha: `Classe::parametro valor`
/// (ex.: `ContourSmoother::prior_strengths 10`). Linhas vazias e iniciadas por '#' são ignoradas.
///
/// Formato (docs/ARQUITETURA.md §1.18):
///  - chave = `Classe::nome` (letras, dígitos e '_'), sensível a maiúsculas; valor = o resto da linha;
///  - listas (`measurement_types`, `spacing_weight`) levam um valor por fase, separados por espaço;
///  - inteiros em decimal, reais em decimal ou científico (ponto, sem locale), booleanos `true|false|1|0`,
///    enumerações por nome (`EP_LOGO_MED_AA`, `spring|generative|none`), `forced_num_colors` aceita `none`;
///  - o que o arquivo não menciona fica como estava em `params` (um .apf pode ser parcial);
///  - chave repetida é erro, e a leitura é transacional: se algo falha, `params` não muda.
class ParameterFiler {
public:
    enum class Mode : std::uint8_t {
        Strict,   ///< chave desconhecida → ParameterFileError
        Lenient,  ///< chave desconhecida → devolvida na lista de avisos (compatibilidade entre versões)
    };

    /// Retorna as chaves ignoradas (sempre vazio no modo Strict). Lança ParameterFileError.
    static std::vector<std::string> readAPF(std::istream& in, EngineParameters& params, Mode mode = Mode::Strict);
    static std::vector<std::string> readAPF(const std::filesystem::path& file, EngineParameters& params,
                                            Mode mode = Mode::Strict);

    /// Escreve todos os campos. Lança std::invalid_argument se `params` viola alguma regra de `validate`
    /// (o leitor não conseguiria reabri-lo) e ParameterFileError em falha de E/S. A versão em arquivo grava
    /// num temporário e renomeia: um arquivo anterior nunca fica pela metade.
    static void writeAPF(std::ostream& out, const EngineParameters& params);
    static void writeAPF(const std::filesystem::path& file, const EngineParameters& params);

    /// Primeira violação das faixas e relações entre campos (as mesmas que os módulos exigem ao construir-se),
    /// ou nullopt. É o que o leitor aplica depois de ler e o escritor antes de escrever.
    static std::optional<std::string> validate(const EngineParameters& params);
};

}  // namespace cnx
