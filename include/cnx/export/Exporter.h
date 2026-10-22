// cnx/export/Exporter.h — orquestração da exportação e protocolo comum dos 5 escritores (§8, docs §1.24).
//
// O Exporter decide a geometria e a ordem de desenho e emite comandos padronizados para um
// IShapeWriter (padrão Builder). SVG, EPS, AI e PDF são escritores próprios em texto puro; o DXF
// usa dxflib (GPLv2/comercial: ver ARQUITETURA.md §0, item 14). Implementado: SVG (SvgWriter.h).
//
// Zero frestas: cada fronteira existe uma vez (Edge); a face da esquerda a escreve no sentido direto e a da
// direita a escreve invertida (P3, P2, P1, P0), com os mesmos números. Modos (`cake_stack_mode`):
//   Stacked          só o laço externo de cada face, pais antes dos filhos (os filhos pintam por cima);
//   Compound         laço externo + furos como subcaminhos (orientação oposta: regra nonzero, sem fill-rule);
//   CompoundByColor  Compound, com as faces da mesma cor num grupo (na ordem da 1ª aparição).
// Faces ocultas (BackgroundSelection) e totalmente transparentes (alfa 0) não são escritas. Em Compound, uma
// face oculta vira buraco; em Stacked, aparece o que está embaixo dela (o pai).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

#include "cnx/engine/Fingerprint.h"
#include "cnx/vector/VectorImage.h"
#include "sla/Vec.h"

namespace cnx {

enum class ExportFormat : std::uint8_t { Svg, Eps, Ai, Pdf, Dxf };

/// `cake_stack_mode`
enum class CakeStackMode : std::uint8_t {
    Stacked = 0,          ///< bolo em camadas: só o laço externo; as faces internas são desenhadas por cima
    Compound = 1,         ///< caminhos compostos: furos como subcaminhos de orientação inversa (nonzero)
    CompoundByColor = 2,  ///< compostos + agrupamento por cor (<g id="#rrggbbaa">)
};

/// `dxf_mode`. A especificação só diz que 1 e 2 discretizam; a diferença entre eles está em aberto.
enum class DxfMode : std::uint8_t { Splines = 0, Polylines = 1, PolylinesMode2 = 2 };

/// Faces marcadas como fundo (isShapeVisible), com desfazer/refazer.
/// Cada estado é um conjunto completo e ordenado: desfazer/refazer O(1), memória O(passos × marcadas).
class BackgroundSelection {
public:
    bool isShapeVisible(FaceId f) const noexcept;
    void setHidden(FaceId f, bool hidden);  ///< registra um passo (descarta o ramo de refazer)
    void clear();                           ///< idem
    bool canUndo() const noexcept { return cursor_ > 0; }
    bool canRedo() const noexcept { return cursor_ + 1 < states_.size(); }
    bool undo();
    bool redo();
    const std::vector<FaceId>& hiddenFaces() const noexcept { return states_[cursor_]; }

    Fingerprint fingerprint(Fingerprint upstream) const;

private:
    std::vector<std::vector<FaceId>> states_ = std::vector<std::vector<FaceId>>(1);
    std::size_t cursor_ = 0;
};

struct ExporterSettings {
    CakeStackMode cakeStackMode = CakeStackMode::Compound;
    bool shapeStroking = true;          ///< `shape_stroking_mode`: traço da cor do preenchimento
    double strokeWidthPt = 0.09375;     ///< 3/32 pt — fecha as frestas de AA dos visualizadores
    bool epsCompatibilityMode = false;  ///< true: operadores por extenso (closepath, setrgbcolor)
    DxfMode dxfMode = DxfMode::Splines;
    double flattenTolerancePx = 0.05;   ///< desvio máximo da discretização (dxf_mode 1/2)
    int coordinateDecimals = 2;         ///< precisão fixa das coordenadas em texto
    double pointsPerPixel = 1.0;        ///< 72 DPI: 1 px = 1 pt
    BackgroundSelection background;

    Fingerprint fingerprint(Fingerprint upstream) const;
};

struct ExportRequest {
    std::filesystem::path file;
    ExportFormat format = ExportFormat::Svg;
    ExporterSettings settings;
};

struct DocumentInfo {
    double widthPx = 0.0;
    double heightPx = 0.0;
    double pointsPerPixel = 1.0;
    std::string title;
    std::string creator = "cnx";
};

struct StrokeStyle {
    sla::Rgba color{};
    double widthPt = 0.09375;
};

class ExportError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Protocolo de desenho comum aos 5 formatos. O Exporter garante a gramática:
///   beginFile ( [beginGroup] ( beginShape ( moveTo (lineTo | curveTo)* closeSubpath )+ endShape )*
///               [endGroup] )* endFile
/// Coordenadas em pixels da imagem; a conversão para pt (e a inversão do eixo y em PS/PDF) é do escritor.
class IShapeWriter {
public:
    virtual ~IShapeWriter() = default;

    virtual void beginFile(const DocumentInfo& info) = 0;
    virtual void beginGroup(const sla::Rgba& color) = 0;
    virtual void endGroup() = 0;
    /// stroke == nullptr: só preenchimento.
    virtual void beginShape(const sla::Rgba& fill, const StrokeStyle* stroke) = 0;
    virtual void moveTo(sla::Vec2 p) = 0;
    virtual void lineTo(sla::Vec2 p) = 0;
    virtual void curveTo(sla::Vec2 c1, sla::Vec2 c2, sla::Vec2 p) = 0;
    virtual void closeSubpath() = 0;
    virtual void endShape() = 0;
    virtual void endFile() = 0;
};

/// Fábrica: SvgWriter, EpsWriter, AiWriter, CompatiblePdfWriter (PDF 1.4, xref byte a byte, /OPM 1),
/// DxfWriter. `out` deve estar em modo binário (o PDF depende dos deslocamentos exatos de bytes).
/// Lança ExportError para os formatos ainda não implementados (hoje: todos menos SVG).
std::unique_ptr<IShapeWriter> makeWriter(ExportFormat format, std::ostream& out, const ExporterSettings& settings);

class Exporter {
public:
    /// `image` deve viver mais que o Exporter. Arestas com curvas (FitBezier) saem como curvas; arestas sem
    /// curvas (antes do ajuste) saem como a poligonal dos nós. Lança ExportError se coordinateDecimals ∉ [0, 9],
    /// strokeWidthPt < 0 ou pointsPerPixel ≤ 0 (ou não finitos).
    Exporter(const VectorImage& image, ExporterSettings settings);

    /// Lança CancelTaskException se `stop` for acionado (verificado a cada face).
    void write(IShapeWriter& writer, const std::stop_token& stop = {}) const;
    /// Escreve num arquivo temporário ao lado do destino e renomeia ao final: nunca deixa um arquivo truncado
    /// (em erro ou cancelamento o temporário é apagado e o destino fica como estava). Título = nome do arquivo.
    /// Lança ExportError (E/S, formato não implementado) e CancelTaskException.
    void writeFile(ExportFormat format, const std::filesystem::path& file, const std::stop_token& stop = {}) const;

    /// Ordem de desenho: por profundidade na árvore de Face::parent (pai antes dos filhos, exigido no modo
    /// Stacked), depois por id. Omite o exterior, as faces ocultas (BackgroundSelection) e as de alfa 0.
    std::vector<FaceId> drawOrder() const;

private:
    /// Emite um laço. Para cada meia-aresta h: se direta, as curvas da aresta em ordem; se inversa,
    /// em ordem reversa e cada uma invertida (P3, P2, P1, P0). As duas faces vizinhas recebem a mesma
    /// geometria: encaixe exato, sem fresta nem sobreposição.
    void emitLoop(IShapeWriter& writer, HalfEdgeId first) const;
    void emitFace(IShapeWriter& writer, FaceId f) const;
    void writeDocument(IShapeWriter& writer, const std::stop_token& stop, std::string title) const;

    const VectorImage& image_;
    ExporterSettings settings_;
};

}  // namespace cnx
