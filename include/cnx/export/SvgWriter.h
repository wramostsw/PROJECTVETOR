// cnx/export/SvgWriter.h — escritor SVG 1.1 (§8, docs §1.24).
//
//   <svg width="{W·ppp}pt" height="{H·ppp}pt" viewBox="0 0 W H">     tamanho em pontos, coordenadas em pixels
//   <path fill="#rrggbb" opacity="1.00" [stroke="#rrggbb" stroke-width="…"] d="M x y C … L x y Z"/>
//   <g id="c_rrggbbaa"> … </g>                                         modo agrupado por cor
//
// Coordenadas com `coordinateDecimals` casas fixas (sem notação científica, independente de locale, nunca
// "-0"). A opacidade sai com 2 casas, como pede a especificação. A largura do traço é dada em pontos
// (0,09375 pt) e convertida para unidades do viewBox (px = pt / pointsPerPixel). O id do grupo não é o
// "#rrggbbaa" da especificação: um id XML não pode começar por '#' nem por dígito, então é "c_rrggbbaa".
#pragma once

#include <iosfwd>
#include <string>

#include "cnx/export/Exporter.h"

namespace cnx {

class SvgWriter final : public IShapeWriter {
public:
    SvgWriter(std::ostream& out, const ExporterSettings& settings);

    void beginFile(const DocumentInfo& info) override;
    void beginGroup(const sla::Rgba& color) override;
    void endGroup() override;
    void beginShape(const sla::Rgba& fill, const StrokeStyle* stroke) override;
    void moveTo(sla::Vec2 p) override;
    void lineTo(sla::Vec2 p) override;
    void curveTo(sla::Vec2 c1, sla::Vec2 c2, sla::Vec2 p) override;
    void closeSubpath() override;
    void endShape() override;
    void endFile() override;

    /// "#rrggbb" (canais arredondados de [0, 1] para 0..255).
    static std::string hexColor(const sla::Rgba& c);
    /// Número com `decimals` casas fixas, sem "-0".
    static std::string number(double v, int decimals);

private:
    void point(sla::Vec2 p);

    std::ostream& out_;
    int decimals_;
    double pointsPerPixel_;
    std::string indent_ = "  ";
    bool first_ = true;  ///< primeiro comando do caminho (sem espaço antes)
};

}  // namespace cnx
