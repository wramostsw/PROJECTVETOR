// cnx/io/PngDecoder.h — PNG (ISO/IEC 15948 / W3C PNG): o decodificador próprio do motor (docs §1.23).
//
// Suporta os 5 tipos de cor (cinza, RGB, paleta, cinza+alfa, RGBA) em todas as profundidades válidas
// (1, 2, 4, 8, 16), os 5 filtros, o entrelaçamento Adam7, tRNS (chave de cor ou alfa da paleta) e pHYs
// (DPI). O CRC de cada chunk é conferido: num chunk crítico, erro; num ancilar, ele é ignorado. 16 bits →
// 8 por arredondamento (v·255 + 32767)/65535; profundidades < 8 de cinza são escaladas exatamente
// (1 → ×255, 2 → ×85, 4 → ×17). Sem gerência de cor (gAMA, cHRM, sRGB, iCCP ignorados); APNG: só a imagem
// padrão. hasAlpha = o arquivo carrega transparência (canal alfa ou tRNS).
#pragma once

#include <span>

#include "cnx/io/ImageDecode.h"

namespace cnx {

/// Lança ImageDecodeError (assinatura, IHDR, chunk crítico desconhecido ou corrompido, IDAT não
/// consecutivos, índice fora da paleta, dados truncados ou a mais) e CancelTaskException.
SourceImage decodePng(std::span<const std::uint8_t> data, const DecodeOptions& options = {});

}  // namespace cnx
