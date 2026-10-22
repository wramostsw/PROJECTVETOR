// cnx/io/JpegDecoder.h — JPEG (ITU-T T.81 / JFIF / EXIF): o decodificador próprio do motor (docs §1.23).
//
// Modos: baseline (SOF0), sequencial estendido de 8 bits com Huffman (SOF1) e progressivo (SOF2: DC e AC,
// primeira passada e refinamento por aproximações sucessivas), com intervalos de reinício (DRI/RSTn). Todos
// decodificam para um buffer de coeficientes; no fim, desquantização (tabela fixada no primeiro scan de cada
// componente, como no libjpeg) e IDCT 8×8 separável em double. Amostragens H, V ∈ 1..4 por componente, com
// o croma ampliado por interpolação linear centrada (o "fancy upsampling" do libjpeg). Cores: 1 componente
// (cinza), 3 (YCbCr do JFIF, ou RGB pelo marcador Adobe ou pelos ids 'R','G','B') e 4 (CMYK ou YCCK na
// convenção Adobe, com os valores invertidos). DPI pelo JFIF (unidades 1 ou 2) ou, sem ele, pelo EXIF; a
// orientação EXIF (1..8) é aplicada: a imagem sai como os visualizadores a mostram.
//
// Não suportados (ImageDecodeError com o motivo): sem perdas, hierárquico, codificação aritmética, precisão
// de 12 bits, DNL (altura 0 no SOF).
#pragma once

#include <span>

#include "cnx/io/ImageDecode.h"

namespace cnx {

/// Lança ImageDecodeError (marcadores, tabelas ausentes ou inválidas, dados truncados, recurso não suportado)
/// e CancelTaskException.
SourceImage decodeJpeg(std::span<const std::uint8_t> data, const DecodeOptions& options = {});

}  // namespace cnx
