// cnx/io/ImageDecode.h — decodificação de imagens: erros, limites e o reconhecimento do formato (docs §1.23).
//
// Os decodificadores (PNG, JPEG e o inflate do zlib) são próprios, em C++20 puro, sem dependências. Entram
// bytes e sai um SourceImage (RGBA de 8 bits straight, DPI, se há transparência e o formato real). Não há gerência
// de cor: os valores saem como estão no arquivo (gAMA, cHRM, sRGB e iCCP do PNG e o perfil ICC do JPEG são
// ignorados).
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>

#include "cnx/engine/CoreEngine.h"

namespace cnx {

/// Arquivo inválido, truncado, corrompido ou com um recurso não suportado (a mensagem diz qual).
class ImageDecodeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct DecodeOptions {
    /// Largura × altura máxima, conferida no cabeçalho antes de alocar (proteção contra arquivos-bomba).
    std::uint64_t maxPixels = std::uint64_t{1} << 28;  // 268 Mpx ≈ 1 GiB em RGBA de 8 bits
    /// Consultado a cada linha (PNG), a cada linha de MCUs (JPEG) e a cada 64 KiB descomprimidos:
    /// pedido de parada → CancelTaskException.
    std::stop_token stop;
};

enum class ImageFormat : std::uint8_t { Png, Jpeg };

/// Pela assinatura: PNG = 89 50 4E 47 0D 0A 1A 0A; JPEG = FF D8 FF. nullopt para o resto.
std::optional<ImageFormat> sniffImageFormat(std::span<const std::uint8_t> data) noexcept;

/// Reconhece o formato e decodifica. formatHint = ".png" ou ".jpeg" (o formato real, não a extensão do
/// arquivo). Lança ImageDecodeError e CancelTaskException.
SourceImage decodeImage(std::span<const std::uint8_t> data, const DecodeOptions& options = {});

namespace detail {
/// Lança CancelTaskException se a parada foi pedida.
void checkStop(const std::stop_token& stop);
/// Lança ImageDecodeError se w·h passa de maxPixels (ou se w ou h for 0).
void checkDimensions(std::uint64_t width, std::uint64_t height, const DecodeOptions& options, const char* format);
}  // namespace detail

}  // namespace cnx
