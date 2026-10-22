// cnx/io/Inflate.h — descompressão zlib (RFC 1950) / DEFLATE (RFC 1951), para o PNG.
//
// Blocos armazenados, Huffman fixo e dinâmico. Huffman canônico com uma tabela rápida de 9 bits e o caminho
// bit a bit (como o puff.c de Mark Adler) para os códigos mais longos. Códigos superdimensionados são erro;
// incompletos só são aceitos no caso que a RFC permite (um único código de comprimento 1). A saída tem um teto
// (tamanho esperado do PNG): passar dele é erro, o que barra as "bombas" de descompressão.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stop_token>
#include <vector>

namespace cnx {

/// Fluxo zlib completo: cabeçalho (CM = 8, janela ≤ 32 KiB, sem dicionário, FCHECK), DEFLATE e Adler-32.
/// A saída deve ter exatamente `expectedSize` bytes (nem mais, nem menos). Bytes depois do Adler-32 são
/// ignorados. Lança ImageDecodeError e CancelTaskException.
std::vector<std::uint8_t> inflateZlib(std::span<const std::uint8_t> data, std::size_t expectedSize, const std::stop_token& stop = {});

/// DEFLATE cru (sem o envelope zlib), até `maxSize` bytes de saída. Devolve o que foi descomprimido.
std::vector<std::uint8_t> inflateRaw(std::span<const std::uint8_t> data, std::size_t maxSize, const std::stop_token& stop = {});

std::uint32_t adler32(std::span<const std::uint8_t> data) noexcept;
/// CRC-32 (polinômio 0xEDB88320, o do PNG e do zlib).
std::uint32_t crc32(std::span<const std::uint8_t> data, std::uint32_t crc = 0) noexcept;

}  // namespace cnx
