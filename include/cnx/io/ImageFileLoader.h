// cnx/io/ImageFileLoader.h — o IImageLoader do motor: lê o arquivo e decodifica PNG ou JPEG (docs §1.23).
//
// O formato vem da assinatura dos bytes, não da extensão (um .jpg que é PNG é lido como PNG, e o classificador
// recebe ".png"). O arquivo é lido inteiro para a memória; o tamanho é limitado por maxFileBytes antes da
// leitura. Respeita o stop_token durante a leitura e a decodificação.
#pragma once

#include <cstdint>
#include <filesystem>
#include <stop_token>
#include <vector>

#include "cnx/engine/CoreEngine.h"
#include "cnx/io/ImageDecode.h"

namespace cnx {

class ImageFileLoader final : public IImageLoader {
public:
    struct Limits {
        std::uintmax_t maxFileBytes = std::uintmax_t{1} << 30;  ///< 1 GiB
        std::uint64_t maxPixels = std::uint64_t{1} << 28;
    };

    ImageFileLoader() = default;
    explicit ImageFileLoader(Limits limits) : limits_(limits) {}

    /// Lança ImageDecodeError (arquivo inexistente, grande demais, formato desconhecido ou inválido) e
    /// CancelTaskException.
    SourceImage load(const std::filesystem::path& file, const std::stop_token& stop) override;

    /// Os bytes do arquivo (exposto para testes e para quem decodifica da memória).
    static std::vector<std::uint8_t> readFile(const std::filesystem::path& file, std::uintmax_t maxBytes, const std::stop_token& stop = {});

private:
    Limits limits_;
};

}  // namespace cnx
