// src/io/ImageDecode.cpp — reconhecimento do formato, limites e o carregador de arquivos.
#include "cnx/io/ImageDecode.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <system_error>

#include "cnx/engine/TaskContext.h"
#include "cnx/io/ImageFileLoader.h"
#include "cnx/io/JpegDecoder.h"
#include "cnx/io/PngDecoder.h"

namespace cnx {

namespace detail {

void checkStop(const std::stop_token& stop) {
    if (stop.stop_requested()) throw CancelTaskException{};
}

void checkDimensions(std::uint64_t width, std::uint64_t height, const DecodeOptions& options, const char* format) {
    if (width == 0 || height == 0) throw ImageDecodeError(std::string(format) + ": largura ou altura 0");
    if (width > 0x7FFFFFFFu || height > 0x7FFFFFFFu || width * height > options.maxPixels) {
        throw ImageDecodeError(std::string(format) + ": imagem de " + std::to_string(width) + " × " + std::to_string(height) + " passa do limite de " +
                               std::to_string(options.maxPixels) + " pixels");
    }
}

}  // namespace detail

std::optional<ImageFormat> sniffImageFormat(std::span<const std::uint8_t> d) noexcept {
    static constexpr std::array<std::uint8_t, 8> kPng{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (d.size() >= kPng.size() && std::equal(kPng.begin(), kPng.end(), d.begin())) return ImageFormat::Png;
    if (d.size() >= 3 && d[0] == 0xFF && d[1] == 0xD8 && d[2] == 0xFF) return ImageFormat::Jpeg;
    return std::nullopt;
}

SourceImage decodeImage(std::span<const std::uint8_t> data, const DecodeOptions& options) {
    const std::optional<ImageFormat> f = sniffImageFormat(data);
    if (!f) throw ImageDecodeError("formato de imagem não reconhecido (só PNG e JPEG)");
    return *f == ImageFormat::Png ? decodePng(data, options) : decodeJpeg(data, options);
}

std::vector<std::uint8_t> ImageFileLoader::readFile(const std::filesystem::path& file, std::uintmax_t maxBytes, const std::stop_token& stop) {
    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(file, ec);
    if (ec) throw ImageDecodeError("não foi possível ler " + file.string() + ": " + ec.message());
    if (size > maxBytes) throw ImageDecodeError(file.string() + ": arquivo de " + std::to_string(size) + " bytes passa do limite");
    std::ifstream in(file, std::ios::binary);
    if (!in) throw ImageDecodeError("não foi possível abrir " + file.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    constexpr std::size_t kChunk = std::size_t{1} << 20;
    for (std::size_t done = 0; done < bytes.size();) {
        detail::checkStop(stop);
        const std::size_t n = std::min(kChunk, bytes.size() - done);
        in.read(reinterpret_cast<char*>(bytes.data() + done), static_cast<std::streamsize>(n));
        if (static_cast<std::size_t>(in.gcount()) != n) throw ImageDecodeError("leitura incompleta de " + file.string());
        done += n;
    }
    return bytes;
}

SourceImage ImageFileLoader::load(const std::filesystem::path& file, const std::stop_token& stop) {
    const std::vector<std::uint8_t> bytes = readFile(file, limits_.maxFileBytes, stop);
    DecodeOptions options;
    options.maxPixels = limits_.maxPixels;
    options.stop = stop;
    return decodeImage(bytes, options);
}

}  // namespace cnx
