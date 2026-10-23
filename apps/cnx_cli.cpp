// cnx_cli — protótipo de linha de comando do motor: lê um PNG/JPEG, roda as sete etapas do CoreEngine e grava um SVG.
//
//   cnx_cli [--perfil 0..8] [--cores N] entrada.png saida.svg
//
// Sem opções o classificador escolhe o perfil e a paleta escolhe o número de tintas.
//
// Só costura peças que já existem (ImageFileLoader, CoreEngine, ExportOutputStage). Não tem lógica de vetorização própria.
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cnx/classify/Classifier.h"
#include "cnx/engine/CoreEngine.h"
#include "cnx/export/ExportOutputStage.h"
#include "cnx/export/Exporter.h"
#include "cnx/export/SvgWriter.h"
#include "cnx/vector/BoundaryTracer.h"
#include "cnx/io/ImageFileLoader.h"
#include "cnx/segment/PixelMap.h"
#include "cnx/vector/VectorImage.h"

namespace {

/// Escreve o progresso de cada etapa no terminal. Os callbacks vêm da thread do orquestrador.
class ConsoleObserver final : public cnx::IEngineObserver {
public:
    void onProgress(cnx::Stage, double) override {}
    void onStageFinished(cnx::Stage stage, double seconds, bool reused) override {
        std::printf("  %-15s %7.3f s%s\n", std::string(cnx::toString(stage)).c_str(), seconds, reused ? "  (reaproveitada)" : "");
    }
    void onRunFinished(cnx::RunStatus, const std::string&) override {}
};

}  // namespace

int main(int argc, char** argv) {
    cnx::EngineParameters params;
    bool stages = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--etapas") {
            stages = true;
        } else if (arg == "--apf" && i + 1 < argc) {
            try {
                cnx::ParameterFiler::readAPF(std::filesystem::path(argv[++i]), params);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "--apf: %s\n", e.what());
                return 2;
            }
        } else if (arg == "--set" && i + 2 < argc) {
            // Um parâmetro avulso no formato do .apf: --set ContourSmoother::prior_strengths 10
            std::istringstream line(std::string(argv[i + 1]) + " " + argv[i + 2]);
            i += 2;
            try {
                cnx::ParameterFiler::readAPF(line, params);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "--set: %s\n", e.what());
                return 2;
            }
        } else if (arg == "--limiar" && i + 1 < argc) {
            // Teto da fusão de regiões, em níveis de cor (0-255): dist² = (níveis/255)².
            const double levels = std::atof(argv[++i]);
            if (levels <= 0.0) {
                std::fprintf(stderr, "--limiar precisa ser > 0\n");
                return 2;
            }
            params.segment.mergeThreshMax = (levels / 255.0) * (levels / 255.0);
            params.segment.mergeThreshStart = std::min(params.segment.mergeThreshStart, params.segment.mergeThreshMax);
        } else if ((arg == "--perfil" || arg == "--cores") && i + 1 < argc) {
            const int value = std::atoi(argv[++i]);
            if (arg == "--perfil") {
                if (value < 0 || value > 9) {
                    std::fprintf(stderr, "--perfil vai de 0 (LogoHighAA) a 8 (PhotoLow), ou 9 (Custom: o perfil não reescreve nada)\n");
                    return 2;
                }
                params.classifier.autoClassify = false;
                params.classifier.forcedProfile = static_cast<cnx::ImageProfile>(value);
            } else {
                if (value < 1) {
                    std::fprintf(stderr, "--cores precisa ser >= 1\n");
                    return 2;
                }
                params.palette.forcedNumColors = value;
                if (value > params.palette.maxColors) params.palette.maxColors = value;  // o K forçado é limitado por maxColors
            }
        } else {
            files.push_back(arg);
        }
    }
    if (files.size() != 2) {
        std::fprintf(stderr,
                     "uso: cnx_cli [--perfil 0..8] [--cores N] [--limiar niveis] [--apf arquivo.apf] [--set Classe::parametro valor] [--etapas] entrada.(png|jpg) saida.svg\n"
                     "  as opções valem na ordem em que aparecem\n"
                     "  perfis: 0-2 logo com AA (alto/médio/baixo), 3-5 logo sem AA, 6-8 foto, 9 Custom (só os parâmetros dados); sem --perfil o classificador escolhe\n");
        return 2;
    }
    const std::filesystem::path input = files[0];
    const std::filesystem::path output = files[1];

    cnx::CoreEngine engine(std::make_shared<cnx::ImageFileLoader>());
    engine.setParameters(params);
    engine.setObserver(std::make_shared<ConsoleObserver>());
    engine.setInput(input);
    engine.setOutputStage(std::make_shared<cnx::ExportOutputStage>());
    cnx::ExportRequest request;
    request.file = output;
    engine.setExportRequests({request});

    std::printf("cnx: %s -> %s\n", input.string().c_str(), output.string().c_str());
    const auto t0 = std::chrono::steady_clock::now();
    engine.start(cnx::Stage::Output);
    const cnx::RunStatus status = engine.wait();
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (status != cnx::RunStatus::Completed) {
        std::fprintf(stderr, "falhou: %s\n", engine.lastMessage().c_str());
        return 1;
    }

    const auto snap = engine.snapshot();
    std::printf("imagem %dx%d, perfil %d, %zu tintas na paleta, %zu faces no vetor\n", snap->source->pixels.width(),
                snap->source->pixels.height(), static_cast<int>(snap->classification->profile), snap->palette->colors.size(),
                snap->fitted->faceCount());
    {
        // Faces minúsculas: as faixas de 1 px de uma rampa de antialiasing (bordas de letra) não têm como ser suavizadas.
        std::size_t tiny = 0;
        std::size_t thin = 0;
        for (cnx::FaceId f = 1; f < static_cast<cnx::FaceId>(snap->fitted->faceCount()); ++f) {
            const double area = std::abs(snap->fitted->face(f).area);
            tiny += area < 10.0 ? 1 : 0;
            thin += area < 60.0 ? 1 : 0;
        }
        std::printf("faces com menos de 10 px2: %zu; com menos de 60 px2: %zu\n", tiny, thin);
    }
    std::printf("total %.3f s; gravado em %s (%ju bytes)\n", total, output.string().c_str(),
                static_cast<std::uintmax_t>(std::filesystem::file_size(output)));
    if (stages) {
        // O que cada etapa entrega ao SVG: a escadinha da segmentação e a poligonal já suavizada (antes do Bézier).
        const cnx::ExporterSettings settings;
        const auto write = [&](const cnx::VectorImage& v, const char* suffix) {
            std::filesystem::path file = output;
            file.replace_extension(std::string(suffix) + ".svg");
            std::ofstream out(file, std::ios::binary);
            cnx::SvgWriter writer(out, settings);
            cnx::Exporter(v, settings).write(writer);
            std::printf("etapa gravada em %s\n", file.string().c_str());
        };
        write(cnx::BoundaryTracer().trace(*snap->segmentation).image, ".escadinha");
        write(*snap->smoothed, ".suavizado");
    }
    return 0;
}
