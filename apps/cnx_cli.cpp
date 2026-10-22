// cnx_cli — protótipo de linha de comando do motor: lê um PNG/JPEG, roda as sete etapas do CoreEngine e grava um SVG.
//
//   cnx_cli [--perfil 0..8] [--cores N] entrada.png saida.svg
//
// Sem opções o classificador escolhe o perfil e a paleta escolhe o número de tintas.
//
// Só costura peças que já existem (ImageFileLoader, CoreEngine, ExportOutputStage). Não tem lógica de vetorização própria.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "cnx/classify/Classifier.h"
#include "cnx/engine/CoreEngine.h"
#include "cnx/export/ExportOutputStage.h"
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
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--perfil" || arg == "--cores") && i + 1 < argc) {
            const int value = std::atoi(argv[++i]);
            if (arg == "--perfil") {
                if (value < 0 || value > 8) {
                    std::fprintf(stderr, "--perfil vai de 0 (LogoHighAA) a 8 (PhotoLow)\n");
                    return 2;
                }
                params.classifier.autoClassify = false;
                params.classifier.forcedProfile = static_cast<cnx::ImageProfile>(value);
            } else {
                params.palette.forcedNumColors = value;
            }
        } else {
            files.push_back(arg);
        }
    }
    if (files.size() != 2) {
        std::fprintf(stderr,
                     "uso: cnx_cli [--perfil 0..8] [--cores N] entrada.(png|jpg) saida.svg\n"
                     "  perfis: 0-2 logo com AA (alto/médio/baixo), 3-5 logo sem AA, 6-8 foto; sem --perfil o classificador escolhe\n");
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
    std::printf("total %.3f s; gravado em %s (%ju bytes)\n", total, output.string().c_str(),
                static_cast<std::uintmax_t>(std::filesystem::file_size(output)));
    return 0;
}
