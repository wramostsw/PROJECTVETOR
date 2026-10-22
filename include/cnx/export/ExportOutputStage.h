// cnx/export/ExportOutputStage.h — o estágio Output do CoreEngine que grava os arquivos pedidos (docs §1.24).
#pragma once

#include <vector>

#include "cnx/engine/CoreEngine.h"

namespace cnx {

/// Para cada ExportRequest, Exporter::writeFile sobre o VectorImage com curvas (snapshot.fitted). O progresso
/// é repartido igualmente entre os pedidos; o cancelamento é verificado a cada face. Lança ExportError se o
/// snapshot não tiver o resultado do ajuste, ou o erro do próprio Exporter.
class ExportOutputStage final : public IOutputStage {
public:
    void write(const PipelineSnapshot& snapshot, const std::vector<ExportRequest>& requests, TaskContext& ctx) override;
};

}  // namespace cnx
