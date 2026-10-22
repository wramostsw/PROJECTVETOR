// src/export/ExportOutputStage.cpp — grava os ExportRequests do CoreEngine.
#include "cnx/export/ExportOutputStage.h"

namespace cnx {

void ExportOutputStage::write(const PipelineSnapshot& snapshot, const std::vector<ExportRequest>& requests, TaskContext& ctx) {
    if (!snapshot.fitted) throw ExportError("exportação: o snapshot não tem o resultado do ajuste de Bézier");
    for (std::size_t i = 0; i < requests.size(); ++i) {
        ctx.checkpoint(static_cast<double>(i) / static_cast<double>(requests.size()));
        const ExportRequest& r = requests[i];
        Exporter(*snapshot.fitted, r.settings).writeFile(r.format, r.file, ctx.stopToken());
    }
    ctx.report(1.0);
}

}  // namespace cnx
