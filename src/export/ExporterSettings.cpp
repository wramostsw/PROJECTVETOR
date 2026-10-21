// src/export/ExporterSettings.cpp — BackgroundSelection (desfazer/refazer) e impressões digitais de exportação.
#include <algorithm>

#include "cnx/export/Exporter.h"

namespace cnx {

bool BackgroundSelection::isShapeVisible(FaceId f) const noexcept {
    const std::vector<FaceId>& hidden = states_[cursor_];
    return !std::binary_search(hidden.begin(), hidden.end(), f);
}

void BackgroundSelection::setHidden(FaceId f, bool hidden) {
    std::vector<FaceId> next = states_[cursor_];
    const auto it = std::lower_bound(next.begin(), next.end(), f);
    const bool present = it != next.end() && *it == f;
    if (hidden && !present) {
        next.insert(it, f);
    } else if (!hidden && present) {
        next.erase(it);
    } else {
        return;  // nada muda: nenhum passo é registrado
    }
    states_.resize(cursor_ + 1);  // descarta o ramo de refazer
    states_.push_back(std::move(next));
    ++cursor_;
}

void BackgroundSelection::clear() {
    if (states_[cursor_].empty()) return;
    states_.resize(cursor_ + 1);
    states_.emplace_back();
    ++cursor_;
}

bool BackgroundSelection::undo() {
    if (!canUndo()) return false;
    --cursor_;
    return true;
}

bool BackgroundSelection::redo() {
    if (!canRedo()) return false;
    ++cursor_;
    return true;
}

Fingerprint BackgroundSelection::fingerprint(Fingerprint upstream) const {
    Fingerprint f;
    f.add(std::string_view("background"));
    for (const FaceId id : states_[cursor_]) f.add(id);
    f.add(static_cast<std::uint64_t>(states_[cursor_].size()));
    return f.add(upstream);
}

Fingerprint ExporterSettings::fingerprint(Fingerprint upstream) const {
    Fingerprint f;
    f.add(std::string_view("export")).add(static_cast<std::uint64_t>(static_cast<std::uint8_t>(cakeStackMode)));
    f.add(static_cast<int>(shapeStroking)).add(strokeWidthPt).add(static_cast<int>(epsCompatibilityMode));
    f.add(static_cast<std::uint64_t>(static_cast<std::uint8_t>(dxfMode))).add(flattenTolerancePx);
    f.add(coordinateDecimals).add(pointsPerPixel);
    return background.fingerprint(f.add(upstream));
}

}  // namespace cnx
