#pragma once
#include <mutex>

#include "gpu/state.h"

namespace d3d {

gpu::State& State();
std::recursive_mutex& Lock();
// a title about to read a surface's memory without a lock: the GPU's pixels are written back to it first
void SyncSurface(uint32_t data_va);
// The display's beam: the line being scanned out now, at 60 Hz with 525 lines a field of which 480 are shown, on the
// grid BlockUntilVerticalBlank returns on (a blank starts at each of its points). in_vblank past the shown lines.
// The NV2A's CRTC register a title reads it from is not emulated
void RasterStatus(bool& in_vblank, uint32_t& line);

}  // namespace d3d
