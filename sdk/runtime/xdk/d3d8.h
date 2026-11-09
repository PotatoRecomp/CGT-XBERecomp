#pragma once
#include <mutex>

#include "gpu/state.h"

namespace d3d {

gpu::State& State();
std::recursive_mutex& Lock();
// a title about to read a surface's memory without a lock: the GPU's pixels are written back to it first
void SyncSurface(uint32_t data_va);

}  // namespace d3d
