#pragma once
#include <cstdint>
#include <functional>

namespace audio {

constexpr uint32_t kRate = 48000;

// the platform output pulls interleaved stereo float frames from render
void StartOutput(std::function<void(float* out, uint32_t frames)> render);

// the longest the mixer held DirectSound's lock for one block since the last take, in microseconds: the title's sound
// calls wait for it (a hitch report); any thread
uint32_t TakeMixMaxUs();

}  // namespace audio
