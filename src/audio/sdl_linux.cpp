// the mixer's stereo output through SDL's audio stream
#include <SDL3/SDL.h>

#include <algorithm>
#include <vector>

#include "audio/audio.h"
#include "common/options.h"
#include "core/log.h"
#include "kernel/kernel.h"

namespace audio {

namespace {

constexpr uint32_t kOutChannels = 2;
constexpr uint32_t kFrames = kRate / 200;  // the mixer renders 5 ms at a time
// the device period asked for: 256 frames, 5.3 ms. PipeWire's default of 1024 put about 50 ms between a sound
// starting and the speakers, a gunshot heard late
constexpr const char* kDeviceFrames = "256";

struct Output {
    SDL_AudioStream* stream = nullptr;
    std::function<void(float*, uint32_t)> render;
    std::vector<float> mix = std::vector<float>(kFrames * kOutChannels);
};
Output g_out;

// SDL asks for more: the mixer renders it 5 ms at a time
void SDLCALL Feed(void* user, SDL_AudioStream* stream, int additional, int) {
    Output& o = *static_cast<Output*>(user);
    while (additional > 0) {
        o.render(o.mix.data(), kFrames);
        // the player's volume, past the mixer's own limit and held within full scale
        float gain = float(game::Opt().volume) / 100.0f;
        if (gain != 1.0f)
            for (float& x : o.mix) x = std::clamp(x * gain, -1.0f, 1.0f);
        int bytes = int(kFrames * kOutChannels * sizeof(float));
        SDL_PutAudioStreamData(stream, o.mix.data(), bytes);
        additional -= bytes;
    }
}

}  // namespace

void StartOutput(std::function<void(float* out, uint32_t frames)> render) {
    g_out.render = std::move(render);
    // the audio device's threads mix on the host and run no guest code: started off the guest threads' one CPU
    kernel::HostThreadsFree free_threads;
    SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, kDeviceFrames);
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        LOGE("audio: SDL audio unavailable, no sound: %s", SDL_GetError());
        return;
    }
    SDL_AudioSpec spec{SDL_AUDIO_F32, int(kOutChannels), int(kRate)};
    g_out.stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, Feed, &g_out);
    if (!g_out.stream) {
        LOGE("audio: no audio device, no sound: %s", SDL_GetError());
        return;
    }
    SDL_ResumeAudioStreamDevice(g_out.stream);
    SDL_AudioSpec dev{};
    int period = 0;
    SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(g_out.stream), &dev, &period);
    LOGR("audio: SDL3 %u Hz stereo, %d frame device period (%s)", kRate, period, SDL_GetCurrentAudioDriver());
}

}  // namespace audio
