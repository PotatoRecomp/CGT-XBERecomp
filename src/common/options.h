#pragma once
#include <atomic>
#include <cstdint>

// The recomp's own settings, shown on the F10 panel (common/menu.cpp) and saved to the settings file; the renderer
// (d3d/vulkan_linux.cpp) reads them every frame.
namespace game {

// the present's look, as MostWanted-Android's screen filters
enum ScreenFilter : uint32_t { kFilterNone, kFilterSharpen, kFilterVivid, kFilterScanlines, kFilterCrt, kFilterHdr };

struct Options {
    bool bloom = true;        // the glow added over bright parts of the frame
    bool fog = true;
    uint32_t msaa = 0;        // samples a pixel of the scene, 0 off
    bool smaa = false;        // over the finished frame
    bool fxaa = false;        // at the present, after SMAA
    uint32_t screen_filter = kFilterNone;
    bool hdr_output = false;  // HDR10 on a display that offers it
    bool fps_counter = false;
    // the output's level in percent. The game mixes for a TV turned up: its effects at -12.5 dB and music at -20 dB,
    // every mix bin 6 dB lower again for headroom; 200 gives that 6 dB back
    uint32_t volume = 200;
    bool frame_rate_60 = false;  // the game's 30 fps pacer run at 60 (game/hooks.cpp)
    // the mouse turns the view by its counts, straight into the game's look angles (game/hooks.cpp)
    bool mouse_look = true;
    bool mouse_invert_y = false;
    uint32_t mouse_sensitivity = 100;  // percent of kMouseDegreesPerCount
};

// degrees of turn for a mouse count at 100% sensitivity, the scale most PC shooters use
constexpr float kMouseDegreesPerCount = 0.022f;

inline Options& Opt() {
    static Options options;
    return options;
}

inline std::atomic<bool> g_hdr_display{false};  // the window's display has HDR on, from the renderer

}  // namespace game
