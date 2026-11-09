#include <algorithm>

#include <SDL3/SDL.h>
#include <toml++/toml.hpp>

#include "app/app.h"
#include "common/menu.h"
#include "common/mouse_keyboard.h"
#include "common/options.h"
#include "core/input.h"
#ifdef __ANDROID__
#include "host.h"
#endif

namespace game {
void InstallCaptureSignal();

namespace {

void Load(const void* table) {
    const toml::table& t = *static_cast<const toml::table*>(table);
    Options& o = Opt();
    o.msaa = uint32_t(t["msaa"].value_or(int64_t(o.msaa)));
    o.screen_filter = std::min(uint32_t(t["screen_filter"].value_or(int64_t(o.screen_filter))), uint32_t(kFilterHdr));
    o.mouse_sensitivity = std::clamp(o.mouse_sensitivity, uint32_t(5), uint32_t(1000));
#ifdef __ANDROID__
    o.touch_sensitivity = std::clamp(o.touch_sensitivity, uint32_t(10), uint32_t(500));
    o.controls_opacity = std::clamp(o.controls_opacity, uint32_t(0), uint32_t(100));
    host::LoadSettings(table);
#endif
}

void Save(void* table) {
    toml::table& t = *static_cast<toml::table*>(table);
    const Options& o = Opt();
    t.insert_or_assign("msaa", int64_t(o.msaa));
    t.insert_or_assign("screen_filter", int64_t(o.screen_filter));
#ifdef __ANDROID__
    host::SaveSettings(table);
#endif
}

}  // namespace
}  // namespace game

// what the runtime asks of the game; strings must be set, hooks may stay null
const app::GameInfo app::kGame = {
    .title = L"Conflict: Global Terror",   // window and splash
    .short_title = L"Conflict",            // message boxes
    .name = L"CGT",                        // window classes, crash dumps, handover event
    .log = "cgt",                          // log file prefix
    .settings = "settings.toml",           // settings file beside the exe
    .splash = L"splash",                   // optional splash.png or splash.bmp beside the exe
    .missing = L"Put the game in the assets folder: its disc image (.iso), or default.xbe with its data folder and mission files beside it.",
    .images = {{"default.xbe", "CGT"}},    // each recompiled xbe and its exe, for title launches
    .options = {                           // game settings kept in the settings file
        {"smaa", &game::Opt().smaa, nullptr},
        {"fxaa", &game::Opt().fxaa, nullptr},
        {"bloom", &game::Opt().bloom, nullptr},
        {"fog", &game::Opt().fog, nullptr},
        {"hdr_output", &game::Opt().hdr_output, nullptr},
        {"fps_counter", &game::Opt().fps_counter, nullptr},
        {"frame_rate_60", &game::Opt().frame_rate_60, nullptr},
        {"mouse_look", &game::Opt().mouse_look, nullptr},
        {"mouse_invert_y", &game::Opt().mouse_invert_y, nullptr},
        {"mouse_sensitivity", nullptr, &game::Opt().mouse_sensitivity},
        {"volume", nullptr, &game::Opt().volume},
#ifdef __ANDROID__
        {"touch_sensitivity", nullptr, &game::Opt().touch_sensitivity},
        {"touch_invert_y", &game::Opt().touch_invert_y, nullptr},
        {"controls_opacity", nullptr, &game::Opt().controls_opacity},
#endif
    },
    .load = game::Load,                    // reads more from the settings file's toml::table
    .save = game::Save,                    // writes more into it
    .init = game::InstallCaptureSignal,    // after the settings load, before the window and boot
    .loaded = nullptr,                     // image in memory, before its entry point runs
    .window = input::StartMouse,           // the game window is up
    .message = nullptr,                    // window messages first; true takes one
#ifdef __ANDROID__
    .ui = host::Ui,                        // the FPS counter; the settings panel is SettingsDialog.java
    .pad = host::Pad,                      // the keyboard's pad with the on-screen controls over it
#else
    .ui = app::DrawUi,                     // ImGui drawing each frame
    .pad = game::PadHook,                  // changes a polled pad
#endif
    .rumble = nullptr,                     // changes a rumble before it is set
    .event = game::Event,                  // Linux and Android: window thread, each SDL_Event after ImGui's
};
