// The on-screen controls' native side. The controls themselves, laid out as the original Xbox pad's (the left stick
// walks, the face buttons, the triggers, White and Black are the pad's), are an Android view over the game,
// TouchControls.java: it draws them from images and reads the fingers, so the game's thread draws nothing for them.
// What the fingers hold comes here over JNI (native/settings_bridge.cpp) and is laid over the pad the keyboard left
// (src/common/keyboard.cpp), so the game reads one report as on the console.
//
// There is no on-screen right stick. A finger dragged on the free right side of the screen, or held on FIRE, turns the
// view as a mouse does: its travel goes to input::OnTouchLook and from there straight into the game's look angles
// (src/game/hooks.cpp), with none of the stick's dead zone, response curve, top turn speed or 0.1 s lag.
#include <toml++/toml.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>

#include <imgui.h>

#include "app/app.h"
#include "common/menu.h"
#include "common/mouse_keyboard.h"
#include "common/options.h"
#include "core/input.h"
#include "core/log.h"
#include "host.h"
#include "loose.h"

namespace touch {

namespace {

std::atomic<float> g_lx{0}, g_ly{0};  // the stick, -1 to 1, y up
std::atomic<uint32_t> g_analog{0};    // a bit per analog button, in input::Pad::analog order: held is full
std::atomic<uint32_t> g_buttons{0};   // input::Pad::buttons bits
std::atomic<int> g_pad{-1};           // a controller is plugged in, as the overlay was last told: -1 not yet told

// where each control sits: its centre as a fraction of the screen, in this order (TouchControls.java has the same).
// These are the pad's own layout, what a fresh install gets and what the editor's RESET puts back: FIRE under the
// right thumb beside the face buttons, so the thumb that fires can drag on to turn the view.
enum Unit : int {
    kUnitStick, kUnitA, kUnitB, kUnitX, kUnitY, kUnitDpad, kUnitStart, kUnitBack, kUnitLT, kUnitRT, kUnitL3, kUnitR3,
    kUnitGear, kUnitWhite, kUnitBlack, kUnitCount
};
struct Point {
    float x, y;
};
constexpr Point kDefaultUnits[kUnitCount] = {
    {0.130f, 0.740f},  // the stick
    {0.880f, 0.840f},  // A, down
    {0.945f, 0.700f},  // B, right
    {0.815f, 0.700f},  // X, left
    {0.880f, 0.560f},  // Y, up
    {0.100f, 0.380f},  // the arrows' middle
    {0.570f, 0.080f},  // START
    {0.430f, 0.080f},  // BACK
    {0.080f, 0.100f},  // LT
    {0.700f, 0.820f},  // RT, FIRE
    {0.260f, 0.910f},  // L3
    {0.920f, 0.330f},  // R3, AIM
    {0.500f, 0.080f},  // the gear
    {0.790f, 0.420f},  // White
    {0.920f, 0.100f},  // Black
};
std::array<Point, kUnitCount> g_units = [] {
    std::array<Point, kUnitCount> a{};
    std::copy(std::begin(kDefaultUnits), std::end(kDefaultUnits), a.begin());
    return a;
}();
// each control's size, a fraction of its kind's: the layout editor's size bar, 0.5 to 1.5
constexpr float kScaleMin = 0.5f, kScaleMax = 1.5f;
std::array<float, kUnitCount> g_scale = [] {
    std::array<float, kUnitCount> a{};
    a.fill(1.0f);
    return a;
}();

// the frame rate in the top left corner, in the HUD's corner the game leaves free
void FpsCounter(double fps) {
    ImGuiIO& io = ImGui::GetIO();
    float size = std::max(io.DisplaySize.y * 0.03f, 12.0f);
    char text[16];
    std::snprintf(text, sizeof(text), "%.0f FPS", fps);
    ImVec2 at(size * 0.5f, size * 0.25f);
    ImVec2 extent = ImGui::GetFont()->CalcTextSizeA(size, FLT_MAX, 0, text);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(ImVec2(at.x - size * 0.25f, at.y), ImVec2(at.x + extent.x + size * 0.25f, at.y + extent.y),
                      IM_COL32(0, 0, 0, 120), size * 0.2f);
    dl->AddText(ImGui::GetFont(), size, at, IM_COL32(115, 242, 140, 230), text);
}

}  // namespace

void Init() { LOGI("touch: on-screen controls drawn by the activity's overlay, swipe look through the mouse path"); }

void Set(uint32_t buttons, uint32_t analog, float lx, float ly) {
    g_buttons = buttons;
    g_analog = analog;
    g_lx = std::clamp(lx, -1.0f, 1.0f);
    g_ly = std::clamp(ly, -1.0f, 1.0f);
}

void Apply(input::Pad& pad) {
    uint32_t analog = g_analog.load();
    uint32_t buttons = g_buttons.load();
    float lx = g_lx.load(), ly = g_ly.load();
    if (analog || buttons || lx || ly) pad.connected = true;
    for (int i = 0; i < 8; i++)
        if (analog & (1u << i)) pad.analog[i] = 0xFF;
    pad.buttons |= uint16_t(buttons);
    if (lx || ly) {
        pad.lx = int16_t(std::lround(lx * 32767.0f));
        pad.ly = int16_t(std::lround(ly * 32767.0f));
    }
    // a controller the phone took over comes in over the top of both
    loose::Merge(pad);
}

void Frame() {
    // The overlay is made with the activity, before this thread has read the settings file: told at the first frame,
    // it reads the layout and the panel's values then, as the file has them. After that a controller plugged in takes
    // the pad's place: the overlay hides its controls, the gear and the swipe look staying.
    const int pad = input::Connected() || loose::Connected() ? 1 : 0;
    if (g_pad.exchange(pad) == pad) return;
    if (pad) Set(0, 0, 0, 0);
    host::ControlsSync(pad != 0);
}

int Units() { return kUnitCount; }

void GetUnit(int unit, float& x, float& y, float& scale) {
    if (unit < 0 || unit >= kUnitCount) return;
    x = g_units[size_t(unit)].x;
    y = g_units[size_t(unit)].y;
    scale = g_scale[size_t(unit)];
}

void SetUnit(int unit, float x, float y, float scale, bool commit) {
    if (unit < 0 || unit >= kUnitCount) return;
    g_units[size_t(unit)] = {std::clamp(x, 0.0f, 1.0f), std::clamp(y, 0.0f, 1.0f)};
    g_scale[size_t(unit)] = std::clamp(scale, kScaleMin, kScaleMax);
    if (commit) app::Save();
}

void ResetUnits() {
    std::copy(std::begin(kDefaultUnits), std::end(kDefaultUnits), g_units.begin());
    g_scale.fill(1.0f);
    app::Save();
}

// the layout as [[controls]] in the settings file, one entry per unit in the order above: its place and its size
void LoadLayout(const void* table) {
    const toml::table& t = *static_cast<const toml::table*>(table);
    const toml::array* a = t["controls"].as_array();
    if (!a) return;
    for (size_t i = 0; i < a->size() && i < size_t(kUnitCount); i++) {
        const toml::table* e = (*a)[i].as_table();
        if (!e) continue;
        g_units[i].x = std::clamp(float((*e)["x"].value_or(double(kDefaultUnits[i].x))), 0.0f, 1.0f);
        g_units[i].y = std::clamp(float((*e)["y"].value_or(double(kDefaultUnits[i].y))), 0.0f, 1.0f);
        g_scale[i] = std::clamp(float((*e)["scale"].value_or(1.0)), kScaleMin, kScaleMax);
    }
}

void SaveLayout(void* table) {
    toml::table& t = *static_cast<toml::table*>(table);
    toml::array a;
    for (int i = 0; i < kUnitCount; i++)
        a.push_back(toml::table{{"x", double(g_units[size_t(i)].x)},
                                {"y", double(g_units[size_t(i)].y)},
                                {"scale", double(g_scale[size_t(i)])}});
    t.insert_or_assign("controls", std::move(a));
}

}  // namespace touch

namespace host {

void Ui(double fps) {
    touch::Frame();
    if (game::Opt().fps_counter) touch::FpsCounter(fps);
}

// the keyboard and a mouse on port 0 as on Linux (an idle pad while the panel is open), the on-screen controls over it
void Pad(int port, input::Pad& p) {
    game::PadHook(port, p);
    if (port == 0 && !app::MenuOpen()) touch::Apply(p);
}

void LoadSettings(const void* table) { touch::LoadLayout(table); }
void SaveSettings(void* table) { touch::SaveLayout(table); }

}  // namespace host
