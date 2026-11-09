// Keyboard and mouse buttons as the controller on port 0, on top of a real pad plugged in there (as UCRR and Black).
// Mouse motion is not a stick: its raw counts are added up here and turned into the game's look angles directly by
// game/hooks.cpp, past the stick's dead zone, response, clamp and turn smoothing.
#include <SDL3/SDL.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "app/app.h"
#include "core/log.h"
#include "common/menu.h"
#include "common/mouse_keyboard.h"
#include "common/options.h"

namespace input {

namespace {

SDL_Window* g_window = nullptr;
std::atomic<uint64_t> g_wheel_until{0};
constexpr uint64_t kWheelPulseMs = 120;  // a notch holds its button this long, long enough for a polled frame

uint64_t NowMs() {
    auto since = std::chrono::steady_clock::now().time_since_epoch();
    return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(since).count()) + 1;
}

// counts since the game last took them; doubles so a long session of small moves loses nothing
std::mutex g_look_lock;
double g_look_x = 0, g_look_y = 0;
uint64_t g_look_taken = 0;           // NowMs of the last take
constexpr uint64_t kLookStaleMs = 250;  // counts older than this were made while nothing turned the view

}  // namespace

bool Focused() { return g_window && (SDL_GetWindowFlags(g_window) & SDL_WINDOW_INPUT_FOCUS); }

void StartMouse(void* window) {
    g_window = static_cast<SDL_Window*>(window);
    // the device's own counts in relative mode: no desktop pointer speed or acceleration curve on top
    SDL_SetHint(SDL_HINT_MOUSE_RELATIVE_SYSTEM_SCALE, "0");
    SDL_SetHint(SDL_HINT_MOUSE_RELATIVE_MODE_CENTER, "1");
    SyncMouseCapture();
}

void SyncMouseCapture() {
    if (!g_window) return;
    bool want = game::Opt().mouse_look && !app::MenuOpen() && Focused();
    if (SDL_GetWindowRelativeMouseMode(g_window) == want) return;
    SDL_SetWindowRelativeMouseMode(g_window, want);
    std::lock_guard lock(g_look_lock);
    g_look_x = g_look_y = 0;
}

void OnMouseMotion(float dx, float dy) {
    if (!g_window || !SDL_GetWindowRelativeMouseMode(g_window)) return;
    std::lock_guard lock(g_look_lock);
    g_look_x += dx;
    g_look_y += dy;
}

bool TakeMouseLook(float& dx, float& dy) {
    uint64_t now = NowMs();
    std::lock_guard lock(g_look_lock);
    bool stale = now - g_look_taken > kLookStaleMs;
    g_look_taken = now;
    dx = float(g_look_x);
    dy = float(g_look_y);
    g_look_x = g_look_y = 0;
    if (stale || !game::Opt().mouse_look) return false;
    return dx != 0 || dy != 0;
}

void OnMouseWheel(float y) {
    if (!app::MenuOpen() && y != 0) g_wheel_until = NowMs() + kWheelPulseMs;
}

bool WheelHeld() { return NowMs() < g_wheel_until.load(); }

}  // namespace input

namespace game {

namespace {

enum Button : uint16_t {
    kUp = 1 << 0,
    kDown = 1 << 1,
    kLeft = 1 << 2,
    kRight = 1 << 3,
    kStart = 1 << 4,
    kBack = 1 << 5,
    kLeftThumb = 1 << 6,
    kRightThumb = 1 << 7,
};
enum Analog { kA, kB, kX, kY, kBlack, kWhite, kLeftTrigger, kRightTrigger };

constexpr struct {
    SDL_Scancode key;
    uint16_t button;
} kButtons[] = {
    // D-pad: menus and picking a squad member
    {SDL_SCANCODE_UP, kUp},           {SDL_SCANCODE_DOWN, kDown},       {SDL_SCANCODE_LEFT, kLeft},
    {SDL_SCANCODE_RIGHT, kRight},     {SDL_SCANCODE_1, kUp},            {SDL_SCANCODE_2, kRight},
    {SDL_SCANCODE_3, kDown},          {SDL_SCANCODE_4, kLeft},
    {SDL_SCANCODE_RETURN, kStart},    {SDL_SCANCODE_ESCAPE, kStart},    {SDL_SCANCODE_TAB, kBack},
    {SDL_SCANCODE_BACKSPACE, kBack},  {SDL_SCANCODE_M, kBack},
    {SDL_SCANCODE_LSHIFT, kLeftThumb}, {SDL_SCANCODE_V, kRightThumb},   {SDL_SCANCODE_RSHIFT, kRightThumb},
};

constexpr struct {
    SDL_Scancode key;
    Analog analog;
} kAnalogs[] = {
    {SDL_SCANCODE_SPACE, kA}, {SDL_SCANCODE_E, kA},     {SDL_SCANCODE_F, kA},
    {SDL_SCANCODE_C, kB},     {SDL_SCANCODE_LCTRL, kB},
    {SDL_SCANCODE_R, kX},
    {SDL_SCANCODE_Q, kY},
    {SDL_SCANCODE_G, kBlack}, {SDL_SCANCODE_T, kWhite},
    {SDL_SCANCODE_Z, kLeftTrigger}, {SDL_SCANCODE_X, kRightTrigger},
};

// a pair of keys as one stick axis, the pad's own deflection kept when no key is down
int16_t Axis(const bool* keys, SDL_Scancode neg, SDL_Scancode pos, int16_t pad) {
    int v = (keys[pos] ? 32767 : 0) - (keys[neg] ? 32767 : 0);
    return v ? int16_t(v) : pad;
}

// XBE_PAD_SCRIPT: a file of "seconds buttons [seconds held]" lines, buttons joined by + (UP DOWN LEFT RIGHT START
// BACK A B X Y, and RX+ RX- RY+ RY- or RX=n RY=n for the right stick), held a tenth of a second by default and timed from
// the first poll; to reach a scene unattended
struct Press {
    double at, len;
    uint16_t buttons;
    uint8_t analogs;  // bit per Analog
    int16_t rx, ry;
};

std::vector<Press> LoadScript() {
    std::vector<Press> v;
    const char* path = std::getenv("XBE_PAD_SCRIPT");
    FILE* f = path ? std::fopen(path, "r") : nullptr;
    if (!f) return v;
    char line[160];
    while (std::fgets(line, sizeof(line), f)) {
        char names[96];
        Press p{0, 0.1, 0, 0, 0, 0};
        if (line[0] == '#' || std::sscanf(line, "%lf %95s %lf", &p.at, names, &p.len) < 2) continue;
        for (char* n = std::strtok(names, "+"); n; n = std::strtok(nullptr, "+")) {
            constexpr struct {
                const char* name;
                uint16_t button;
                int analog;
            } kNames[] = {{"UP", kUp, -1},       {"DOWN", kDown, -1}, {"LEFT", kLeft, -1}, {"RIGHT", kRight, -1},
                          {"START", kStart, -1}, {"BACK", kBack, -1}, {"A", 0, kA},        {"B", 0, kB},
                          {"X", 0, kX},          {"Y", 0, kY}};
            if (!std::strncmp(n, "RX=", 3)) p.rx = int16_t(std::atoi(n + 3));
            if (!std::strncmp(n, "RY=", 3)) p.ry = int16_t(std::atoi(n + 3));
            if (!std::strcmp(n, "RX+")) p.rx = 32767;
            if (!std::strcmp(n, "RX-")) p.rx = -32767;
            if (!std::strcmp(n, "RY+")) p.ry = 32767;
            if (!std::strcmp(n, "RY-")) p.ry = -32767;
            for (auto& k : kNames)
                if (!std::strcmp(n, k.name)) {
                    p.buttons |= k.button;
                    if (k.analog >= 0) p.analogs |= uint8_t(1 << k.analog);
                }
        }
        v.push_back(p);
    }
    std::fclose(f);
    return v;
}

void Script(input::Pad& p) {
    static const std::vector<Press> presses = LoadScript();
    if (presses.empty()) return;
    static const auto start = std::chrono::steady_clock::now();
    double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    p.connected = true;
    for (const Press& s : presses) {
        if (now < s.at || now >= s.at + s.len) continue;
        p.buttons |= s.buttons;
        if (s.rx) p.rx = s.rx;
        if (s.ry) p.ry = s.ry;
        for (int a = 0; a < 8; a++)
            if (s.analogs >> a & 1) p.analog[a] = 0xFF;
    }
}

// the keys and mouse buttons held, added to the pad; only while the game window has focus
void Keyboard(input::Pad& p) {
    if (!input::Focused()) return;
    const bool* keys = SDL_GetKeyboardState(nullptr);
    if (!keys) return;
    p.connected = true;
    for (auto [key, button] : kButtons)
        if (keys[key]) p.buttons |= button;
    for (auto [key, analog] : kAnalogs)
        if (keys[key]) p.analog[analog] = 0xFF;
    p.lx = Axis(keys, SDL_SCANCODE_A, SDL_SCANCODE_D, p.lx);
    p.ly = Axis(keys, SDL_SCANCODE_S, SDL_SCANCODE_W, p.ly);  // y up, as the console's stick
    // the right stick from the keypad
    p.rx = Axis(keys, SDL_SCANCODE_KP_4, SDL_SCANCODE_KP_6, p.rx);
    p.ry = Axis(keys, SDL_SCANCODE_KP_2, SDL_SCANCODE_KP_8, p.ry);

    SDL_MouseButtonFlags mouse = SDL_GetMouseState(nullptr, nullptr);
    if (mouse & SDL_BUTTON_LMASK) p.analog[kRightTrigger] = 0xFF;  // fire
    if (mouse & SDL_BUTTON_RMASK) p.buttons |= kRightThumb;  // aim / scope: the right stick click, as V
    if (mouse & SDL_BUTTON_MMASK) p.analog[kBlack] = 0xFF;
    if (mouse & SDL_BUTTON_X1MASK) p.analog[kWhite] = 0xFF;
    if (input::WheelHeld()) p.analog[kY] = 0xFF;  // the wheel changes weapon
}

}  // namespace

// GameInfo::pad: the keyboard and mouse on port 0; while the F10 panel is open the game gets an idle pad, the keys
// and the controller work the panel
void PadHook(int port, input::Pad& p) {
    if (app::MenuOpen()) {
        bool connected = p.connected || port == 0;
        p = {};
        p.connected = connected;
        return;
    }
    if (port != 0) return;
    Keyboard(p);
    Script(p);
}

// GameInfo::event: F10 and Escape for the panel, the mouse wheel and motion
bool Event(const void* sdl_event) {
    const SDL_Event& e = *static_cast<const SDL_Event*>(sdl_event);
    input::SyncMouseCapture();  // focus, the panel or the setting may have changed since the last event
    switch (e.type) {
        case SDL_EVENT_MOUSE_MOTION:
            input::OnMouseMotion(e.motion.xrel, e.motion.yrel);
            break;
        case SDL_EVENT_KEY_DOWN:
            if (!e.key.repeat && (e.key.key == SDLK_F10 || (e.key.key == SDLK_ESCAPE && app::MenuOpen()))) {
                {
                    std::lock_guard lock(app::UiLock());
                    app::ToggleMenu();
                }
                return true;
            }
            break;
        case SDL_EVENT_KEY_UP:  // TMPDBG F6 marks a sound heard wrong
            if (e.key.key == SDLK_F6 && std::getenv("XBE_SOUND_LOG")) LOGR("snd ===== MARK =====");
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            input::OnMouseWheel(e.wheel.y);
            break;
    }
    return false;
}

}  // namespace game
