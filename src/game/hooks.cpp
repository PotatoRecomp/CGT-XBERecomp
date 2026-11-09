// Hooks for default.xbe. A strong definition replaces the recompiled function at that address, for example:
//
// #include "xdk/hle.h"
//
// X86_EXTERN_FUNC(__imp__sub_00012345);
// X86_FUNC(sub_00012345) {
//     __imp__sub_00012345(ctx, base);  // the original
// }
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "common/mouse_keyboard.h"
#include "common/options.h"
#include "core/log.h"
#include "xdk/hle.h"

namespace {

using kernel::R32;
using kernel::W32;

float RF(uint32_t va) {
    uint32_t v = R32(va);
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}
void WF(uint32_t va, float f) {
    uint32_t v;
    std::memcpy(&v, &f, 4);
    W32(va, v);
}

}  // namespace

// 60 fps. The renderer's frame pacer (0x112D2D, thiscall on the renderer) waits for vertical blanks until the
// present interval its present parameters hold at +0xAC has passed since the last frame, then swaps: 2 on the
// console, 30 fps. Run with 1 it waits one blank, 60 fps. The game's clock (0x2426xx) steps the simulation by the
// milliseconds that passed, averaged over 30 frames, so the game keeps its speed. The interval is given back after,
// so nothing else of the game sees it changed.
X86_EXTERN_FUNC(__imp__sub_00112D2D);
X86_FUNC(sub_00112D2D) {
    using kernel::R32;
    using kernel::W32;
    constexpr uint32_t kInterval = 0xAC;
    uint32_t renderer = ctx.ecx;
    uint32_t interval = renderer ? R32(renderer + kInterval) : 0;
    bool unlock = game::Opt().frame_rate_60 && (interval == 2 || interval == 0x80000002u);
    if (unlock) W32(renderer + kInterval, interval - 1);
    __imp__sub_00112D2D(ctx, base);
    if (unlock) W32(renderer + kInterval, interval);
}

// Mouse look, straight into the game's look angles rather than through the right stick.
//
// The stick's path: the input manager reads the pad (0x13C2EB) with a dead zone, the player control (0xC41B9) takes
// its look actions into +0x3C (turn) and +0x40 (pitch) from -1 to 1, scaled down by the zoom (+0x40C) and nudged by
// aim assist, and the character's look (0x5F51F) turns those into degrees: the camera's pitch by
// dt * look * speed * 0.75, clamped to +-45 (or to a window round a mounted gun or cover), and its turn rate, kept at
// +0x678 in degrees a second. On foot the camera takes its yaw from the body, and the body (0x7ADEA) turns by a rate
// at +0x67C that follows +0x678 with a 0.1 s lag, as an angular velocity its physics body integrates.
//
// The mouse (and on Android a finger's swipe, input::OnTouchLook) skips all of it: its counts times the sensitivity are
// degrees, added to the camera's pitch and yaw before the look function's own clamps and wrap run on them, and handed
// to the body as exactly that much turn this frame, outside the 0.1 s lag. Positive yaw turns right and positive pitch
// looks down, so counts right and down add.
namespace {

constexpr uint32_t kGameApp = 0x4DA6B0;        // +8 the input manager
constexpr uint32_t kInputAnyPad = 0x4BC;       // input manager: one player on whichever controller pressed start
constexpr uint32_t kClock = 0x4DA708;          // +0xB0 the frame's seconds
constexpr uint32_t kPlayerControl = 0x3A0;     // character: its player control, 0 for the AI
constexpr uint32_t kControlState = 0x418;      // player control: 2 while the player has no say (scripted)
constexpr uint32_t kControlPlayer = 0x454;     // player control: player number, -1 none
constexpr uint32_t kControlZoom = 0x40C;       // player control: tan(fov) / tan(40 degrees), 0.25 to 1
constexpr uint32_t kTurnTarget = 0x678;        // character: degrees a second the look asks of the body
constexpr uint32_t kTurnRate = 0x67C;          // character: the body's turn rate, following kTurnTarget

// the turn the look gave the body this frame, for the body's update; game thread only
struct {
    uint32_t character = 0;
    float yaw = 0;
} g_body_turn;

// XBE_MOUSE_LOG: each frame the mouse turned the view, the yaw it asked for the frame before and the yaw the camera
// came round by since (equal while the stick rests), to check the body turned by exactly the mouse's degrees
struct {
    bool on = std::getenv("XBE_MOUSE_LOG") != nullptr;
    uint32_t character = 0;
    float asked = 0, yaw = 0;
} g_mouse_log;

// the character's mouse look in degrees, when it is the player's and the mouse moved
bool MouseLook(uint32_t character, float& yaw, float& pitch) {
    uint32_t control = R32(character + kPlayerControl);
    if (!control || R32(control + kControlState) == 2) return false;
    uint32_t player = R32(control + kControlPlayer);
    uint32_t input = R32(R32(kGameApp) + 8);
    bool any_pad = input && R32(input + kInputAnyPad) != 0;
    // the mouse is on port 0: player 1, or the one player of a game taken by any controller
    if (player == 0xFFFFFFFFu || (player != 0 && !any_pad)) return false;
    if (!input::TakeLook(yaw, pitch)) return false;
    float zoom = RF(control + kControlZoom);
    if (!(zoom > 0.0f && zoom <= 1.0f)) zoom = 1.0f;
    yaw *= zoom;
    pitch *= zoom;
    return true;
}

}  // namespace

// the character's look: thiscall (pitch*, yaw*, turn speed), ret 0xC, 1 while the player looks about (the camera
// does not swing back behind the player then)
X86_EXTERN_FUNC(__imp__sub_0005F51F);
X86_FUNC(sub_0005F51F) {
    uint32_t character = ctx.ecx;
    uint32_t pitch_at = R32(ctx.esp + 4), yaw_at = R32(ctx.esp + 8);
    float yaw = 0, pitch = 0;
    bool mouse = MouseLook(character, yaw, pitch);
    if (g_mouse_log.on && character == g_mouse_log.character) {
        float camera_yaw = RF(yaw_at);
        if (g_mouse_log.asked != 0 || mouse)
            LOGR("mouse: asked %.3f turned %.3f", g_mouse_log.asked,
                 std::remainder(camera_yaw - g_mouse_log.yaw, 360.0f));
        g_mouse_log.asked = 0;
        g_mouse_log.yaw = camera_yaw;
    }
    if (mouse) {
        if (g_mouse_log.on) {
            g_mouse_log.character = character;
            g_mouse_log.asked = yaw;
            g_mouse_log.yaw = RF(yaw_at);
        }
        WF(pitch_at, RF(pitch_at) + pitch);
        WF(yaw_at, RF(yaw_at) + yaw);
        // set, not added: a turn the body never took is not carried into a later frame
        g_body_turn = {character, yaw};
    }
    __imp__sub_0005F51F(ctx, base);
    if (mouse) ctx.eax = 1;
}

// the character's body update: the mouse's turn added to the turn rate and its target for this frame only, so the
// body's 0.1 s follow passes it through whole and the physics turns the body by exactly those degrees in this frame's
// seconds; taken back after, so the stick's turn keeps its own smoothing
X86_EXTERN_FUNC(__imp__sub_0007ADEA);
X86_FUNC(sub_0007ADEA) {
    uint32_t character = ctx.ecx;
    float rate = 0;
    if (g_body_turn.character == character && g_body_turn.yaw != 0) {
        float dt = RF(R32(kClock) + 0xB0);
        if (dt > 0) rate = g_body_turn.yaw / dt;
        g_body_turn.yaw = 0;
    }
    if (rate != 0) {
        WF(character + kTurnTarget, RF(character + kTurnTarget) + rate);
        WF(character + kTurnRate, RF(character + kTurnRate) + rate);
    }
    __imp__sub_0007ADEA(ctx, base);
    if (rate != 0) {
        WF(character + kTurnTarget, RF(character + kTurnTarget) - rate);
        WF(character + kTurnRate, RF(character + kTurnRate) - rate);
    }
}
