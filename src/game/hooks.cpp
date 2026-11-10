// Hooks for default.xbe. A strong definition replaces the recompiled function at that address, for example:
//
// #include "xdk/hle.h"
//
// X86_EXTERN_FUNC(__imp__sub_00012345);
// X86_FUNC(sub_00012345) {
//     __imp__sub_00012345(ctx, base);  // the original
// }
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "common/mouse_keyboard.h"
#include "common/options.h"
#include "core/log.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"
#include "xdk/d3d8.h"
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

#ifdef __ANDROID__
// CanTakeDamage (thiscall, ret): 0x905DD checks this before subtracting damage from +0x13C;
// bullets/explosions (0xAA4E7) and environmental damage also consult it. +4 == 1 is a character,
// +0x15 its team; 0x43B7E8 is the player's team, as the game's own cheats check it (0x95700).
X86_EXTERN_FUNC(__imp__sub_0008F938);
X86_FUNC(sub_0008F938) {
    uint32_t unit = ctx.ecx;
    if (game::Opt().infinite_health && R32(unit + 4) == 1
            && kernel::G<uint8_t>(unit + 0x15) == kernel::G<uint8_t>(0x43B7E8)) {
        ctx.eax = 0;
        ctx.esp += 4;
        return;
    }
    __imp__sub_0008F938(ctx, base);
}

// +0x54 is total ammo, including reserve; +0x78 counts spent magazine rounds (0x9D2AC).
// Refill before readiness checks and after firing so an empty magazine never requests a reload.
namespace {
bool FillMagazine(uint32_t weapon, uint32_t unit) {
    if (!game::Opt().infinite_ammo || !unit || R32(unit + 4) != 1
            || kernel::G<uint8_t>(unit + 0x15) != kernel::G<uint8_t>(0x43B7E8)
            || R32(weapon + 4) != 6) return false;
    uint32_t capacity = kernel::R16(R32(weapon + 0x24) + 0xB0);
    if (!capacity) return false;
    W32(weapon + 0x54, std::max(R32(weapon + 0x54), capacity));
    W32(weapon + 0x78, 0);
    return true;
}
}  // namespace

X86_EXTERN_FUNC(__imp__sub_00095700);
X86_FUNC(sub_00095700) {
    FillMagazine(ctx.ecx, R32(ctx.esp + 4));
    __imp__sub_00095700(ctx, base);
}

X86_EXTERN_FUNC(__imp__sub_0009D37A);
X86_FUNC(sub_0009D37A) {
    uint32_t weapon = ctx.ecx, unit = R32(ctx.esp + 4);
    bool freeze = FillMagazine(weapon, unit);
    uint32_t ammo = freeze ? R32(weapon + 0x54) : 0;
    __imp__sub_0009D37A(ctx, base);
    if (freeze) {
        // Restoring only the magazine lets total ammo drain until reserve reaches zero.
        W32(weapon + 0x54, ammo);
        W32(weapon + 0x78, 0);
    }
}

// Throwable use (thiscall, ret 0xC): 0x971B9 decrements +0x54, outside the gun's firing path.
// Preserve the starting count rather than waiting for the built-in cheat's empty-item refill.
X86_EXTERN_FUNC(__imp__sub_00096975);
X86_FUNC(sub_00096975) {
    uint32_t item = ctx.ecx, unit = R32(ctx.esp + 4);
    bool freeze = game::Opt().infinite_ammo && unit && R32(unit + 4) == 1
            && kernel::G<uint8_t>(unit + 0x15) == kernel::G<uint8_t>(0x43B7E8);
    uint32_t count = freeze ? R32(item + 0x54) : 0;
    __imp__sub_00096975(ctx, base);
    if (freeze) W32(item + 0x54, count);
}

// The built-in infinite-ammo query (cdecl, ret). 0x95700 supplies rounds when checking a player's
// weapon; 0x96975 refills it when the last round fires, and 0x9D37A suppresses the empty-ammo action.
X86_EXTERN_FUNC(__imp__sub_000A5522);
X86_FUNC(sub_000A5522) {
    if (game::Opt().infinite_ammo) {
        ctx.eax = 1;
        ctx.esp += 4;
        return;
    }
    __imp__sub_000A5522(ctx, base);
}
#endif

// 60 fps. The renderer's frame pacer (0x112D2D, thiscall on the renderer) waits for vertical blanks until the
// present interval its present parameters hold at +0xAC has passed since the last frame, then swaps: 2 on the
// console, 30 fps. Run with 1 it waits one blank, 60 fps. The game's clock (0x2426xx) steps the simulation by the
// milliseconds that passed, averaged over 30 frames, so the game keeps its speed. The interval is given back after,
// so nothing else of the game sees it changed.
//
// The pacer's clocks. It times frames on the title's millisecond tick (0x10739C, KeTickCount) against a blank period
// it measures once, at its first frame, as the least of four timings of BlockUntilVerticalBlank, kept at 0x4A5308.
// With interval 2 a frame that took under one period waits for the blank to end and then for the beam; one that took
// one to two periods waits (0x112C55) for the beam to reach line 460, the bottom of the picture, so the swap lands on
// the next blank; two periods or more swaps at once. The beam is GetRasterStatus (0x2D9210, below), which reads the
// NV2A's CRTC line register; nothing wrote it, every read said "in the blank", and that wait never waited: each frame
// that took 16 ms or more was swapped at once, and the game ran at 45 to 80 frames a second wherever the phone could
// draw that fast, with uneven steps, the view jumping back and forth at the first mission's doorway. The period is also
// set to what the console measures (16.67 ms blanks on a 1 ms tick time 16 or 17; the least is 16) before the first
// frame, as a thread held up between a blank and its tick read measured it short, and the tick comes from the guest
// clock itself rather than a copy a host thread refreshes
constexpr uint32_t kBlankPeriod = 0x4A5308;  // ms, 0 until measured

X86_EXTERN_FUNC(__imp__sub_00112D2D);
X86_FUNC(sub_00112D2D) {
    using kernel::R32;
    using kernel::W32;
    constexpr uint32_t kInterval = 0xAC;
    if (R32(kBlankPeriod) == 0) W32(kBlankPeriod, 16);
    uint32_t renderer = ctx.ecx;
    uint32_t interval = renderer ? R32(renderer + kInterval) : 0;
    bool unlock = game::Opt().frame_rate_60 && (interval == 2 || interval == 0x80000002u);
    if (unlock) W32(renderer + kInterval, interval - 1);
    __imp__sub_00112D2D(ctx, base);
    if (unlock) W32(renderer + kInterval, interval);
}

// The title's millisecond tick (mov eax, [KeTickCount]; mov eax, [eax]; ret), 76 callers: its frame pacer, its game
// clock and its timers. Read from the guest clock on each call rather than from the copy in memory, which a host thread
// refreshes and which on a busy phone stood still for several milliseconds and then jumped: frames of 0 ms followed by
// one with all the time that passed
X86_FUNC(sub_0010739C) {
    ctx.eax = kernel::TickCount();
    ctx.esp += 4;
}

// D3DDevice_GetRasterStatus(D3DRASTER_STATUS*), stdcall: { InVBlank, ScanLine } from the display's emulated beam
// (d3d::RasterStatus) instead of the NV2A register no host code keeps. The pacer polls it in a loop while it waits for
// the beam, running the title's idle callback between reads: the loop is the console's, and a yield keeps it from
// holding a phone's core to itself
X86_FUNC(sub_002D9210) {
    uint32_t status = kernel::R32(ctx.esp + 4);
    bool in_vblank;
    uint32_t line;
    d3d::RasterStatus(in_vblank, line);
    kernel::W32(status, in_vblank ? 1 : 0);
    kernel::W32(status + 4, line);
    std::this_thread::yield();
    ctx.eax = 0;
    ctx.esp += 8;  // ret 4
}

// Detail distance. An object's on-screen size (+0x158) is 0x121945's cam+0x140 / cam+0x144 / |object - camera| (on
// the x87 stack) times the object's +0xF0, worked out by its three callers before a draw. Two things read it:
// - the object draw (0x18B033, 0x18F2B0) leaves out an object under 0.01 (0x3E8DE0, 0x3E8EE8) entirely;
// - the mesh renderer (0x1DAEB1) draws the full mesh at +0xBC above 0.2 (0x43DDBC), reduced levels at +0xC0 below it.
// Both drop short pieces, such as the joint between a pipe's elbow and its straight run in the first mission, which
// opened as a gap a few steps back from the wall: a pixel at 640x480, a clear slit at the resolutions played now. The
// constants are shared (0x3E8DE0 is also a cm to m scale at 0x18B703), so the size is made four times larger instead,
// keeping every piece and detail level to four times the distance
X86_EXTERN_FUNC(__imp__sub_00121945);
X86_FUNC(sub_00121945) {
    __imp__sub_00121945(ctx, base);
    X87_ST(0) *= 4.0;
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

// The look history, for a report (SAVE REPORT, game::LogLookHistory): each frame the player's look runs, which camera
// mode called it (its return address), the camera's yaw and pitch after, the turn the mouse or a swipe gave, the
// body's turn target and rate, the frame's seconds and the body's position. A view that swings between two states
// shows here as the value that swings, and whether input or the game moved it
struct LookNote {
    uint32_t tick, caller;
    float dt, yaw, pitch, asked, target, rate, pos[3];
    bool looking;
};
constexpr int kLookNotes = 180;
LookNote g_look_notes[kLookNotes];
uint64_t g_look_count = 0;
std::mutex g_look_lock;

bool Readable(uint32_t va, uint32_t size) { return va >= 0x10000 && mem::IsCommitted(va) && mem::IsCommitted(va + size); }

void NoteLook(uint32_t character, uint32_t caller, float yaw, float pitch, float asked, bool looking) {
    uint32_t control = R32(character + kPlayerControl);
    if (!control || R32(control + kControlPlayer) == 0xFFFFFFFFu) return;  // the AI's characters
    LookNote n{};
    n.tick = kernel::TickCount();
    n.caller = caller;
    uint32_t clock = R32(kClock);
    n.dt = Readable(clock, 0xB4) ? RF(clock + 0xB0) : 0;
    n.yaw = yaw;
    n.pitch = pitch;
    n.asked = asked;
    n.target = RF(character + kTurnTarget);
    n.rate = RF(character + kTurnRate);
    n.looking = looking;
    // the body, as the camera code reads it (0x18C2C): [character + 0x30], its position at +0x100, or +0x60 while
    // the flags at +0xC8 & 0x3C0 say so
    uint32_t body = R32(character + 0x30);
    if (Readable(body, 0x10C)) {
        uint32_t at = (R32(body + 0xC8) & 0x3C0) ? body + 0x60 : body + 0x100;
        for (int i = 0; i < 3; i++) n.pos[i] = RF(at + 4 * i);
    }
    std::lock_guard lock(g_look_lock);
    g_look_notes[g_look_count++ % kLookNotes] = n;
}

}  // namespace

namespace game {
void LogLookHistory() {
    std::lock_guard lock(g_look_lock);
    uint64_t first = g_look_count > kLookNotes ? g_look_count - kLookNotes : 0;
    LOGDIAG("look: last %llu: tick caller dt yaw pitch asked target rate looking | x y z",
         (unsigned long long)(g_look_count - first));
    for (uint64_t i = first; i < g_look_count; i++) {
        const LookNote& n = g_look_notes[i % kLookNotes];
        LOGDIAG("look: %u %06X %.4f %.2f %.2f %.2f %.1f %.1f %d | %.2f %.2f %.2f", n.tick, n.caller, n.dt, n.yaw, n.pitch,
             n.asked, n.target, n.rate, int(n.looking), n.pos[0], n.pos[1], n.pos[2]);
    }
}
}  // namespace game

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
    uint32_t caller = R32(ctx.esp);
    __imp__sub_0005F51F(ctx, base);
    if (mouse) ctx.eax = 1;
    NoteLook(character, caller, RF(yaw_at), RF(pitch_at), mouse ? yaw : 0, ctx.eax != 0);
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
