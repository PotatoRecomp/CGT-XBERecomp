// The settings screen is the phone's own, not ImGui's: CGTActivity draws it in Java (SettingsDialog.java), the same
// rows as the desktop F10 panel (src/common/menu.cpp) where they mean something on a phone, plus the touch controls'.
// A value is read once when the panel opens and written one at a time as it changes, across JNI rather than every
// frame. The gear on the on-screen controls (TouchControls.java) opens it, and F10 on a keyboard does as on the
// desktop. The controls' own JNI is here too.
#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_system.h>

#include "app/app.h"
#include "common/menu.h"
#include "common/mouse_keyboard.h"
#include "common/options.h"
#include "core/log.h"
#include "host.h"

namespace {

// the order the dialog's values are in, the same numbers SettingsDialog.java has
enum Key : int {
    kResolution, kMsaa, kSmaa, kFxaa, kAniso, kBloom, kFog, kHdr, kHdrAvailable, kScreenFilter, kVsync, kFpsCap,
    kFrameRate60, kFpsCounter, kTouchSensitivity, kTouchInvertY, kControlsOpacity, kMouseLook, kMouseSensitivity,
    kMouseInvertY, kVolume, kOutWidth, kOutHeight, kFrameSkip, kAsyncShaders, kShadows, kKeyCount
};

// the names for the settings log, in Key order
const char* const kKeyNames[] = {
    "render_percent", "msaa", "smaa", "fxaa", "anisotropic_filtering", "bloom", "fog", "hdr_output", "hdr_available",
    "screen_filter", "vsync", "fps_cap", "frame_rate_60", "fps_counter", "touch_sensitivity", "touch_invert_y",
    "controls_opacity", "mouse_look", "mouse_sensitivity", "mouse_invert_y", "volume", "out_width", "out_height",
    "frame_skip", "async_shaders", "shadows",
};
static_assert(sizeof(kKeyNames) / sizeof(kKeyNames[0]) == kKeyCount);

std::atomic<bool> g_panel_open{false};  // SettingsDialog is up

// a value from a list, the list's first when it is not one of them
template <size_t N>
uint32_t OneOf(jint v, const uint32_t (&values)[N]) {
    for (uint32_t x : values)
        if (int64_t(x) == int64_t(v)) return x;
    return values[0];
}

JNIEnv* Env() { return static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv()); }

// an instance method of the activity taking a boolean, from any thread
void CallActivity(const char* method, bool arg) {
    JNIEnv* env = Env();
    if (!env) return;
    jobject activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!activity) return;
    jclass cls = env->GetObjectClass(activity);
    if (cls) {
        jmethodID mid = env->GetMethodID(cls, method, "(Z)V");
        if (mid) env->CallVoidMethod(activity, mid, jboolean(arg));
        env->DeleteLocalRef(cls);
    }
    if (env->ExceptionCheck()) env->ExceptionClear();
    env->DeleteLocalRef(activity);
}

}  // namespace

// src/common/menu.h on Android: the panel is the activity's dialog. keyboard.cpp idles the game's pad while it is open
// and opens or closes it with F10 and Escape, as on the desktop
namespace app {

bool MenuOpen() { return g_panel_open.load(std::memory_order_relaxed); }

void ToggleMenu() { host::ShowSettings(!MenuOpen()); }

}  // namespace app

void host::ControlsSync(bool pad) { CallActivity("controlsSync", pad); }

void host::ShowSettings(bool show) { CallActivity("showSettingsFromNative", show); }

// the dialog opened or closed: the game's pad goes idle while it is up, and the mouse is given back to the panel
extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_cgt_CGTActivity_nativeSettingsOpen(JNIEnv*, jclass,
                                                                                         jboolean open) {
    bool was = g_panel_open.exchange(open != JNI_FALSE);
    if (was == (open != JNI_FALSE)) return;
    if (!open) app::Save();
    app::Post(input::SyncMouseCapture);
}

// the panel's values as one array, read when it opens
extern "C" JNIEXPORT jintArray JNICALL Java_com_datfooldive_cgt_CGTActivity_nativeSettings(JNIEnv* env, jclass) {
    const app::Settings& s = app::Get();
    const game::Options& o = game::Opt();
    const app::Mode out = app::Output();
    jint v[kKeyCount];
    v[kResolution] = jint(s.render_percent);
    v[kMsaa] = jint(o.msaa);
    v[kSmaa] = o.smaa;
    v[kFxaa] = o.fxaa;
    v[kAniso] = jint(s.aniso);
    v[kBloom] = o.bloom;
    v[kFog] = o.fog;
    v[kHdr] = o.hdr_output;
    v[kHdrAvailable] = game::g_hdr_display.load();
    v[kScreenFilter] = jint(o.screen_filter);
    v[kVsync] = s.vsync;
    v[kFpsCap] = jint(s.fps_cap);
    v[kFrameRate60] = o.frame_rate_60;
    v[kFpsCounter] = o.fps_counter;
    v[kTouchSensitivity] = jint(o.touch_sensitivity);
    v[kTouchInvertY] = o.touch_invert_y;
    v[kControlsOpacity] = jint(o.controls_opacity);
    v[kMouseLook] = o.mouse_look;
    v[kMouseSensitivity] = jint(o.mouse_sensitivity);
    v[kMouseInvertY] = o.mouse_invert_y;
    v[kVolume] = jint(o.volume);
    v[kOutWidth] = jint(out.width);
    v[kOutHeight] = jint(out.height);
    v[kFrameSkip] = o.frame_skip;
    v[kAsyncShaders] = o.async_shaders;
    v[kShadows] = o.shadows;
    jintArray arr = env->NewIntArray(kKeyCount);
    if (arr) env->SetIntArrayRegion(arr, 0, kKeyCount, v);
    return arr;
}

// one value as the panel moves it; commit writes the file, which a slider only does when it is let go
extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_cgt_CGTActivity_nativeSettingsSet(JNIEnv*, jclass, jint key,
                                                                                        jint value, jboolean commit) {
    app::Settings& s = app::Get();
    game::Options& o = game::Opt();
    switch (key) {
        case kResolution: s.render_percent = uint32_t(std::clamp<jint>(value, 25, 200)); break;
        case kMsaa: o.msaa = OneOf(value, {0u, 2u, 4u, 8u}); break;
        case kSmaa: o.smaa = value != 0; break;
        case kFxaa: o.fxaa = value != 0; break;
        case kAniso: s.aniso = OneOf(value, {0u, 2u, 4u, 8u, 16u}); break;
        case kBloom: o.bloom = value != 0; break;
        case kFog: o.fog = value != 0; break;
        case kFrameSkip: o.frame_skip = value != 0; break;
        case kAsyncShaders: o.async_shaders = value != 0; break;
        case kShadows: o.shadows = value != 0; break;
        case kHdr: o.hdr_output = value != 0 && game::g_hdr_display.load(); break;
        case kScreenFilter: o.screen_filter = uint32_t(std::clamp<jint>(value, 0, game::kFilterHdr)); break;
        case kVsync: s.vsync = value != 0; break;
        case kFpsCap: s.fps_cap = OneOf(value, {30u, 60u, 90u, 120u, 0u}); break;
        case kFrameRate60: o.frame_rate_60 = value != 0; break;
        case kFpsCounter: o.fps_counter = value != 0; break;
        case kTouchSensitivity: o.touch_sensitivity = uint32_t(std::clamp<jint>(value, 10, 500)); break;
        case kTouchInvertY: o.touch_invert_y = value != 0; break;
        case kControlsOpacity: o.controls_opacity = uint32_t(std::clamp<jint>(value, 0, 100)); break;
        case kMouseLook: o.mouse_look = value != 0; break;
        case kMouseSensitivity: o.mouse_sensitivity = uint32_t(std::clamp<jint>(value, 5, 1000)); break;
        case kMouseInvertY: o.mouse_invert_y = value != 0; break;
        case kVolume: o.volume = uint32_t(std::clamp<jint>(value, 50, 400)); break;
        default: return;
    }
    if (key == kVsync) app::ApplyDisplay();
    if (key == kMouseLook) app::Post(input::SyncMouseCapture);
    if (commit) {
        LOGR("Setting: %s = %d", kKeyNames[key], int(value));
        app::Save();
    }
}

// the settings back to a fresh install's (native/main.cpp's first settings file), the controls' layout with them
extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_cgt_CGTActivity_nativeSettingsDefaults(JNIEnv*, jclass) {
    app::Settings& s = app::Get();
    const app::Settings ds;
    s.render_percent = 70;
    s.fps_cap = ds.fps_cap;
    s.vsync = ds.vsync;
    s.aniso = ds.aniso;
    game::Opt() = game::Options{};
    game::Opt().fps_counter = true;
    app::Post(input::SyncMouseCapture);
    touch::ResetUnits();  // writes the file
}

// True while the game still counts itself in the background. CGTActivity asks a moment after it regains the window
// focus: a rapid surface loss on resume can leave SDL short of a ready surface, so it never moves the app back to the
// front and the game hangs in its background wait. Still background then means the activity is in front but the
// surface did not come back, and the Java side makes a new one.
extern "C" JNIEXPORT jboolean JNICALL Java_com_datfooldive_cgt_CGTActivity_nativeStuckInBackground(JNIEnv*, jclass) {
    return host::Foreground() ? JNI_FALSE : JNI_TRUE;
}

// the on-screen controls (TouchControls.java): what the fingers hold, as it changes
extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_cgt_TouchControls_nativeTouchPad(JNIEnv*, jclass, jint buttons,
                                                                                       jint analog, jfloat lx,
                                                                                       jfloat ly) {
    touch::Set(uint32_t(buttons), uint32_t(analog), lx, ly);
}

// a look finger's travel since its last move event, in dp, x right and y down: added to the mouse look's sum
extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_cgt_TouchControls_nativeTouchLook(JNIEnv*, jclass, jfloat dx,
                                                                                        jfloat dy) {
    input::OnTouchLook(dx, dy);
}

// their layout: x, y and size for each control in touch.cpp's Unit order
extern "C" JNIEXPORT jfloatArray JNICALL Java_com_datfooldive_cgt_TouchControls_nativeControls(JNIEnv* env, jclass) {
    const int n = touch::Units();
    std::vector<jfloat> v(size_t(n) * 3);
    for (int i = 0; i < n; i++) {
        float x = 0, y = 0, scale = 1;
        touch::GetUnit(i, x, y, scale);
        v[size_t(i) * 3] = x;
        v[size_t(i) * 3 + 1] = y;
        v[size_t(i) * 3 + 2] = scale;
    }
    jfloatArray arr = env->NewFloatArray(jsize(v.size()));
    if (arr) env->SetFloatArrayRegion(arr, 0, jsize(v.size()), v.data());
    return arr;
}

// one control moved or sized in the layout editor; commit writes the file, as a drag or a size bar is let go
extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_cgt_TouchControls_nativeControlSet(JNIEnv*, jclass, jint unit,
                                                                                         jfloat x, jfloat y,
                                                                                         jfloat scale,
                                                                                         jboolean commit) {
    touch::SetUnit(unit, x, y, scale, commit);
}

extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_cgt_TouchControls_nativeControlsReset(JNIEnv*, jclass) {
    touch::ResetUnits();
}
