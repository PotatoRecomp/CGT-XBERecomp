// The Android host: SDL's window and event loop, the app's lifecycle, the settings and folders; the game is described
// by app::kGame (src/common/game_info.cpp), as on Linux, whose host (sdk/runtime/app/main_linux.cpp) this follows
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_system.h>
#include <dlfcn.h>
#include <jni.h>
#include <sys/system_properties.h>
#include <unistd.h>
#include <unwind.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include <imgui.h>

#include "app/app.h"
#include "common/mouse_keyboard.h"
#include "core/input.h"
#include "core/log.h"
#include "host.h"

namespace gpu {
void LogFrameHistory();  // native/vulkan.cpp
}  // namespace gpu
extern const char kBuildId[];  // build_id.cpp, written by android/build_id.cmake
#include "image_symbols.h"
#include "kernel/kernel.h"

namespace app {

std::filesystem::path g_game_dir;  // native/game_files.cpp: where the game's files were found

namespace {

SDL_Window* g_window = nullptr;
// SDL_RegisterEvents' range: what other threads ask of the window thread
uint32_t g_event_base = 0;
enum : uint32_t { kOpenWindow, kRun, kEventCount };
std::atomic<bool> g_running{true};
std::atomic<bool> g_foreground{true};
// each return to the front: Android hands a new window, whose surface the renderer makes again (host::SurfaceEpoch)
std::atomic<uint32_t> g_surface_epoch{0};
std::mutex g_foreground_lock;
std::condition_variable g_foreground_cv;
std::atomic<bool> g_focus_paused{false};  // the game's thread waits in WaitWhileUnfocused

std::atomic<uint32_t> g_pixel_w{1280};
std::atomic<uint32_t> g_pixel_h{720};
// the screen the render scale is taken from: the last landscape size the window had. Coming back from the background
// Android may show the window portrait or 1 pixel for a moment, and a render scale from that remade every target
std::atomic<uint32_t> g_out_w{1280};
std::atomic<uint32_t> g_out_h{720};

std::mutex g_open_lock;
std::condition_variable g_open_cv;
uint64_t g_open_asked = 0;
uint64_t g_open_done = 0;

void PushEvent(uint32_t kind, int32_t code = 0, void* data = nullptr) {
    SDL_Event e{};
    e.type = g_event_base + kind;
    e.user.code = code;
    e.user.data1 = data;
    SDL_PushEvent(&e);
}

// window thread: what other threads read of the window
void UpdateSizes() {
    if (!g_window) return;
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g_window, &w, &h);
    if (w > 1 && h > 1 && w >= h) {
        g_pixel_w = uint32_t(w);
        g_pixel_h = uint32_t(h);
        g_out_w = uint32_t(w);
        g_out_h = uint32_t(h);
    }
}

// Android's pause and resume, seen as SDL queues them: SDL blocks this thread inside the pump right after queueing
// the pause, so the queue hands it over only after the resume, and the game thread went on drawing into a window
// Android had taken away. A watch runs at the queueing, before the block.
void SetForeground(bool on) {
    {
        std::lock_guard lock(g_foreground_lock);
        if (g_foreground == on) return;
        if (on) g_surface_epoch++;
        g_foreground = on;
    }
    g_foreground_cv.notify_all();
    LOGR("app: %s", on ? "in front" : "in the background");
}

bool Lifecycle(void*, SDL_Event* e) {
    switch (e->type) {
        case SDL_EVENT_WILL_ENTER_BACKGROUND:
            SetForeground(false);
            break;
        case SDL_EVENT_DID_ENTER_FOREGROUND:
            UpdateSizes();
            SetForeground(true);
            break;
        default:
            break;
    }
    return true;
}

void Handle(const SDL_Event& e) {
    // the game's own events first, as on Linux: F10 and Escape for the settings panel, a mouse's motion and wheel
    if (kGame.event && kGame.event(&e)) return;
    if (e.type >= g_event_base && e.type < g_event_base + kEventCount) {
        switch (e.type - g_event_base) {
            case kOpenWindow: {
                SDL_ShowWindow(g_window);
                SDL_RaiseWindow(g_window);
                std::lock_guard lock(g_open_lock);
                g_open_done++;
                g_open_cv.notify_all();
                break;
            }
            case kRun: {
                std::unique_ptr<std::function<void()>> fn(static_cast<std::function<void()>*>(e.user.data1));
                (*fn)();
                break;
            }
        }
        return;
    }
    switch (e.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            LOGR("window: quit event 0x%x", unsigned(e.type));
            g_running = false;
            break;
        case SDL_EVENT_TERMINATING:
            g_running = false;
            // the game's thread may be waiting for the front: it goes on to the exit
            SetForeground(true);
            break;
        case SDL_EVENT_LOW_MEMORY:
            LOGW("app: low memory");
            break;
        case SDL_EVENT_GAMEPAD_ADDED:
        case SDL_EVENT_GAMEPAD_REMOVED:
            input::OnDevice(e.type == SDL_EVENT_GAMEPAD_ADDED, e.gdevice.which);
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
            UpdateSizes();
            break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED:
            // the window has the focus only in front: a return SDL's lifecycle did not tell is taken here
            if (!g_foreground) {
                LOGDIAG("app: the window has the focus while counted in the background: resumed");
                UpdateSizes();
                SetForeground(true);
            }
            break;
        default:
            break;
    }
}

// UTF-8 from a wide string: wchar_t is UTF-32 here
std::string Narrow(const std::wstring& w) {
    std::string s;
    for (wchar_t c : w) {
        auto u = uint32_t(c);
        if (u < 0x80) {
            s += char(u);
        } else if (u < 0x800) {
            s += char(0xC0 | (u >> 6));
            s += char(0x80 | (u & 0x3F));
        } else if (u < 0x10000) {
            s += char(0xE0 | (u >> 12));
            s += char(0x80 | ((u >> 6) & 0x3F));
            s += char(0x80 | (u & 0x3F));
        } else {
            s += char(0xF0 | (u >> 18));
            s += char(0x80 | ((u >> 12) & 0x3F));
            s += char(0x80 | ((u >> 6) & 0x3F));
            s += char(0x80 | (u & 0x3F));
        }
    }
    return s;
}

// the settings file the port starts with: the screen's own size and 70% of its height rendered, which a phone's GPU
// keeps up with
void WriteDefaultSettings(const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) return;
    f << "display_mode = \"borderless\"\n"
         "width = 0\n"
         "height = 0\n"
         "render_percent = 70\n"
         "fps_cap = 60\n"
         "vsync = true\n"
         "anisotropic_filtering = 0\n"
         "fps_counter = true\n";
}

}  // namespace

std::string Utf8(const std::wstring& text) { return Narrow(text); }

// the saves were kept in the app's own files folder, which an uninstall wipes: the first run that finds the game
// takes them into the game's folder, so the progress already made is not left behind. True when the game's folder is
// the one to use, false when the old saves could not be moved and stay where they are
bool TakeSaves(const std::filesystem::path& to) {
    std::error_code ec;
    const std::filesystem::path from = ExeDir() / "hdd";
    if (std::filesystem::exists(to, ec)) return true;           // already there, the app's own left alone
    if (!std::filesystem::is_directory(from, ec)) return true;  // a fresh install: made there
    std::filesystem::rename(from, to, ec);
    if (ec) {
        ec.clear();
        std::filesystem::copy(from, to, std::filesystem::copy_options::recursive, ec);
        if (ec) {
            LOGW("saves: %s did not move to %s: %s", from.c_str(), to.c_str(), ec.message().c_str());
            std::error_code ignored;
            std::filesystem::remove_all(to, ignored);  // what was half copied
            return false;
        }
        std::filesystem::remove_all(from, ec);
    }
    LOGI("saves: moved from %s to %s", from.c_str(), to.c_str());
    return true;
}

std::filesystem::path ExeDir() {
    static const std::filesystem::path dir = [] {
        const char* internal = SDL_GetAndroidInternalStoragePath();
        if (internal && *internal) return std::filesystem::path(internal);
        return std::filesystem::path(SDL_GetBasePath() ? SDL_GetBasePath() : ".");
    }();
    return dir;
}

int Instance() { return 0; }

void* Window() { return g_window; }

void Quit() {
    SDL_Event e{};
    e.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&e);
}

void Post(std::function<void()> fn) { PushEvent(kRun, 0, new std::function<void()>(std::move(fn))); }

void ShowError(const std::wstring& text) {
    std::string utf8 = Narrow(text);
    LOGE("error: %s", utf8.c_str());
    host::Toast(utf8.c_str());
    // the toast outlives the process only while it is up: give it a moment, then let the activity close
    std::this_thread::sleep_for(std::chrono::seconds(3));
}

bool AskYesNo(const std::wstring& text) {
    ShowError(text);
    return false;
}

std::recursive_mutex& UiLock() {
    static std::recursive_mutex m;
    return m;
}

void ApplyDisplay() {}  // Android shows the window full screen and does not resize it

void OpenGameWindow(bool) {
    std::unique_lock lock(g_open_lock);
    uint64_t ticket = ++g_open_asked;
    PushEvent(kOpenWindow);
    g_open_cv.wait(lock, [&] { return g_open_done >= ticket || !g_running; });
}

Mode Desktop() { return {g_out_w.load(), g_out_h.load()}; }
Mode WindowPixels() { return {g_pixel_w.load(), g_pixel_h.load()}; }

// ImGui draws only the FPS counter here, which takes no input: the settings panel is SettingsDialog.java
void UiFrame() {
    using clock = std::chrono::steady_clock;
    static clock::time_point last = clock::now();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(g_pixel_w), float(g_pixel_h));
    io.DisplayFramebufferScale = ImVec2(1, 1);
    auto now = clock::now();
    io.DeltaTime = std::max(std::chrono::duration<float>(now - last).count(), 1e-4f);
    last = now;
}

namespace {

std::atomic<uint64_t> g_swaps{0};  // the game's Swaps so far, for the freeze watchdog
std::atomic<uint64_t> g_shown{0};  // frames the renderer put on the screen (host::NotePresent)
std::atomic<int> g_acquire_result{0}, g_present_result{0};  // the renderer's last VkResults
// the render thread's current sub-step and when it entered it, for the freeze report (host::NoteRenderPhase)
std::atomic<const char*> g_render_phase{"start"};
std::atomic<int64_t> g_render_phase_ns{0};

// A game that stops swapping in front for kFrozen is written up once in cgt.log: each guest thread's stack and each
// native thread's state (R running, S sleeping) and what it sleeps in. That tells a loop in the title's code from a
// wait on a lock or on the renderer
void LogThreads() {
    const int64_t now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const int64_t in_ns = now - g_render_phase_ns.load(std::memory_order_relaxed);
    LOGDIAG("freeze: render thread in '%s' for %lld ms, last acquire %d, present %d, surface epoch %u",
         g_render_phase.load(std::memory_order_relaxed), (long long)(in_ns / 1'000'000), g_acquire_result.load(),
         g_present_result.load(), g_surface_epoch.load());
    for (auto& t : kernel::Threads()) {
        if (!t || !t->ctx) continue;
        LOGDIAG("freeze: guest thread %X", t->id);
        kernel::LogGuestStack(*t->ctx, 48, true);
    }
    kernel::LogRecentFiles();
    std::error_code ec;
    for (const auto& task : std::filesystem::directory_iterator("/proc/self/task", ec)) {
        std::string name, stat, wchan;
        std::ifstream(task.path() / "comm") >> name;
        std::getline(std::ifstream(task.path() / "stat"), stat);
        std::ifstream(task.path() / "wchan") >> wchan;
        // the state follows the name in parentheses
        size_t close = stat.rfind(')');
        char state = close != std::string::npos && close + 2 < stat.size() ? stat[close + 2] : '?';
        LOGDIAG("freeze: native %s (%s) %c %s", task.path().filename().c_str(), name.c_str(), state, wchan.c_str());
    }
}

// SAVE REPORT on the settings panel (SettingsDialog.java): the freeze report on demand, for what sets off no watchdog,
// a game that goes on drawing but gets nowhere (a load it waits on for ever, frames still swapped) or a drop in frame
// rate. The overlay's stats, each guest thread's stack, the native threads and the title's last file calls go into
// cgt.log in the game's folder, where the player finds it to send; the path comes back for the panel to show
std::string WriteReport() {
    char stats[256];
    touch::StatsLine(stats, sizeof(stats));
    LOGDIAG("report: asked for at swap %llu, %s", (unsigned long long)g_swaps.load(), stats);
    LOGDIAG("report: settings now %s", app::SettingsLine().c_str());
    LogThreads();
    gpu::LogFrameHistory();
    game::LogLookHistory();
    logx::Flush();
    return logx::Path();
}

void StartFreezeWatchdog() {
    std::thread([] {
        using clock = std::chrono::steady_clock;
        constexpr auto kFrozen = std::chrono::seconds(3);
        uint64_t seen = g_swaps, shown = g_shown;
        auto since = clock::now(), shown_since = since, behind_since = since;
        bool reported = false, reported_shown = false;
        int reports = 0;  // a long load without frames looks the same: a few per run at most
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            auto t = clock::now();
            // Android has the window in front (it has the input focus) while the app still counts itself in the
            // background: a return to the front SDL's lifecycle did not tell, and the game would wait in its Swap for
            // ever with the last picture up
            bool focused = g_window && (SDL_GetWindowFlags(g_window) & SDL_WINDOW_INPUT_FOCUS);
            if (g_foreground || !focused) behind_since = t;
            if (t - behind_since >= std::chrono::seconds(2)) {
                LOGDIAG("freeze: in front with the input focus but counted in the background for 2 s: resumed");
                SetForeground(true);
                behind_since = t;
            }
            uint64_t now = g_swaps, now_shown = g_shown;
            // the background, a focus pause and the time before the first frame are no freeze
            bool idle = !g_foreground || g_focus_paused || !now;
            if (now != seen || idle) {
                if (reported && now != seen) {
                    LOGDIAG("freeze: the game swaps again at %llu", (unsigned long long)now);
                    reported = false;
                }
                seen = now;
                since = t;
            } else if (!reported && reports < 3 && t - since >= kFrozen) {
                reported = true;
                reports++;
                LOGE("freeze: no Swap for 3 s at swap %llu, in front", (unsigned long long)now);
                LogThreads();
                logx::Flush();
            }
            // the game swapping and nothing reaching the screen: the picture stands still, the game and its sound go on
            if (now_shown != shown || idle || now == seen) {
                if (reported_shown && now_shown != shown) {
                    LOGDIAG("freeze: frames reach the screen again");
                    reported_shown = false;
                }
                shown = now_shown;
                shown_since = t;
            } else if (!reported_shown && reports < 3 && t - shown_since >= kFrozen) {
                reported_shown = true;
                reports++;
                LOGE("freeze: the game swaps (%llu) and no frame reached the screen for 3 s: acquire %d, present %d, "
                     "foreground %d, surface epoch %u, window %ux%u",
                     (unsigned long long)now, g_acquire_result.load(), g_present_result.load(),
                     int(g_foreground.load()), g_surface_epoch.load(), g_pixel_w.load(), g_pixel_h.load());
                logx::Flush();
            }
        }
    }).detach();
}

}  // namespace

void Pace() {
    g_swaps.fetch_add(1, std::memory_order_relaxed);
    using clock = std::chrono::steady_clock;
    static clock::time_point next = clock::now();
    uint32_t cap = Get().fps_cap;
    auto now = clock::now();
    if (!cap) {
        next = now;
        return;
    }
    auto period = std::chrono::nanoseconds(1000000000ull / cap);
    next += period;
    if (next < now - period) next = now;
    while (clock::now() + std::chrono::milliseconds(1) < next) std::this_thread::sleep_for(std::chrono::microseconds(500));
    while (clock::now() < next) std::this_thread::yield();
}

// In the background Android takes the window away. The game's thread would go on into its Swap and wait in the
// renderer for the front, holding the D3D lock and whatever locks of the title it had, and the other threads would
// give up on those after their patience and run unlocked. It waits here instead, at the top of Swap before any lock,
// with the guest clock held so the title does not come back to seconds that passed while it was not drawn. The sound
// is left to SDL, which stops Android's audio in the background itself.
void WaitWhileUnfocused() {
    if (g_foreground) return;
    LOGR("app: paused in the background");
    g_focus_paused = true;
    kernel::HoldGuestClock(true);
    {
        // a wait as the kernel's: the locks the title's thread holds go for it
        kernel::Blocking blocking;
        std::unique_lock lock(g_foreground_lock);
        g_foreground_cv.wait(lock, [] { return g_foreground.load() || !g_running; });
    }
    kernel::HoldGuestClock(false);
    g_focus_paused = false;
    LOGR("app: resumed");
}

bool FocusPaused() { return g_focus_paused; }

// the port has no splash picture and one instance only
bool OpenSplash() { return false; }
void SplashFirstFrame() {}
void CloseSplash() {}
bool StartHandover() { return false; }
bool HandingOver() { return false; }
void FinishHandover() {}
bool PrepareGameCode(const GameFiles&) { return true; }

}  // namespace app

namespace host {

SDL_Window* Window() { return app::g_window; }
bool Foreground() { return app::g_foreground; }
uint32_t SurfaceEpoch() { return app::g_surface_epoch; }
std::filesystem::path FilesDir() { return app::ExeDir(); }
std::filesystem::path GameDir() { return app::g_game_dir; }

void NotePresent(int acquire, int present, bool shown) {
    app::g_acquire_result = acquire;
    app::g_present_result = present;
    if (shown) app::g_shown++;
}

namespace {
char g_gpu_name[256] = "";
}  // namespace
void SetGpuName(const char* name) { std::snprintf(g_gpu_name, sizeof(g_gpu_name), "%s", name ? name : ""); }
const char* GpuName() { return g_gpu_name; }

void NoteRenderPhase(const char* phase) {
    app::g_render_phase.store(phase ? phase : "idle", std::memory_order_relaxed);
    app::g_render_phase_ns.store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(),
        std::memory_order_relaxed);
}

void Toast(const char* text) {
    JNIEnv* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
    if (!env) return;
    jobject activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!activity) return;
    jclass cls = env->GetObjectClass(activity);
    if (cls) {
        jmethodID mid = env->GetStaticMethodID(cls, "showToast", "(Ljava/lang/String;)V");
        if (mid) {
            jstring s = env->NewStringUTF(text ? text : "");
            if (s) {
                env->CallStaticVoidMethod(cls, mid, s);
                env->DeleteLocalRef(s);
            }
        }
        env->DeleteLocalRef(cls);
    }
    if (env->ExceptionCheck()) env->ExceptionClear();
    env->DeleteLocalRef(activity);
}

}  // namespace host

// SettingsDialog.java's SAVE REPORT: the report written, the file it went to for the toast
extern "C" JNIEXPORT jstring JNICALL Java_com_datfooldive_cgt_CGTActivity_nativeSaveReport(JNIEnv* env, jclass) {
    std::string path = app::WriteReport();
    return env->NewStringUTF(path.c_str());
}

namespace {

// a native frame of the crash backtrace: the library and the offset in it, what ndk-stack and addr2line take
_Unwind_Reason_Code LogFrame(_Unwind_Context* uc, void* arg) {
    int& depth = *static_cast<int*>(arg);
    uintptr_t pc = _Unwind_GetIP(uc);
    if (!pc || depth >= 32) return _URC_END_OF_STACK;
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(pc), &info) && info.dli_fname) {
        const char* lib = std::strrchr(info.dli_fname, '/');
        LOGE("  #%02d %s+0x%zx", depth, lib ? lib + 1 : info.dli_fname,
             size_t(pc - reinterpret_cast<uintptr_t>(info.dli_fbase)));
    } else {
        LOGE("  #%02d %p", depth, reinterpret_cast<void*>(pc));
    }
    depth++;
    return _URC_NO_REASON;
}

// a crash goes to the log file with the guest thread's stack and the native backtrace, then on to Android's own
// handler (the tombstone) as if this one were not there
void OnCrash(int sig, siginfo_t* info, void*) {
    void* at = info ? info->si_addr : nullptr;
    if (kernel::Thread* t = kernel::Current(); t && t->ctx) {
        LOGE("crash: signal %d (%s) at %p on guest thread %X", sig, strsignal(sig), at, t->id);
        if (sig == SIGSEGV || sig == SIGBUS) {
            uintptr_t va = reinterpret_cast<uintptr_t>(at), g = reinterpret_cast<uintptr_t>(kernel::g_base);
            if (va >= g && va - g < (uintptr_t(1) << 32)) LOGE("  guest address %08zX", size_t(va - g));
        }
        kernel::LogGuestStack(*t->ctx, 64);
    } else {
        LOGE("crash: signal %d (%s) at %p", sig, strsignal(sig), at);
    }
    int depth = 0;
    _Unwind_Backtrace(LogFrame, &depth);
    logx::Flush();
    signal(sig, SIG_DFL);
    raise(sig);
}

void InstallCrashHandler() {
    struct sigaction sa{};
    sa.sa_sigaction = OnCrash;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT}) sigaction(sig, &sa, nullptr);
}

// errors and crashes go to cgt.log in the game's folder, beside its files, where a player finds it to send; the header
// says which phone, which build and which settings it came from
void LogToGameFolder(const std::filesystem::path& dir) {
    char model[PROP_VALUE_MAX] = "", maker[PROP_VALUE_MAX] = "", release[PROP_VALUE_MAX] = "", sdk[PROP_VALUE_MAX] = "";
    __system_property_get("ro.product.manufacturer", maker);
    __system_property_get("ro.product.model", model);
    __system_property_get("ro.build.version.release", release);
    __system_property_get("ro.build.version.sdk", sdk);
    std::time_t now = std::time(nullptr);
    char date[32];
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    std::string header = app::Utf8(app::kGame.title) + " (Android), " + date + "\n" + maker + " " + model +
                         ", Android " + release + " (API " + sdk + ")\nBuild: " + kBuildId + "\nSettings: " +
                         app::SettingsLine() + "\n";
    logx::ErrorFile((dir / (std::string(app::kGame.log) + ".log")).string(), header);
}

}  // namespace

namespace prof {
void StartIfAsked();  // native/profiler.cpp
void SetDir(const std::string& dir);
}  // namespace prof

int main(int, char**) {
    // an app gets no environment of its own: `adb shell setprop debug.cgt.perf 1` stands in for XBE_PERF, read before
    // anything asks diag::Perf
    char perf[PROP_VALUE_MAX] = "";
    if (__system_property_get("debug.cgt.perf", perf) > 0 && perf[0] != '0') setenv("XBE_PERF", "1", 1);
    prof::StartIfAsked();
    // SDL sets the activity's orientation from the window it is asked for, over the manifest's sensorLandscape
    SDL_SetHint(SDL_HINT_ORIENTATIONS, "LandscapeLeft LandscapeRight");
    // the screen's fingers are TouchControls.java's: none of them reaches the game as a mouse, and a real mouse's
    // clicks stay a mouse's (src/common/keyboard.cpp)
    SDL_SetHint(SDL_HINT_TOUCH_MOUSE_EVENTS, "0");
    SDL_SetHint(SDL_HINT_MOUSE_TOUCH_EVENTS, "0");

    std::string title = app::Utf8(app::kGame.title);
    SDL_SetAppMetadata(title.c_str(), nullptr, nullptr);
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        LOGE("SDL video failed: %s", SDL_GetError());
        app::ShowError(L"No display to open the game window on.");
        return 1;
    }
    app::g_event_base = SDL_RegisterEvents(app::kEventCount);
    SDL_AddEventWatch(app::Lifecycle, nullptr);
    app::UpdateSizes();

    std::error_code ec;
    if (!std::filesystem::exists(app::ExeDir() / app::kGame.settings, ec))
        app::WriteDefaultSettings(app::ExeDir() / app::kGame.settings);
    app::Load();
    // a full log in the app's internal storage beside settings.toml, as well as logcat; the error log in the game's
    // folder (LogToGameFolder) is the one a player finds and sends
    logx::Init((app::ExeDir() / "logs").string(), app::kGame.log);
    LOGR("Started");
    if (app::kGame.init) app::kGame.init();
    app::LogSystem();

    app::GameFiles game;
    if (!app::FindGame(game)) return 1;
    LogToGameFolder(game.dir);
    prof::SetDir(game.dir.string());
    InstallCrashHandler();
    if (!app::PrepareGameCode(game)) return 1;
    // the saves go in the game's own folder, beside its files: an uninstall takes the app's folder, not this one
    std::filesystem::path hdd = game.dir / "hdd";
    if (!app::TakeSaves(hdd)) hdd = app::ExeDir() / "hdd";

    app::g_window = SDL_CreateWindow(title.c_str(), 1280, 720, SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!app::g_window) {
        LOGE("SDL window failed: %s", SDL_GetError());
        app::ShowError(L"The game window could not be opened, see cgt.log in the CGT folder.");
        return 1;
    }
    app::UpdateSizes();
    LOGR("Window: %ux%u", app::g_pixel_w.load(), app::g_pixel_h.load());

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.BackendPlatformName = "sdl3 (android)";

    input::Init();
    touch::Init();
    if (app::kGame.window) app::kGame.window(app::g_window);
    if (!kernel::Boot(std::move(game.xbe), IMAGE_XBE, game.dir.string(), hdd.string())) {
        app::ShowError(L"The game failed to start, see cgt.log in the CGT folder.");
        return 1;
    }
    app::StartFreezeWatchdog();

    // Polled, then a short sleep: SDL_WaitEventTimeout on Android pumps its events again each time anything is
    // queued, its own bookkeeping included, and kept this thread busy on a big core while the game ran. A key or a
    // controller waits 4 ms at most; the touch overlay's input goes to the game over JNI, not through here
    constexpr auto kIdle = std::chrono::milliseconds(4);
    while (app::g_running) {
        SDL_Event e;
        bool any = false;
        while (app::g_running && SDL_PollEvent(&e)) {
            app::Handle(e);
            any = true;
        }
        if (!any) std::this_thread::sleep_for(kIdle);
    }
    {
        std::lock_guard lock(app::g_open_lock);
        app::g_open_cv.notify_all();
    }
    {
        std::lock_guard lock(app::g_foreground_lock);
        app::g_foreground_cv.notify_all();
    }
    LOGR("Exiting");
    logx::Flush();
    kernel::Shutdown(0);
}
