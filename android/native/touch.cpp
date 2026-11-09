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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#include <unistd.h>

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
// The stats overlay (FPS COUNTER), one row centred at the top: the frame rate and time, then what a drop can be told by, the
// CPU (this app's share of all cores, the fastest core's clock, the SoC), the GPU (busy share and clock where the
// driver lets an app read them, Adreno's kgsl and Mali's and MediaTek's files, and its name), the app's memory and the
// battery's temperature, a phone's own throttle. A value Android keeps from apps is left out rather than shown wrong.
// Sampled twice a second: a few small files, read on the render thread
struct Stats {
    double cpu = -1;      // percent of all cores
    double cpu_ghz = -1;  // the fastest core now
    double gpu = -1;      // percent busy
    double gpu_mhz = -1;
    double ram_mb = -1;
    double temp_c = -1;  // battery
};

std::mutex g_stats_lock;
Stats g_stats;
double g_stats_fps = 0;

// the first number in a small file, NaN when it cannot be read
double ReadNumber(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) return NAN;
    double v = NAN;
    if (std::fscanf(f, "%lf", &v) != 1) v = NAN;
    std::fclose(f);
    return v;
}

void SampleStats() {
    using clock = std::chrono::steady_clock;
    static clock::time_point last{};
    static double last_ticks = -1;
    static const long cores = std::max(sysconf(_SC_NPROCESSORS_CONF), 1L);
    static const double tick = double(std::max(sysconf(_SC_CLK_TCK), 1L));
    auto now = clock::now();
    double dt = std::chrono::duration<double>(now - last).count();
    if (last != clock::time_point{} && dt < 0.5) return;
    Stats st;
    // utime and stime, fields 14 and 15 of /proc/self/stat, after the name in parentheses
    if (FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[1024];
        size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        buf[n] = 0;
        unsigned long long ut = 0, sys = 0;
        if (const char* close = std::strrchr(buf, ')');
            close && std::sscanf(close + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &ut, &sys) == 2) {
            double ticks = double(ut + sys);
            if (last_ticks >= 0 && dt > 0) st.cpu = (ticks - last_ticks) / tick / dt / double(cores) * 100.0;
            last_ticks = ticks;
        }
    }
    for (long c = 0; c < cores; c++) {
        char path[96];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/cpufreq/scaling_cur_freq", c);
        double khz = ReadNumber(path);
        if (khz > 0) st.cpu_ghz = std::max(st.cpu_ghz, khz / 1e6);
    }
    // Adreno, Mali, MediaTek's GED, Samsung's and others' kernel node
    for (const char* p : {"/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage", "/sys/class/misc/mali0/device/utilization",
                          "/sys/kernel/ged/hal/gpu_utilization", "/sys/kernel/gpu/gpu_busy"}) {
        double v = ReadNumber(p);
        if (v >= 0 && v <= 100) {
            st.gpu = v;
            break;
        }
    }
    double hz = ReadNumber("/sys/class/kgsl/kgsl-3d0/gpuclk");
    if (hz > 0) st.gpu_mhz = hz / 1e6;
    if (st.gpu_mhz < 0) {
        double mhz = ReadNumber("/sys/kernel/gpu/gpu_clock");
        if (mhz > 0) st.gpu_mhz = mhz;
    }
    // resident pages, the second field of statm
    if (FILE* f = std::fopen("/proc/self/statm", "r")) {
        unsigned long long size = 0, rss = 0;
        if (std::fscanf(f, "%llu %llu", &size, &rss) == 2) st.ram_mb = double(rss) * double(sysconf(_SC_PAGESIZE)) / 1048576.0;
        std::fclose(f);
    }
    double tenths = ReadNumber("/sys/class/power_supply/battery/temp");
    if (tenths > 0 && tenths < 1000) st.temp_c = tenths / 10.0;
    last = now;
    std::lock_guard lock(g_stats_lock);
    g_stats = st;
}

// the overlay's lines; parts the phone does not give are left out
int StatsLines(double fps, char lines[4][128]) {
    Stats st;
    {
        std::lock_guard lock(g_stats_lock);
        st = g_stats;
    }
    auto add = [](char* line, const char* fmt, double v) {
        size_t n = std::strlen(line);
        std::snprintf(line + n, 128 - n, fmt, v);
    };
    int count = 0;
    std::snprintf(lines[count++], 128, "%.0f FPS  %.1f ms", fps, fps > 0 ? 1000.0 / fps : 0.0);
    char* cpu = lines[count++];
    std::snprintf(cpu, 128, "CPU");
    if (st.cpu >= 0) add(cpu, " %.0f%%", st.cpu);
    if (st.cpu_ghz > 0) add(cpu, "  %.2f GHz", st.cpu_ghz);
    if (!host::SocName().empty()) std::snprintf(cpu + std::strlen(cpu), 128 - std::strlen(cpu), "  %s", host::SocName().c_str());
    char* gpu = lines[count++];
    std::snprintf(gpu, 128, "GPU");
    if (st.gpu >= 0) add(gpu, " %.0f%%", st.gpu);
    if (st.gpu_mhz > 0) add(gpu, "  %.0f MHz", st.gpu_mhz);
    if (*host::GpuName()) std::snprintf(gpu + std::strlen(gpu), 128 - std::strlen(gpu), "  %s", host::GpuName());
    char* rest = lines[count];
    rest[0] = 0;
    if (st.ram_mb > 0) add(rest, "RAM %.0f MB", st.ram_mb);
    if (st.temp_c > 0) add(rest, *rest ? "  %.1f C" : "%.1f C", st.temp_c);
    if (*rest) count++;
    return count;
}

void FpsCounter(double fps) {
    SampleStats();
    {
        std::lock_guard lock(g_stats_lock);
        g_stats_fps = fps;
    }
    char lines[4][128];
    int count = StatsLines(fps, lines);
    // one row, the parts apart by a bar, never wrapped: a row wider than the screen (a long SoC or GPU name on a narrow
    // one) is drawn smaller until it fits
    ImGuiIO& io = ImGui::GetIO();
    ImFont* font = ImGui::GetFont();
    const char* const kBar = "  |  ";
    float size = std::max(io.DisplaySize.y * 0.026f, 11.0f);
    auto row_width = [&](float sz) {
        float w = 0;
        for (int i = 0; i < count; i++) {
            w += font->CalcTextSizeA(sz, FLT_MAX, 0, lines[i]).x;
            if (i + 1 < count) w += font->CalcTextSizeA(sz, FLT_MAX, 0, kBar).x;
        }
        return w;
    };
    float width = row_width(size);
    float room = io.DisplaySize.x * 0.96f;
    if (width > room) {
        size *= room / width;
        width = row_width(size);
    }
    float pad = size * 0.35f;
    ImVec2 at((io.DisplaySize.x - width) * 0.5f, size * 0.3f);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(ImVec2(at.x - pad, at.y - pad * 0.5f), ImVec2(at.x + width + pad, at.y + size + pad * 0.5f),
                      IM_COL32(0, 0, 0, 130), size * 0.25f);
    const ImU32 kFps = IM_COL32(115, 242, 140, 235), kText = IM_COL32(225, 232, 245, 220), kSep = IM_COL32(225, 232, 245, 110);
    float x = at.x;
    for (int i = 0; i < count; i++) {
        dl->AddText(font, size, ImVec2(x, at.y), i == 0 ? kFps : kText, lines[i]);
        x += font->CalcTextSizeA(size, FLT_MAX, 0, lines[i]).x;
        if (i + 1 < count) {
            dl->AddText(font, size, ImVec2(x, at.y), kSep, kBar);
            x += font->CalcTextSizeA(size, FLT_MAX, 0, kBar).x;
        }
    }
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

void StatsLine(char* out, size_t size) {
    SampleStats();
    double fps;
    {
        std::lock_guard lock(g_stats_lock);
        fps = g_stats_fps;
    }
    char lines[4][128];
    int count = StatsLines(fps, lines);
    size_t at = 0;
    out[0] = 0;
    for (int i = 0; i < count && at + 1 < size; i++)
        at += size_t(std::snprintf(out + at, size - at, i ? " | %s" : "%s", lines[i]));
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
