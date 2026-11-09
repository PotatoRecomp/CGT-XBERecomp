// The F10 settings panel: the recomp's general settings only (display, graphics, screen filter). The game's own
// options stay in its menus. Rows and controls as in UCRR and Black: a row per setting, its value stepped with the
// arrows (keys, D-pad or the < > buttons), saved to the settings file when the panel closes.
#include "common/menu.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <iterator>

#include "app/app.h"
#include "common/mouse_keyboard.h"
#include "common/options.h"

namespace app {

namespace {

std::atomic<bool> g_open{false};
float g_fps = 0;

// the game's menus: steel blue bars over a dark night map
constexpr ImVec4 kPanelBg{0.03f, 0.05f, 0.09f, 0.95f};
constexpr ImVec4 kBorder{0.45f, 0.55f, 0.95f, 0.85f};
constexpr ImVec4 kText{0.90f, 0.93f, 0.98f, 1.0f};
constexpr ImVec4 kTextDisabled{0.45f, 0.50f, 0.60f, 1.0f};
constexpr ImVec4 kHighlight{0.16f, 0.30f, 0.62f, 1.0f};
constexpr ImVec4 kHeading{0.62f, 0.72f, 1.0f, 1.0f};
constexpr ImVec4 kButton{0.10f, 0.16f, 0.32f, 1.0f};
constexpr ImVec4 kButtonHovered{0.18f, 0.28f, 0.54f, 1.0f};
constexpr ImVec4 kSelected{0.85f, 0.78f, 0.50f, 1.0f};  // the game's gold for the chosen item

constexpr float kValueEms = 12.0f;  // the value's room between the arrows

struct RowBox {
    float row_w, row_h, value_w, block_w, row_x, row_y;
};

RowBox MeasureRow() {
    const ImGuiStyle& style = ImGui::GetStyle();
    RowBox box;
    box.row_w = ImGui::GetContentRegionAvail().x;
    box.row_h = ImGui::GetFrameHeight();
    box.value_w = ImGui::GetFontSize() * kValueEms;
    box.block_w = box.row_h * 2 + box.value_w + style.ItemSpacing.x * 2;
    box.row_x = ImGui::GetCursorPosX();
    box.row_y = ImGui::GetCursorPosY();
    return box;
}

void Section(const char* title) {
    float em = ImGui::GetFontSize();
    ImGui::Dummy(ImVec2(0, em * 0.4f));
    ImGui::PushStyleColor(ImGuiCol_Text, kHeading);
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0, em * 0.2f));
}

// label on the left, < value > on the right; -1 or 1 when stepped this frame
int OptionRow(const char* label, const char* value, const char* hint = nullptr) {
    const ImGuiStyle& style = ImGui::GetStyle();
    RowBox box = MeasureRow();
    ImGui::PushID(label);
    int step = 0;
    ImGui::Selectable("##row", false, ImGuiSelectableFlags_AllowOverlap, ImVec2(box.row_w, box.row_h));
    bool focused = ImGui::IsItemFocused();
    if (hint && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", hint);
    if (focused) {
        ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
                                                  ImGui::GetColorU32(ImGuiCol_Header));
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft)) step = -1;
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) || ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight)) step = 1;
    }
    ImGui::SetCursorPos(ImVec2(box.row_x + style.ItemInnerSpacing.x, box.row_y));
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);

    ImGui::PushItemFlag(ImGuiItemFlags_NoNav | ImGuiItemFlags_ButtonRepeat, true);
    float block_x = box.row_x + box.row_w - box.block_w;
    ImGui::SetCursorPos(ImVec2(block_x, box.row_y));
    if (ImGui::ArrowButton("##dec", ImGuiDir_Left)) step = -1;
    float value_x = block_x + box.row_h + style.ItemSpacing.x;
    float pad = (box.value_w - ImGui::CalcTextSize(value).x) * 0.5f;
    ImGui::SetCursorPos(ImVec2(value_x + std::max(pad, 0.0f), box.row_y));
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(focused ? kSelected : kText, "%s", value);
    ImGui::SetCursorPos(ImVec2(value_x + box.value_w + style.ItemSpacing.x, box.row_y));
    if (ImGui::ArrowButton("##inc", ImGuiDir_Right)) step = 1;
    ImGui::PopItemFlag();
    ImGui::PopID();

    ImGui::SetCursorPos(ImVec2(box.row_x, box.row_y));
    ImGui::Dummy(ImVec2(box.row_w, box.row_h));
    return step;
}

const char* OnOff(bool on) { return on ? "ON" : "OFF"; }

int Wrap(int index, int step, int count) {
    index = (index + step) % count;
    return index < 0 ? index + count : index;
}

// a value from a list: its index, the value itself when it is not in the list taken as the first
template <typename T, size_t N>
int IndexOf(const T (&values)[N], T v) {
    for (size_t i = 0; i < N; i++)
        if (values[i] == v) return int(i);
    return 0;
}

// a row over a list of values with their names
template <typename T, size_t N>
void ListRow(const char* label, T& v, const T (&values)[N], const char* const (&names)[N], const char* hint = nullptr) {
    int i = IndexOf(values, v);
    if (int step = OptionRow(label, names[i], hint)) v = values[Wrap(i, step, int(N))];
}

void FpsOverlay() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    float em = ImGui::GetFontSize();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - em, vp->WorkPos.y + em), ImGuiCond_Always,
                            ImVec2(1, 0));
    ImGui::SetNextWindowBgAlpha(0.55f);
    constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs;
    if (ImGui::Begin("##fps", nullptr, kFlags)) ImGui::TextColored(ImVec4(0.45f, 0.95f, 0.55f, 1), "%.0f FPS", g_fps);
    ImGui::End();
}

void RestoreDefaults() {
    Settings& s = Get();
    const Settings ds;
    s.display_mode = ds.display_mode;
    s.render_percent = ds.render_percent;
    s.fps_cap = ds.fps_cap;
    s.vsync = ds.vsync;
    s.aniso = ds.aniso;
    game::Opt() = game::Options{};
    ApplyDisplay();
}

void Rows() {
    Settings& s = Get();
    game::Options& o = game::Opt();

    Section("DISPLAY");
    {
        constexpr DisplayMode kModes[] = {DisplayMode::Fullscreen, DisplayMode::Borderless, DisplayMode::Windowed};
        const char* const kNames[] = {"FULLSCREEN", "BORDERLESS", "WINDOWED"};
        DisplayMode was = s.display_mode;
        ListRow("DISPLAY MODE", s.display_mode, kModes, kNames);
        if (s.display_mode != was) ApplyDisplay();
    }
    if (int step = OptionRow("VERTICAL SYNC", OnOff(s.vsync))) {
        (void)step;
        s.vsync = !s.vsync;
        ApplyDisplay();
    }
    {
        constexpr uint32_t kCaps[] = {30, 60, 75, 90, 120, 144, 165, 240, 0};
        const char* const kNames[] = {"30", "60", "75", "90", "120", "144", "165", "240", "UNLIMITED"};
        ListRow("MAX FRAME RATE", s.fps_cap, kCaps, kNames, "The console ran the game at 30.");
    }
    if (OptionRow("GAME FRAME RATE", o.frame_rate_60 ? "60" : "30 (CONSOLE)",
                  "The game paces itself at 30 like the console; 60 runs it twice as smooth at the same speed."))
        o.frame_rate_60 = !o.frame_rate_60;
    if (OptionRow("FRAME SKIP", OnOff(o.frame_skip),
                  "While the drawing falls behind, frames are left undrawn so the picture keeps up with the game."))
        o.frame_skip = !o.frame_skip;
    {
        char text[16];
        std::snprintf(text, sizeof text, "%u%%", s.render_percent);
        if (int step = OptionRow("RENDER RESOLUTION", text, "Of the output's height: below 100 is faster, above sharper."))
            s.render_percent = uint32_t(std::clamp(int(s.render_percent) + step * 10, 50, 200));
    }
    if (OptionRow("FPS COUNTER", OnOff(o.fps_counter))) o.fps_counter = !o.fps_counter;

    Section("GRAPHICS");
    {
        constexpr uint32_t kSamples[] = {0, 2, 4, 8};
        const char* const kNames[] = {"OFF", "2x", "4x", "8x"};
        ListRow("MSAA", o.msaa, kSamples, kNames, "Multisampling: smooth geometry edges, sharp image.");
    }
    if (OptionRow("SMAA", OnOff(o.smaa), "Smooths every edge of the frame, alpha tested ones too.")) o.smaa = !o.smaa;
    if (OptionRow("FXAA", OnOff(o.fxaa), "A cheap edge blur at the present.")) o.fxaa = !o.fxaa;
    {
        constexpr uint32_t kLevels[] = {0, 2, 4, 8, 16};
        const char* const kNames[] = {"GAME DEFAULT", "2x", "4x", "8x", "16x"};
        ListRow("ANISOTROPIC FILTERING", s.aniso, kLevels, kNames);
    }
    if (OptionRow("BLOOM", OnOff(o.bloom))) o.bloom = !o.bloom;
    if (OptionRow("FOG", OnOff(o.fog))) o.fog = !o.fog;
    if (OptionRow("SHADOWS", OnOff(o.shadows), "The shadow the room's main light casts.")) o.shadows = !o.shadows;
    if (game::g_hdr_display.load() || o.hdr_output) {
        if (OptionRow("HDR OUTPUT", OnOff(o.hdr_output), "HDR10 on a display with HDR turned on.")) o.hdr_output = !o.hdr_output;
    }

    Section("CONTROLS");
    if (OptionRow("MOUSE LOOK", OnOff(o.mouse_look), "The mouse turns the view directly; off leaves the pointer free."))
        o.mouse_look = !o.mouse_look;
    {
        char text[16];
        std::snprintf(text, sizeof text, "%u.%02u", o.mouse_sensitivity / 100, o.mouse_sensitivity % 100);
        if (int step = OptionRow("MOUSE SENSITIVITY", text,
                                 "1.00 turns 0.022 degrees a count, as most PC shooters. Scopes and zoom scale it down "
                                 "with the view."))
            o.mouse_sensitivity = uint32_t(std::clamp(int(o.mouse_sensitivity) + step * 5, 5, 1000));
    }
    if (OptionRow("MOUSE INVERT Y", OnOff(o.mouse_invert_y))) o.mouse_invert_y = !o.mouse_invert_y;

    Section("AUDIO");
    {
        char text[16];
        std::snprintf(text, sizeof text, "%u%%", o.volume);
        if (int step = OptionRow("VOLUME", text, "The game mixes quietly for a TV: 200% matches a PC's other sound."))
            o.volume = uint32_t(std::clamp(int(o.volume) + step * 25, 50, 400));
    }

    Section("SCREEN FILTER");
    {
        constexpr uint32_t kFilters[] = {game::kFilterNone,      game::kFilterSharpen, game::kFilterVivid,
                                         game::kFilterScanlines, game::kFilterCrt,     game::kFilterHdr};
        const char* const kNames[] = {"OFF", "SHARPEN", "VIVID", "SCANLINES", "CRT", "HDR LOOK"};
        ListRow("FILTER", o.screen_filter, kFilters, kNames,
                "SHARPEN crisper edges, VIVID richer colour, SCANLINES and CRT a TV's look, HDR LOOK deeper "
                "shadows and held back highlights.");
    }
}

}  // namespace

bool MenuOpen() { return g_open.load(std::memory_order_relaxed); }

void ToggleMenu() {
    bool open = !MenuOpen();
    g_open.store(open, std::memory_order_relaxed);
    if (!open) Save();
    Post(input::SyncMouseCapture);  // the panel takes the pointer, the game gets it back
}

void DrawUi(double fps) {
    g_fps = float(fps);
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    // sized by the window: ImGui's 13 px font at 480 lines, scaled up with the window
    ImGui::GetIO().FontGlobalScale = std::max(1.0f, vp->WorkSize.y / 480.0f);
    if (game::Opt().fps_counter) FpsOverlay();
    if (!MenuOpen()) return;

    float em = ImGui::GetFontSize();
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(std::min(em * 44.0f, vp->WorkSize.x * 0.95f), std::min(em * 36.0f, vp->WorkSize.y * 0.92f)),
                             ImGuiCond_Always);

    const std::pair<ImGuiCol, ImVec4> colors[] = {
        {ImGuiCol_WindowBg, kPanelBg},         {ImGuiCol_Border, kBorder},
        {ImGuiCol_Text, kText},                {ImGuiCol_TextDisabled, kTextDisabled},
        {ImGuiCol_Header, kHighlight},         {ImGuiCol_HeaderHovered, kHighlight},
        {ImGuiCol_HeaderActive, kHighlight},   {ImGuiCol_Button, kButton},
        {ImGuiCol_ButtonHovered, kButtonHovered}, {ImGuiCol_ButtonActive, kHighlight},
        {ImGuiCol_Separator, kBorder},         {ImGuiCol_NavHighlight, kSelected},
    };
    for (auto& [col, v] : colors) ImGui::PushStyleColor(col, v);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(em * 1.5f, em * 1.2f));

    constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar |
                                        ImGuiWindowFlags_NoMove;
    if (ImGui::Begin("##settings", nullptr, kFlags)) {
        const char* title = "RECOMP SETTINGS";
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                             std::max((ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(title).x) * 0.5f, 0.0f));
        ImGui::TextColored(kHeading, "%s", title);

        float footer = ImGui::GetFrameHeightWithSpacing() + em * 1.2f;
        ImGui::BeginChild("##rows", ImVec2(0, -footer), ImGuiChildFlags_NavFlattened);
        // the first row has the cursor as the panel opens, so the arrows work at once
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        Rows();
        ImGui::EndChild();

        ImGui::Separator();
        ImGui::TextDisabled("F10 / ESC close");
        float bw = em * 8.0f, gap = ImGui::GetStyle().ItemSpacing.x;
        ImGui::SameLine(ImGui::GetWindowWidth() - ImGui::GetStyle().WindowPadding.x - bw * 2 - gap);
        if (ImGui::Button("DEFAULTS", ImVec2(bw, 0))) RestoreDefaults();
        ImGui::SameLine();
        if (ImGui::Button("CLOSE", ImVec2(bw, 0))) ToggleMenu();
    }
    ImGui::End();

    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(int(std::size(colors)));
}

}  // namespace app
