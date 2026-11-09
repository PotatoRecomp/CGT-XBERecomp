#pragma once
#include <filesystem>
#include <string>

#include "core/input.h"

struct SDL_Window;

// the Android host, shared between native/main.cpp, the Vulkan backend, the settings bridge and the touch controls
namespace host {

SDL_Window* Window();
bool Foreground();                 // the activity is in front: the renderer waits while it is not
uint32_t SurfaceEpoch();           // counts returns to the front: a new window, its surface made again
std::filesystem::path FilesDir();  // internal, writable: logs, settings, shader cache
std::filesystem::path GameDir();   // where the game's files were found, empty until FindGame finds them
void Toast(const char* text);      // the activity's toast, any thread
// the renderer's present: its last acquire and present VkResults, and whether the frame reached the screen (the
// freeze watchdog in native/main.cpp)
void NotePresent(int acquire, int present, bool shown);
// the render thread's current sub-step, a static string ("acquire", "wait fence", "present", ...); nullptr is idle.
// The freeze watchdog names it and how long the thread has sat there. Any thread.
void NoteRenderPhase(const char* phase);
// the on-screen controls' overlay told to read the layout and settings again and whether a controller is plugged in:
// once the settings file is loaded, and as a controller comes or goes; any thread
void ControlsSync(bool pad);
// the settings panel (SettingsDialog.java) opened or closed from native: F10, or Escape while it is open, on a
// keyboard (src/common/keyboard.cpp through app::ToggleMenu); any thread
void ShowSettings(bool show);
// the GPU the renderer picked, for the stats overlay; set once at its start
void SetGpuName(const char* name);
const char* GpuName();
// the SoC as Android names it (ro.soc.model, the board platform, /proc/cpuinfo), empty when none says
const std::string& SocName();

// app::kGame's hooks on Android (src/common/game_info.cpp): each frame's ImGui drawing (the FPS counter, the overlay
// told of a controller), the pad with the on-screen controls laid over it, and the touch layout kept in the settings
// file's table
void Ui(double fps);
void Pad(int port, input::Pad& p);
void LoadSettings(const void* table);
void SaveSettings(void* table);

}  // namespace host

namespace touch {

// The controls are TouchControls.java's, an overlay the activity draws; what they hold and where they sit come here
// (native/settings_bridge.cpp). There is no on-screen right stick: a finger dragged on the free right side of the
// screen (or held on FIRE or AIM) turns the view through the mouse look path (input::OnTouchLook), not the stick.
void Init();
// what the fingers hold: input::Pad::buttons bits, a bit per analog button held, the left stick -1 to 1 with y up
void Set(uint32_t buttons, uint32_t analog, float lx, float ly);
// the layout, each control's centre as a fraction of the screen and its size as a fraction of its kind's: read from
// the settings file's table before the first frame, written back into it
void LoadLayout(const void* table);
void SaveLayout(void* table);
int Units();
void GetUnit(int unit, float& x, float& y, float& scale);
void SetUnit(int unit, float x, float y, float scale, bool commit);  // commit writes the file
void ResetUnits();  // the pad's own layout back, every control at its kind's size
// gameplay: the stick and buttons of the on-screen controls, over the pad the keyboard left
void Apply(input::Pad& pad);
void Frame();  // once a frame: tells the overlay when a controller comes or goes
// the stats overlay's lines as one, for a report (native/main.cpp)
void StatsLine(char* out, size_t size);

}  // namespace touch
