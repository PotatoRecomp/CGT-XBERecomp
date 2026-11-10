#pragma once
#include "core/input.h"

namespace input {

void StartMouse(void* window);  // GameInfo::window
bool Focused();                 // the game window has the keyboard
void OnMouseWheel(float y);
bool WheelHeld();               // a wheel notch, held as a button for a moment

// Mouse look. The window thread holds the mouse in relative mode while the game has it (focused, F10 panel shut,
// mouse look on) and adds up its raw counts; the game thread takes them where it turns the view (game/hooks.cpp).
void SyncMouseCapture();                 // window thread: grab or let go of the mouse as the state says
void OnMouseMotion(float dx, float dy);  // window thread: a relative motion event's counts
// Touch look (Android): a finger's travel in dp, x right and y down, added up beside the mouse's counts and turned the
// same way, straight into the look angles rather than through the right stick; any thread
void OnTouchLook(float dx, float dy);
// Android gyro: integrated rotation in degrees, x right and y down. Zeroes clear pending gyro motion.
void OnGyroLook(float yaw, float pitch);
// game thread: the turn asked for since the last take in degrees, each source's sensitivity and invert Y applied, x
// right and y down, before the zoom's scaling; false when there is none. Motion that waited longer than a moment (the
// game paused, a cutscene, a menu) is dropped rather than turned all at once.
bool TakeLook(float& yaw, float& pitch);

}  // namespace input

namespace game {

void PadHook(int port, input::Pad& p);  // GameInfo::pad
bool Event(const void* sdl_event);      // GameInfo::event
void LogLookHistory();                  // game/hooks.cpp: the player's look, frame by frame, for a report

}  // namespace game
