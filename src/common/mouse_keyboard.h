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
// game thread: the counts since the last take, x right and y down; false when there are none. Counts that waited
// longer than a moment (the game paused, a cutscene, a menu) are dropped rather than turned all at once.
bool TakeMouseLook(float& dx, float& dy);

}  // namespace input

namespace game {

void PadHook(int port, input::Pad& p);  // GameInfo::pad
bool Event(const void* sdl_event);      // GameInfo::event

}  // namespace game
