// A relaunch on Linux replaces the process (kernel::Relaunch), so no previous instance holds a window for this one
#include "app/app.h"

namespace app {

bool StartHandover() { return false; }
bool HandingOver() { return false; }
void FinishHandover() {}

// a desktop window keeps running behind other windows: the game never waits for the front (Android's host does)
void WaitWhileUnfocused() {}
bool FocusPaused() { return false; }

}  // namespace app
