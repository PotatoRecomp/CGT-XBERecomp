# Conflict: Global Terror - XBE Recomp
An unofficial native PC port of the original Xbox version of Conflict: Global Terror.

This port was created using the [Xbox XBE Recomp](https://github.com/TGP482/XboxXBE-Recomp) tool.

> [!NOTE]
> This project does not include any game assets. You must provide the files from your own legally acquired copy of the game to install or build.

## Installation
- Put your game files in the `assets` folder: the disc image (`.iso`) of the USA release, or `default.xbe` with the `data` folder and the mission files beside it.
- Run the executable. Saves (`hdd`), logs and `settings.toml` are written beside it.

## Settings
Press **F10** in game to open the recomp settings (F10 or Esc closes them, which saves `settings.toml`). They cover the recomp's general settings only; the game's own options stay in its menus. While the panel is open the game gets no input, and the keyboard or controller works the panel.
- **Display:** display mode, vertical sync, max frame rate, game frame rate (30 as on the console, or 60), render resolution, FPS counter.
- **Graphics:** MSAA, SMAA, FXAA, anisotropic filtering, bloom, fog, and HDR output on displays with HDR on.
- **Controls:** mouse look, mouse sensitivity, mouse invert Y.
- **Audio:** volume, 50% to 400%, 200% by default. The game mixes for a TV turned up (its effects at -12.5 dB, music at -20 dB, every mix bin 6 dB lower again for headroom), which on a PC sounds almost silent; 200% gives the headroom back.
- **Screen filter** (from MostWanted-Android): Sharpen, Vivid, Scanlines, CRT and HDR Look, applied at the present.

## Keyboard and Mouse
The keyboard and mouse buttons work as the controller on port 1, together with a real controller. The mouse turns the view directly: its raw counts (no desktop pointer speed or acceleration) are turned into degrees and added to the game's own look angles, not passed through the right stick, so there is no dead zone, response curve, top turn speed or the 0.1 s lag the game gives the stick's turn. Scopes and zoom scale it down with the view, as the game does the stick. **MOUSE LOOK**, **MOUSE SENSITIVITY** (1.00 is 0.022 degrees a count) and **MOUSE INVERT Y** are on the F10 panel; the pointer is held while the game has focus and given back while the panel is open. The keypad and the controller's right stick still look as before.

| Action | Keys |
|---|---|
| Move | W A S D |
| Look | Mouse, keypad 4 6 8 2 |
| Fire / Aim (scope) | Left mouse or X / Right mouse, V or Right Shift |
| Left trigger | Z |
| A / B / X / Y | Space, E, F / C, Ctrl / R / Q, mouse wheel |
| Black / White | G, middle mouse / T, mouse side button |
| D-pad (menus, squad) | Arrows, 1 2 3 4 |
| Start / Back | Enter, Esc / Tab, Backspace, M |
| Stick clicks | Left Shift / V, Right Shift |

## Linux
The game runs natively on Linux, drawn with Vulkan. The recompiled game code is linked into the executable when it is built, so there is no first-start compile.

### Building on Linux
1. Add the game to `assets`, as above. With only a disc image there, `build.sh` takes `default.xbe` off it for the recompiler.
2. Install Clang, LLD, CMake, Ninja, Git, pkg-config, the Vulkan loader and headers, and shaderc, plus SDL3's build dependencies (X11, Wayland, xkbcommon, PipeWire/PulseAudio/ALSA, udev). `build.sh` lists the packages for Arch, Debian/Ubuntu and Fedora when something is missing.
3. Run `./build.sh` (`./build.sh debug` for full logs). The first build writes `config/game.toml` with `xbe_symbols`, then recompiles the entire game: 15 to 60 minutes on weaker systems.
4. Run `out/build/release/cgt`.

### Requirements
- A 64-bit distribution with a Vulkan 1.3 driver supporting `VK_KHR_push_descriptor` (Mesa 23 or newer, NVIDIA 535 or newer)
- A CPU with SSE4.2 and POPCNT, 4 GB of memory

## Status
Boots through the intro movies and menus into missions, with sound. Menu titles and items appear as scrambled letters for about a second before settling; that is the game's own reveal animation.

Fixes over the stock SDK and the `xbe_symbols` output:
- `BinkWait` (0x3B08D0) set by hand: the signatures miss it, and the SDK's Bink frame pacing called the trap address on the first movie.
- `D3DDevice_DrawIndexedVerticesUP` (0x2E2F50) set by hand, with its host version ported from the Warriors and Midnight Club 3 SDKs: left to the library it ran the console's push buffer code on the loading screen.
- `D3DDevice_SetVertexShaderInput` (0x2DB670) and `D3DDevice_SetVertexShaderInputDirect` (0x2DB0E0), new in this SDK copy: the game binds some meshes' streams and attribute layout through them rather than `SetStreamSource`. Left to the library, the host drew those meshes with stale streams, as large black shapes across the scene in missions.
- `BlockUntilVerticalBlank` waits for the next 60 Hz blank instead of sleeping a flat 16 ms: the game paces itself by counting blanks, and the flat sleep held it at 25 fps instead of the console's 30.
- **60 fps** (optional, GAME FRAME RATE in the F10 panel): the game's frame pacer (0x112D2D) waits for two vertical blanks per frame; run with one it draws 60. The game's clock steps the simulation by measured milliseconds, so it keeps its speed (checked: game time 1.000x real time at 30 and at 60). Menus stay at 30.
- Frame copies between targets kept at different scales are drawn across instead of copied as image regions, which read the source rect at the destination's scale: the game's light glow test saves a patch of the frame and puts it back, and the doorway in the first mission showed another part of the scene, as if seen through water.
- Audio latency: the mixer renders 5 ms blocks and asks for a 256 frame (5.3 ms) device period. PipeWire's default of 1024 frames and 10 ms blocks put about 50 ms between a sound starting and the speakers, gunshots heard late.
- Sound buffer `Lock` reads its flags as the library does: 1 starts at the write cursor, 2 takes the entire buffer (the SDK had them the other way round). The game streams every sound through a 1 to 2 second ring per voice and fills the whole ring with flag 2 before it plays it; that fill wrote nothing, so each sound played a ring of the voice's previous sound, at the new sound's pitch, before its own start: a sound played at 44 kHz started a second late, one pitched down to 10 kHz 4.4 seconds late, and what played first was another sound at the wrong pitch. The write cursor is the library's too (32 frames ahead while playing, the play cursor when stopped).
- Compiled shaders (`shadercache.bin`) and the driver's pipeline cache (`pipelinecache.bin`) are kept beside the executable: a new effect on screen (a muzzle flash, an explosion) compiled its shaders and stalled the frame every run; now only the first time ever.
- Lights kept by index: Direct3D's light index is any number, and the game lights its props and characters with lights 8 and 71 (others past 200). The SDK only kept 0 to 7, so chairs, tables, bodies and the player's helmet drew black; up to 8 enabled lights now fill the renderer's slots in the order they were enabled.
- Pre-transformed (XYZRHW) vertices moved half a pixel right and down, as in the Dynasty Warriors 5 and Warriors SDKs: the console samples on its pixel centre, so drawn at the host's the small font lost the top row of every glyph (the bar of F and E).
- DirectSound taken from Urban Chaos' SDK, the most complete of the ports: Doppler shift from the sounds' and listener's velocities, distance, rolloff and doppler factors, cones, pitch, mix-in buffers, and the DSOUND exports named only by id (`B0073` is `SetMode`, which crashed the first level load when left to the guest; `B0080` passes on to `FlushEx`).

## Game Hooks
Game specific fixes go in `src/game`. Define `X86_FUNC(sub_XXXXXXXX)` to replace a recompiled function and call `__imp__sub_XXXXXXXX` to reach the original. Functions the analysis misses go in `[functions]` of `config/game.toml`.
