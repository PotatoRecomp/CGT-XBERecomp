#!/usr/bin/env bash
# Linux build: build.sh [debug]  (debug: the release build's optimisation with full logs)
set -e
GAME_TITLE="Conflict - Global Terror"
GAME_XBE=default.xbe
GAME_EXE=cgt
cd "$(dirname "$0")"
PRESET=release
[ "$1" = debug ] && PRESET=debug
OUT=out/build/$PRESET

echo
echo "  $GAME_TITLE"
echo "  =================="
echo

fail() {
    echo
    echo "  BUILD FAILED. $1"
    echo
    exit 1
}

# the recompiler reads the xbe itself; with only a disc image in assets it is taken off the image
if [ ! -f "assets/$GAME_XBE" ]; then
    ISO=$(ls assets/*.iso assets/*.ISO 2>/dev/null | head -1 || true)
    if [ -n "$ISO" ] && command -v python3 >/dev/null; then
        echo "  Taking $GAME_XBE off $(basename "$ISO") ..."
        python3 -I tools/xiso_get.py "$ISO" "$GAME_XBE" "assets/$GAME_XBE" >/dev/null ||
            fail "Could not read $GAME_XBE from $ISO."
    fi
fi
if [ ! -f "assets/$GAME_XBE" ]; then
    echo "  Put the game in the assets folder first:"
    echo "    $PWD/assets"
    echo "  Its disc image (.iso), or $GAME_XBE with the data folder and the mission files beside it."
    fail ""
fi

# ---------------------------------------------------------------- toolchain
echo "[1/3] Toolchain"
missing=
for tool in clang++ cmake ninja git pkg-config; do
    command -v $tool >/dev/null || missing="$missing $tool"
done
for lib in vulkan shaderc; do
    pkg-config --exists $lib 2>/dev/null || missing="$missing $lib"
done
if [ -n "$missing" ]; then
    echo "      missing:$missing"
    echo "      Arch:          sudo pacman -S clang lld cmake ninja git pkgconf vulkan-headers vulkan-icd-loader shaderc"
    echo "      Debian/Ubuntu: sudo apt install clang lld cmake ninja-build git pkg-config libvulkan-dev libshaderc-dev"
    echo "                     and SDL3's build dependencies (libx11-dev libxext-dev libwayland-dev libxkbcommon-dev"
    echo "                     libpipewire-0.3-dev libpulse-dev libasound2-dev libudev-dev libdecor-0-dev)"
    echo "      Fedora:        sudo dnf install clang lld cmake ninja-build git pkgconf vulkan-loader-devel libshaderc-devel"
    fail "Install the missing tools, then run this again."
fi
echo "      $(clang++ --version | head -1)"

# ---------------------------------------------------------------- symbols
echo "[2/3] XDK symbols"
cmake --preset $PRESET >/dev/null || fail "Read the last message above."
if grep -q '^\[symbols\]' config/game.toml 2>/dev/null; then
    echo "      config/game.toml already written"
else
    cmake --build "$OUT" --target xbe_symbols >/dev/null || fail "Read the last message above."
    SDK=$(head -1 "$OUT/sdk_dir.txt")
    mkdir -p config
    "$OUT/xbe_symbols" "assets/$GAME_XBE" "$SDK/xdk" --library CGT.dll --out config/game.toml ||
        fail "Read the last message above."
fi

# ---------------------------------------------------------------- game
echo "[3/3] Game code"
echo "      the first build recompiles $GAME_XBE: 15 to 60 minutes on weaker systems ..."
cmake --build "$OUT" || fail "Read the last message above."

echo
echo "  $GAME_TITLE built."
echo "  $OUT/$GAME_EXE"
echo
