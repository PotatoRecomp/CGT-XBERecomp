#!/usr/bin/env bash
# Conflict: Global Terror for Android: writes the XDK symbols and recompiles default.xbe on this PC when the desktop
# build has not already, then builds the APK. Everything runs on Linux: the recompiler is built with the host's
# compiler, the APK with the NDK Gradle points at. generated/game and config/game.toml are shared with the desktop build.
set -e
GAME_TITLE="Conflict - Global Terror"
GAME_XBE=default.xbe
cd "$(dirname "$0")"
ANDROID_DIR=$PWD
REPO=$(cd .. && pwd)

echo
echo "  $GAME_TITLE (Android)"
echo "  =============================="
echo

fail() {
    echo
    echo "  BUILD FAILED. $1"
    echo
    exit 1
}

# the recompiler reads the xbe itself; with only a disc image in assets it is taken off the image
if [ ! -f "$REPO/assets/$GAME_XBE" ]; then
    ISO=$(ls "$REPO"/assets/*.iso "$REPO"/assets/*.ISO 2>/dev/null | head -1 || true)
    if [ -n "$ISO" ] && command -v python3 >/dev/null; then
        echo "  Taking $GAME_XBE off $(basename "$ISO") ..."
        python3 -I "$REPO/tools/xiso_get.py" "$ISO" "$GAME_XBE" "$REPO/assets/$GAME_XBE" >/dev/null ||
            fail "Could not read $GAME_XBE from $ISO."
    fi
fi
if [ ! -f "$REPO/assets/$GAME_XBE" ]; then
    echo "  Put the game in the assets folder first:"
    echo "    $REPO/assets"
    echo "  Its disc image (.iso), or $GAME_XBE with the data folder and the mission files beside it."
    fail ""
fi

export ANDROID_HOME="${ANDROID_HOME:-${HOME}/Android/Sdk}"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME}}"
[ -f local.properties ] || echo "sdk.dir=${ANDROID_HOME}" > local.properties

# ---------------------------------------------------------------- recompiled code
echo "[1/2] Recompiled code"
host_tools() {
    for tool in cmake ninja git c++; do
        command -v $tool >/dev/null || fail "$tool is missing: install CMake, Ninja, Git and a C++ compiler."
    done
    cmake -S tools/host -B out/host -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null || fail "Read the last message above."
    cmake --build out/host || fail "Read the last message above."
}
if grep -q '^\[symbols\]' "$REPO/config/game.toml" 2>/dev/null; then
    echo "      config/game.toml already written"
else
    host_tools
    (cd "$REPO" && "$ANDROID_DIR/out/host/xbe_symbols" "assets/$GAME_XBE" sdk/xdk --library CGT.dll \
        --out config/game.toml) || fail "Read the last message above."
fi
# regenerate whenever the config or the xbe changed since the last run, as the desktop build's codegen.stamp rule does
STAMP="$REPO/generated/game/codegen.stamp"
if [ -f "$REPO/generated/game/x86_func_mapping.cpp" ] && [ -f "$STAMP" ] &&
   [ ! "$REPO/config/game.toml" -nt "$STAMP" ] && [ ! "$REPO/assets/$GAME_XBE" -nt "$STAMP" ]; then
    echo "      generated/game is up to date"
else
    echo "      recompiling $GAME_XBE: 15 to 60 minutes on weaker systems ..."
    [ -x out/host/xbe_recomp ] || host_tools
    (cd "$REPO" && "$ANDROID_DIR/out/host/xbe_recomp" config/game.toml --xdk sdk/xdk) ||
        fail "Read the last message above."
    touch "$STAMP"
fi

# ---------------------------------------------------------------- apk
echo "[2/2] APK"
./gradlew assembleRelease || fail "Read the last message above."

echo
echo "  $GAME_TITLE built."
echo "  android/app/build/outputs/apk/release/app-release.apk"
echo
