// The game on the phone: its folder on the shared storage or in the app's own, the extracted files or a disc image.
// The xbe must be the one this executable was recompiled from; another release's would run its own data under this code.
#include <algorithm>
#include <cstdlib>
#include <cwctype>
#include <fstream>

#include <jni.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_system.h>

#include "app/app.h"
#include "core/log.h"
#include "host.h"
#include "image_symbols.h"
#include "kernel/crypto.h"
#include "kernel/disc.h"

extern const char kImageSha1[];  // the xbe recompiled into this executable, written by CMake

namespace app {

extern std::filesystem::path g_game_dir;  // native/main.cpp: what host::GameDir reports

namespace {

namespace fs = std::filesystem;

// where a player's copy is looked for: a CGT folder at the top of the phone's storage and of each card, the app's
// own files folder, and the internal one
// the CGT folder on each SD card or USB stick, as the activity has it from Android's storage volumes: some phones
// don't let /storage be listed, though each card's folder opens
std::vector<fs::path> CardFolders() {
    std::vector<fs::path> out;
    JNIEnv* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
    jobject activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!env || !activity) return out;
    jclass cls = env->GetObjectClass(activity);
    jmethodID m = cls ? env->GetMethodID(cls, "cardFolderPaths", "()[Ljava/lang/String;") : nullptr;
    auto paths = m ? static_cast<jobjectArray>(env->CallObjectMethod(activity, m)) : nullptr;
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        paths = nullptr;
    }
    for (jsize i = 0; paths && i < env->GetArrayLength(paths); i++) {
        auto path = static_cast<jstring>(env->GetObjectArrayElement(paths, i));
        if (const char* c = path ? env->GetStringUTFChars(path, nullptr) : nullptr) {
            out.emplace_back(c);
            env->ReleaseStringUTFChars(path, c);
        }
        if (path) env->DeleteLocalRef(path);
    }
    if (paths) env->DeleteLocalRef(paths);
    if (cls) env->DeleteLocalRef(cls);
    env->DeleteLocalRef(activity);
    return out;
}

std::vector<fs::path> Candidates() {
    std::vector<fs::path> out = {
        "/storage/emulated/0/CGT",
        "/storage/emulated/0/Android/data/com.datfooldive.cgt/files/assets",
        "/sdcard/CGT",
    };
    std::error_code ec;
    for (auto& e : fs::directory_iterator("/storage", ec))
        if (e.is_directory(ec) && e.path().filename() != "emulated" && e.path().filename() != "self")
            out.push_back(e.path() / "CGT");
    for (fs::path& card : CardFolders()) out.push_back(std::move(card));
    out.push_back(ExeDir() / "assets");
    return out;
}

bool ReadHostFile(const fs::path& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

bool HasIso(const fs::path& dir, fs::path& iso) {
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        std::wstring ext = e.path().extension().wstring();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](wchar_t c) { return wchar_t(std::towlower(c)); });
        if (ext != L".iso" || !e.is_regular_file(ec)) continue;
        iso = e.path();
        return true;
    }
    return false;
}

// the folder is the game only when it holds more than the xbe and the images: an xbe left beside an image has none of
// the game's data with it
bool Extracted(const fs::path& dir) {
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        std::wstring ext = e.path().extension().wstring();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](wchar_t c) { return wchar_t(std::towlower(c)); });
        if (ext != L".iso" && ext != L".xbe") return true;
    }
    return false;
}

}  // namespace

std::string Sha1(const std::vector<uint8_t>& data) {
    alignas(8) uint8_t ctx[128];
    uint8_t digest[20];
    crypto::ShaInit(ctx);
    crypto::ShaUpdate(ctx, data.data(), data.size());
    crypto::ShaFinal(ctx, digest);
    std::string hex;
    for (uint8_t b : digest) {
        hex += "0123456789abcdef"[b >> 4];
        hex += "0123456789abcdef"[b & 15];
    }
    return hex;
}

namespace {
// no game yet: the activity's dialog says where it goes and offers to import it; this waits a second before looking
// again, keeping SDL's events moving, and ends the app if it is closed meanwhile
void WaitForGame() {
    for (int i = 0; i < 10; i++) {
        SDL_Event e;
        while (SDL_PollEvent(&e))
            if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_TERMINATING) std::_Exit(0);
        SDL_Delay(100);
    }
}

// another release boots its own data under this code, which usually crashes: the player decides
bool AskOtherRelease(const fs::path& name) {
    std::string text = name.filename().string() +
                       " is a different release of the game than this build was made from, and will most likely "
                       "crash.\n\nStart it anyway?";
    const SDL_MessageBoxButtonData buttons[] = {
        {SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT | SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, 0, "No"},
        {0, 1, "Yes"},
    };
    std::string title = Utf8(kGame.short_title);
    SDL_MessageBoxData box{SDL_MESSAGEBOX_WARNING, nullptr, title.c_str(), text.c_str(), 2, buttons, nullptr};
    int answer = 0;
    return SDL_ShowMessageBox(&box, &answer) && answer == 1;
}

// 1 found, 0 none there yet, -1 the player turned down another release
int Search(GameFiles& game) {
    std::vector<fs::path> dirs = Candidates();
    std::sort(dirs.begin(), dirs.end());
    dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());

    // the first copy with this xbe wins, the extracted files before any image; another release is kept to offer
    GameFiles other;
    std::unique_ptr<kernel::Disc> other_disc;
    fs::path other_name;
    auto consider = [&](const fs::path& dir, const fs::path& name, std::vector<uint8_t> xbe,
                        std::unique_ptr<kernel::Disc> disc) {
        std::string sha = Sha1(xbe);
        bool match = sha == kImageSha1;
        LOGR("Game: %s in %s, sha1 %s%s", IMAGE_XBE, name.string().c_str(), sha.c_str(), match ? "" : ", another release");
        if (match) {
            game = {dir, std::move(xbe)};
            kernel::MountDisc(std::move(disc));
            return true;
        }
        if (other_name.empty()) {
            other = {dir, std::move(xbe)};
            other_disc = std::move(disc);
            other_name = name;
        }
        return false;
    };

    for (const fs::path& dir : dirs) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec) || !Extracted(dir)) continue;
        std::vector<uint8_t> xbe;
        if (ReadHostFile(dir / IMAGE_XBE, xbe) && consider(dir, dir / IMAGE_XBE, std::move(xbe), nullptr)) {
            g_game_dir = dir;
            return 1;
        }
    }
    for (const fs::path& dir : dirs) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) continue;
        fs::path iso;
        if (!HasIso(dir, iso)) continue;
        auto disc = std::make_unique<kernel::Disc>();
        std::string error;
        std::vector<uint8_t> xbe;
        if (!disc->Open(iso, error)) {
            LOGW("game: %s: %s", iso.string().c_str(), error.c_str());
            continue;
        }
        if (!disc->Load(IMAGE_XBE, xbe)) {
            LOGW("game: %s has no %s", iso.string().c_str(), IMAGE_XBE);
            continue;
        }
        if (consider(dir, iso, std::move(xbe), std::move(disc))) {
            g_game_dir = dir;
            return 1;
        }
    }

    if (other_name.empty()) return 0;
    if (!AskOtherRelease(other_name)) return -1;
    game = std::move(other);
    kernel::MountDisc(std::move(other_disc));
    g_game_dir = game.dir;
    return 1;
}

}  // namespace

bool FindGame(GameFiles& game) {
    for (;;) {
        int found = Search(game);
        if (found) return found > 0;
        WaitForGame();
    }
}

}  // namespace app
