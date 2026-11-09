#pragma once
#include <cstdlib>
#include <filesystem>

// diagnostics switched on by environment variables, read once
namespace diag {

// XBE_PERF set: perf log lines once a second and the read-backs named
inline bool Perf() {
    static const bool on = std::getenv("XBE_PERF") != nullptr;
    return on;
}

// XBE_CAPTURE_DIR: where F3 saves a frame, "captures" beside the executable when unset
inline const std::filesystem::path& CaptureDir() {
    static const std::filesystem::path dir = [] {
#ifdef _WIN32
        const wchar_t* v = _wgetenv(L"XBE_CAPTURE_DIR");
#else
        const char* v = std::getenv("XBE_CAPTURE_DIR");
#endif
        return std::filesystem::path(v && *v ? v : "captures");
    }();
    return dir;
}

}  // namespace diag
