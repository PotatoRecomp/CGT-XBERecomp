// A sampling profiler for phones whose kernel keeps perf events from the shell (simpleperf finds no event to count).
// `adb shell setprop debug.cgt.prof 1` before the game starts: every thread of the process gets a timer on its own CPU
// time, so a thread that waits is never sampled and one that works is sampled each kIntervalNs of its work. The signal
// takes the interrupted pc and the frame pointer chain's return addresses; a writer thread appends them to
// cgt_prof.bin in the game's folder every two seconds, with the threads' names in cgt_prof.txt. Offline,
// tools/prof_report.py symbolizes them against the unstripped libmain.so, where the recompiled code's functions are
// named by their guest address (sub_XXXXXXXX)
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include <dirent.h>
#include <dlfcn.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/system_properties.h>
#include <ucontext.h>
#include <unistd.h>

#include "core/log.h"

namespace prof {

namespace {

constexpr long kIntervalNs = 2'000'000;  // 2 ms of a thread's CPU time
constexpr int kDepth = 14;               // the pc and 13 return addresses
constexpr uint32_t kRecords = 250'000;   // 28 MB, about 8 minutes of one busy thread

struct Record {
    uint32_t tid;
    uint32_t depth;
    uint64_t pc[kDepth];
};

Record* g_ring = nullptr;
std::atomic<uint32_t> g_next{0};       // claimed by the handler
std::atomic<uint32_t> g_done[kRecords / 1024 + 1];  // per block of 1024: records finished, for the writer
std::atomic<bool> g_on{false};
std::string g_dir;
std::mutex g_dir_lock;

// async signal safe: atomics and plain stores only
void OnSample(int, siginfo_t*, void* uc_void) {
    if (!g_on.load(std::memory_order_relaxed)) return;
    uint32_t i = g_next.fetch_add(1, std::memory_order_relaxed);
    if (i >= kRecords) return;
    Record& r = g_ring[i];
    auto* uc = static_cast<ucontext_t*>(uc_void);
    r.tid = uint32_t(syscall(SYS_gettid));
    r.pc[0] = uc->uc_mcontext.pc;
    uint32_t n = 1;
    // a leaf without a frame record of its own leaves its caller in lr
    r.pc[n++] = uc->uc_mcontext.regs[30];
    // frame records: [fp] the caller's fp, [fp + 8] the return address. Followed while they climb the stack within
    // reach of sp, which bounds every read to the thread's own mapped stack
    uint64_t sp = uc->uc_mcontext.sp;
    uint64_t fp = uc->uc_mcontext.regs[29];
    while (n < kDepth && fp >= sp && fp - sp < (8ull << 20) && (fp & 15) == 0) {
        const uint64_t* rec = reinterpret_cast<const uint64_t*>(fp);
        uint64_t next = rec[0], ret = rec[1];
        if (!ret) break;
        r.pc[n++] = ret;
        if (next <= fp) break;
        fp = next;
    }
    r.depth = n;
    g_done[i / 1024].fetch_add(1, std::memory_order_release);
}

std::string ThreadName(int tid) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
    char name[64] = "";
    if (FILE* f = std::fopen(path, "r")) {
        if (std::fgets(name, sizeof(name), f)) name[std::strcspn(name, "\n")] = 0;
        std::fclose(f);
    }
    return name;
}

// a timer on the thread's CPU clock that signals that thread; false when the thread is gone
bool Watch(int tid) {
    clockid_t clock = clockid_t((~unsigned(tid) << 3) | 6);  // the kernel's MAKE_THREAD_CPUCLOCK(tid, SCHED)
    sigevent ev{};
    ev.sigev_notify = SIGEV_THREAD_ID;
    ev.sigev_signo = SIGPROF;
    ev.sigev_notify_thread_id = tid;
    timer_t timer;
    if (timer_create(clock, &ev, &timer) != 0) return false;
    itimerspec spec{};
    spec.it_interval.tv_nsec = kIntervalNs;
    spec.it_value.tv_nsec = kIntervalNs;
    timer_settime(timer, 0, &spec, nullptr);
    return true;
}

// new threads watched, every 2 s the finished records appended and the names written out
void Run() {
    std::set<int> watched;
    std::string names;
    uint32_t written = 0;
    uint64_t base = 0;
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&OnSample), &info)) base = reinterpret_cast<uint64_t>(info.dli_fbase);
    for (;;) {
        if (DIR* d = opendir("/proc/self/task")) {
            while (dirent* e = readdir(d)) {
                int tid = std::atoi(e->d_name);
                if (tid <= 0 || watched.count(tid)) continue;
                if (Watch(tid)) {
                    watched.insert(tid);
                    names += std::to_string(tid) + " " + ThreadName(tid) + "\n";
                }
            }
            closedir(d);
        }
        std::string dir;
        {
            std::lock_guard lock(g_dir_lock);
            dir = g_dir;
        }
        // records claimed but not yet finished are left for the next pass: whole blocks of 1024 only
        uint32_t claimed = std::min(g_next.load(std::memory_order_relaxed), kRecords);
        uint32_t ready = written;
        while (ready / 1024 < claimed / 1024 && g_done[ready / 1024].load(std::memory_order_acquire) == 1024)
            ready = (ready / 1024 + 1) * 1024;
        if (claimed == kRecords && ready / 1024 == kRecords / 1024 && g_done[kRecords / 1024].load() == kRecords % 1024)
            ready = kRecords;
        if (!dir.empty() && ready > written) {
            std::filesystem::path bin = std::filesystem::path(dir) / "cgt_prof.bin";
            if (FILE* f = std::fopen(bin.c_str(), written ? "ab" : "wb")) {
                if (!written) {
                    // header: magic, the library's load address, the record layout
                    uint64_t head[4] = {0x31464f5250544743ull, base, kDepth, sizeof(Record)};
                    std::fwrite(head, sizeof(head), 1, f);
                }
                std::fwrite(g_ring + written, sizeof(Record), ready - written, f);
                std::fclose(f);
                written = ready;
            }
            if (FILE* f = std::fopen((std::filesystem::path(dir) / "cgt_prof.txt").c_str(), "w")) {
                std::fputs(names.c_str(), f);
                // the code each sample outside libmain.so may be in: "map <start> <end> <path>", the shell cannot read
                // an app's maps itself
                if (FILE* m = std::fopen("/proc/self/maps", "r")) {
                    char line[512];
                    while (std::fgets(line, sizeof(line), m)) {
                        unsigned long long lo, hi;
                        char perms[8], path[384] = "";
                        if (std::sscanf(line, "%llx-%llx %7s %*s %*s %*s %383s", &lo, &hi, perms, path) >= 3 &&
                            perms[2] == 'x' && path[0])
                            std::fprintf(f, "map %llx %llx %s\n", lo, hi, path);
                    }
                    std::fclose(m);
                }
                std::fclose(f);
            }
        }
        if (written == kRecords) {
            LOGR("prof: %u samples written, the ring is full", kRecords);
            g_on = false;
            return;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

}  // namespace

void StartIfAsked() {
    char v[PROP_VALUE_MAX] = "";
    if (__system_property_get("debug.cgt.prof", v) <= 0 || v[0] == '0') return;
    g_ring = new (std::nothrow) Record[kRecords];
    if (!g_ring) return;
    struct sigaction sa{};
    sa.sa_sigaction = OnSample;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, nullptr);
    g_on = true;
    std::thread(Run).detach();
    LOGR("prof: sampling every thread each %ld us of its CPU time", kIntervalNs / 1000);
}

void SetDir(const std::string& dir) {
    std::lock_guard lock(g_dir_lock);
    g_dir = dir;
}

}  // namespace prof
