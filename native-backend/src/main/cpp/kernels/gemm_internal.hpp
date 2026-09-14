#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {

constexpr long long PARALLEL_THRESHOLD_FLOPS = 5'000'000LL;
using ProfileClock = std::chrono::steady_clock;

enum class GemmKernelOption {
    Kernel5x4,
    Kernel5x8,
    Kernel6x8,
    Kernel6x8NoInline,
    Kernel6x8U2,
    Kernel6x8U4,
    Kernel6x8U4Asm,
};

struct ResolvedGemmPlan {
    int mc;
    int kc;
    int nc;
    int mr;
    int nr;
    enum class Tile { Scalar, M5N4, M5N8, M6N8 } tile = Tile::Scalar;
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    enum class Kernel { Default, NoInline, U2, U4, KMajor6, KMajor6Asm } kernel = Kernel::Default;
    enum class BLayout { Panel, NR8Old, NR8Fast } b_layout = BLayout::Panel;
    int nr8_prefetch = 0;
    bool kmajor6() const { return kernel == Kernel::KMajor6 || kernel == Kernel::KMajor6Asm; }
    bool nr8() const { return b_layout != BLayout::Panel; }
#endif
};

enum class GemmSchedulerMode {
    Panel,
    SharedTail,
};

enum class GemmAPackingMode {
    Private,
    SharedL3,
};

enum class GemmWorkerAffinityMode {
    None,
    Physical,
};

// Per-call control snapshot. Production defaults are panel scheduling,
// private A packing, no worker pinning, and the intrinsic 6x8 kernel. The
// other values are explicit diagnostics or reproducibility controls.
struct GemmRuntimeConfig {
    GemmSchedulerMode scheduler = GemmSchedulerMode::Panel;
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    GemmAPackingMode a_packing = GemmAPackingMode::Private;
    bool a_packing_diagnostics = false;
#endif
    GemmWorkerAffinityMode worker_affinity = GemmWorkerAffinityMode::None;
    bool scheduler_diagnostics = false;
    bool worker_affinity_diagnostics = false;
};

struct MatrixDescriptor {
    const double* data;
    double* mutable_data;
    int offset;
    int ld;
    int rows;
    int cols;
    bool col_major;
    bool transpose;
};

struct ProfileAccumulator {
    std::uint64_t calls = 0;
    std::uint64_t wall_ns = 0;
    std::uint64_t vendor_calls = 0;
    std::uint64_t vendor_ns = 0;
    std::uint64_t scale_c_ns = 0;
    std::uint64_t pack_a_ns = 0;
    std::uint64_t pack_b_ns = 0;
    std::uint64_t kernel_ns = 0;
    std::uint64_t thread_launch_ns = 0;
    std::uint64_t thread_join_ns = 0;
    std::uint64_t pack_a_calls = 0;
    std::uint64_t pack_b_calls = 0;
    std::uint64_t microtile_calls = 0;
    std::uint64_t pack_a_bytes = 0;
    std::uint64_t pack_b_bytes = 0;
};

inline std::uint64_t elapsed_ns(ProfileClock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(ProfileClock::now() - start).count()
    );
}

struct ScopedProfileTimer {
    std::uint64_t* target;
    ProfileClock::time_point start;

    ScopedProfileTimer(bool enabled, std::uint64_t& destination)
        : target(enabled ? &destination : nullptr) {
        if (target != nullptr) {
            start = ProfileClock::now();
        }
    }

    ~ScopedProfileTimer() {
        if (target != nullptr) {
            *target += elapsed_ns(start);
        }
    }
};

inline int round_up(int value, int multiple) {
    if (multiple <= 0) {
        return value;
    }
    const int rem = value % multiple;
    return rem == 0 ? value : value + multiple - rem;
}

inline int logical_rows(const MatrixDescriptor& desc) {
    return desc.transpose ? desc.cols : desc.rows;
}

inline int logical_cols(const MatrixDescriptor& desc) {
    return desc.transpose ? desc.rows : desc.cols;
}

inline const double* data_ptr(const MatrixDescriptor& desc) {
    return desc.data + desc.offset;
}

inline double* mutable_data_ptr(const MatrixDescriptor& desc) {
    return desc.mutable_data + desc.offset;
}

inline double load_element(const MatrixDescriptor& desc, int row, int col) {
    const double* base = data_ptr(desc);
    if (!desc.transpose) {
        return desc.col_major
            ? base[col * desc.ld + row]
            : base[row * desc.ld + col];
    }
    return desc.col_major
        ? base[row * desc.ld + col]
        : base[col * desc.ld + row];
}

inline std::size_t packed_elements(int rows, int columns) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
}

inline double* ensure_elements(std::vector<double>& buffer, std::size_t elements) {
    if (buffer.size() < elements) buffer.resize(elements);
    return buffer.data();
}

struct Scratch {
    std::vector<double> a_pack;
    std::vector<double> b_pack;

    double* ensure_a(std::size_t size) { return ensure_elements(a_pack, size); }
    double* ensure_b(std::size_t size) { return ensure_elements(b_pack, size); }
};


// Only leftover panels are shared. Each buffer remains KC*packed_N doubles.
struct CancellableBarrier {
    std::mutex mutex;
    std::condition_variable cv;
    int arrived = 0;
    unsigned generation = 0;
    bool cancelled = false;

    bool barrier(int participants, std::uint64_t* events = nullptr) {
        std::unique_lock<std::mutex> lock(mutex);
        if (cancelled) return false;
        const unsigned observed = generation;
        if (++arrived == participants) {
            arrived = 0;
            ++generation;
            if (events != nullptr) ++*events;
            cv.notify_all();
        } else {
            cv.wait(lock, [&] { return cancelled || generation != observed; });
        }
        return !cancelled;
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled = true;
        cv.notify_all();
    }
};

struct SharedTailPanel : CancellableBarrier {
    std::vector<double> packed_b;
};

struct SchedulerWork {
    std::uint64_t tasks = 0, rows = 0, microtiles = 0, panels = 0, flops = 0;
    std::uint64_t elapsed_ns = 0, wait_ns = 0;
};

struct SharedTailSchedule {
    int full_panels = 0;
    std::vector<std::unique_ptr<SharedTailPanel>> panels;
    void cancel() {
        for (auto& panel : panels) panel->cancel();
    }
};

struct SharedASchedule;

struct ThreadPoolJob {
    const MatrixDescriptor* a = nullptr;
    const MatrixDescriptor* b = nullptr;
    const MatrixDescriptor* c = nullptr;
    int m = 0;
    int k = 0;
    int n = 0;
    double alpha = 0.0;
    ResolvedGemmPlan blocks{};
    int panel_stride = 0;
    SharedTailSchedule* tail = nullptr;
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    SharedASchedule* shared_a = nullptr;
#endif
    std::vector<SchedulerWork>* work = nullptr;
    bool profile_enabled = false;
    std::vector<ProfileAccumulator>* profiles = nullptr;
    std::atomic<bool>* failed = nullptr;
};

struct WorkerAffinityRecord {
    long tid = -1;
    int intended_cpu = -1;
    int observed_cpu = -1;
    int status = 0;
    std::string affinity_mask;
};

ResolvedGemmPlan resolve_gemm_plan(int m, int n, int k);
GemmRuntimeConfig resolve_gemm_runtime_config(bool parallel);
bool parse_env_bool(const char*, bool);
void report_scheduler_work(const std::vector<SchedulerWork>&, bool);
void report_worker_affinity(const std::vector<WorkerAffinityRecord>&);
void process_column_panels(const MatrixDescriptor&, const MatrixDescriptor&, const MatrixDescriptor&,
                           int, int, int, double, const ResolvedGemmPlan&, int, int, Scratch&, bool, ProfileAccumulator&);
} // namespace jlc_gemm
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
