#include "jlc_native.h"
#include "gemm_internal.hpp"
#include "gemm_packing.hpp"
#include "gemm_panel.hpp"
#include "gemm_test_hooks.hpp"
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
#include "gemm_experimental.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#if defined(JLC_NATIVE_TEST_POOL_FAULT_INJECTION)
#include <climits>
#endif
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <fstream>
#include <new>
#include <mutex>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(JLC_NATIVE_HAS_VENDOR_BLAS)
#include <cblas.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {
struct ProfileState {
    std::atomic<bool> enabled{false};
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> wall_ns{0};
    std::atomic<std::uint64_t> vendor_calls{0};
    std::atomic<std::uint64_t> vendor_ns{0};
    std::atomic<std::uint64_t> scale_c_ns{0};
    std::atomic<std::uint64_t> pack_a_ns{0};
    std::atomic<std::uint64_t> pack_b_ns{0};
    std::atomic<std::uint64_t> kernel_ns{0};
    std::atomic<std::uint64_t> thread_launch_ns{0};
    std::atomic<std::uint64_t> thread_join_ns{0};
    std::atomic<std::uint64_t> pack_a_calls{0};
    std::atomic<std::uint64_t> pack_b_calls{0};
    std::atomic<std::uint64_t> microtile_calls{0};
    std::atomic<std::uint64_t> pack_a_bytes{0};
    std::atomic<std::uint64_t> pack_b_bytes{0};
    std::atomic<std::uint64_t> last_requested_threads{0};
    std::atomic<std::uint64_t> last_actual_threads{0};
    std::atomic<std::uint64_t> last_panel_count{0};
    std::atomic<std::uint64_t> last_mc{0};
    std::atomic<std::uint64_t> last_kc{0};
    std::atomic<std::uint64_t> last_nc{0};
    std::atomic<std::uint64_t> last_mr{0};
    std::atomic<std::uint64_t> last_nr{0};
};

struct NativeMatrix {
    int rows;
    int cols;
    int ld;
    int order;
    int alignment_bytes;
    std::uint64_t bytes;
    double* data;
};

ProfileState g_profile;

int normalize_alignment(int alignment_bytes) {
    if (alignment_bytes <= 0) {
        return 64;
    }
    int value = 1;
    while (value < alignment_bytes && value > 0) {
        value <<= 1;
    }
    return value <= 0 ? 64 : value;
}

std::size_t round_up_size(std::size_t value, std::size_t multiple) {
    if (multiple == 0) {
        return value;
    }
    const std::size_t rem = value % multiple;
    return rem == 0 ? value : value + multiple - rem;
}

void* aligned_allocate(std::size_t alignment, std::size_t size) {
    if (size == 0) {
        return nullptr;
    }
    const std::size_t adjusted = round_up_size(size, alignment);
#if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(adjusted, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, adjusted) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_release(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

inline bool profiling_enabled() {
    return g_profile.enabled.load(std::memory_order_relaxed);
}

inline void add_profile(std::atomic<std::uint64_t>& target, std::uint64_t value) {
    if (value != 0) {
        target.fetch_add(value, std::memory_order_relaxed);
    }
}

void merge_profile(const ProfileAccumulator& profile) {
    add_profile(g_profile.calls, profile.calls);
    add_profile(g_profile.wall_ns, profile.wall_ns);
    add_profile(g_profile.vendor_calls, profile.vendor_calls);
    add_profile(g_profile.vendor_ns, profile.vendor_ns);
    add_profile(g_profile.scale_c_ns, profile.scale_c_ns);
    add_profile(g_profile.pack_a_ns, profile.pack_a_ns);
    add_profile(g_profile.pack_b_ns, profile.pack_b_ns);
    add_profile(g_profile.kernel_ns, profile.kernel_ns);
    add_profile(g_profile.thread_launch_ns, profile.thread_launch_ns);
    add_profile(g_profile.thread_join_ns, profile.thread_join_ns);
    add_profile(g_profile.pack_a_calls, profile.pack_a_calls);
    add_profile(g_profile.pack_b_calls, profile.pack_b_calls);
    add_profile(g_profile.microtile_calls, profile.microtile_calls);
    add_profile(g_profile.pack_a_bytes, profile.pack_a_bytes);
    add_profile(g_profile.pack_b_bytes, profile.pack_b_bytes);
}

void record_last_profile_metadata(int requested_threads, int actual_threads, int panel_count, const ResolvedGemmPlan& blocks) {
    g_profile.last_requested_threads.store(static_cast<std::uint64_t>(requested_threads), std::memory_order_relaxed);
    g_profile.last_actual_threads.store(static_cast<std::uint64_t>(actual_threads), std::memory_order_relaxed);
    g_profile.last_panel_count.store(static_cast<std::uint64_t>(panel_count), std::memory_order_relaxed);
    g_profile.last_mc.store(static_cast<std::uint64_t>(blocks.mc), std::memory_order_relaxed);
    g_profile.last_kc.store(static_cast<std::uint64_t>(blocks.kc), std::memory_order_relaxed);
    g_profile.last_nc.store(static_cast<std::uint64_t>(blocks.nc), std::memory_order_relaxed);
    g_profile.last_mr.store(static_cast<std::uint64_t>(blocks.mr), std::memory_order_relaxed);
    g_profile.last_nr.store(static_cast<std::uint64_t>(blocks.nr), std::memory_order_relaxed);
}

constexpr const char* provider_description() {
#if defined(JLC_NATIVE_HAS_VENDOR_BLAS) && defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
    return "vendor BLAS+LAPACK available (" JLC_NATIVE_VENDOR_BLAS_NAME ")";
#elif defined(JLC_NATIVE_HAS_VENDOR_BLAS)
    return "vendor BLAS available (" JLC_NATIVE_VENDOR_BLAS_NAME ")";
#else
    return "builtin only";
#endif
}

inline bool should_try_vendor(int flags) {
#if defined(JLC_NATIVE_HAS_VENDOR_BLAS)
    if ((flags & JLC_GEMM_FLAG_FORCE_BUILTIN) != 0) {
        return false;
    }
    return (flags & (JLC_GEMM_FLAG_PREFER_VENDOR | JLC_GEMM_FLAG_FORCE_VENDOR)) != 0;
#else
    (void) flags;
    return false;
#endif
}

inline bool vendor_supported(const MatrixDescriptor& a, const MatrixDescriptor& b, const MatrixDescriptor& c) {
    return !a.col_major && !b.col_major && !c.col_major && !c.transpose;
}

bool try_vendor_gemm(const MatrixDescriptor& a, const MatrixDescriptor& b, const MatrixDescriptor& c,
                     double alpha, double beta) {
#if defined(JLC_NATIVE_HAS_VENDOR_BLAS)
    if (!vendor_supported(a, b, c)) {
        return false;
    }
    cblas_dgemm(
        CblasRowMajor,
        a.transpose ? CblasTrans : CblasNoTrans,
        b.transpose ? CblasTrans : CblasNoTrans,
        logical_rows(a),
        logical_cols(b),
        logical_cols(a),
        alpha,
        data_ptr(a), a.ld,
        data_ptr(b), b.ld,
        beta,
        mutable_data_ptr(c), c.ld
    );
    return true;
#else
    (void) a;
    (void) b;
    (void) c;
    (void) alpha;
    (void) beta;
    return false;
#endif
}

inline void scale_c(const MatrixDescriptor& c, int m, int n, double beta) {
    if (beta == 1.0) {
        return;
    }
    double* base = mutable_data_ptr(c);
    if (!c.col_major) {
        if (beta == 0.0) {
            for (int row = 0; row < m; ++row) {
                std::fill(base + row * c.ld, base + row * c.ld + n, 0.0);
            }
            return;
        }
        for (int row = 0; row < m; ++row) {
            double* row_ptr = base + row * c.ld;
            for (int col = 0; col < n; ++col) {
                row_ptr[col] *= beta;
            }
        }
        return;
    }

    if (beta == 0.0) {
        for (int col = 0; col < n; ++col) {
            std::fill(base + col * c.ld, base + col * c.ld + m, 0.0);
        }
        return;
    }
    for (int col = 0; col < n; ++col) {
        double* col_ptr = base + col * c.ld;
        for (int row = 0; row < m; ++row) {
            col_ptr[row] *= beta;
        }
    }
}

inline void add_to_c(const MatrixDescriptor& c, int row, int col, double value) {
    double* base = mutable_data_ptr(c);
    if (c.col_major) {
        base[col * c.ld + row] += value;
    } else {
        base[row * c.ld + col] += value;
    }
}

void process_generic_output(const MatrixDescriptor& a, const MatrixDescriptor& b, const MatrixDescriptor& c,
                            int m, int k, int n, double alpha) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                sum += load_element(a, row, p) * load_element(b, p, col);
            }
            add_to_c(c, row, col, alpha * sum);
        }
    }
}

void cancel_shared_job(const ThreadPoolJob& job) {
    job.failed->store(true, std::memory_order_relaxed);
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    if (job.shared_a != nullptr) job.shared_a->cancel();
#endif
    if (job.tail != nullptr) job.tail->cancel();
}

#if defined(__linux__)
int read_topology_id(int cpu, const char* name) {
    const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
        + "/topology/" + name;
    std::ifstream input(path);
    int value = -1;
    input >> value;
    return input ? value : -1;
}

std::vector<int> select_allowed_physical_cpus(int threads) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        throw std::runtime_error("sched_getaffinity failed while selecting physical CPUs");
    }

    std::vector<int> selected;
    std::set<std::pair<int, int>> seen_cores;
    for (int cpu = 0; cpu < CPU_SETSIZE && static_cast<int>(selected.size()) < threads; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        const int package = read_topology_id(cpu, "physical_package_id");
        const int core = read_topology_id(cpu, "core_id");
        if (package < 0 || core < 0) {
            continue;
        }
        if (seen_cores.emplace(package, core).second) {
            selected.push_back(cpu);
        }
    }
    if (static_cast<int>(selected.size()) != threads) {
        std::fprintf(stderr,
                     "JLC worker affinity error: requested %d workers but only %zu distinct allowed physical cores were discovered\n",
                     threads, selected.size());
        throw std::runtime_error("insufficient distinct allowed physical cores");
    }
    return selected;
}

std::string affinity_mask_string(const cpu_set_t& mask) {
    std::string result;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) {
            if (!result.empty()) {
                result += ',';
            }
            result += std::to_string(cpu);
        }
    }
    return result;
}
#endif

WorkerAffinityRecord configure_current_worker_affinity(bool physical_affinity, int intended_cpu) {
    WorkerAffinityRecord record;
    record.intended_cpu = physical_affinity ? intended_cpu : -1;
#if defined(__linux__)
    record.tid = static_cast<long>(syscall(SYS_gettid));
    if (physical_affinity) {
        cpu_set_t requested_mask;
        CPU_ZERO(&requested_mask);
        CPU_SET(intended_cpu, &requested_mask);
        record.status = pthread_setaffinity_np(pthread_self(), sizeof(requested_mask), &requested_mask);
    }
    cpu_set_t observed_mask;
    CPU_ZERO(&observed_mask);
    const int get_status = pthread_getaffinity_np(pthread_self(), sizeof(observed_mask), &observed_mask);
    if (record.status == 0 && get_status != 0) {
        record.status = get_status;
    }
    record.affinity_mask = get_status == 0 ? affinity_mask_string(observed_mask) : "unavailable";
    record.observed_cpu = sched_getcpu();
#else
    record.status = physical_affinity ? -1 : 0;
    record.affinity_mask = "unsupported";
#endif
    return record;
}

std::vector<int> select_worker_cpus(bool physical, int threads) {
    if (!physical) return {};
#if defined(__linux__)
    return select_allowed_physical_cpus(threads);
#else
    throw std::runtime_error("physical worker affinity unsupported on this platform");
#endif
}

WorkerAffinityRecord initialize_worker_affinity(bool physical, int intended_cpu) noexcept {
    try {
#if defined(JLC_NATIVE_TEST_WORKER_FAULT_INJECTION)
        const char* fault = std::getenv("JLC_NATIVE_TEST_WORKER_FAIL");
        if (fault != nullptr && std::strcmp(fault, "startup") == 0) throw std::bad_alloc();
#endif
        return configure_current_worker_affinity(physical, intended_cpu);
    } catch (...) {
        WorkerAffinityRecord failed;
        failed.status = -1;
        return failed;
    }
}

void process_column_panels(const MatrixDescriptor& a, const MatrixDescriptor& b, const MatrixDescriptor& c,
                           int m, int k, int n, double alpha,
                           const ResolvedGemmPlan& blocks,
                           int panel_start, int panel_stride,
                           Scratch& scratch, bool profile_enabled, ProfileAccumulator& profile);

void process_scheduled_panels(const ThreadPoolJob& job, int worker, Scratch& scratch,
                              ProfileAccumulator& profile);

// One task boundary for both lifetimes. Completion/join remains the caller's job.
void run_worker_task(const ThreadPoolJob& job, int index, Scratch& scratch) noexcept {
    try {
#if defined(JLC_NATIVE_TEST_WORKER_FAULT_INJECTION)
        const char* fault = std::getenv("JLC_NATIVE_TEST_WORKER_FAIL");
        if (index == 1 && fault != nullptr && std::strcmp(fault, "task") == 0) throw std::bad_alloc();
#endif
        ProfileAccumulator local_profile;
        process_scheduled_panels(job, index, scratch, local_profile);
        if (job.profile_enabled && job.profiles != nullptr)
            (*job.profiles)[static_cast<std::size_t>(index)] = local_profile;
    } catch (...) {
        cancel_shared_job(job);
    }
}

struct NativeWorkspace {
    int preferred_threads;
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    SharedAStorage shared_a_storage;
#endif
    std::mutex mutex;
    std::mutex pool_mutex;
    std::condition_variable pool_cv;
    std::condition_variable pool_done_cv;
    std::condition_variable pool_startup_cv;
    bool pool_stop = false;
    int pool_active = 0;
    std::uint64_t pool_generation = 0;
    int pool_threads = 0;
    int pool_started = 0;
    bool pool_startup_failed = false;
    bool pool_physical_affinity = false;
    ThreadPoolJob pool_job{};
    std::vector<std::thread> pool_workers;
    std::vector<Scratch> pool_scratch;
    std::vector<int> pool_selected_cpus;
    std::vector<WorkerAffinityRecord> pool_affinity_records;
};

std::atomic<NativeWorkspace*> g_default_workspace{nullptr};

NativeWorkspace* current_workspace() {
    return g_default_workspace.load(std::memory_order_acquire);
}

void pool_worker(NativeWorkspace* workspace, int worker_index) {
    const int intended_cpu = workspace->pool_physical_affinity
        ? workspace->pool_selected_cpus[static_cast<std::size_t>(worker_index)]
        : -1;
    WorkerAffinityRecord affinity_record = initialize_worker_affinity(workspace->pool_physical_affinity, intended_cpu);

    {
        std::lock_guard<std::mutex> lock(workspace->pool_mutex);
        workspace->pool_startup_failed = workspace->pool_startup_failed || affinity_record.status != 0;
        workspace->pool_affinity_records[static_cast<std::size_t>(worker_index)] = std::move(affinity_record);
        ++workspace->pool_started;
        workspace->pool_startup_cv.notify_one();
    }

    std::uint64_t observed_generation = 0;
    while (true) {
        ThreadPoolJob job;
        {
            std::unique_lock<std::mutex> lock(workspace->pool_mutex);
            workspace->pool_cv.wait(lock, [&]() {
                return workspace->pool_stop || workspace->pool_generation != observed_generation;
            });
            if (workspace->pool_stop) {
                return;
            }
            observed_generation = workspace->pool_generation;
            job = workspace->pool_job;
        }

        run_worker_task(job, worker_index, workspace->pool_scratch[static_cast<std::size_t>(worker_index)]);

        {
            std::lock_guard<std::mutex> lock(workspace->pool_mutex);
            --workspace->pool_active;
            if (workspace->pool_active == 0) {
                workspace->pool_done_cv.notify_one();
            }
        }
    }
}

void shutdown_thread_pool(NativeWorkspace* workspace) {
    if (workspace == nullptr) {
        return;
    }
#if defined(JLC_NATIVE_TEST_POOL_FAULT_INJECTION)
    const std::size_t workers_to_join = workspace->pool_workers.size();
    if (pool_test_diagnostics_requested()) {
        std::fprintf(stderr,
                     "JLC_POOL_SHUTDOWN_BEGIN requested=%d created=%zu started=%d\n",
                     workspace->pool_threads, workers_to_join, workspace->pool_started);
    }
#endif
    {
        std::lock_guard<std::mutex> lock(workspace->pool_mutex);
        workspace->pool_stop = true;
    }
    workspace->pool_cv.notify_all();
    for (std::thread& worker : workspace->pool_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workspace->pool_workers.clear();
    workspace->pool_scratch.clear();
    workspace->pool_threads = 0;
    workspace->pool_started = 0;
    workspace->pool_startup_failed = false;
    workspace->pool_physical_affinity = false;
    workspace->pool_selected_cpus.clear();
    workspace->pool_affinity_records.clear();
    workspace->pool_stop = false;
    workspace->pool_active = 0;
    workspace->pool_generation = 0;
#if defined(JLC_NATIVE_TEST_POOL_FAULT_INJECTION)
    if (pool_test_diagnostics_requested()) {
        std::fprintf(stderr, "JLC_POOL_SHUTDOWN_END joined=%zu\n", workers_to_join);
    }
#endif
}

void ensure_thread_pool(NativeWorkspace* workspace, int threads,
                        GemmWorkerAffinityMode affinity_mode,
                        bool affinity_diagnostics) {
    if (workspace == nullptr) {
        return;
    }
    if (threads <= 0) {
        shutdown_thread_pool(workspace);
        return;
    }
    const bool physical_affinity = affinity_mode == GemmWorkerAffinityMode::Physical;
    if (workspace->pool_threads == threads && !workspace->pool_workers.empty()
        && workspace->pool_physical_affinity == physical_affinity) {
        return;
    }
    shutdown_thread_pool(workspace);
    workspace->pool_threads = threads;
    workspace->pool_physical_affinity = physical_affinity;
    workspace->pool_selected_cpus = select_worker_cpus(physical_affinity, threads);
    workspace->pool_scratch.assign(static_cast<std::size_t>(threads), Scratch{});
    workspace->pool_affinity_records.assign(static_cast<std::size_t>(threads), WorkerAffinityRecord{});
    workspace->pool_workers.reserve(static_cast<std::size_t>(threads));
#if defined(JLC_NATIVE_TEST_POOL_FAULT_INJECTION)
    const int fail_at = pool_creation_failure_index();
#endif
    try {
        for (int index = 0; index < threads; ++index) {
#if defined(JLC_NATIVE_TEST_POOL_FAULT_INJECTION)
            if (index == fail_at) {
                std::fprintf(stderr,
                             "JLC_POOL_CREATE_INJECT requested=%d fail_at=%d created=%zu started=%d\n",
                             threads, fail_at, workspace->pool_workers.size(), workspace->pool_started);
                throw std::runtime_error("injected persistent-pool thread creation failure");
            }
#endif
            workspace->pool_workers.emplace_back([workspace, index]() { pool_worker(workspace, index); });
        }
    } catch (...) {
        shutdown_thread_pool(workspace);
        throw;
    }
    {
        std::unique_lock<std::mutex> lock(workspace->pool_mutex);
        workspace->pool_startup_cv.wait(lock, [&]() { return workspace->pool_started == threads; });
    }
    if (affinity_diagnostics) report_worker_affinity(workspace->pool_affinity_records);
    if (workspace->pool_startup_failed) {
        std::fprintf(stderr, "JLC worker affinity error: one or more workers failed affinity setup\n");
        shutdown_thread_pool(workspace);
        throw std::runtime_error("worker affinity setup failed");
    }
}

void process_column_panels(const MatrixDescriptor& a, const MatrixDescriptor& b, const MatrixDescriptor& c,
                           int m, int k, int n, double alpha,
                           const ResolvedGemmPlan& blocks,
                           int panel_start, int panel_stride,
                           Scratch& scratch,
                           bool profile_enabled,
                           ProfileAccumulator& profile) {
    double* c_base = mutable_data_ptr(c);
    for (int jj = panel_start; jj < n; jj += panel_stride) {
        const int n_panel = std::min(blocks.nc, n - jj);
        const int packed_n = round_up(n_panel, blocks.nr);

        for (int kk = 0; kk < k; kk += blocks.kc) {
            const int k_block = std::min(blocks.kc, k - kk);
            double* b_pack = scratch.ensure_b(static_cast<std::size_t>(k_block) * packed_n);
            {
                ScopedProfileTimer timer(profile_enabled, profile.pack_b_ns);
                pack_b_selected(blocks, b, kk, k_block, jj, n_panel, packed_n, b_pack);
            }
            if (profile_enabled) {
                ++profile.pack_b_calls;
                profile.pack_b_bytes += static_cast<std::uint64_t>(k_block) * static_cast<std::uint64_t>(packed_n) * sizeof(double);
            }

            for (int ii = 0; ii < m; ii += blocks.mc) {
                const int row_end = std::min(ii + blocks.mc, m);
                for (int i = ii; i < row_end; i += blocks.mr) {
                    const int m_block = std::min(blocks.mr, row_end - i);
                    double* a_pack = scratch.ensure_a(static_cast<std::size_t>(m_block) * k_block);
                    {
                        ScopedProfileTimer timer(profile_enabled, profile.pack_a_ns);
                        pack_a_selected(blocks, a, i, m_block, kk, k_block, alpha, a_pack);
                    }
                    if (profile_enabled) {
                        ++profile.pack_a_calls;
                        profile.pack_a_bytes += static_cast<std::uint64_t>(m_block) * static_cast<std::uint64_t>(k_block) * sizeof(double);
                    }

                    const ProfileClock::time_point kernel_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
                    int j = 0;
                    for (; j + blocks.nr <= n_panel; j += blocks.nr) {
                        double* c_tile = c_base + i * c.ld + jj + j;
                        compute_microtile(blocks, m_block, k_block, blocks.nr, packed_n,
                                          a_pack, b_micro_panel(blocks, b_pack, j, k_block), c_tile, c.ld);
                        if (profile_enabled) {
                            ++profile.microtile_calls;
                        }
                    }
                    if (j < n_panel) {
                        double* c_tile = c_base + i * c.ld + jj + j;
                        compute_microtile(blocks, m_block, k_block, n_panel - j, packed_n,
                                          a_pack, b_micro_panel(blocks, b_pack, j, k_block), c_tile, c.ld);
                        if (profile_enabled) {
                            ++profile.microtile_calls;
                        }
                    }
                    if (profile_enabled) {
                        profile.kernel_ns += elapsed_ns(kernel_start);
                    }
                }
            }
        }
    }
}


void process_shared_rows(const ThreadPoolJob& job, int jj, int kk, const double* b_pack,
                         int first_tile, int last_tile, Scratch& scratch, ProfileAccumulator& profile) {
    const int n_panel = std::min(job.blocks.nc, job.n - jj);
    const int packed_n = round_up(n_panel, job.blocks.nr);
    const int kb = std::min(job.blocks.kc, job.k - kk);
    process_shared_private_rows(*job.a, *job.c, job.m, jj, kk, kb, n_panel, packed_n,
                              job.alpha, job.blocks, b_pack, first_tile, last_tile,
                              scratch, job.profile_enabled, profile);
}

int row_tiles(int m, const ResolvedGemmPlan& blocks) {
    return (m / blocks.mc) * ((blocks.mc + blocks.mr - 1) / blocks.mr)
        + ((m % blocks.mc) + blocks.mr - 1) / blocks.mr;
}

// Convert an original MC/MR tile boundary into a row, retaining MC tail tiles.
int tile_row(int tile, int m, const ResolvedGemmPlan& blocks) {
    const int per_mc = (blocks.mc + blocks.mr - 1) / blocks.mr;
    return std::min(m, (tile / per_mc) * blocks.mc + (tile % per_mc) * blocks.mr);
}

void process_scheduled_panels(const ThreadPoolJob& job, int worker, Scratch& scratch,
                              ProfileAccumulator& profile) {
    const int threads = job.panel_stride / job.blocks.nc;
    const int tiles = row_tiles(job.m, job.blocks);
    SchedulerWork stats;
    const bool diagnostic = job.work != nullptr;
    const auto start = diagnostic ? ProfileClock::now() : ProfileClock::time_point{};
    const int full_n = job.tail == nullptr ? job.n : job.tail->full_panels * job.blocks.nc;
    const auto count_work = [&](int width, int first, int last) {
        if (!diagnostic || first == last) return;
        ++stats.tasks;
        ++stats.panels;
        const int rows = tile_row(last, job.m, job.blocks) - tile_row(first, job.m, job.blocks);
        stats.rows += rows;
        stats.microtiles += static_cast<std::uint64_t>(last - first) * ((width + job.blocks.nr - 1) / job.blocks.nr);
        stats.flops += 2ULL * rows * width * job.k;
    };
    // Shared-buffer paths must release peers waiting at a barrier after any
    // worker-side failure.
    const auto run = [&] {
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
        if (job.shared_a != nullptr) {
            process_l3_column_panels(job, worker, full_n, scratch, profile);
            if (job.failed->load(std::memory_order_relaxed)) return;
        } else
#endif
        {
            process_column_panels(*job.a, *job.b, *job.c, job.m, job.k, full_n, job.alpha,
                                  job.blocks, worker * job.blocks.nc, job.panel_stride,
                                  scratch, job.profile_enabled, profile);
        }
        if (diagnostic) {
            for (int jj = worker * job.blocks.nc; jj < full_n; jj += job.panel_stride)
                count_work(std::min(job.blocks.nc, full_n - jj), 0, tiles);
        }
        if (job.tail == nullptr) return;
        const int remainder = static_cast<int>(job.tail->panels.size());
        const int group = worker % remainder;
        const int rank = worker / remainder;
        const int members = (threads - 1 - group) / remainder + 1;
        const int first = static_cast<int>(static_cast<long long>(tiles) * rank / members);
        const int last = static_cast<int>(static_cast<long long>(tiles) * (rank + 1) / members);
        const int jj = (job.tail->full_panels + group) * job.blocks.nc;
        const int width = std::min(job.blocks.nc, job.n - jj);
        const int packed_n = round_up(width, job.blocks.nr);
        auto& shared = *job.tail->panels[group];
        count_work(width, first, last);
        const auto barrier = [&] {
            const auto wait_start = diagnostic ? ProfileClock::now() : ProfileClock::time_point{};
            const bool ok = shared.barrier(members);
            if (diagnostic) stats.wait_ns += elapsed_ns(wait_start);
            return ok;
        };
        for (int kk = 0; kk < job.k; kk += job.blocks.kc) {
            const int kb = std::min(job.blocks.kc, job.k - kk);
            if (rank == 0) {
                ScopedProfileTimer timer(job.profile_enabled, profile.pack_b_ns);
                pack_b_selected(job.blocks, *job.b, kk, kb, jj, width, packed_n, shared.packed_b.data());
                if (job.profile_enabled) {
                    ++profile.pack_b_calls;
                    profile.pack_b_bytes += static_cast<std::uint64_t>(kb) * packed_n * sizeof(double);
                }
            }
            if (!barrier()) return; // publish B before readers start
#if defined(JLC_NATIVE_TEST_A_PACKING_FAULT_INJECTION)
            const char* fault = std::getenv("JLC_NATIVE_TEST_A_FAIL");
            if (job.shared_a != nullptr && worker == 1 && kk == 0 && fault != nullptr
                && std::strcmp(fault, "tail") == 0) throw std::runtime_error("injected tail consumer failure");
#endif
            process_shared_rows(job, jj, kk, shared.packed_b.data(), first, last, scratch, profile);
            if (!barrier()) return; // all readers finish before B is overwritten
        }
    };
    run();
    if (diagnostic) {
        stats.elapsed_ns = elapsed_ns(start);
        (*job.work)[worker] = stats;
    }
}

jlc_status native_gemm_impl(const MatrixDescriptor& a, const MatrixDescriptor& b, const MatrixDescriptor& c,
                            double alpha, double beta, int threads, int flags) {
    if (a.data == nullptr || b.data == nullptr || c.mutable_data == nullptr) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (a.ld < 0 || b.ld < 0 || c.ld < 0 || a.rows < 0 || a.cols < 0 || b.rows < 0 || b.cols < 0 || c.rows < 0 || c.cols < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (c.transpose) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }

    const int m = logical_rows(a);
    const int k = logical_cols(a);
    const int b_rows = logical_rows(b);
    const int n = logical_cols(b);

    if (k != b_rows || c.rows != m || c.cols != n) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }

    const bool profile_enabled = profiling_enabled();
    ProfileAccumulator call_profile;
    const ProfileClock::time_point wall_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
    if (profile_enabled) {
        call_profile.calls = 1;
    }

    const auto finalize_profile = [&]() {
        if (profile_enabled) {
            call_profile.wall_ns = elapsed_ns(wall_start);
            merge_profile(call_profile);
        }
    };

    if (should_try_vendor(flags)) {
        const ProfileClock::time_point vendor_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
        if (try_vendor_gemm(a, b, c, alpha, beta)) {
            if (profile_enabled) {
                call_profile.vendor_calls = 1;
                call_profile.vendor_ns = elapsed_ns(vendor_start);
            }
            finalize_profile();
            return JLC_STATUS_SUCCESS;
        }
    }

    {
        ScopedProfileTimer timer(profile_enabled, call_profile.scale_c_ns);
        scale_c(c, m, n, beta);
    }
    if (alpha == 0.0 || m == 0 || k == 0 || n == 0) {
        finalize_profile();
        return JLC_STATUS_SUCCESS;
    }

#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    const bool small_k_update_enabled = small_k_update_gemm_enabled();
    if (supports_small_k_update_gemm(a, b, c, m, k, n, small_k_update_enabled)) {
        try {
            const ResolvedGemmPlan blocks = resolve_gemm_plan(m, n, k);
            NativeWorkspace* workspace = current_workspace();
            const int requested_threads = std::max(1, threads > 0 ? threads : (workspace == nullptr ? 1 : workspace->preferred_threads));
            record_last_profile_metadata(requested_threads, 1, 1, blocks);
            small_k_update_gemm(a, b, c, m, k, n, alpha, profile_enabled, call_profile);
        } catch (...) {
            finalize_profile();
            return JLC_STATUS_INTERNAL_ERROR;
        }
        finalize_profile();
        return JLC_STATUS_SUCCESS;
    }

#endif

    if (c.col_major) {
        try {
            process_generic_output(a, b, c, m, k, n, alpha);
        } catch (...) {
            finalize_profile();
            return JLC_STATUS_INTERNAL_ERROR;
        }
        finalize_profile();
        return JLC_STATUS_SUCCESS;
    }

    try {
        const ResolvedGemmPlan blocks = resolve_gemm_plan(m, n, k);
        const long long flops = 2LL * m * n * k;
        const int panel_count = std::max(1, (n + blocks.nc - 1) / blocks.nc);
        NativeWorkspace* workspace = current_workspace();
        const int requested_threads = std::max(1, threads > 0 ? threads : (workspace == nullptr ? 1 : workspace->preferred_threads));
        const int actual_threads = (flops >= PARALLEL_THRESHOLD_FLOPS)
            ? std::min(requested_threads, panel_count)
            : 1;
        record_last_profile_metadata(requested_threads, actual_threads, panel_count, blocks);

        const GemmRuntimeConfig config = resolve_gemm_runtime_config(actual_threads > 1);
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
        const bool use_l3_a = config.a_packing == GemmAPackingMode::SharedL3;
        const bool a_diagnostics = config.a_packing_diagnostics;
#endif
        if (actual_threads <= 1) {
            ProfileAccumulator local_profile;
            Scratch local_scratch;
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
            if (a_diagnostics) report_a_packing(SharedASchedule{}, "single-worker");
#endif
            process_column_panels(a, b, c, m, k, n, alpha, blocks, 0, blocks.nc,
                                  local_scratch, profile_enabled, local_profile);
            if (profile_enabled) {
                merge_profile(local_profile);
            }
            finalize_profile();
            return JLC_STATUS_SUCCESS;
        }

        std::atomic<bool> worker_failed{false};
        SharedTailSchedule tail;
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
        SharedASchedule shared_a;
        SharedAStorage local_a_storage;
#endif
        const bool use_shared_tail = config.scheduler == GemmSchedulerMode::SharedTail
            && panel_count % actual_threads != 0;
        if (use_shared_tail) {
            const int remainder = panel_count % actual_threads;
            tail.full_panels = panel_count - remainder;
            for (int r = 0; r < remainder; ++r) {
                auto panel = std::make_unique<SharedTailPanel>();
                panel->packed_b.resize(packed_elements(std::min(k, blocks.kc),
                    round_up(std::min(blocks.nc, n - (tail.full_panels + r) * blocks.nc), blocks.nr)));
                tail.panels.push_back(std::move(panel));
            }
        }
        std::vector<SchedulerWork> work(config.scheduler_diagnostics ? actual_threads : 0);
        ThreadPoolJob job;
        job.a = &a; job.b = &b; job.c = &c;
        job.m = m; job.k = k; job.n = n; job.alpha = alpha;
        job.blocks = blocks;
        job.panel_stride = actual_threads * blocks.nc;
        job.profile_enabled = profile_enabled;
        job.tail = use_shared_tail ? &tail : nullptr;
        job.work = work.empty() ? nullptr : &work;
        job.failed = &worker_failed;

        if (workspace != nullptr) {
            std::lock_guard<std::mutex> lock(workspace->mutex);
            ensure_thread_pool(workspace, actual_threads,
                               config.worker_affinity, config.worker_affinity_diagnostics);
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
            if (use_l3_a && workspace->pool_physical_affinity) {
                shared_a.prepare(workspace->shared_a_storage, workspace->pool_selected_cpus,
                    packed_elements(std::min(m, blocks.mc), std::min(k, blocks.kc)), a_diagnostics);
                job.shared_a = &shared_a;
            }
#endif
            std::vector<ProfileAccumulator> worker_profiles(static_cast<std::size_t>(actual_threads));
            job.profiles = profile_enabled ? &worker_profiles : nullptr;

            const ProfileClock::time_point launch_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
            const ProfileClock::time_point join_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
            {
                std::unique_lock<std::mutex> pool_lock(workspace->pool_mutex);
                workspace->pool_job = job;
                workspace->pool_active = actual_threads;
                ++workspace->pool_generation;
                workspace->pool_cv.notify_all();
                if (profile_enabled) {
                    call_profile.thread_launch_ns += elapsed_ns(launch_start);
                }
                workspace->pool_done_cv.wait(pool_lock, [&]() { return workspace->pool_active == 0; });
                if (profile_enabled) {
                    call_profile.thread_join_ns += elapsed_ns(join_start);
                }
            }

            if (profile_enabled) {
                for (const ProfileAccumulator& worker_profile : worker_profiles) {
                    merge_profile(worker_profile);
                }
            }
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
            if (a_diagnostics) report_a_packing(shared_a, use_l3_a ? "physical-affinity-required" : "selector-private");
#endif
        } else {
            std::vector<std::thread> workers;
            std::vector<ProfileAccumulator> worker_profiles(static_cast<std::size_t>(actual_threads));
            const bool physical_affinity = config.worker_affinity == GemmWorkerAffinityMode::Physical;
            const bool affinity_diagnostics = config.worker_affinity_diagnostics;
            const std::vector<int> selected_cpus = select_worker_cpus(physical_affinity, actual_threads);
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
            if (use_l3_a && physical_affinity) {
                shared_a.prepare(local_a_storage, selected_cpus,
                    packed_elements(std::min(m, blocks.mc), std::min(k, blocks.kc)), a_diagnostics);
                job.shared_a = &shared_a;
            }
#endif
            std::vector<WorkerAffinityRecord> affinity_records(static_cast<std::size_t>(actual_threads));
            job.profiles = profile_enabled ? &worker_profiles : nullptr;
            workers.reserve(static_cast<std::size_t>(actual_threads));
            const ProfileClock::time_point launch_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
            try {
            for (int thread_index = 0; thread_index < actual_threads; ++thread_index) {
#if defined(JLC_NATIVE_TEST_A_PACKING_FAULT_INJECTION)
                const char* fault = std::getenv("JLC_NATIVE_TEST_A_FAIL");
                if (job.shared_a != nullptr && thread_index == 2 && fault != nullptr
                    && std::strcmp(fault, "create") == 0) throw std::runtime_error("injected per-call creation failure");
#endif
#if defined(JLC_NATIVE_TEST_WORKER_FAULT_INJECTION)
                const char* worker_fault = std::getenv("JLC_NATIVE_TEST_WORKER_FAIL");
                if (thread_index == 2 && worker_fault != nullptr && std::strcmp(worker_fault, "create") == 0)
                    throw std::runtime_error("injected fallback creation failure");
#endif
                workers.emplace_back([job, physical_affinity, thread_index, &selected_cpus, &affinity_records]() {
                    try {
                        const int intended_cpu = physical_affinity ? selected_cpus[static_cast<std::size_t>(thread_index)] : -1;
                        auto record = initialize_worker_affinity(physical_affinity, intended_cpu);
#if defined(JLC_NATIVE_TEST_A_PACKING_FAULT_INJECTION)
                        const char* fault = std::getenv("JLC_NATIVE_TEST_A_FAIL");
                        if (job.shared_a != nullptr && thread_index == 1 && fault != nullptr
                            && std::strcmp(fault, "affinity") == 0) record.status = 1;
#endif
                        const bool failed = record.status != 0;
                        affinity_records[static_cast<std::size_t>(thread_index)] = std::move(record);
                        if (failed) { cancel_shared_job(job); return; }
                        Scratch thread_scratch;
                        run_worker_task(job, thread_index, thread_scratch);
                    } catch (...) { cancel_shared_job(job); }
                });
            }
            } catch (...) {
                cancel_shared_job(job);
                for (auto& worker : workers) if (worker.joinable()) worker.join();
                throw;
            }
            if (profile_enabled) {
                call_profile.thread_launch_ns += elapsed_ns(launch_start);
            }
            const ProfileClock::time_point join_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
            for (std::thread& worker : workers) {
                worker.join();
            }
            if (affinity_diagnostics) report_worker_affinity(affinity_records);
            if (profile_enabled) {
                call_profile.thread_join_ns += elapsed_ns(join_start);
                for (const ProfileAccumulator& worker_profile : worker_profiles) {
                    merge_profile(worker_profile);
                }
            }
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
            if (a_diagnostics) report_a_packing(shared_a, use_l3_a ? "physical-affinity-required" : "selector-private");
#endif
        }
        if (!work.empty()) report_scheduler_work(work, use_shared_tail);
        if (worker_failed.load(std::memory_order_relaxed)) throw std::runtime_error("GEMM worker failed");
    } catch (const std::bad_alloc&) {
        finalize_profile();
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        finalize_profile();
        return JLC_STATUS_INTERNAL_ERROR;
    }

    finalize_profile();
    return JLC_STATUS_SUCCESS;
}
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
using namespace jlc_gemm;

bool jlc_native_is_available() {
    return true;
}

const char* jlc_native_runtime_description() {
#if defined(__AVX512F__)
    return "jlc_native packed AVX-512 GEMM";
#elif defined(__AVX2__)
    return "jlc_native packed AVX2 GEMM";
#else
    return "jlc_native packed scalar GEMM";
#endif
}

const char* jlc_native_provider_description() {
    return provider_description();
}

jlc_context_handle jlc_native_context_create(int preferred_threads, int /*alignment_bytes*/, int /*flags*/) {
    auto* workspace = new (std::nothrow) NativeWorkspace();
    if (workspace == nullptr) {
        return 0;
    }
    workspace->preferred_threads = std::max(1, preferred_threads);
    NativeWorkspace* expected = nullptr;
    g_default_workspace.compare_exchange_strong(expected, workspace, std::memory_order_acq_rel);
    return reinterpret_cast<jlc_context_handle>(workspace);
}

jlc_matrix_handle jlc_native_matrix_create(int rows, int cols, int order, int alignment_bytes) {
    if (rows < 0 || cols < 0) {
        return 0;
    }
    const int normalized_alignment = normalize_alignment(alignment_bytes);
    const int ld = (order == static_cast<int>(JLC_MATRIX_COL_MAJOR)) ? rows : cols;
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
    const std::uint64_t bytes = elements * static_cast<std::uint64_t>(sizeof(double));
    double* data = static_cast<double*>(aligned_allocate(static_cast<std::size_t>(normalized_alignment),
                                                        static_cast<std::size_t>(bytes)));
    if (bytes > 0 && data == nullptr) {
        return 0;
    }
    auto* matrix = new (std::nothrow) NativeMatrix{
        rows,
        cols,
        ld,
        order,
        normalized_alignment,
        bytes,
        data
    };
    if (matrix == nullptr) {
        aligned_release(data);
        return 0;
    }
    return reinterpret_cast<jlc_matrix_handle>(matrix);
}

void jlc_native_matrix_destroy(jlc_matrix_handle handle) {
    auto* matrix = reinterpret_cast<NativeMatrix*>(handle);
    if (matrix == nullptr) {
        return;
    }
    aligned_release(matrix->data);
    delete matrix;
}

double* jlc_native_matrix_data(jlc_matrix_handle handle) {
    auto* matrix = reinterpret_cast<NativeMatrix*>(handle);
    if (matrix == nullptr) {
        return nullptr;
    }
    return matrix->data;
}

std::uint64_t jlc_native_matrix_bytes(jlc_matrix_handle handle) {
    auto* matrix = reinterpret_cast<NativeMatrix*>(handle);
    if (matrix == nullptr) {
        return 0;
    }
    return matrix->bytes;
}

void jlc_native_context_destroy(jlc_context_handle handle) {
    auto* workspace = reinterpret_cast<NativeWorkspace*>(handle);
    if (workspace == nullptr) {
        return;
    }
    if (!workspace->pool_workers.empty()) {
        shutdown_thread_pool(workspace);
    }
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    workspace->shared_a_storage = SharedAStorage{};
#endif
    NativeWorkspace* expected = workspace;
    g_default_workspace.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
}

void jlc_native_profile_set_enabled(bool enabled) {
    g_profile.enabled.store(enabled, std::memory_order_relaxed);
}

void jlc_native_profile_reset() {
    g_profile.calls.store(0, std::memory_order_relaxed);
    g_profile.wall_ns.store(0, std::memory_order_relaxed);
    g_profile.vendor_calls.store(0, std::memory_order_relaxed);
    g_profile.vendor_ns.store(0, std::memory_order_relaxed);
    g_profile.scale_c_ns.store(0, std::memory_order_relaxed);
    g_profile.pack_a_ns.store(0, std::memory_order_relaxed);
    g_profile.pack_b_ns.store(0, std::memory_order_relaxed);
    g_profile.kernel_ns.store(0, std::memory_order_relaxed);
    g_profile.thread_launch_ns.store(0, std::memory_order_relaxed);
    g_profile.thread_join_ns.store(0, std::memory_order_relaxed);
    g_profile.pack_a_calls.store(0, std::memory_order_relaxed);
    g_profile.pack_b_calls.store(0, std::memory_order_relaxed);
    g_profile.microtile_calls.store(0, std::memory_order_relaxed);
    g_profile.pack_a_bytes.store(0, std::memory_order_relaxed);
    g_profile.pack_b_bytes.store(0, std::memory_order_relaxed);
    g_profile.last_requested_threads.store(0, std::memory_order_relaxed);
    g_profile.last_actual_threads.store(0, std::memory_order_relaxed);
    g_profile.last_panel_count.store(0, std::memory_order_relaxed);
    g_profile.last_mc.store(0, std::memory_order_relaxed);
    g_profile.last_kc.store(0, std::memory_order_relaxed);
    g_profile.last_nc.store(0, std::memory_order_relaxed);
    g_profile.last_mr.store(0, std::memory_order_relaxed);
    g_profile.last_nr.store(0, std::memory_order_relaxed);
}

void jlc_native_profile_snapshot(jlc_gemm_profile* out_profile) {
    if (out_profile == nullptr) {
        return;
    }
    out_profile->calls = g_profile.calls.load(std::memory_order_relaxed);
    out_profile->wall_ns = g_profile.wall_ns.load(std::memory_order_relaxed);
    out_profile->vendor_calls = g_profile.vendor_calls.load(std::memory_order_relaxed);
    out_profile->vendor_ns = g_profile.vendor_ns.load(std::memory_order_relaxed);
    out_profile->scale_c_ns = g_profile.scale_c_ns.load(std::memory_order_relaxed);
    out_profile->pack_a_ns = g_profile.pack_a_ns.load(std::memory_order_relaxed);
    out_profile->pack_b_ns = g_profile.pack_b_ns.load(std::memory_order_relaxed);
    out_profile->kernel_ns = g_profile.kernel_ns.load(std::memory_order_relaxed);
    out_profile->thread_launch_ns = g_profile.thread_launch_ns.load(std::memory_order_relaxed);
    out_profile->thread_join_ns = g_profile.thread_join_ns.load(std::memory_order_relaxed);
    out_profile->pack_a_calls = g_profile.pack_a_calls.load(std::memory_order_relaxed);
    out_profile->pack_b_calls = g_profile.pack_b_calls.load(std::memory_order_relaxed);
    out_profile->microtile_calls = g_profile.microtile_calls.load(std::memory_order_relaxed);
    out_profile->pack_a_bytes = g_profile.pack_a_bytes.load(std::memory_order_relaxed);
    out_profile->pack_b_bytes = g_profile.pack_b_bytes.load(std::memory_order_relaxed);
    out_profile->last_requested_threads = g_profile.last_requested_threads.load(std::memory_order_relaxed);
    out_profile->last_actual_threads = g_profile.last_actual_threads.load(std::memory_order_relaxed);
    out_profile->last_panel_count = g_profile.last_panel_count.load(std::memory_order_relaxed);
    out_profile->last_mc = g_profile.last_mc.load(std::memory_order_relaxed);
    out_profile->last_kc = g_profile.last_kc.load(std::memory_order_relaxed);
    out_profile->last_nc = g_profile.last_nc.load(std::memory_order_relaxed);
    out_profile->last_mr = g_profile.last_mr.load(std::memory_order_relaxed);
    out_profile->last_nr = g_profile.last_nr.load(std::memory_order_relaxed);
}

jlc_status jlc_native_gemm(const double* a, int a_rows, int a_cols,
                           const double* b, int b_rows, int b_cols,
                           double* c, int c_rows, int c_cols,
                           double alpha, double beta,
                           int threads, int flags) {
    MatrixDescriptor a_desc{a, nullptr, 0, a_cols, a_rows, a_cols, false, false};
    MatrixDescriptor b_desc{b, nullptr, 0, b_cols, b_rows, b_cols, false, false};
    MatrixDescriptor c_desc{c, c, 0, c_cols, c_rows, c_cols, false, false};
    return native_gemm_impl(a_desc, b_desc, c_desc, alpha, beta, threads, flags);
}

jlc_status jlc_native_gemm_strided(const double* a, int a_offset, int a_ld, int a_rows, int a_cols, int a_flags,
                                   const double* b, int b_offset, int b_ld, int b_rows, int b_cols, int b_flags,
                                   double* c, int c_offset, int c_ld, int c_rows, int c_cols, int c_flags,
                                   double alpha, double beta,
                                   int threads, int flags) {
    MatrixDescriptor a_desc{
        a, nullptr, a_offset, a_ld, a_rows, a_cols,
        (a_flags & JLC_GEMM_FLAG_A_COL_MAJOR) != 0,
        (a_flags & JLC_GEMM_FLAG_A_TRANSPOSE) != 0
    };
    MatrixDescriptor b_desc{
        b, nullptr, b_offset, b_ld, b_rows, b_cols,
        (b_flags & JLC_GEMM_FLAG_B_COL_MAJOR) != 0,
        (b_flags & JLC_GEMM_FLAG_B_TRANSPOSE) != 0
    };
    MatrixDescriptor c_desc{
        c, c, c_offset, c_ld, c_rows, c_cols,
        (c_flags & JLC_GEMM_FLAG_C_COL_MAJOR) != 0,
        false
    };
    return native_gemm_impl(a_desc, b_desc, c_desc, alpha, beta, threads, flags);
}

jlc_status jlc_native_gemm_strided_batched(const double* a, int a_offset, int a_ld, int a_rows, int a_cols, int a_flags, int a_stride,
                                           const double* b, int b_offset, int b_ld, int b_rows, int b_cols, int b_flags, int b_stride,
                                           double* c, int c_offset, int c_ld, int c_rows, int c_cols, int c_flags, int c_stride,
                                           double alpha, double beta,
                                           int batch_count,
                                           int threads, int flags) {
    if (batch_count < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    for (int batch = 0; batch < batch_count; ++batch) {
        const int a_batch_offset = a_offset + batch * a_stride;
        const int b_batch_offset = b_offset + batch * b_stride;
        const int c_batch_offset = c_offset + batch * c_stride;
        MatrixDescriptor a_desc{
            a, nullptr, a_batch_offset, a_ld, a_rows, a_cols,
            (a_flags & JLC_GEMM_FLAG_A_COL_MAJOR) != 0,
            (a_flags & JLC_GEMM_FLAG_A_TRANSPOSE) != 0
        };
        MatrixDescriptor b_desc{
            b, nullptr, b_batch_offset, b_ld, b_rows, b_cols,
            (b_flags & JLC_GEMM_FLAG_B_COL_MAJOR) != 0,
            (b_flags & JLC_GEMM_FLAG_B_TRANSPOSE) != 0
        };
        MatrixDescriptor c_desc{
            c, c, c_batch_offset, c_ld, c_rows, c_cols,
            (c_flags & JLC_GEMM_FLAG_C_COL_MAJOR) != 0,
            false
        };
        const jlc_status status = native_gemm_impl(a_desc, b_desc, c_desc, alpha, beta, threads, flags);
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
    }
    return JLC_STATUS_SUCCESS;
}
