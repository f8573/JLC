#include "jlc_native.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <new>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(JLC_NATIVE_HAS_VENDOR_BLAS)
#include <cblas.h>
#endif

namespace {
constexpr int L1_CACHE = 32 * 1024;
constexpr int L2_CACHE = 256 * 1024;
constexpr int L3_CACHE = 8 * 1024 * 1024;
constexpr long long PARALLEL_THRESHOLD_FLOPS = 5'000'000LL;
using ProfileClock = std::chrono::steady_clock;

struct BlockSizes {
    int mc;
    int kc;
    int nc;
    int mr;
    int nr;
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

void record_last_profile_metadata(int requested_threads, int actual_threads, int panel_count, const BlockSizes& blocks) {
    g_profile.last_requested_threads.store(static_cast<std::uint64_t>(requested_threads), std::memory_order_relaxed);
    g_profile.last_actual_threads.store(static_cast<std::uint64_t>(actual_threads), std::memory_order_relaxed);
    g_profile.last_panel_count.store(static_cast<std::uint64_t>(panel_count), std::memory_order_relaxed);
    g_profile.last_mc.store(static_cast<std::uint64_t>(blocks.mc), std::memory_order_relaxed);
    g_profile.last_kc.store(static_cast<std::uint64_t>(blocks.kc), std::memory_order_relaxed);
    g_profile.last_nc.store(static_cast<std::uint64_t>(blocks.nc), std::memory_order_relaxed);
    g_profile.last_mr.store(static_cast<std::uint64_t>(blocks.mr), std::memory_order_relaxed);
    g_profile.last_nr.store(static_cast<std::uint64_t>(blocks.nr), std::memory_order_relaxed);
}

inline int round_up(int value, int multiple) {
    if (multiple <= 0) {
        return value;
    }
    const int rem = value % multiple;
    return rem == 0 ? value : value + multiple - rem;
}

int parse_env_positive_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > 1'000'000L) {
        return fallback;
    }
    return static_cast<int>(parsed);
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

BlockSizes compute_block_sizes(int m, int n, int k) {
    int nr = 1;
    int mr = 4;
#if defined(__AVX512F__)
    nr = 8;
    mr = 6;
#elif defined(__AVX2__)
    nr = 4;
    mr = 5;
#endif

    const int l1_constraint = std::max(64, L1_CACHE / (8 * mr * 2));
    const int l2_constraint = static_cast<int>(std::sqrt(L2_CACHE / (8.0 * 2.0)));

    int kc = round_up(std::min(l1_constraint, l2_constraint), 8);
    kc = std::clamp(kc, 64, 512);

    int nc = round_up((L2_CACHE / 2) / (kc * 8), nr);
    nc = std::clamp(nc, nr * 4, 4096);

    int mc = round_up(L3_CACHE / (8 * std::max(1, kc + nc)), mr);
    mc = std::clamp(mc, mr * 4, 2048);

#if defined(__AVX2__) && !defined(__AVX512F__)
    const int max_dim = std::max(m, std::max(n, k));
    if (max_dim >= 4096) {
        kc = std::max(kc, 384);
    } else if (max_dim >= 2048) {
        kc = std::max(kc, 256);
    }
#endif

    const int mr_override = parse_env_positive_int("JLC_NATIVE_MR", 0);
    const int nr_override = parse_env_positive_int("JLC_NATIVE_NR", 0);
    const int kc_override = parse_env_positive_int("JLC_NATIVE_KC", 0);
    const int nc_override = parse_env_positive_int("JLC_NATIVE_NC", 0);
    const int mc_override = parse_env_positive_int("JLC_NATIVE_MC", 0);

    if (mr_override > 0) {
        mr = mr_override;
    }
    if (nr_override > 0) {
        nr = nr_override;
    }
    if (kc_override > 0) {
        kc = round_up(kc_override, 4);
    }
    if (nc_override > 0) {
        nc = round_up(nc_override, std::max(1, nr));
    }
    if (mc_override > 0) {
        mc = round_up(mc_override, std::max(1, mr));
    }
    mr = std::clamp(mr, 1, 64);
    nr = std::clamp(nr, 1, 64);
    kc = std::clamp(kc, 8, 4096);
    nc = std::clamp(nc, nr, 4096);
    mc = std::clamp(mc, mr, 4096);

    return BlockSizes{mc, kc, nc, mr, nr};
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

inline void pack_a(const MatrixDescriptor& a, int row_start, int rows,
                   int col_start, int k_block, double alpha, double* a_pack) {
    int dst = 0;
    const double* base = data_ptr(a);
    if (!a.transpose && !a.col_major) {
        if (alpha == 1.0) {
            for (int r = 0; r < rows; ++r) {
                const int src = (row_start + r) * a.ld + col_start;
                std::memcpy(a_pack + dst, base + src, sizeof(double) * static_cast<std::size_t>(k_block));
                dst += k_block;
            }
        } else {
            for (int r = 0; r < rows; ++r) {
                const int src = (row_start + r) * a.ld + col_start;
                for (int p = 0; p < k_block; ++p) {
                    a_pack[dst++] = base[src + p] * alpha;
                }
            }
        }
        return;
    }

    for (int r = 0; r < rows; ++r) {
        for (int p = 0; p < k_block; ++p) {
            a_pack[dst++] = load_element(a, row_start + r, col_start + p) * alpha;
        }
    }
}

inline void pack_b(const MatrixDescriptor& b, int row_start, int k_block,
                   int col_start, int cols, int packed_cols, double* b_pack) {
    int dst = 0;
    const double* base = data_ptr(b);
    if (!b.transpose && !b.col_major) {
        for (int p = 0; p < k_block; ++p) {
            const int src = (row_start + p) * b.ld + col_start;
            std::memcpy(b_pack + dst, base + src, sizeof(double) * static_cast<std::size_t>(cols));
            if (packed_cols > cols) {
                std::fill(b_pack + dst + cols, b_pack + dst + packed_cols, 0.0);
            }
            dst += packed_cols;
        }
        return;
    }

    for (int p = 0; p < k_block; ++p) {
        for (int col = 0; col < cols; ++col) {
            b_pack[dst + col] = load_element(b, row_start + p, col_start + col);
        }
        if (packed_cols > cols) {
            std::fill(b_pack + dst + cols, b_pack + dst + packed_cols, 0.0);
        }
        dst += packed_cols;
    }
}

inline void microkernel_scalar(int mr, int k_block, int n_block, int packed_n,
                               const double* a_pack, const double* b_pack,
                               double* c, int ldc) {
    for (int r = 0; r < mr; ++r) {
        const double* a_row = a_pack + r * k_block;
        double* c_row = c + r * ldc;
        for (int col = 0; col < n_block; ++col) {
            double accum = 0.0;
            for (int p = 0; p < k_block; ++p) {
                accum += a_row[p] * b_pack[p * packed_n + col];
            }
            c_row[col] += accum;
        }
    }
}

#if defined(__AVX2__)
inline void microkernel_5x4_avx2(int k_block, const double* a_pack,
                                 const double* b_pack, int packed_n,
                                 double* c, int ldc) {
    __m256d c0 = _mm256_loadu_pd(c);
    __m256d c1 = _mm256_loadu_pd(c + ldc);
    __m256d c2 = _mm256_loadu_pd(c + 2 * ldc);
    __m256d c3 = _mm256_loadu_pd(c + 3 * ldc);
    __m256d c4 = _mm256_loadu_pd(c + 4 * ldc);

    const int off1 = k_block;
    const int off2 = 2 * k_block;
    const int off3 = 3 * k_block;
    const int off4 = 4 * k_block;

    for (int p = 0; p < k_block; ++p) {
        const __m256d b = _mm256_loadu_pd(b_pack + p * packed_n);
        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(a_pack + p), b, c0);
        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(a_pack + off1 + p), b, c1);
        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(a_pack + off2 + p), b, c2);
        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(a_pack + off3 + p), b, c3);
        c4 = _mm256_fmadd_pd(_mm256_broadcast_sd(a_pack + off4 + p), b, c4);
    }

    _mm256_storeu_pd(c, c0);
    _mm256_storeu_pd(c + ldc, c1);
    _mm256_storeu_pd(c + 2 * ldc, c2);
    _mm256_storeu_pd(c + 3 * ldc, c3);
    _mm256_storeu_pd(c + 4 * ldc, c4);
}
#endif

#if defined(__AVX512F__)
inline void microkernel_6x8_avx512(int k_block, const double* a_pack,
                                   const double* b_pack, int packed_n,
                                   double* c, int ldc) {
    __m512d c0 = _mm512_loadu_pd(c);
    __m512d c1 = _mm512_loadu_pd(c + ldc);
    __m512d c2 = _mm512_loadu_pd(c + 2 * ldc);
    __m512d c3 = _mm512_loadu_pd(c + 3 * ldc);
    __m512d c4 = _mm512_loadu_pd(c + 4 * ldc);
    __m512d c5 = _mm512_loadu_pd(c + 5 * ldc);

    const int off1 = k_block;
    const int off2 = 2 * k_block;
    const int off3 = 3 * k_block;
    const int off4 = 4 * k_block;
    const int off5 = 5 * k_block;

    for (int p = 0; p < k_block; ++p) {
        const __m512d b = _mm512_loadu_pd(b_pack + p * packed_n);
        c0 = _mm512_fmadd_pd(_mm512_set1_pd(a_pack[p]), b, c0);
        c1 = _mm512_fmadd_pd(_mm512_set1_pd(a_pack[off1 + p]), b, c1);
        c2 = _mm512_fmadd_pd(_mm512_set1_pd(a_pack[off2 + p]), b, c2);
        c3 = _mm512_fmadd_pd(_mm512_set1_pd(a_pack[off3 + p]), b, c3);
        c4 = _mm512_fmadd_pd(_mm512_set1_pd(a_pack[off4 + p]), b, c4);
        c5 = _mm512_fmadd_pd(_mm512_set1_pd(a_pack[off5 + p]), b, c5);
    }

    _mm512_storeu_pd(c, c0);
    _mm512_storeu_pd(c + ldc, c1);
    _mm512_storeu_pd(c + 2 * ldc, c2);
    _mm512_storeu_pd(c + 3 * ldc, c3);
    _mm512_storeu_pd(c + 4 * ldc, c4);
    _mm512_storeu_pd(c + 5 * ldc, c5);
}
#endif

inline void compute_microtile(const BlockSizes& blocks,
                              int m_block, int k_block, int n_block, int packed_n,
                              const double* a_pack, const double* b_pack,
                              double* c, int ldc) {
#if defined(__AVX512F__)
    if (blocks.nr == 8 && blocks.mr == 6 && m_block == 6 && n_block == 8) {
        microkernel_6x8_avx512(k_block, a_pack, b_pack, packed_n, c, ldc);
        return;
    }
#endif
#if defined(__AVX2__)
    if (blocks.nr == 4 && blocks.mr == 5 && m_block == 5 && n_block == 4) {
        microkernel_5x4_avx2(k_block, a_pack, b_pack, packed_n, c, ldc);
        return;
    }
#endif
    microkernel_scalar(m_block, k_block, n_block, packed_n, a_pack, b_pack, c, ldc);
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

struct Scratch {
    std::vector<double> a_pack;
    std::vector<double> b_pack;

    double* ensure_a(std::size_t size) {
        if (a_pack.size() < size) {
            a_pack.resize(size);
        }
        return a_pack.data();
    }

    double* ensure_b(std::size_t size) {
        if (b_pack.size() < size) {
            b_pack.resize(size);
        }
        return b_pack.data();
    }
};

struct ThreadPoolJob {
    const MatrixDescriptor* a = nullptr;
    const MatrixDescriptor* b = nullptr;
    const MatrixDescriptor* c = nullptr;
    int m = 0;
    int k = 0;
    int n = 0;
    double alpha = 0.0;
    BlockSizes blocks{};
    int panel_stride = 0;
    bool profile_enabled = false;
    std::vector<ProfileAccumulator>* profiles = nullptr;
};

struct NativeWorkspace {
    int preferred_threads;
    int alignment_bytes;
    Scratch scratch;
    std::mutex mutex;
    std::mutex pool_mutex;
    std::condition_variable pool_cv;
    std::condition_variable pool_done_cv;
    bool pool_stop = false;
    bool pool_has_job = false;
    int pool_active = 0;
    int pool_threads = 0;
    ThreadPoolJob pool_job{};
    std::vector<std::thread> pool_workers;
    std::vector<Scratch> pool_scratch;
};

std::mutex g_workspace_guard;
NativeWorkspace* g_default_workspace = nullptr;

NativeWorkspace* current_workspace() {
    std::lock_guard<std::mutex> lock(g_workspace_guard);
    return g_default_workspace;
}

void pool_worker(NativeWorkspace* workspace, int worker_index) {
    while (true) {
        ThreadPoolJob job;
        {
            std::unique_lock<std::mutex> lock(workspace->pool_mutex);
            workspace->pool_cv.wait(lock, [&]() { return workspace->pool_has_job || workspace->pool_stop; });
            if (workspace->pool_stop) {
                return;
            }
            job = workspace->pool_job;
        }

        ProfileAccumulator local_profile;
        Scratch& scratch = workspace->pool_scratch[static_cast<std::size_t>(worker_index)];
        process_column_panels(
            *job.a, *job.b, *job.c,
            job.m, job.k, job.n, job.alpha,
            job.blocks,
            worker_index * job.blocks.nc,
            job.panel_stride,
            scratch,
            job.profile_enabled,
            local_profile
        );
        if (job.profile_enabled && job.profiles != nullptr) {
            (*job.profiles)[static_cast<std::size_t>(worker_index)] = local_profile;
        }

        {
            std::lock_guard<std::mutex> lock(workspace->pool_mutex);
            --workspace->pool_active;
            if (workspace->pool_active == 0) {
                workspace->pool_has_job = false;
                workspace->pool_done_cv.notify_one();
            }
        }
    }
}

void shutdown_thread_pool(NativeWorkspace* workspace) {
    if (workspace == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(workspace->pool_mutex);
        workspace->pool_stop = true;
        workspace->pool_has_job = true;
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
    workspace->pool_stop = false;
    workspace->pool_has_job = false;
    workspace->pool_active = 0;
}

void ensure_thread_pool(NativeWorkspace* workspace, int threads) {
    if (workspace == nullptr) {
        return;
    }
    if (threads <= 0) {
        shutdown_thread_pool(workspace);
        return;
    }
    if (workspace->pool_threads == threads && !workspace->pool_workers.empty()) {
        return;
    }
    shutdown_thread_pool(workspace);
    workspace->pool_threads = threads;
    workspace->pool_scratch.assign(static_cast<std::size_t>(threads), Scratch{});
    workspace->pool_workers.reserve(static_cast<std::size_t>(threads));
    for (int index = 0; index < threads; ++index) {
        workspace->pool_workers.emplace_back([workspace, index]() { pool_worker(workspace, index); });
    }
}

void process_column_panels(const MatrixDescriptor& a, const MatrixDescriptor& b, const MatrixDescriptor& c,
                           int m, int k, int n, double alpha,
                           const BlockSizes& blocks,
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
                pack_b(b, kk, k_block, jj, n_panel, packed_n, b_pack);
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
                        pack_a(a, i, m_block, kk, k_block, alpha, a_pack);
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
                                          a_pack, b_pack + j, c_tile, c.ld);
                        if (profile_enabled) {
                            ++profile.microtile_calls;
                        }
                    }
                    if (j < n_panel) {
                        double* c_tile = c_base + i * c.ld + jj + j;
                        compute_microtile(blocks, m_block, k_block, n_panel - j, packed_n,
                                          a_pack, b_pack + j, c_tile, c.ld);
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
        const BlockSizes blocks = compute_block_sizes(m, n, k);
        const long long flops = 2LL * m * n * k;
        const int panel_count = std::max(1, (n + blocks.nc - 1) / blocks.nc);
        NativeWorkspace* workspace = current_workspace();
        const int requested_threads = std::max(1, threads > 0 ? threads : (workspace == nullptr ? 1 : workspace->preferred_threads));
        const int actual_threads = (flops >= PARALLEL_THRESHOLD_FLOPS)
            ? std::min(requested_threads, panel_count)
            : 1;
        record_last_profile_metadata(requested_threads, actual_threads, panel_count, blocks);

        if (actual_threads <= 1) {
            ProfileAccumulator local_profile;
            Scratch local_scratch;
            if (workspace == nullptr) {
                process_column_panels(a, b, c, m, k, n, alpha, blocks, 0, blocks.nc,
                                      local_scratch, profile_enabled, local_profile);
            } else {
                std::lock_guard<std::mutex> lock(workspace->mutex);
                process_column_panels(a, b, c, m, k, n, alpha, blocks, 0, blocks.nc,
                                      workspace->scratch, profile_enabled, local_profile);
            }
            if (profile_enabled) {
                merge_profile(local_profile);
            }
            finalize_profile();
            return JLC_STATUS_SUCCESS;
        }

        if (workspace != nullptr) {
            std::lock_guard<std::mutex> lock(workspace->mutex);
            ensure_thread_pool(workspace, actual_threads);
            std::vector<ProfileAccumulator> worker_profiles(static_cast<std::size_t>(actual_threads));
            ThreadPoolJob job;
            job.a = &a;
            job.b = &b;
            job.c = &c;
            job.m = m;
            job.k = k;
            job.n = n;
            job.alpha = alpha;
            job.blocks = blocks;
            job.panel_stride = actual_threads * blocks.nc;
            job.profile_enabled = profile_enabled;
            job.profiles = profile_enabled ? &worker_profiles : nullptr;

            const ProfileClock::time_point launch_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
            const ProfileClock::time_point join_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
            {
                std::unique_lock<std::mutex> pool_lock(workspace->pool_mutex);
                workspace->pool_job = job;
                workspace->pool_active = actual_threads;
                workspace->pool_has_job = true;
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
        } else {
            std::vector<std::thread> workers;
            std::vector<ProfileAccumulator> worker_profiles(static_cast<std::size_t>(actual_threads));
            workers.reserve(static_cast<std::size_t>(actual_threads));
            const ProfileClock::time_point launch_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
            for (int thread_index = 0; thread_index < actual_threads; ++thread_index) {
                workers.emplace_back([=, &a, &b, &c, &blocks, &worker_profiles]() {
                    Scratch thread_scratch;
                    ProfileAccumulator local_profile;
                    process_column_panels(
                        a, b, c, m, k, n, alpha, blocks,
                        thread_index * blocks.nc, actual_threads * blocks.nc,
                        thread_scratch,
                        profile_enabled,
                        local_profile
                    );
                    if (profile_enabled) {
                        worker_profiles[static_cast<std::size_t>(thread_index)] = local_profile;
                    }
                });
            }
            if (profile_enabled) {
                call_profile.thread_launch_ns += elapsed_ns(launch_start);
            }
            const ProfileClock::time_point join_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
            for (std::thread& worker : workers) {
                worker.join();
            }
            if (profile_enabled) {
                call_profile.thread_join_ns += elapsed_ns(join_start);
                for (const ProfileAccumulator& worker_profile : worker_profiles) {
                    merge_profile(worker_profile);
                }
            }
        }
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

jlc_context_handle jlc_native_context_create(int preferred_threads, int alignment_bytes, int /*flags*/) {
    auto* workspace = new (std::nothrow) NativeWorkspace{
        std::max(1, preferred_threads),
        std::max(8, alignment_bytes),
        {},
        {}
    };
    if (workspace == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_workspace_guard);
    if (g_default_workspace == nullptr) {
        g_default_workspace = workspace;
    }
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
    shutdown_thread_pool(workspace);
    {
        std::lock_guard<std::mutex> lock(g_workspace_guard);
        if (g_default_workspace == workspace) {
            g_default_workspace = nullptr;
        }
    }
    delete workspace;
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
