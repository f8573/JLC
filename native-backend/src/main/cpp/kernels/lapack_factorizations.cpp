#include "jlc_native.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
extern "C" {
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgeqrf_(const int* m, const int* n, double* a, const int* lda, double* tau,
             double* work, const int* lwork, int* info);
void dorgqr_(const int* m, const int* n, const int* k, double* a, const int* lda, const double* tau,
             double* work, const int* lwork, int* info);
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
}
#endif

namespace {
using QrProfileClock = std::chrono::steady_clock;

constexpr int DEFAULT_NATIVE_LU_BLOCK_THRESHOLD = 256;
constexpr int DEFAULT_NATIVE_LU_BLOCK_SIZE = 64;
constexpr int DEFAULT_NATIVE_QR_BLOCK_THRESHOLD = 128;
constexpr int DEFAULT_NATIVE_QR_BLOCK_SIZE = 48;
constexpr int DEFAULT_NATIVE_QR_TRAILING_BLOCK = 96;
constexpr int DEFAULT_NATIVE_QR_Q_BLOCK = 96;
constexpr int DEFAULT_NATIVE_QR_COPY_TILE = 16;
constexpr int DEFAULT_NATIVE_QR_DIRECT_PARALLEL_THRESHOLD_FLOPS = 25'000'000;
constexpr int MAX_STACK_QR_PANEL_SIZE = 128;
constexpr int DEFAULT_NATIVE_CHOLESKY_BLOCK_THRESHOLD = 128;
constexpr int DEFAULT_NATIVE_CHOLESKY_BLOCK_SIZE = 64;

struct QrProfileCounters {
    std::atomic<bool> enabled{false};
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> wall_ns{0};
    std::atomic<std::uint64_t> factorize_ns{0};
    std::atomic<std::uint64_t> input_transpose_ns{0};
    std::atomic<std::uint64_t> panel_ns{0};
    std::atomic<std::uint64_t> reflector_pack_ns{0};
    std::atomic<std::uint64_t> t_build_ns{0};
    std::atomic<std::uint64_t> trailing_pack_ns{0};
    std::atomic<std::uint64_t> trailing_unpack_ns{0};
    std::atomic<std::uint64_t> trailing_gemm_ns{0};
    std::atomic<std::uint64_t> trailing_t_apply_ns{0};
    std::atomic<std::uint64_t> r_extract_ns{0};
    std::atomic<std::uint64_t> q_init_ns{0};
    std::atomic<std::uint64_t> q_build_ns{0};
    std::atomic<std::uint64_t> q_gemm_ns{0};
    std::atomic<std::uint64_t> q_t_apply_ns{0};
};

QrProfileCounters g_qr_profile;
std::atomic<int> g_qr_block_size_override{0};
std::atomic<int> g_qr_gemm_threads_override{1};

inline std::uint64_t qr_elapsed_ns(QrProfileClock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(QrProfileClock::now() - start).count()
    );
}

inline void qr_profile_add(std::atomic<std::uint64_t>& counter, QrProfileClock::time_point start) {
    if (g_qr_profile.enabled.load(std::memory_order_relaxed)) {
        counter.fetch_add(qr_elapsed_ns(start), std::memory_order_relaxed);
    }
}

inline QrProfileClock::time_point qr_profile_start() {
    return g_qr_profile.enabled.load(std::memory_order_relaxed)
        ? QrProfileClock::now()
        : QrProfileClock::time_point{};
}

inline void transpose_row_to_col(const double* src, double* dst, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const int row_offset = row * cols;
        for (int col = 0; col < cols; ++col) {
            dst[col * rows + row] = src[row_offset + col];
        }
    }
}

inline void transpose_col_to_row(const double* src, double* dst, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const int row_offset = row * cols;
        for (int col = 0; col < cols; ++col) {
            dst[row_offset + col] = src[col * rows + row];
        }
    }
}

inline void extract_r_from_col_major(const double* factor, int m, int n, int r_rows, double* r_row) {
    for (int col = 0; col < n; ++col) {
        const int src_base = col * m;
        const int row_limit = std::min(col + 1, r_rows);
        for (int row = 0; row < row_limit; ++row) {
            r_row[row * n + col] = factor[src_base + row];
        }
        for (int row = row_limit; row < r_rows; ++row) {
            r_row[row * n + col] = 0.0;
        }
    }
}

inline int parse_env_positive_int(const char* name, int fallback) {
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

inline bool env_is_set(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0';
}

inline int native_lu_block_threshold() {
    return parse_env_positive_int("JLC_NATIVE_LU_BLOCK_THRESHOLD", DEFAULT_NATIVE_LU_BLOCK_THRESHOLD);
}

inline int native_lu_block_size() {
    return parse_env_positive_int("JLC_NATIVE_LU_BLOCK_SIZE", DEFAULT_NATIVE_LU_BLOCK_SIZE);
}

inline int native_qr_block_threshold() {
    if (g_qr_block_size_override.load(std::memory_order_relaxed) > 0) {
        return 1;
    }
    return parse_env_positive_int("JLC_NATIVE_QR_BLOCK_THRESHOLD", DEFAULT_NATIVE_QR_BLOCK_THRESHOLD);
}

inline int native_qr_block_size() {
    const int override_value = g_qr_block_size_override.load(std::memory_order_relaxed);
    if (override_value > 0) {
        return override_value;
    }
    return parse_env_positive_int("JLC_NATIVE_QR_BLOCK_SIZE", DEFAULT_NATIVE_QR_BLOCK_SIZE);
}

inline int native_qr_block_size_for_shape(int m, int n, int k) {
    const int override_value = g_qr_block_size_override.load(std::memory_order_relaxed);
    if (override_value > 0) {
        return override_value;
    }
    if (env_is_set("JLC_NATIVE_QR_BLOCK_SIZE")) {
        return parse_env_positive_int("JLC_NATIVE_QR_BLOCK_SIZE", DEFAULT_NATIVE_QR_BLOCK_SIZE);
    }
    const int max_dim = std::max(m, n);
    return max_dim >= 1536 ? std::min(96, std::max(1, k)) : std::min(48, std::max(1, k));
}

inline int native_qr_trailing_block() {
    return parse_env_positive_int("JLC_NATIVE_QR_TRAILING_BLOCK", DEFAULT_NATIVE_QR_TRAILING_BLOCK);
}

inline int native_qr_q_block() {
    return parse_env_positive_int("JLC_NATIVE_QR_Q_BLOCK", DEFAULT_NATIVE_QR_Q_BLOCK);
}

inline int native_qr_copy_tile() {
    static const int value = parse_env_positive_int("JLC_NATIVE_QR_COPY_TILE", DEFAULT_NATIVE_QR_COPY_TILE);
    return value;
}

inline bool parse_env_bool(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    if (value[0] == '0' || value[0] == 'f' || value[0] == 'F'
        || value[0] == 'n' || value[0] == 'N') {
        return false;
    }
    return value[0] == '1' || value[0] == 't' || value[0] == 'T'
        || value[0] == 'y' || value[0] == 'Y';
}

inline bool native_qr_use_t_transpose_for_trailing() {
    return parse_env_bool("JLC_NATIVE_QR_USE_T_TRANSPOSE_FOR_TRAILING", true);
}

inline bool native_qr_use_direct_trailing_update() {
    return parse_env_bool("JLC_NATIVE_QR_DIRECT_TRAILING_UPDATE", false);
}

inline int native_qr_direct_update_threads() {
    const unsigned int hardware = std::max(1u, std::thread::hardware_concurrency());
    return parse_env_positive_int("JLC_NATIVE_QR_DIRECT_UPDATE_THREADS", static_cast<int>(hardware));
}

inline int native_qr_gemm_threads() {
    return std::max(1, g_qr_gemm_threads_override.load(std::memory_order_relaxed));
}

inline int native_qr_direct_parallel_threshold_flops() {
    return parse_env_positive_int(
        "JLC_NATIVE_QR_DIRECT_PARALLEL_THRESHOLD_FLOPS",
        DEFAULT_NATIVE_QR_DIRECT_PARALLEL_THRESHOLD_FLOPS
    );
}

inline int native_cholesky_block_threshold() {
    return parse_env_positive_int("JLC_NATIVE_CHOLESKY_BLOCK_THRESHOLD", DEFAULT_NATIVE_CHOLESKY_BLOCK_THRESHOLD);
}

inline int native_cholesky_block_size() {
    return parse_env_positive_int("JLC_NATIVE_CHOLESKY_BLOCK_SIZE", DEFAULT_NATIVE_CHOLESKY_BLOCK_SIZE);
}

inline void swap_rows(double* data, int ld, int row_a, int row_b, int col_start, int col_end) {
    if (row_a == row_b || col_start >= col_end) {
        return;
    }
    const int offset_a = row_a * ld;
    const int offset_b = row_b * ld;
    for (int col = col_start; col < col_end; ++col) {
        std::swap(data[offset_a + col], data[offset_b + col]);
    }
}

inline void subtract_scaled_row_segment(double* data, int ld, int row, int previous_row,
                                        int col_start, int col_end, double factor) {
    if (factor == 0.0 || col_start >= col_end) {
        return;
    }
    const int row_offset = row * ld;
    const int previous_offset = previous_row * ld;
    for (int col = col_start; col < col_end; ++col) {
        data[row_offset + col] = std::fma(-factor, data[previous_offset + col], data[row_offset + col]);
    }
}

void solve_upper_panel(double* lu, int n, int panel_start, int panel_end) {
    for (int row = panel_start + 1; row < panel_end; ++row) {
        for (int previous = panel_start; previous < row; ++previous) {
            const double factor = lu[row * n + previous];
            subtract_scaled_row_segment(lu, n, row, previous, panel_end, n, factor);
        }
    }
}

int select_pivot_row(const double* lu, int n, int column) {
    int pivot_row = column;
    double pivot_abs = std::abs(lu[column * n + column]);
    for (int row = column + 1; row < n; ++row) {
        const double candidate_abs = std::abs(lu[row * n + column]);
        if (candidate_abs > pivot_abs) {
            pivot_abs = candidate_abs;
            pivot_row = row;
        }
    }
    return pivot_row;
}

void factor_cholesky_panel(double* a, int n, int block_start, int panel_width, int* out_info) {
    const int block_end = block_start + panel_width;
    for (int col = block_start; col < block_end; ++col) {
        double diag = a[col * n + col];
        for (int k = block_start; k < col; ++k) {
            const double value = a[col * n + k];
            diag = std::fma(-value, value, diag);
        }
        if (!(diag > 0.0) || std::isnan(diag)) {
            if (*out_info == 0) {
                *out_info = col + 1;
            }
            return;
        }

        const double diag_sqrt = std::sqrt(diag);
        a[col * n + col] = diag_sqrt;
        for (int row = col + 1; row < n; ++row) {
            double value = a[row * n + col];
            for (int k = block_start; k < col; ++k) {
                value = std::fma(-a[row * n + k], a[col * n + k], value);
            }
            a[row * n + col] = value / diag_sqrt;
        }
    }
}

double dot_contiguous(const double* lhs, const double* rhs, int length) {
    double sum = 0.0;
    for (int index = 0; index < length; ++index) {
        sum = std::fma(lhs[index], rhs[index], sum);
    }
    return sum;
}

void compute_householder_column(double* a_col, int m, int k, double* tau) {
    const int col_base = k * m;
    const int tail_length = m - k;
    if (tail_length <= 1) {
        tau[k] = 0.0;
        return;
    }

    double scale = 0.0;
    for (int row = k; row < m; ++row) {
        scale = std::max(scale, std::abs(a_col[col_base + row]));
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        tau[k] = 0.0;
        return;
    }

    const double alpha = a_col[col_base + k];
    const double inv_scale = 1.0 / scale;
    const double scaled_alpha = alpha * inv_scale;

    double sigma = 0.0;
    for (int row = k + 1; row < m; ++row) {
        const double value = a_col[col_base + row] * inv_scale;
        sigma = std::fma(value, value, sigma);
    }

    if (sigma == 0.0) {
        if (alpha >= 0.0) {
            tau[k] = 0.0;
            return;
        }
        tau[k] = 2.0;
        a_col[col_base + k] = -alpha;
        return;
    }

    const double beta_scaled = -std::copysign(std::hypot(scaled_alpha, std::sqrt(sigma)), scaled_alpha);
    const double beta = beta_scaled * scale;
    tau[k] = (beta - alpha) / beta;

    const double inv_v0 = 1.0 / (alpha - beta);
    for (int row = k + 1; row < m; ++row) {
        a_col[col_base + row] *= inv_v0;
    }
    a_col[col_base + k] = beta;
}

void apply_householder_to_column(double* a_col, int m, int reflector_col, int target_col, double tau_k) {
    if (tau_k == 0.0) {
        return;
    }
    const int reflector_base = reflector_col * m;
    const int target_base = target_col * m;

    double dot = a_col[target_base + reflector_col];
    for (int row = reflector_col + 1; row < m; ++row) {
        dot = std::fma(a_col[reflector_base + row], a_col[target_base + row], dot);
    }
    dot *= tau_k;

    a_col[target_base + reflector_col] -= dot;
    for (int row = reflector_col + 1; row < m; ++row) {
        a_col[target_base + row] = std::fma(-dot, a_col[reflector_base + row], a_col[target_base + row]);
    }
}

void factorize_qr_panel(double* a_col, double* tau, int m, int k_start, int k_end) {
    for (int k = k_start; k < k_end; ++k) {
        compute_householder_column(a_col, m, k, tau);
        for (int col = k + 1; col < k_end; ++col) {
            apply_householder_to_column(a_col, m, k, col, tau[k]);
        }
    }
}

void pack_qr_reflectors(const double* a_col, int m, int k_start, int panel_size_actual,
                        double* vt_pack) {
    const int active_rows = m - k_start;
    for (int panel_col = 0; panel_col < panel_size_actual; ++panel_col) {
        const int reflector_col = k_start + panel_col;
        const int reflector_base = reflector_col * m;
        for (int local_row = 0; local_row < active_rows; ++local_row) {
            double value = 0.0;
            if (local_row == panel_col) {
                value = 1.0;
            } else if (local_row > panel_col) {
                value = a_col[reflector_base + k_start + local_row];
            }
            vt_pack[panel_col * active_rows + local_row] = value;
        }
    }
}

void pack_col_major_block_to_row_major(const double* a_col, int m, int row_start, int col_start,
                                       int rows, int cols, int stride, double* packed) {
    const int tile = native_qr_copy_tile();
    for (int row_block = 0; row_block < rows; row_block += tile) {
        const int row_limit = std::min(row_block + tile, rows);
        for (int col_block = 0; col_block < cols; col_block += tile) {
            const int col_limit = std::min(col_block + tile, cols);
            for (int row = row_block; row < row_limit; ++row) {
                const int packed_base = row * stride;
                for (int col = col_block; col < col_limit; ++col) {
                    packed[packed_base + col] = a_col[(col_start + col) * m + row_start + row];
                }
            }
        }
    }
}

void unpack_row_major_block_to_col_major(const double* packed, int stride, int rows, int cols,
                                         double* a_col, int m, int row_start, int col_start) {
    const int tile = native_qr_copy_tile();
    for (int row_block = 0; row_block < rows; row_block += tile) {
        const int row_limit = std::min(row_block + tile, rows);
        for (int col_block = 0; col_block < cols; col_block += tile) {
            const int col_limit = std::min(col_block + tile, cols);
            for (int row = row_block; row < row_limit; ++row) {
                const int packed_base = row * stride;
                for (int col = col_block; col < col_limit; ++col) {
                    a_col[(col_start + col) * m + row_start + row] = packed[packed_base + col];
                }
            }
        }
    }
}

void build_qr_block_t(const double* vt_pack, const double* tau, int k_start, int panel_size_actual,
                      int active_rows, double* t, double* work, double* column) {
    std::fill(t, t + static_cast<std::size_t>(panel_size_actual) * static_cast<std::size_t>(panel_size_actual), 0.0);
    for (int panel_col = 0; panel_col < panel_size_actual; ++panel_col) {
        const double tau_k = tau[k_start + panel_col];
        if (tau_k == 0.0) {
            t[panel_col * panel_size_actual + panel_col] = 0.0;
            continue;
        }

        const int tail_length = active_rows - panel_col;
        const double* current = vt_pack + panel_col * active_rows + panel_col;
        for (int prev = 0; prev < panel_col; ++prev) {
            const double* previous = vt_pack + prev * active_rows + panel_col;
            work[prev] = -tau_k * dot_contiguous(previous, current, tail_length);
        }

        for (int prev = 0; prev < panel_col; ++prev) {
            double sum = 0.0;
            for (int mid = prev; mid < panel_col; ++mid) {
                sum = std::fma(t[prev * panel_size_actual + mid], work[mid], sum);
            }
            column[prev] = sum;
        }

        for (int prev = 0; prev < panel_col; ++prev) {
            t[prev * panel_size_actual + panel_col] = column[prev];
        }
        t[panel_col * panel_size_actual + panel_col] = tau_k;
    }
}

void apply_qr_t(const double* t, double* w, int panel_size_actual, int block_cols, int stride) {
    for (int row = 0; row < panel_size_actual; ++row) {
        double* target_row = w + row * stride;
        const double diag = t[row * panel_size_actual + row];
        for (int col = 0; col < block_cols; ++col) {
            target_row[col] *= diag;
        }

        for (int k = row + 1; k < panel_size_actual; ++k) {
            const double coeff = t[row * panel_size_actual + k];
            if (coeff == 0.0) {
                continue;
            }
            const double* source_row = w + k * stride;
            for (int col = 0; col < block_cols; ++col) {
                target_row[col] = std::fma(coeff, source_row[col], target_row[col]);
            }
        }
    }
}

void apply_qr_t_transpose(const double* t, double* w, int panel_size_actual, int block_cols, int stride) {
    for (int row = panel_size_actual - 1; row >= 0; --row) {
        double* target_row = w + row * stride;
        const double diag = t[row * panel_size_actual + row];
        for (int col = 0; col < block_cols; ++col) {
            target_row[col] *= diag;
        }

        for (int k = 0; k < row; ++k) {
            const double coeff = t[k * panel_size_actual + row];
            if (coeff == 0.0) {
                continue;
            }
            const double* source_row = w + k * stride;
            for (int col = 0; col < block_cols; ++col) {
                target_row[col] = std::fma(coeff, source_row[col], target_row[col]);
            }
        }
    }
}

void apply_qr_block_t(const double* t, double* w, int panel_size_actual, int block_cols, int stride) {
    if (native_qr_use_t_transpose_for_trailing()) {
        apply_qr_t_transpose(t, w, panel_size_actual, block_cols, stride);
    } else {
        apply_qr_t(t, w, panel_size_actual, block_cols, stride);
    }
}

struct QrWorkspace {
    std::vector<double> a_col;
    std::vector<double> tau;
    std::vector<double> vt_pack;
    std::vector<double> t;
    std::vector<double> w;
    std::vector<double> c_block;
    std::vector<double> work;
    std::vector<double> column;

    void prepare(int m, int n, int max_cols, int panel_size_max, int k) {
        a_col.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        tau.resize(static_cast<std::size_t>(std::max(1, k)));
        vt_pack.resize(static_cast<std::size_t>(panel_size_max) * static_cast<std::size_t>(m));
        t.resize(static_cast<std::size_t>(panel_size_max) * static_cast<std::size_t>(panel_size_max));
        w.resize(static_cast<std::size_t>(panel_size_max) * static_cast<std::size_t>(max_cols));
        c_block.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(max_cols));
        work.resize(static_cast<std::size_t>(panel_size_max));
        column.resize(static_cast<std::size_t>(panel_size_max));
    }
};

QrWorkspace& qr_workspace() {
    thread_local QrWorkspace workspace;
    return workspace;
}

void qr_compute_w_vtc_range(const double* vt_pack, int panel_size_actual, int active_rows,
                            const double* a_col, int m, int row_start, int col_start,
                            int local_col_begin, int local_col_end, int stride, double* w) {
    if (panel_size_actual <= MAX_STACK_QR_PANEL_SIZE) {
        double sums[MAX_STACK_QR_PANEL_SIZE];
        for (int local_col = local_col_begin; local_col < local_col_end; ++local_col) {
            std::fill(sums, sums + panel_size_actual, 0.0);
            const double* c = a_col + (col_start + local_col) * m + row_start;
            for (int row = 0; row < active_rows; ++row) {
                const double c_value = c[row];
                for (int panel_row = 0; panel_row < panel_size_actual; ++panel_row) {
                    sums[panel_row] = std::fma(vt_pack[panel_row * active_rows + row], c_value, sums[panel_row]);
                }
            }
            for (int panel_row = 0; panel_row < panel_size_actual; ++panel_row) {
                w[panel_row * stride + local_col] = sums[panel_row];
            }
        }
        return;
    }

    for (int panel_row = 0; panel_row < panel_size_actual; ++panel_row) {
        const double* v = vt_pack + panel_row * active_rows;
        double* w_row = w + panel_row * stride;
        for (int local_col = local_col_begin; local_col < local_col_end; ++local_col) {
            const double* c = a_col + (col_start + local_col) * m + row_start;
            double sum = 0.0;
            for (int row = 0; row < active_rows; ++row) {
                sum = std::fma(v[row], c[row], sum);
            }
            w_row[local_col] = sum;
        }
    }
}

void qr_update_c_vw_range(const double* vt_pack, int panel_size_actual, int active_rows,
                          const double* w, int stride, double* a_col, int m, int row_start,
                          int col_start, int local_col_begin, int local_col_end) {
    for (int local_col = local_col_begin; local_col < local_col_end; ++local_col) {
        double* c = a_col + (col_start + local_col) * m + row_start;
        for (int panel_row = 0; panel_row < panel_size_actual; ++panel_row) {
            const double w_value = w[panel_row * stride + local_col];
            if (w_value == 0.0) {
                continue;
            }
            const double* v = vt_pack + panel_row * active_rows;
            for (int row = 0; row < active_rows; ++row) {
                c[row] = std::fma(-v[row], w_value, c[row]);
            }
        }
    }
}

template <typename Work>
void run_qr_column_parallel(int block_cols, int worker_count, Work&& work) {
    if (worker_count <= 1 || block_cols <= 1) {
        work(0, block_cols);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_count - 1));
    for (int worker = 1; worker < worker_count; ++worker) {
        const int begin = (block_cols * worker) / worker_count;
        const int end = (block_cols * (worker + 1)) / worker_count;
        workers.emplace_back([&, begin, end]() {
            work(begin, end);
        });
    }
    work(0, block_cols / worker_count);
    for (std::thread& worker : workers) {
        worker.join();
    }
}

jlc_status apply_qr_block_to_col_major_direct(const double* vt_pack, const double* t,
                                              int panel_size_actual, int active_rows, int max_block_cols,
                                              double* w, double* a_col, int m, int row_start,
                                              int trailing_col_start, int trailing_cols) {
    const long long block_flops = 2LL * static_cast<long long>(active_rows)
        * static_cast<long long>(panel_size_actual)
        * static_cast<long long>(max_block_cols);
    const int max_workers = block_flops >= native_qr_direct_parallel_threshold_flops()
        ? std::max(1, native_qr_direct_update_threads())
        : 1;

    for (int col_start = 0; col_start < trailing_cols; col_start += max_block_cols) {
        const int block_cols = std::min(max_block_cols, trailing_cols - col_start);
        const int worker_count = std::min(max_workers, block_cols);
        const int absolute_col_start = trailing_col_start + col_start;

        const auto vtc_start = qr_profile_start();
        run_qr_column_parallel(block_cols, worker_count, [&](int begin, int end) {
            qr_compute_w_vtc_range(
                vt_pack, panel_size_actual, active_rows,
                a_col, m, row_start, absolute_col_start,
                begin, end, max_block_cols, w
            );
        });
        qr_profile_add(g_qr_profile.trailing_gemm_ns, vtc_start);

        const auto t_apply_start = qr_profile_start();
        apply_qr_block_t(t, w, panel_size_actual, block_cols, max_block_cols);
        qr_profile_add(g_qr_profile.trailing_t_apply_ns, t_apply_start);

        const auto vw_start = qr_profile_start();
        run_qr_column_parallel(block_cols, worker_count, [&](int begin, int end) {
            qr_update_c_vw_range(
                vt_pack, panel_size_actual, active_rows,
                w, max_block_cols, a_col, m, row_start,
                absolute_col_start, begin, end
            );
        });
        qr_profile_add(g_qr_profile.trailing_gemm_ns, vw_start);
    }
    return JLC_STATUS_SUCCESS;
}

jlc_status apply_qr_block_to_col_major_gemm(const double* vt_pack, const double* t,
                                            int panel_size_actual, int active_rows, int max_block_cols,
                                            double* w, double* c_block,
                                            double* a_col, int m, int row_start,
                                            int trailing_col_start, int trailing_cols) {
    for (int col_start = 0; col_start < trailing_cols; col_start += max_block_cols) {
        const int block_cols = std::min(max_block_cols, trailing_cols - col_start);
        const auto pack_start = qr_profile_start();
        pack_col_major_block_to_row_major(
            a_col, m, row_start, trailing_col_start + col_start, active_rows, block_cols, max_block_cols, c_block
        );
        qr_profile_add(g_qr_profile.trailing_pack_ns, pack_start);

        const auto gemm_vtc_start = qr_profile_start();
        jlc_status status = jlc_native_gemm_strided(
            vt_pack, 0, active_rows, panel_size_actual, active_rows, 0,
            c_block, 0, max_block_cols, active_rows, block_cols, 0,
            w, 0, max_block_cols, panel_size_actual, block_cols, 0,
            1.0, 0.0, native_qr_gemm_threads(), 0
        );
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        qr_profile_add(g_qr_profile.trailing_gemm_ns, gemm_vtc_start);

        const auto t_apply_start = qr_profile_start();
        apply_qr_block_t(t, w, panel_size_actual, block_cols, max_block_cols);
        qr_profile_add(g_qr_profile.trailing_t_apply_ns, t_apply_start);

        const auto gemm_vw_start = qr_profile_start();
        status = jlc_native_gemm_strided(
            vt_pack, 0, active_rows, panel_size_actual, active_rows, JLC_GEMM_FLAG_A_TRANSPOSE,
            w, 0, max_block_cols, panel_size_actual, block_cols, 0,
            c_block, 0, max_block_cols, active_rows, block_cols, 0,
            -1.0, 1.0, native_qr_gemm_threads(), 0
        );
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        qr_profile_add(g_qr_profile.trailing_gemm_ns, gemm_vw_start);

        const auto unpack_start = qr_profile_start();
        unpack_row_major_block_to_col_major(
            c_block, max_block_cols, active_rows, block_cols, a_col, m, row_start, trailing_col_start + col_start
        );
        qr_profile_add(g_qr_profile.trailing_unpack_ns, unpack_start);
    }
    return JLC_STATUS_SUCCESS;
}

jlc_status apply_qr_block_to_col_major(const double* vt_pack, const double* t,
                                       int panel_size_actual, int active_rows, int max_block_cols,
                                       double* w, double* c_block,
                                       double* a_col, int m, int row_start, int trailing_col_start, int trailing_cols) {
    if (native_qr_use_direct_trailing_update()) {
        return apply_qr_block_to_col_major_direct(
            vt_pack, t,
            panel_size_actual, active_rows, max_block_cols,
            w, a_col, m, row_start, trailing_col_start, trailing_cols
        );
    }
    return apply_qr_block_to_col_major_gemm(
        vt_pack, t,
        panel_size_actual, active_rows, max_block_cols,
        w, c_block,
        a_col, m, row_start, trailing_col_start, trailing_cols
    );
}

jlc_status apply_qr_block_to_row_major(const double* vt_pack, const double* t,
                                       int panel_size_actual, int active_rows, int max_block_cols,
                                       double* w,
                                       double* q, int q_cols, int row_start, int col_start_initial) {
    for (int col_start = col_start_initial; col_start < q_cols; col_start += max_block_cols) {
        const int block_cols = std::min(max_block_cols, q_cols - col_start);
        const int offset = row_start * q_cols + col_start;

        const auto gemm_vtq_start = qr_profile_start();
        jlc_status status = jlc_native_gemm_strided(
            vt_pack, 0, active_rows, panel_size_actual, active_rows, 0,
            q, offset, q_cols, active_rows, block_cols, 0,
            w, 0, max_block_cols, panel_size_actual, block_cols, 0,
            1.0, 0.0, native_qr_gemm_threads(), 0
        );
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        qr_profile_add(g_qr_profile.q_gemm_ns, gemm_vtq_start);

        const auto t_apply_start = qr_profile_start();
        apply_qr_t(t, w, panel_size_actual, block_cols, max_block_cols);
        qr_profile_add(g_qr_profile.q_t_apply_ns, t_apply_start);

        const auto gemm_vw_start = qr_profile_start();
        status = jlc_native_gemm_strided(
            vt_pack, 0, active_rows, panel_size_actual, active_rows, JLC_GEMM_FLAG_A_TRANSPOSE,
            w, 0, max_block_cols, panel_size_actual, block_cols, 0,
            q, offset, q_cols, active_rows, block_cols, 0,
            -1.0, 1.0, native_qr_gemm_threads(), 0
        );
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        qr_profile_add(g_qr_profile.q_gemm_ns, gemm_vw_start);
    }
    return JLC_STATUS_SUCCESS;
}

jlc_status qr_factorize_core(const double* a, int m, int n,
                             std::vector<double>& a_col, std::vector<double>& tau) {
    try {
        const int k = std::min(m, n);
        if (k == 0) {
            return JLC_STATUS_SUCCESS;
        }

        const bool use_blocked = std::max(m, n) >= native_qr_block_threshold();
        const int panel_size_max = use_blocked ? std::max(1, std::min(native_qr_block_size_for_shape(m, n, k), k)) : 1;
        const int max_block_cols = use_blocked
            ? std::max(1, std::min(n, native_qr_trailing_block()))
            : std::max(1, n);

        QrWorkspace& workspace = qr_workspace();
        workspace.prepare(m, n, max_block_cols, panel_size_max, k);
        std::vector<double>& vt_pack = workspace.vt_pack;
        std::vector<double>& t = workspace.t;
        std::vector<double>& w = workspace.w;
        std::vector<double>& c_block = workspace.c_block;
        std::vector<double>& work = workspace.work;
        std::vector<double>& column = workspace.column;

        a_col.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        const auto transpose_start = qr_profile_start();
        transpose_row_to_col(a, a_col.data(), m, n);
        qr_profile_add(g_qr_profile.input_transpose_ns, transpose_start);

        tau.resize(static_cast<std::size_t>(std::max(1, k)));
        std::fill(tau.begin(), tau.end(), 0.0);

        for (int k_start = 0; k_start < k; k_start += panel_size_max) {
            const int k_end = std::min(k_start + panel_size_max, k);
            const int panel_size_actual = k_end - k_start;
            const auto panel_start = qr_profile_start();
            factorize_qr_panel(a_col.data(), tau.data(), m, k_start, k_end);
            qr_profile_add(g_qr_profile.panel_ns, panel_start);

            if (k_end >= n) {
                continue;
            }

            const int active_rows = m - k_start;
            const auto reflector_pack_start = qr_profile_start();
            pack_qr_reflectors(a_col.data(), m, k_start, panel_size_actual, vt_pack.data());
            qr_profile_add(g_qr_profile.reflector_pack_ns, reflector_pack_start);
            const auto t_build_start = qr_profile_start();
            build_qr_block_t(vt_pack.data(), tau.data(), k_start, panel_size_actual, active_rows,
                             t.data(), work.data(), column.data());
            qr_profile_add(g_qr_profile.t_build_ns, t_build_start);

            const jlc_status status = apply_qr_block_to_col_major(
                vt_pack.data(), t.data(),
                panel_size_actual, active_rows, max_block_cols,
                w.data(), c_block.data(),
                a_col.data(), m, k_start, k_end, n - k_end
            );
            if (status != JLC_STATUS_SUCCESS) {
                return status;
            }
        }

        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
}

jlc_status qr_factorize_builtin(const double* a, int m, int n, int q_cols,
                                double* q, double* r) {
    QrWorkspace& workspace = qr_workspace();
    std::vector<double>& a_col = workspace.a_col;
    std::vector<double>& tau = workspace.tau;
    const jlc_status factor_status = qr_factorize_core(a, m, n, a_col, tau);
    if (factor_status != JLC_STATUS_SUCCESS) {
        return factor_status;
    }

    const int k = std::min(m, n);
    if (k == 0) {
        std::fill(q, q + static_cast<std::size_t>(m) * static_cast<std::size_t>(q_cols), 0.0);
        std::fill(r, r + static_cast<std::size_t>(q_cols) * static_cast<std::size_t>(n), 0.0);
        return JLC_STATUS_SUCCESS;
    }

    try {
        const auto r_extract_start = qr_profile_start();
        extract_r_from_col_major(a_col.data(), m, n, q_cols, r);
        qr_profile_add(g_qr_profile.r_extract_ns, r_extract_start);

        const auto q_init_start = qr_profile_start();
        std::fill(q, q + static_cast<std::size_t>(m) * static_cast<std::size_t>(q_cols), 0.0);
        const int diag_limit = std::min(m, q_cols);
        for (int diag = 0; diag < diag_limit; ++diag) {
            q[diag * q_cols + diag] = 1.0;
        }
        qr_profile_add(g_qr_profile.q_init_ns, q_init_start);

        const bool use_blocked = std::max(m, n) >= native_qr_block_threshold();
        const int panel_size_max = use_blocked ? std::max(1, std::min(native_qr_block_size_for_shape(m, n, k), k)) : 1;
        const int max_block_cols = use_blocked
            ? std::max(1, std::min(q_cols, native_qr_q_block()))
            : std::max(1, q_cols);
        QrWorkspace& workspace = qr_workspace();
        workspace.prepare(m, n, max_block_cols, panel_size_max, k);
        std::vector<double>& vt_pack = workspace.vt_pack;
        std::vector<double>& t = workspace.t;
        std::vector<double>& w = workspace.w;
        std::vector<double>& work = workspace.work;
        std::vector<double>& column = workspace.column;

        const auto q_build_start = qr_profile_start();
        for (int block_end = k; block_end > 0; block_end -= panel_size_max) {
            const int k_start = std::max(0, block_end - panel_size_max);
            const int panel_size_actual = block_end - k_start;
            const int active_rows = m - k_start;

            const auto reflector_pack_start = qr_profile_start();
            pack_qr_reflectors(a_col.data(), m, k_start, panel_size_actual, vt_pack.data());
            qr_profile_add(g_qr_profile.reflector_pack_ns, reflector_pack_start);
            const auto t_build_start = qr_profile_start();
            build_qr_block_t(vt_pack.data(), tau.data(), k_start, panel_size_actual, active_rows,
                             t.data(), work.data(), column.data());
            qr_profile_add(g_qr_profile.t_build_ns, t_build_start);

            const jlc_status status = apply_qr_block_to_row_major(
                vt_pack.data(), t.data(),
                panel_size_actual, active_rows, max_block_cols,
                w.data(),
                q, q_cols, k_start, k_start
            );
            if (status != JLC_STATUS_SUCCESS) {
                return status;
            }
        }
        qr_profile_add(g_qr_profile.q_build_ns, q_build_start);

        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
}

}  // namespace

#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
namespace {
int query_dgeqrf_workspace(int m, int n, double* a, double* tau) {
    const int lda = m;
    int lwork = -1;
    int info = 0;
    double work_query = 0.0;
    dgeqrf_(&m, &n, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) {
        return -1;
    }
    return std::max(1, static_cast<int>(work_query));
}

int query_dorgqr_workspace(int m, int n, int k, double* a, double* tau) {
    const int lda = m;
    int lwork = -1;
    int info = 0;
    double work_query = 0.0;
    dorgqr_(&m, &n, &k, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) {
        return -1;
    }
    return std::max(1, static_cast<int>(work_query));
}
}  // namespace
#endif

jlc_status jlc_native_lu_factor(double* packed_lu, int n, int* pivots, int pivot_length, int* out_info) {
    if (packed_lu == nullptr || pivots == nullptr || out_info == nullptr || n < 0 || pivot_length != n) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    *out_info = 0;
    if (n == 0) {
        return JLC_STATUS_SUCCESS;
    }
    const bool use_blocked = n >= native_lu_block_threshold();
    const int block_size = std::max(1, std::min(native_lu_block_size(), n));

    if (!use_blocked) {
        for (int k = 0; k < n; ++k) {
            const int pivot_row = select_pivot_row(packed_lu, n, k);
            pivots[k] = pivot_row;
            swap_rows(packed_lu, n, k, pivot_row, 0, n);

            const double pivot = packed_lu[k * n + k];
            if (pivot == 0.0) {
                if (*out_info == 0) {
                    *out_info = k + 1;
                }
                continue;
            }

            for (int row = k + 1; row < n; ++row) {
                const std::size_t row_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(n);
                const double factor = packed_lu[row_base + k] / pivot;
                packed_lu[row_base + k] = factor;
                for (int col = k + 1; col < n; ++col) {
                    packed_lu[row_base + col] -= factor * packed_lu[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + col];
                }
            }
        }
        return JLC_STATUS_SUCCESS;
    }

    for (int panel_start = 0; panel_start < n; panel_start += block_size) {
        const int panel_end = std::min(panel_start + block_size, n);
        for (int k = panel_start; k < panel_end; ++k) {
            const int pivot_row = select_pivot_row(packed_lu, n, k);
            pivots[k] = pivot_row;
            swap_rows(packed_lu, n, k, pivot_row, 0, n);

            const double pivot = packed_lu[k * n + k];
            if (pivot == 0.0) {
                if (*out_info == 0) {
                    *out_info = k + 1;
                }
                continue;
            }

            for (int row = k + 1; row < n; ++row) {
                const std::size_t row_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(n);
                const double factor = packed_lu[row_base + k] / pivot;
                packed_lu[row_base + k] = factor;
                for (int col = k + 1; col < panel_end; ++col) {
                    packed_lu[row_base + col] -= factor * packed_lu[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + col];
                }
            }
        }

        if (panel_end >= n) {
            continue;
        }

        solve_upper_panel(packed_lu, n, panel_start, panel_end);

        const int trailing = n - panel_end;
        const int panel_width = panel_end - panel_start;
        if (trailing > 0 && panel_width > 0) {
            const jlc_status status = jlc_native_gemm_strided(
                packed_lu, panel_end * n + panel_start, n, trailing, panel_width, 0,
                packed_lu, panel_start * n + panel_end, n, panel_width, trailing, 0,
                packed_lu, panel_end * n + panel_end, n, trailing, trailing, 0,
                -1.0, 1.0, 0, 0
            );
            if (status != JLC_STATUS_SUCCESS) {
                return status;
            }
        }
    }
    return JLC_STATUS_SUCCESS;
}

jlc_status jlc_native_lu_factor_vendor(double* packed_lu, int n, int* pivots, int pivot_length, int* out_info) {
    if (packed_lu == nullptr || pivots == nullptr || out_info == nullptr || n < 0 || pivot_length != n) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    *out_info = 0;
    if (n == 0) {
        return JLC_STATUS_SUCCESS;
    }
#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
    try {
        std::vector<double> a_col(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        std::vector<int> ipiv(static_cast<std::size_t>(n), 0);
        transpose_row_to_col(packed_lu, a_col.data(), n, n);

        const int m = n;
        const int lda = n;
        int info = 0;
        dgetrf_(&m, &m, a_col.data(), &lda, ipiv.data(), &info);
        if (info < 0) {
            return JLC_STATUS_INVALID_ARGUMENT;
        }

        transpose_col_to_row(a_col.data(), packed_lu, n, n);
        for (int i = 0; i < n; ++i) {
            pivots[i] = ipiv[static_cast<std::size_t>(i)] - 1;
        }
        *out_info = info;
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
#else
    return JLC_STATUS_INTERNAL_ERROR;
#endif
}

jlc_status jlc_native_qr_factorize_only(const double* a, int m, int n) {
    const bool profile_enabled = g_qr_profile.enabled.load(std::memory_order_relaxed);
    const auto wall_start = profile_enabled ? QrProfileClock::now() : QrProfileClock::time_point{};
    if (a == nullptr || m < 0 || n < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0 || n == 0) {
        return JLC_STATUS_SUCCESS;
    }
    QrWorkspace& workspace = qr_workspace();
    std::vector<double>& a_col = workspace.a_col;
    std::vector<double>& tau = workspace.tau;
    const auto factor_start = profile_enabled ? QrProfileClock::now() : QrProfileClock::time_point{};
    const jlc_status status = qr_factorize_core(a, m, n, a_col, tau);
    if (profile_enabled) {
        g_qr_profile.calls.fetch_add(1, std::memory_order_relaxed);
        g_qr_profile.factorize_ns.fetch_add(qr_elapsed_ns(factor_start), std::memory_order_relaxed);
        g_qr_profile.wall_ns.fetch_add(qr_elapsed_ns(wall_start), std::memory_order_relaxed);
    }
    return status;
}

jlc_status jlc_native_qr_factorize_only_vendor(const double* a, int m, int n) {
    if (a == nullptr || m < 0 || n < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0 || n == 0) {
        return JLC_STATUS_SUCCESS;
    }
#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
    try {
        const int k = std::min(m, n);
        std::vector<double> a_col(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);
        transpose_row_to_col(a, a_col.data(), m, n);

        std::vector<double> tau(static_cast<std::size_t>(std::max(1, k)), 0.0);
        int lwork = query_dgeqrf_workspace(m, n, a_col.data(), tau.data());
        if (lwork < 1) {
            return JLC_STATUS_INTERNAL_ERROR;
        }
        std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
        const int lda = m;
        int info = 0;
        dgeqrf_(&m, &n, a_col.data(), &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0) {
            return info < 0 ? JLC_STATUS_INVALID_ARGUMENT : JLC_STATUS_INTERNAL_ERROR;
        }
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
#else
    return JLC_STATUS_INTERNAL_ERROR;
#endif
}

jlc_status jlc_native_qr_decompose(const double* a, int m, int n, int q_cols,
                                   double* q, double* r) {
    const bool profile_enabled = g_qr_profile.enabled.load(std::memory_order_relaxed);
    const auto wall_start = profile_enabled ? QrProfileClock::now() : QrProfileClock::time_point{};
    if (a == nullptr || q == nullptr || r == nullptr || m < 0 || n < 0 || q_cols < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    const int k = std::min(m, n);
    if (!(q_cols == k || q_cols == m)) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0 || n == 0) {
        return JLC_STATUS_SUCCESS;
    }
    const jlc_status status = qr_factorize_builtin(a, m, n, q_cols, q, r);
    if (profile_enabled) {
        g_qr_profile.calls.fetch_add(1, std::memory_order_relaxed);
        g_qr_profile.wall_ns.fetch_add(qr_elapsed_ns(wall_start), std::memory_order_relaxed);
    }
    return status;
}

jlc_status jlc_native_qr_decompose_vendor(const double* a, int m, int n, int q_cols,
                                          double* q, double* r) {
    if (a == nullptr || q == nullptr || r == nullptr || m < 0 || n < 0 || q_cols < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    const int k = std::min(m, n);
    if (!(q_cols == k || q_cols == m)) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0 || n == 0) {
        return JLC_STATUS_SUCCESS;
    }
#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
    try {
        std::vector<double> a_col(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);
        transpose_row_to_col(a, a_col.data(), m, n);

        std::vector<double> tau(static_cast<std::size_t>(std::max(1, k)), 0.0);
        int lwork = query_dgeqrf_workspace(m, n, a_col.data(), tau.data());
        if (lwork < 1) {
            return JLC_STATUS_INTERNAL_ERROR;
        }
        std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
        const int lda = m;
        int info = 0;
        dgeqrf_(&m, &n, a_col.data(), &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0) {
            return info < 0 ? JLC_STATUS_INVALID_ARGUMENT : JLC_STATUS_INTERNAL_ERROR;
        }

        extract_r_from_col_major(a_col.data(), m, n, q_cols, r);

        std::vector<double> q_col(static_cast<std::size_t>(m) * static_cast<std::size_t>(q_cols), 0.0);
        for (int col = 0; col < k; ++col) {
            const double* src = a_col.data() + col * m;
            double* dst = q_col.data() + col * m;
            std::copy(src, src + m, dst);
        }

        lwork = query_dorgqr_workspace(m, q_cols, k, q_col.data(), tau.data());
        if (lwork < 1) {
            return JLC_STATUS_INTERNAL_ERROR;
        }
        work.assign(static_cast<std::size_t>(lwork), 0.0);
        dorgqr_(&m, &q_cols, &k, q_col.data(), &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0) {
            return info < 0 ? JLC_STATUS_INVALID_ARGUMENT : JLC_STATUS_INTERNAL_ERROR;
        }

        transpose_col_to_row(q_col.data(), q, m, q_cols);
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
#else
    return JLC_STATUS_INTERNAL_ERROR;
#endif
}

jlc_status jlc_native_cholesky_decompose(double* packed_l, int n, int* out_info) {
    if (packed_l == nullptr || out_info == nullptr || n < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    *out_info = 0;
    if (n == 0) {
        return JLC_STATUS_SUCCESS;
    }
    const bool use_blocked = n >= native_cholesky_block_threshold();
    const int block_size = std::max(1, std::min(native_cholesky_block_size(), n));

    if (!use_blocked) {
        for (int col = 0; col < n; ++col) {
            double diag = packed_l[col * n + col];
            for (int k = 0; k < col; ++k) {
                const double value = packed_l[col * n + k];
                diag = std::fma(-value, value, diag);
            }
            if (!(diag > 0.0) || std::isnan(diag)) {
                *out_info = col + 1;
                return JLC_STATUS_SUCCESS;
            }

            const double diag_sqrt = std::sqrt(diag);
            packed_l[col * n + col] = diag_sqrt;
            for (int row = col + 1; row < n; ++row) {
                double value = packed_l[row * n + col];
                for (int k = 0; k < col; ++k) {
                    value = std::fma(-packed_l[row * n + k], packed_l[col * n + k], value);
                }
                packed_l[row * n + col] = value / diag_sqrt;
            }
        }
        return JLC_STATUS_SUCCESS;
    }

    for (int block_start = 0; block_start < n; block_start += block_size) {
        const int panel_width = std::min(block_size, n - block_start);
        factor_cholesky_panel(packed_l, n, block_start, panel_width, out_info);
        if (*out_info != 0) {
            return JLC_STATUS_SUCCESS;
        }

        const int trailing_start = block_start + panel_width;
        if (trailing_start >= n) {
            continue;
        }

        const int trailing_rows = n - trailing_start;
        const jlc_status status = jlc_native_gemm_strided(
            packed_l, trailing_start * n + block_start, n, trailing_rows, panel_width, 0,
            packed_l, trailing_start * n + block_start, n, panel_width, trailing_rows, JLC_GEMM_FLAG_B_COL_MAJOR,
            packed_l, trailing_start * n + trailing_start, n, trailing_rows, trailing_rows, 0,
            -1.0, 1.0, 0, 0
        );
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
    }
    return JLC_STATUS_SUCCESS;
}

jlc_status jlc_native_cholesky_decompose_vendor(double* packed_l, int n, int* out_info) {
    if (packed_l == nullptr || out_info == nullptr || n < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    *out_info = 0;
    if (n == 0) {
        return JLC_STATUS_SUCCESS;
    }
#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
    try {
        std::vector<double> a_col(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        transpose_row_to_col(packed_l, a_col.data(), n, n);
        const int lda = n;
        int info = 0;
        const char lower = 'L';
        dpotrf_(&lower, &n, a_col.data(), &lda, &info);
        if (info < 0) {
            return JLC_STATUS_INVALID_ARGUMENT;
        }
        transpose_col_to_row(a_col.data(), packed_l, n, n);
        *out_info = info;
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
#else
    return JLC_STATUS_INTERNAL_ERROR;
#endif
}

void jlc_native_qr_profile_set_enabled(bool enabled) {
    g_qr_profile.enabled.store(enabled, std::memory_order_relaxed);
}

void jlc_native_qr_profile_reset() {
    g_qr_profile.calls.store(0, std::memory_order_relaxed);
    g_qr_profile.wall_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.factorize_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.input_transpose_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.panel_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.reflector_pack_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.t_build_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.trailing_pack_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.trailing_unpack_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.trailing_gemm_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.trailing_t_apply_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.r_extract_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.q_init_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.q_build_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.q_gemm_ns.store(0, std::memory_order_relaxed);
    g_qr_profile.q_t_apply_ns.store(0, std::memory_order_relaxed);
}

void jlc_native_qr_profile_snapshot(jlc_qr_profile* out_profile) {
    if (out_profile == nullptr) {
        return;
    }
    out_profile->calls = g_qr_profile.calls.load(std::memory_order_relaxed);
    out_profile->wall_ns = g_qr_profile.wall_ns.load(std::memory_order_relaxed);
    out_profile->factorize_ns = g_qr_profile.factorize_ns.load(std::memory_order_relaxed);
    out_profile->input_transpose_ns = g_qr_profile.input_transpose_ns.load(std::memory_order_relaxed);
    out_profile->panel_ns = g_qr_profile.panel_ns.load(std::memory_order_relaxed);
    out_profile->reflector_pack_ns = g_qr_profile.reflector_pack_ns.load(std::memory_order_relaxed);
    out_profile->t_build_ns = g_qr_profile.t_build_ns.load(std::memory_order_relaxed);
    out_profile->trailing_pack_ns = g_qr_profile.trailing_pack_ns.load(std::memory_order_relaxed);
    out_profile->trailing_unpack_ns = g_qr_profile.trailing_unpack_ns.load(std::memory_order_relaxed);
    out_profile->trailing_gemm_ns = g_qr_profile.trailing_gemm_ns.load(std::memory_order_relaxed);
    out_profile->trailing_t_apply_ns = g_qr_profile.trailing_t_apply_ns.load(std::memory_order_relaxed);
    out_profile->r_extract_ns = g_qr_profile.r_extract_ns.load(std::memory_order_relaxed);
    out_profile->q_init_ns = g_qr_profile.q_init_ns.load(std::memory_order_relaxed);
    out_profile->q_build_ns = g_qr_profile.q_build_ns.load(std::memory_order_relaxed);
    out_profile->q_gemm_ns = g_qr_profile.q_gemm_ns.load(std::memory_order_relaxed);
    out_profile->q_t_apply_ns = g_qr_profile.q_t_apply_ns.load(std::memory_order_relaxed);
}

void jlc_native_qr_set_block_size_override(int block_size) {
    g_qr_block_size_override.store(std::max(0, block_size), std::memory_order_relaxed);
}

void jlc_native_qr_set_gemm_threads_override(int threads) {
    g_qr_gemm_threads_override.store(std::max(1, threads), std::memory_order_relaxed);
}
