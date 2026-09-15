#include "jlc_native.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

namespace {
void set_identity(double* out, int n) {
    std::fill(out, out + static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        out[i * n + i] = 1.0;
    }
}

void transpose_row_major(const double* src, double* dst, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const int row_base = row * cols;
        for (int col = 0; col < cols; ++col) {
            dst[col * rows + row] = src[row_base + col];
        }
    }
}

void apply_householder_left(double* m, int cols, int start_row, int start_col,
                            const std::vector<double>& v, int len, double tau) {
    for (int col = start_col; col < cols; ++col) {
        double dot = 0.0;
        for (int i = 0; i < len; ++i) {
            dot = std::fma(v[static_cast<std::size_t>(i)], m[(start_row + i) * cols + col], dot);
        }
        dot *= tau;
        for (int i = 0; i < len; ++i) {
            m[(start_row + i) * cols + col] -= dot * v[static_cast<std::size_t>(i)];
        }
    }
}

void apply_householder_right(double* m, int rows, int cols, int start_row, int start_col,
                             const std::vector<double>& v, int len, double tau) {
    for (int row = start_row; row < rows; ++row) {
        const int row_base = row * cols + start_col;
        double dot = 0.0;
        for (int i = 0; i < len; ++i) {
            dot = std::fma(m[row_base + i], v[static_cast<std::size_t>(i)], dot);
        }
        dot *= tau;
        for (int i = 0; i < len; ++i) {
            m[row_base + i] -= dot * v[static_cast<std::size_t>(i)];
        }
    }
}

jlc_status bidiagonal_upper(const double* a, int m, int n, double* u, double* b, double* v) {
    std::copy(a, a + static_cast<std::size_t>(m) * static_cast<std::size_t>(n), b);
    set_identity(u, m);
    set_identity(v, n);

    const int limit = std::min(m, n);
    std::vector<double> v_col(static_cast<std::size_t>(m), 0.0);
    std::vector<double> v_row(static_cast<std::size_t>(n), 0.0);

    for (int k = 0; k < limit; ++k) {
        const int len = m - k;
        double x0 = b[k * n + k];
        double norm_sq = 0.0;
        for (int i = 1; i < len; ++i) {
            const double value = b[(k + i) * n + k];
            norm_sq = std::fma(value, value, norm_sq);
        }
        const double xnorm = std::sqrt(norm_sq);
        if (xnorm != 0.0) {
            const double beta = x0 >= 0.0 ? -std::hypot(x0, xnorm) : std::hypot(x0, xnorm);
            const double tau = (beta - x0) / beta;
            const double inv_v0 = 1.0 / (x0 - beta);

            v_col[0] = 1.0;
            for (int i = 1; i < len; ++i) {
                v_col[static_cast<std::size_t>(i)] = b[(k + i) * n + k] * inv_v0;
            }

            apply_householder_left(b, n, k, k, v_col, len, tau);
            apply_householder_right(u, m, m, 0, k, v_col, len, tau);

            b[k * n + k] = beta;
            for (int i = 1; i < len; ++i) {
                b[(k + i) * n + k] = 0.0;
            }
        }

        if (k >= n - 1) {
            continue;
        }

        const int len_row = n - k - 1;
        x0 = b[k * n + k + 1];
        norm_sq = 0.0;
        for (int j = 1; j < len_row; ++j) {
            const double value = b[k * n + k + 1 + j];
            norm_sq = std::fma(value, value, norm_sq);
        }
        const double xnorm_row = std::sqrt(norm_sq);
        if (xnorm_row != 0.0) {
            const double beta = x0 >= 0.0 ? -std::hypot(x0, xnorm_row) : std::hypot(x0, xnorm_row);
            const double tau = (beta - x0) / beta;
            const double inv_v0 = 1.0 / (x0 - beta);

            v_row[0] = 1.0;
            for (int j = 1; j < len_row; ++j) {
                v_row[static_cast<std::size_t>(j)] = b[k * n + k + 1 + j] * inv_v0;
            }

            apply_householder_right(b, m, n, k, k + 1, v_row, len_row, tau);
            apply_householder_right(v, n, n, 0, k + 1, v_row, len_row, tau);

            b[k * n + k + 1] = beta;
            for (int j = 1; j < len_row; ++j) {
                b[k * n + k + 1 + j] = 0.0;
            }
        }
    }

    return JLC_STATUS_SUCCESS;
}
}

jlc_status jlc_native_bidiagonal_decompose(const double* a, int m, int n,
                                           double* u, double* b, double* v) {
    if (a == nullptr || u == nullptr || b == nullptr || v == nullptr || m < 0 || n < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0 || n == 0) {
        return JLC_STATUS_SUCCESS;
    }
    try {
        if (m >= n) {
            return bidiagonal_upper(a, m, n, u, b, v);
        }

        std::vector<double> a_t(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);
        std::vector<double> u_t(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        std::vector<double> b_t(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);
        std::vector<double> v_t(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
        transpose_row_major(a, a_t.data(), m, n);
        const jlc_status status = bidiagonal_upper(a_t.data(), n, m, u_t.data(), b_t.data(), v_t.data());
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        transpose_row_major(b_t.data(), b, n, m);
        std::copy(v_t.begin(), v_t.end(), u);
        std::copy(u_t.begin(), u_t.end(), v);
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
}
