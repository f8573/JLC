#include "jlc_native.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
extern "C" {
void dgehrd_(const int* n, const int* ilo, const int* ihi,
             double* a, const int* lda, double* tau,
             double* work, const int* lwork, int* info);
void dorghr_(const int* n, const int* ilo, const int* ihi,
             double* a, const int* lda, const double* tau,
             double* work, const int* lwork, int* info);
}
#endif

namespace {
inline void transpose_row_to_col(const double* src, double* dst, int n) {
    for (int row = 0; row < n; ++row) {
        const int row_offset = row * n;
        for (int col = 0; col < n; ++col) {
            dst[col * n + row] = src[row_offset + col];
        }
    }
}

inline void transpose_col_to_row(const double* src, double* dst, int n) {
    for (int row = 0; row < n; ++row) {
        const int row_offset = row * n;
        for (int col = 0; col < n; ++col) {
            dst[row_offset + col] = src[col * n + row];
        }
    }
}

#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
int query_dgehrd_workspace(int n, double* a, double* tau) {
    const int ilo = 1;
    const int ihi = n;
    const int lda = n;
    int lwork = -1;
    int info = 0;
    double work_query = 0.0;
    dgehrd_(&n, &ilo, &ihi, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) {
        return -1;
    }
    return std::max(1, static_cast<int>(work_query));
}

int query_dorghr_workspace(int n, double* a, const double* tau) {
    const int ilo = 1;
    const int ihi = n;
    const int lda = n;
    int lwork = -1;
    int info = 0;
    double work_query = 0.0;
    dorghr_(&n, &ilo, &ihi, a, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) {
        return -1;
    }
    return std::max(1, static_cast<int>(work_query));
}

jlc_status compute_hessenberg_col_major(int n,
                                        std::vector<double>& a_col,
                                        std::vector<double>& tau_vendor) {
    const int ilo = 1;
    const int ihi = n;
    const int lda = n;
    const int tau_size = std::max(1, n - 1);
    tau_vendor.assign(static_cast<std::size_t>(tau_size), 0.0);

    int lwork = query_dgehrd_workspace(n, a_col.data(), tau_vendor.data());
    if (lwork < 1) {
        return JLC_STATUS_INTERNAL_ERROR;
    }

    std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
    int info = 0;
    dgehrd_(&n, &ilo, &ihi, a_col.data(), &lda, tau_vendor.data(), work.data(), &lwork, &info);
    if (info < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    return info == 0 ? JLC_STATUS_SUCCESS : JLC_STATUS_INTERNAL_ERROR;
}
#endif
}

bool jlc_native_vendor_lapack_available() {
#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
    return true;
#else
    return false;
#endif
}

namespace {
void set_identity(double* q, int n) {
    std::fill(q, q + static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        q[i * n + i] = 1.0;
    }
}

void zero_below_subdiagonal(double* h, int n) {
    for (int col = 0; col < n - 2; ++col) {
        for (int row = col + 2; row < n; ++row) {
            h[row * n + col] = 0.0;
        }
    }
}

void hessenberg_reduce_builtin_core(double* h, int n, std::vector<double>& tau, double* q) {
    std::vector<double> v(static_cast<std::size_t>(n), 0.0);
    if (q != nullptr) {
        set_identity(q, n);
    }

    for (int k = 0; k < n - 2; ++k) {
        const int start = k + 1;
        const int len = n - start;
        const int base = start * n + k;
        const double x0 = h[base];

        double norm_sq = 0.0;
        for (int i = 1; i < len; ++i) {
            const double value = h[(start + i) * n + k];
            norm_sq = std::fma(value, value, norm_sq);
        }
        const double xnorm = std::sqrt(norm_sq);
        if (xnorm == 0.0) {
            tau[static_cast<std::size_t>(k)] = 0.0;
            continue;
        }

        const double beta = x0 >= 0.0 ? -std::hypot(x0, xnorm) : std::hypot(x0, xnorm);
        const double tau_k = (beta - x0) / beta;
        const double inv_v0 = 1.0 / (x0 - beta);

        tau[static_cast<std::size_t>(k)] = tau_k;
        v[0] = 1.0;
        h[base] = beta;
        for (int i = 1; i < len; ++i) {
            const double vi = h[(start + i) * n + k] * inv_v0;
            v[static_cast<std::size_t>(i)] = vi;
            h[(start + i) * n + k] = vi;
        }

        for (int col = start; col < n; ++col) {
            double dot = h[start * n + col];
            for (int i = 1; i < len; ++i) {
                dot = std::fma(v[static_cast<std::size_t>(i)], h[(start + i) * n + col], dot);
            }
            dot *= tau_k;
            h[start * n + col] -= dot;
            for (int i = 1; i < len; ++i) {
                h[(start + i) * n + col] -= dot * v[static_cast<std::size_t>(i)];
            }
        }

        for (int row = 0; row < n; ++row) {
            const int idx = row * n + start;
            double dot = h[idx];
            for (int i = 1; i < len; ++i) {
                dot = std::fma(h[idx + i], v[static_cast<std::size_t>(i)], dot);
            }
            dot *= tau_k;
            h[idx] -= dot;
            for (int i = 1; i < len; ++i) {
                h[idx + i] -= dot * v[static_cast<std::size_t>(i)];
            }
        }

        if (q != nullptr) {
            for (int row = 0; row < n; ++row) {
                const int idx = row * n + start;
                double dot = q[idx];
                for (int i = 1; i < len; ++i) {
                    dot = std::fma(q[idx + i], v[static_cast<std::size_t>(i)], dot);
                }
                dot *= tau_k;
                q[idx] -= dot;
                for (int i = 1; i < len; ++i) {
                    q[idx + i] -= dot * v[static_cast<std::size_t>(i)];
                }
            }
        }
    }

    zero_below_subdiagonal(h, n);
}
}

jlc_status jlc_native_hessenberg_reduce_vendor(double* h, int n) {
    if (h == nullptr || n < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (n <= 2) {
        return JLC_STATUS_SUCCESS;
    }
#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
    try {
        std::vector<double> a_col(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        std::vector<double> tau_vendor;
        transpose_row_to_col(h, a_col.data(), n);
        jlc_status status = compute_hessenberg_col_major(n, a_col, tau_vendor);
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        transpose_col_to_row(a_col.data(), h, n);
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
#else
    (void) h;
    (void) n;
    return JLC_STATUS_INTERNAL_ERROR;
#endif
}

jlc_status jlc_native_hessenberg_reduce(double* h, int n) {
    if (h == nullptr || n < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (n <= 2) {
        return JLC_STATUS_SUCCESS;
    }
    try {
        std::vector<double> tau(static_cast<std::size_t>(std::max(1, n - 2)), 0.0);
        hessenberg_reduce_builtin_core(h, n, tau, nullptr);
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
}

jlc_status jlc_native_hessenberg_decompose(double* h, int n, double* q) {
    if (h == nullptr || q == nullptr || n < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (n <= 2) {
        set_identity(q, n);
        return JLC_STATUS_SUCCESS;
    }
    try {
        std::vector<double> tau(static_cast<std::size_t>(std::max(1, n - 2)), 0.0);
        hessenberg_reduce_builtin_core(h, n, tau, q);
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
}

jlc_status jlc_native_hessenberg_decompose_vendor(double* h, int n, double* q) {
    if (h == nullptr || q == nullptr || n < 0) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (n <= 2) {
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                q[row * n + col] = row == col ? 1.0 : 0.0;
            }
        }
        return JLC_STATUS_SUCCESS;
    }
#if defined(JLC_NATIVE_HAS_VENDOR_LAPACK)
    try {
        std::vector<double> a_col(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        std::vector<double> tau_vendor;
        transpose_row_to_col(h, a_col.data(), n);
        jlc_status status = compute_hessenberg_col_major(n, a_col, tau_vendor);
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }

        transpose_col_to_row(a_col.data(), h, n);

        std::vector<double> q_col = a_col;
        const int ilo = 1;
        const int ihi = n;
        const int lda = n;
        int lwork = query_dorghr_workspace(n, q_col.data(), tau_vendor.data());
        if (lwork < 1) {
            return JLC_STATUS_INTERNAL_ERROR;
        }
        std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
        int info = 0;
        dorghr_(&n, &ilo, &ihi, q_col.data(), &lda, tau_vendor.data(), work.data(), &lwork, &info);
        if (info < 0) {
            return JLC_STATUS_INVALID_ARGUMENT;
        }
        if (info != 0) {
            return JLC_STATUS_INTERNAL_ERROR;
        }

        transpose_col_to_row(q_col.data(), q, n);
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
#else
    (void) h;
    (void) n;
    (void) q;
    return JLC_STATUS_INTERNAL_ERROR;
#endif
}
