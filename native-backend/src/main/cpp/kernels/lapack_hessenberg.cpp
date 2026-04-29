#include "jlc_native.h"

#include <algorithm>
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
