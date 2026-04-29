#pragma once

#include <cstdint>

#include "jlc_status.h"

enum jlc_gemm_flag : std::uint32_t {
    JLC_GEMM_FLAG_A_TRANSPOSE = 1u << 0,
    JLC_GEMM_FLAG_A_COL_MAJOR = 1u << 1,
    JLC_GEMM_FLAG_B_TRANSPOSE = 1u << 2,
    JLC_GEMM_FLAG_B_COL_MAJOR = 1u << 3,
    JLC_GEMM_FLAG_C_COL_MAJOR = 1u << 4,
    JLC_GEMM_FLAG_PREFER_VENDOR = 1u << 8,
    JLC_GEMM_FLAG_FORCE_VENDOR = 1u << 9,
    JLC_GEMM_FLAG_FORCE_BUILTIN = 1u << 10
};

enum jlc_matrix_order : std::uint32_t {
    JLC_MATRIX_ROW_MAJOR = 0,
    JLC_MATRIX_COL_MAJOR = 1
};

using jlc_context_handle = std::uintptr_t;
using jlc_matrix_handle = std::uintptr_t;

struct jlc_gemm_profile {
    std::uint64_t calls;
    std::uint64_t wall_ns;
    std::uint64_t vendor_calls;
    std::uint64_t vendor_ns;
    std::uint64_t scale_c_ns;
    std::uint64_t pack_a_ns;
    std::uint64_t pack_b_ns;
    std::uint64_t kernel_ns;
    std::uint64_t thread_launch_ns;
    std::uint64_t thread_join_ns;
    std::uint64_t pack_a_calls;
    std::uint64_t pack_b_calls;
    std::uint64_t microtile_calls;
    std::uint64_t pack_a_bytes;
    std::uint64_t pack_b_bytes;
    std::uint64_t last_requested_threads;
    std::uint64_t last_actual_threads;
    std::uint64_t last_panel_count;
    std::uint64_t last_mc;
    std::uint64_t last_kc;
    std::uint64_t last_nc;
    std::uint64_t last_mr;
    std::uint64_t last_nr;
};

bool jlc_native_is_available();
bool jlc_native_vendor_lapack_available();
const char* jlc_native_runtime_description();
const char* jlc_native_provider_description();
jlc_context_handle jlc_native_context_create(int preferred_threads, int alignment_bytes, int flags);
void jlc_native_context_destroy(jlc_context_handle handle);
void jlc_native_profile_set_enabled(bool enabled);
void jlc_native_profile_reset();
void jlc_native_profile_snapshot(jlc_gemm_profile* out_profile);
jlc_matrix_handle jlc_native_matrix_create(int rows, int cols, int order, int alignment_bytes);
void jlc_native_matrix_destroy(jlc_matrix_handle handle);
double* jlc_native_matrix_data(jlc_matrix_handle handle);
std::uint64_t jlc_native_matrix_bytes(jlc_matrix_handle handle);
jlc_status jlc_native_lu_factor(double* packed_lu, int n, int* pivots, int pivot_length, int* out_info);
jlc_status jlc_native_lu_factor_vendor(double* packed_lu, int n, int* pivots, int pivot_length, int* out_info);
jlc_status jlc_native_qr_factorize_only(const double* a, int m, int n);
jlc_status jlc_native_qr_factorize_only_vendor(const double* a, int m, int n);
jlc_status jlc_native_qr_decompose(const double* a, int m, int n, int q_cols,
                                   double* q, double* r);
jlc_status jlc_native_qr_decompose_vendor(const double* a, int m, int n, int q_cols,
                                          double* q, double* r);
jlc_status jlc_native_cholesky_decompose(double* packed_l, int n, int* out_info);
jlc_status jlc_native_cholesky_decompose_vendor(double* packed_l, int n, int* out_info);
jlc_status jlc_native_hessenberg_reduce_vendor(double* h, int n);
jlc_status jlc_native_hessenberg_decompose_vendor(double* h, int n, double* q);
jlc_status jlc_native_gemm(const double* a, int a_rows, int a_cols,
                           const double* b, int b_rows, int b_cols,
                           double* c, int c_rows, int c_cols,
                           double alpha, double beta,
                           int threads, int flags);
jlc_status jlc_native_gemm_strided(const double* a, int a_offset, int a_ld, int a_rows, int a_cols, int a_flags,
                                   const double* b, int b_offset, int b_ld, int b_rows, int b_cols, int b_flags,
                                   double* c, int c_offset, int c_ld, int c_rows, int c_cols, int c_flags,
                                   double alpha, double beta,
                                   int threads, int flags);
jlc_status jlc_native_gemm_strided_batched(const double* a, int a_offset, int a_ld, int a_rows, int a_cols, int a_flags, int a_stride,
                                           const double* b, int b_offset, int b_ld, int b_rows, int b_cols, int b_flags, int b_stride,
                                           double* c, int c_offset, int c_ld, int c_rows, int c_cols, int c_flags, int c_stride,
                                           double alpha, double beta,
                                           int batch_count,
                                           int threads, int flags);
