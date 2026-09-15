#pragma once

#include "gemm_internal.hpp"
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {

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

inline void microkernel_5x8_avx2(int k_block, const double* a_pack,
                                 const double* b_pack, int packed_n,
                                 double* c, int ldc) {
    __m256d c00 = _mm256_loadu_pd(c);
    __m256d c01 = _mm256_loadu_pd(c + 4);
    __m256d c10 = _mm256_loadu_pd(c + ldc);
    __m256d c11 = _mm256_loadu_pd(c + ldc + 4);
    __m256d c20 = _mm256_loadu_pd(c + 2 * ldc);
    __m256d c21 = _mm256_loadu_pd(c + 2 * ldc + 4);
    __m256d c30 = _mm256_loadu_pd(c + 3 * ldc);
    __m256d c31 = _mm256_loadu_pd(c + 3 * ldc + 4);
    __m256d c40 = _mm256_loadu_pd(c + 4 * ldc);
    __m256d c41 = _mm256_loadu_pd(c + 4 * ldc + 4);

    const int off1 = k_block;
    const int off2 = 2 * k_block;
    const int off3 = 3 * k_block;
    const int off4 = 4 * k_block;

    for (int p = 0; p < k_block; ++p) {
        const __m256d b0 = _mm256_loadu_pd(b_pack + p * packed_n);
        const __m256d b1 = _mm256_loadu_pd(b_pack + p * packed_n + 4);
        __m256d a = _mm256_broadcast_sd(a_pack + p);
        c00 = _mm256_fmadd_pd(a, b0, c00);
        c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(a_pack + off1 + p);
        c10 = _mm256_fmadd_pd(a, b0, c10);
        c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(a_pack + off2 + p);
        c20 = _mm256_fmadd_pd(a, b0, c20);
        c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(a_pack + off3 + p);
        c30 = _mm256_fmadd_pd(a, b0, c30);
        c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(a_pack + off4 + p);
        c40 = _mm256_fmadd_pd(a, b0, c40);
        c41 = _mm256_fmadd_pd(a, b1, c41);
    }

    _mm256_storeu_pd(c, c00);
    _mm256_storeu_pd(c + 4, c01);
    _mm256_storeu_pd(c + ldc, c10);
    _mm256_storeu_pd(c + ldc + 4, c11);
    _mm256_storeu_pd(c + 2 * ldc, c20);
    _mm256_storeu_pd(c + 2 * ldc + 4, c21);
    _mm256_storeu_pd(c + 3 * ldc, c30);
    _mm256_storeu_pd(c + 3 * ldc + 4, c31);
    _mm256_storeu_pd(c + 4 * ldc, c40);
    _mm256_storeu_pd(c + 4 * ldc + 4, c41);
}

inline void microkernel_6x8_avx2(int k_block, const double* a_pack,
                                 const double* b_pack, int packed_n,
                                 double* c, int ldc) {
    __m256d c00 = _mm256_loadu_pd(c);
    __m256d c01 = _mm256_loadu_pd(c + 4);
    __m256d c10 = _mm256_loadu_pd(c + ldc);
    __m256d c11 = _mm256_loadu_pd(c + ldc + 4);
    __m256d c20 = _mm256_loadu_pd(c + 2 * ldc);
    __m256d c21 = _mm256_loadu_pd(c + 2 * ldc + 4);
    __m256d c30 = _mm256_loadu_pd(c + 3 * ldc);
    __m256d c31 = _mm256_loadu_pd(c + 3 * ldc + 4);
    __m256d c40 = _mm256_loadu_pd(c + 4 * ldc);
    __m256d c41 = _mm256_loadu_pd(c + 4 * ldc + 4);
    __m256d c50 = _mm256_loadu_pd(c + 5 * ldc);
    __m256d c51 = _mm256_loadu_pd(c + 5 * ldc + 4);

    const int off1 = k_block;
    const int off2 = 2 * k_block;
    const int off3 = 3 * k_block;
    const int off4 = 4 * k_block;
    const int off5 = 5 * k_block;

    for (int p = 0; p < k_block; ++p) {
        const __m256d b0 = _mm256_loadu_pd(b_pack + p * packed_n);
        const __m256d b1 = _mm256_loadu_pd(b_pack + p * packed_n + 4);
        __m256d a = _mm256_broadcast_sd(a_pack + p);
        c00 = _mm256_fmadd_pd(a, b0, c00);
        c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(a_pack + off1 + p);
        c10 = _mm256_fmadd_pd(a, b0, c10);
        c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(a_pack + off2 + p);
        c20 = _mm256_fmadd_pd(a, b0, c20);
        c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(a_pack + off3 + p);
        c30 = _mm256_fmadd_pd(a, b0, c30);
        c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(a_pack + off4 + p);
        c40 = _mm256_fmadd_pd(a, b0, c40);
        c41 = _mm256_fmadd_pd(a, b1, c41);
        a = _mm256_broadcast_sd(a_pack + off5 + p);
        c50 = _mm256_fmadd_pd(a, b0, c50);
        c51 = _mm256_fmadd_pd(a, b1, c51);
    }

    _mm256_storeu_pd(c, c00);
    _mm256_storeu_pd(c + 4, c01);
    _mm256_storeu_pd(c + ldc, c10);
    _mm256_storeu_pd(c + ldc + 4, c11);
    _mm256_storeu_pd(c + 2 * ldc, c20);
    _mm256_storeu_pd(c + 2 * ldc + 4, c21);
    _mm256_storeu_pd(c + 3 * ldc, c30);
    _mm256_storeu_pd(c + 3 * ldc + 4, c31);
    _mm256_storeu_pd(c + 4 * ldc, c40);
    _mm256_storeu_pd(c + 4 * ldc + 4, c41);
    _mm256_storeu_pd(c + 5 * ldc, c50);
    _mm256_storeu_pd(c + 5 * ldc + 4, c51);
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

#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
bool compute_experimental_microtile(const ResolvedGemmPlan&, int, int, int, int,
                                     const double*, const double*, double*, int);
#endif
inline void compute_microtile(const ResolvedGemmPlan& blocks,
                              int m_block, int k_block, int n_block, int packed_n,
                              const double* a_pack, const double* b_pack,
                              double* c, int ldc) {
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    if (blocks.nr8()) packed_n = 8;
    if (blocks.kernel != ResolvedGemmPlan::Kernel::Default
        && compute_experimental_microtile(blocks, m_block, k_block, n_block,
                                           packed_n, a_pack, b_pack, c, ldc)) return;
#endif
#if defined(__AVX512F__)
    if (blocks.tile == ResolvedGemmPlan::Tile::M6N8 && m_block == 6 && n_block == 8) {
        microkernel_6x8_avx512(k_block, a_pack, b_pack, packed_n, c, ldc);
        return;
    }
#endif
#if defined(__AVX2__)
    if (blocks.tile == ResolvedGemmPlan::Tile::M5N8 && m_block == 5 && n_block == 8) {
        microkernel_5x8_avx2(k_block, a_pack, b_pack, packed_n, c, ldc);
        return;
    }
    if (blocks.tile == ResolvedGemmPlan::Tile::M6N8 && m_block == 6 && n_block == 8) {
        microkernel_6x8_avx2(k_block, a_pack, b_pack, packed_n, c, ldc);
        return;
    }
    if (blocks.tile == ResolvedGemmPlan::Tile::M5N4 && m_block == 5 && n_block == 4) {
        microkernel_5x4_avx2(k_block, a_pack, b_pack, packed_n, c, ldc);
        return;
    }
#endif
    microkernel_scalar(m_block, k_block, n_block, packed_n, a_pack, b_pack, c, ldc);
}

} // namespace jlc_gemm
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
