#include "gemm_experimental.hpp"
#include "gemm_packing.hpp"
#include "gemm_panel.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {

bool small_k_update_gemm_enabled() {
    return parse_env_bool("JLC_NATIVE_SMALL_K_UPDATE_GEMM", false);
}

inline int selected_nr8_prefetch_distance() {
    const char* value = std::getenv("JLC_NATIVE_NR8_PREFETCH");
    if (value == nullptr || std::strcmp(value, "0") == 0) return 0;
    if (std::strcmp(value, "group4") == 0) return 5;
    if (std::strcmp(value, "1") == 0) return 1;
    if (std::strcmp(value, "2") == 0) return 2;
    if (std::strcmp(value, "4") == 0) return 4;
    return 0;
}

void resolve_experimental_layout(ResolvedGemmPlan& plan, GemmKernelOption requested) {
    if (plan.tile != ResolvedGemmPlan::Tile::M6N8) return;
    using Kernel = ResolvedGemmPlan::Kernel;
    const bool u4 = requested == GemmKernelOption::Kernel6x8U4 || requested == GemmKernelOption::Kernel6x8U4Asm;
    plan.kernel = requested == GemmKernelOption::Kernel6x8NoInline ? Kernel::NoInline
        : requested == GemmKernelOption::Kernel6x8U2 ? Kernel::U2 : u4 ? Kernel::U4 : Kernel::Default;
    const char* a = std::getenv("JLC_NATIVE_A_LAYOUT");
    if (!u4 || a == nullptr || std::strcmp(a, "kmajor6") != 0) return;
    plan.kernel = Kernel::KMajor6;
#if defined(JLC_NATIVE_HAS_KMAJOR6_ASM)
    if (requested == GemmKernelOption::Kernel6x8U4Asm) plan.kernel = Kernel::KMajor6Asm;
#endif
    const char* b = std::getenv("JLC_NATIVE_B_LAYOUT");
    if (b == nullptr || std::strcmp(b, "nr8") != 0) return;
    const char* packer = std::getenv("JLC_NATIVE_NR8_PACKER");
    plan.b_layout = packer != nullptr && std::strcmp(packer, "old") == 0
        ? ResolvedGemmPlan::BLayout::NR8Old : ResolvedGemmPlan::BLayout::NR8Fast;
    plan.nr8_prefetch = selected_nr8_prefetch_distance();
}

// Full MR panels are K-major; M tails retain the existing row format.
void pack_a_experimental(const ResolvedGemmPlan& blocks, const MatrixDescriptor& a,
                            int row_start, int rows, int col_start, int k_block,
                            double alpha, double* a_pack) {
    if (!blocks.kmajor6() || rows != 6) {
        pack_a(a, row_start, rows, col_start, k_block, alpha, a_pack);
        return;
    }
    const double* base = data_ptr(a);
    if (!a.transpose && !a.col_major) {
        for (int p = 0; p < k_block; ++p) {
            for (int r = 0; r < 6; ++r) {
                const double value = base[(row_start + r) * a.ld + col_start + p];
                a_pack[p * 6 + r] = alpha == 1.0 ? value : value * alpha;
            }
        }
        return;
    }
    for (int p = 0; p < k_block; ++p) {
        for (int r = 0; r < 6; ++r) {
            a_pack[p * 6 + r] = load_element(a, row_start + r, col_start + p) * alpha;
        }
    }
}

// Copy every complete NR=8 source row as one compiler-visible fixed-width
// operation. One call covers all complete micro-panels in an NC/KC panel.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void pack_b_nr8_contiguous_full(const double* src, int ld, int kb,
                                int full_cols, double* dst) {
    for (int j = 0; j < full_cols; j += 8) {
        const double* src_row = src + j;
        double* dst_row = dst + static_cast<std::size_t>(j) * kb;
        for (int p = 0; p < kb; ++p) {
            std::memcpy(dst_row, src_row, 8 * sizeof(double));
            src_row += ld;
            dst_row += 8;
        }
    }
}

// Each NR8 micro-panel is kb*8 doubles, ordered by ascending column tile.
void pack_b_experimental(const ResolvedGemmPlan& blocks, const MatrixDescriptor& b,
                            int row_start, int kb, int col_start, int cols,
                            int packed_cols, double* dst) {
    if (!blocks.nr8()) {
        pack_b(b, row_start, kb, col_start, cols, packed_cols, dst);
        return;
    }
    // Fast-path predicate: NR8, non-transposed row-major storage, and each
    // selected micro-panel has all eight columns. Descriptor validation makes
    // each fixed 64-byte source row safe; any N tail stays on pack_b below.
    if (blocks.b_layout == ResolvedGemmPlan::BLayout::NR8Fast && !b.transpose && !b.col_major) {
        const int full_cols = cols & ~7;
        if (full_cols != 0) {
            const double* src = data_ptr(b) +
                static_cast<std::size_t>(row_start) * b.ld + col_start;
            pack_b_nr8_contiguous_full(src, b.ld, kb, full_cols, dst);
        }
        if (full_cols != cols) {
            pack_b(b, row_start, kb, col_start + full_cols, cols - full_cols, 8,
                   dst + static_cast<std::size_t>(full_cols) * kb);
        }
        return;
    }
    for (int j = 0; j < cols; j += 8) {
        pack_b(b, row_start, kb, col_start + j, std::min(8, cols - j), 8,
               dst + static_cast<std::size_t>(j) * kb);
    }
}

#if defined(__AVX2__)
// Experimental noinline control: isolate the otherwise identical kernel from
// the large caller so its loop-invariant pointers and strides remain in registers.
__attribute__((noinline)) void microkernel_6x8_noinline_avx2(
        int k_block, const double* a_pack, const double* b_pack, int packed_n,
        double* c, int ldc) {
    microkernel_6x8_avx2(k_block, a_pack, b_pack, packed_n, c, ldc);
}

inline void microkernel_6x8_u2_avx2(int k_block, const double* a_pack,
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

    int p = 0;
    for (; p + 1 < k_block; p += 2) {
        {
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
        {
            const int p1 = p + 1;
            const __m256d b0 = _mm256_loadu_pd(b_pack + p1 * packed_n);
            const __m256d b1 = _mm256_loadu_pd(b_pack + p1 * packed_n + 4);
            __m256d a = _mm256_broadcast_sd(a_pack + p1);
            c00 = _mm256_fmadd_pd(a, b0, c00);
            c01 = _mm256_fmadd_pd(a, b1, c01);
            a = _mm256_broadcast_sd(a_pack + off1 + p1);
            c10 = _mm256_fmadd_pd(a, b0, c10);
            c11 = _mm256_fmadd_pd(a, b1, c11);
            a = _mm256_broadcast_sd(a_pack + off2 + p1);
            c20 = _mm256_fmadd_pd(a, b0, c20);
            c21 = _mm256_fmadd_pd(a, b1, c21);
            a = _mm256_broadcast_sd(a_pack + off3 + p1);
            c30 = _mm256_fmadd_pd(a, b0, c30);
            c31 = _mm256_fmadd_pd(a, b1, c31);
            a = _mm256_broadcast_sd(a_pack + off4 + p1);
            c40 = _mm256_fmadd_pd(a, b0, c40);
            c41 = _mm256_fmadd_pd(a, b1, c41);
            a = _mm256_broadcast_sd(a_pack + off5 + p1);
            c50 = _mm256_fmadd_pd(a, b0, c50);
            c51 = _mm256_fmadd_pd(a, b1, c51);
        }
    }

    if (p < k_block) {
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

// U4 scheduling experiment. The six packed-A rows advance independently, so
// the hot loop does not reconstruct row addresses from k_block every K. The
// spare sixteenth YMM register carries the low half of the following B row
// across the final FMA pair of each of the first three K steps.
__attribute__((noinline, optimize("align-loops=32")))
void microkernel_6x8_u4_avx2(int k_block, const double* a_pack,
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

    const double* a0 = a_pack;
    const double* a1 = a0 + k_block;
    const double* a2 = a1 + k_block;
    const double* a3 = a2 + k_block;
    const double* a4 = a3 + k_block;
    const double* a5 = a4 + k_block;
    const double* bp = b_pack;

#define JLC_U4_ROWS_0_TO_4(AOFF)                                              \
    do {                                                                       \
        __m256d av = _mm256_broadcast_sd(a0 + (AOFF));                         \
        c00 = _mm256_fmadd_pd(av, b0, c00);                                    \
        c01 = _mm256_fmadd_pd(av, b1, c01);                                    \
        av = _mm256_broadcast_sd(a1 + (AOFF));                                 \
        c10 = _mm256_fmadd_pd(av, b0, c10);                                    \
        c11 = _mm256_fmadd_pd(av, b1, c11);                                    \
        av = _mm256_broadcast_sd(a2 + (AOFF));                                 \
        c20 = _mm256_fmadd_pd(av, b0, c20);                                    \
        c21 = _mm256_fmadd_pd(av, b1, c21);                                    \
        av = _mm256_broadcast_sd(a3 + (AOFF));                                 \
        c30 = _mm256_fmadd_pd(av, b0, c30);                                    \
        c31 = _mm256_fmadd_pd(av, b1, c31);                                    \
        av = _mm256_broadcast_sd(a4 + (AOFF));                                 \
        c40 = _mm256_fmadd_pd(av, b0, c40);                                    \
        c41 = _mm256_fmadd_pd(av, b1, c41);                                    \
    } while (false)

#define JLC_U4_ROW_5(AOFF)                                                     \
    do {                                                                       \
        const __m256d av = _mm256_broadcast_sd(a5 + (AOFF));                   \
        c50 = _mm256_fmadd_pd(av, b0, c50);                                    \
        c51 = _mm256_fmadd_pd(av, b1, c51);                                    \
    } while (false)

    int p = 0;
    const int main_k = k_block & ~3;
    for (; p < main_k; p += 4) {
        __m256d b0 = _mm256_loadu_pd(bp);
        __m256d b1 = _mm256_loadu_pd(bp + 4);

        JLC_U4_ROWS_0_TO_4(0);
        __m256d next_b0 = _mm256_loadu_pd(bp + packed_n);
        JLC_U4_ROW_5(0);
        b1 = _mm256_loadu_pd(bp + packed_n + 4);
        b0 = next_b0;
        bp += packed_n;

        JLC_U4_ROWS_0_TO_4(1);
        next_b0 = _mm256_loadu_pd(bp + packed_n);
        JLC_U4_ROW_5(1);
        b1 = _mm256_loadu_pd(bp + packed_n + 4);
        b0 = next_b0;
        bp += packed_n;

        JLC_U4_ROWS_0_TO_4(2);
        next_b0 = _mm256_loadu_pd(bp + packed_n);
        JLC_U4_ROW_5(2);
        b1 = _mm256_loadu_pd(bp + packed_n + 4);
        b0 = next_b0;
        bp += packed_n;

        JLC_U4_ROWS_0_TO_4(3);
        JLC_U4_ROW_5(3);
        bp += packed_n;

        a0 += 4;
        a1 += 4;
        a2 += 4;
        a3 += 4;
        a4 += 4;
        a5 += 4;
    }

    for (; p < k_block; ++p) {
        const __m256d b0 = _mm256_loadu_pd(bp);
        const __m256d b1 = _mm256_loadu_pd(bp + 4);
        __m256d av = _mm256_broadcast_sd(a0++);
        c00 = _mm256_fmadd_pd(av, b0, c00);
        c01 = _mm256_fmadd_pd(av, b1, c01);
        av = _mm256_broadcast_sd(a1++);
        c10 = _mm256_fmadd_pd(av, b0, c10);
        c11 = _mm256_fmadd_pd(av, b1, c11);
        av = _mm256_broadcast_sd(a2++);
        c20 = _mm256_fmadd_pd(av, b0, c20);
        c21 = _mm256_fmadd_pd(av, b1, c21);
        av = _mm256_broadcast_sd(a3++);
        c30 = _mm256_fmadd_pd(av, b0, c30);
        c31 = _mm256_fmadd_pd(av, b1, c31);
        av = _mm256_broadcast_sd(a4++);
        c40 = _mm256_fmadd_pd(av, b0, c40);
        c41 = _mm256_fmadd_pd(av, b1, c41);
        av = _mm256_broadcast_sd(a5++);
        c50 = _mm256_fmadd_pd(av, b0, c50);
        c51 = _mm256_fmadd_pd(av, b1, c51);
        bp += packed_n;
    }

#undef JLC_U4_ROW_5
#undef JLC_U4_ROWS_0_TO_4

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

#if defined(JLC_NATIVE_HAS_KMAJOR6_ASM)
// Same full-tile contract as the intrinsic consumer below; Linux x86-64 SysV.
extern "C" __attribute__((visibility("hidden")))
void jlc_kmajor6_6x8_u4_asm(int k_block, const double* a_pack,
                         const double* b_pack, int packed_n, double* c, int ldc);
extern "C" __attribute__((visibility("hidden")))
void jlc_kmajor6_6x8_u4_nr8_asm(int k_block, const double* a_pack,
                             const double* b_micro, double* c, int ldc);
extern "C" __attribute__((visibility("hidden")))
void jlc_kmajor6_6x8_u4_nr8_pf1_asm(int k_block, const double* a_pack,
                                 const double* b_micro, double* c, int ldc);
extern "C" __attribute__((visibility("hidden")))
void jlc_kmajor6_6x8_u4_nr8_pf2_asm(int k_block, const double* a_pack,
                                 const double* b_micro, double* c, int ldc);
extern "C" __attribute__((visibility("hidden")))
void jlc_kmajor6_6x8_u4_nr8_pf4_asm(int k_block, const double* a_pack,
                                 const double* b_micro, double* c, int ldc);
extern "C" __attribute__((visibility("hidden")))
void jlc_kmajor6_6x8_u4_nr8_group4_asm(int k_block, const double* a_pack,
                                    const double* b_micro, double* c, int ldc);
#endif

// Matched U4 arithmetic with one contiguous K-major A stream.
__attribute__((noinline, optimize("align-loops=32")))
void microkernel_6x8_u4_kmajor6_avx2(int k_block, const double* a_pack,
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

    const double* ap = a_pack;
    const double* bp = b_pack;

#define JLC_KMAJOR6_U4_ROWS_0_TO_4(AOFF)                                              \
    do {                                                                       \
        __m256d av = _mm256_broadcast_sd(ap + 6 * (AOFF) + 0);                         \
        c00 = _mm256_fmadd_pd(av, b0, c00);                                    \
        c01 = _mm256_fmadd_pd(av, b1, c01);                                    \
        av = _mm256_broadcast_sd(ap + 6 * (AOFF) + 1);                                 \
        c10 = _mm256_fmadd_pd(av, b0, c10);                                    \
        c11 = _mm256_fmadd_pd(av, b1, c11);                                    \
        av = _mm256_broadcast_sd(ap + 6 * (AOFF) + 2);                                 \
        c20 = _mm256_fmadd_pd(av, b0, c20);                                    \
        c21 = _mm256_fmadd_pd(av, b1, c21);                                    \
        av = _mm256_broadcast_sd(ap + 6 * (AOFF) + 3);                                 \
        c30 = _mm256_fmadd_pd(av, b0, c30);                                    \
        c31 = _mm256_fmadd_pd(av, b1, c31);                                    \
        av = _mm256_broadcast_sd(ap + 6 * (AOFF) + 4);                                 \
        c40 = _mm256_fmadd_pd(av, b0, c40);                                    \
        c41 = _mm256_fmadd_pd(av, b1, c41);                                    \
    } while (false)

#define JLC_KMAJOR6_U4_ROW_5(AOFF)                                                     \
    do {                                                                       \
        const __m256d av = _mm256_broadcast_sd(ap + 6 * (AOFF) + 5);                   \
        c50 = _mm256_fmadd_pd(av, b0, c50);                                    \
        c51 = _mm256_fmadd_pd(av, b1, c51);                                    \
    } while (false)

    int p = 0;
    const int main_k = k_block & ~3;
    for (; p < main_k; p += 4) {
        __m256d b0 = _mm256_loadu_pd(bp);
        __m256d b1 = _mm256_loadu_pd(bp + 4);

        JLC_KMAJOR6_U4_ROWS_0_TO_4(0);
        __m256d next_b0 = _mm256_loadu_pd(bp + packed_n);
        JLC_KMAJOR6_U4_ROW_5(0);
        b1 = _mm256_loadu_pd(bp + packed_n + 4);
        b0 = next_b0;
        bp += packed_n;

        JLC_KMAJOR6_U4_ROWS_0_TO_4(1);
        next_b0 = _mm256_loadu_pd(bp + packed_n);
        JLC_KMAJOR6_U4_ROW_5(1);
        b1 = _mm256_loadu_pd(bp + packed_n + 4);
        b0 = next_b0;
        bp += packed_n;

        JLC_KMAJOR6_U4_ROWS_0_TO_4(2);
        next_b0 = _mm256_loadu_pd(bp + packed_n);
        JLC_KMAJOR6_U4_ROW_5(2);
        b1 = _mm256_loadu_pd(bp + packed_n + 4);
        b0 = next_b0;
        bp += packed_n;

        JLC_KMAJOR6_U4_ROWS_0_TO_4(3);
        JLC_KMAJOR6_U4_ROW_5(3);
        bp += packed_n;

        ap += 24;
    }

    for (; p < k_block; ++p) {
        const __m256d b0 = _mm256_loadu_pd(bp);
        const __m256d b1 = _mm256_loadu_pd(bp + 4);
        __m256d av = _mm256_broadcast_sd(ap + 0);
        c00 = _mm256_fmadd_pd(av, b0, c00);
        c01 = _mm256_fmadd_pd(av, b1, c01);
        av = _mm256_broadcast_sd(ap + 1);
        c10 = _mm256_fmadd_pd(av, b0, c10);
        c11 = _mm256_fmadd_pd(av, b1, c11);
        av = _mm256_broadcast_sd(ap + 2);
        c20 = _mm256_fmadd_pd(av, b0, c20);
        c21 = _mm256_fmadd_pd(av, b1, c21);
        av = _mm256_broadcast_sd(ap + 3);
        c30 = _mm256_fmadd_pd(av, b0, c30);
        c31 = _mm256_fmadd_pd(av, b1, c31);
        av = _mm256_broadcast_sd(ap + 4);
        c40 = _mm256_fmadd_pd(av, b0, c40);
        c41 = _mm256_fmadd_pd(av, b1, c41);
        av = _mm256_broadcast_sd(ap + 5);
        c50 = _mm256_fmadd_pd(av, b0, c50);
        c51 = _mm256_fmadd_pd(av, b1, c51);
        bp += packed_n;
        ap += 6;
    }

#undef JLC_KMAJOR6_U4_ROW_5
#undef JLC_KMAJOR6_U4_ROWS_0_TO_4

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

// N tails preserve the original scalar reduction by reconstructing one row.
// KC is bounded by 4096 in resolve_gemm_plan. This cold adapter has no heap
// allocation and is never entered by the full 6x8 hot loop.
__attribute__((noinline)) void microkernel_kmajor6_n_tail(
    int k_block, int n_block, int packed_n, const double* a_pack,
    const double* b_pack, double* c, int ldc) {
    double row[4096];
    for (int r = 0; r < 6; ++r) {
        for (int p = 0; p < k_block; ++p) row[p] = a_pack[p * 6 + r];
        microkernel_scalar(1, k_block, n_block, packed_n, row, b_pack, c + r * ldc, ldc);
    }
}

bool compute_experimental_microtile(const ResolvedGemmPlan& blocks,
        int m_block, int k_block, int n_block, int packed_n,
        const double* a_pack, const double* b_pack, double* c, int ldc) {
#if defined(__AVX2__)
    if (blocks.kmajor6() && m_block == 6) {
        if (n_block == 8) {
#if defined(JLC_NATIVE_HAS_KMAJOR6_ASM)
            if (blocks.kernel == ResolvedGemmPlan::Kernel::KMajor6Asm) {
                if (blocks.nr8()) {
                    switch (blocks.nr8_prefetch) {
                        case 5:
                            jlc_kmajor6_6x8_u4_nr8_group4_asm(k_block, a_pack, b_pack, c, ldc);
                            break;
                        case 1:
                            jlc_kmajor6_6x8_u4_nr8_pf1_asm(k_block, a_pack, b_pack, c, ldc);
                            break;
                        case 2:
                            jlc_kmajor6_6x8_u4_nr8_pf2_asm(k_block, a_pack, b_pack, c, ldc);
                            break;
                        case 4:
                            jlc_kmajor6_6x8_u4_nr8_pf4_asm(k_block, a_pack, b_pack, c, ldc);
                            break;
                        default:
                            jlc_kmajor6_6x8_u4_nr8_asm(k_block, a_pack, b_pack, c, ldc);
                            break;
                    }
                } else {
                    jlc_kmajor6_6x8_u4_asm(k_block, a_pack, b_pack, packed_n, c, ldc);
                }
            } else
#endif
            {
                microkernel_6x8_u4_kmajor6_avx2(k_block, a_pack, b_pack, packed_n, c, ldc);
            }
        } else {
            microkernel_kmajor6_n_tail(k_block, n_block, packed_n, a_pack, b_pack, c, ldc);
        }
        return true;
    }
    if (m_block == 6 && n_block == 8) {
        if (blocks.kernel == ResolvedGemmPlan::Kernel::NoInline) {
            microkernel_6x8_noinline_avx2(k_block, a_pack, b_pack, packed_n, c, ldc);
            return true;
        }
        if (blocks.kernel == ResolvedGemmPlan::Kernel::U2) {
            microkernel_6x8_u2_avx2(k_block, a_pack, b_pack, packed_n, c, ldc);
            return true;
        }
        if (blocks.kernel == ResolvedGemmPlan::Kernel::U4) {
            microkernel_6x8_u4_avx2(k_block, a_pack, b_pack, packed_n, c, ldc);
            return true;
        }
    }
#endif
    return false;
}

bool supports_small_k_update_gemm(const MatrixDescriptor& a, const MatrixDescriptor& b,
                                         const MatrixDescriptor& c, int m, int k, int n,
                                         bool enabled) {
    if (!enabled) {
        return false;
    }
    if (c.col_major || c.transpose || b.col_major || b.transpose || a.col_major) {
        return false;
    }
    if (k <= 0 || k > 64 || m < 128 || n < 32 || n > 128) {
        return false;
    }
    return true;
}

inline double load_small_k_a(const MatrixDescriptor& a, const double* base, int row, int p) {
    return a.transpose
        ? base[p * a.ld + row]
        : base[row * a.ld + p];
}

#if defined(__AVX2__)
inline void small_k_update_tile_4x4_avx2(const MatrixDescriptor& a, const double* a_base,
                                         const double* b_base, double* c_base,
                                         int row, int col, int k, double alpha,
                                         int b_ld, int c_ld) {
    __m256d c0 = _mm256_loadu_pd(c_base + row * c_ld + col);
    __m256d c1 = _mm256_loadu_pd(c_base + (row + 1) * c_ld + col);
    __m256d c2 = _mm256_loadu_pd(c_base + (row + 2) * c_ld + col);
    __m256d c3 = _mm256_loadu_pd(c_base + (row + 3) * c_ld + col);

    for (int p = 0; p < k; ++p) {
        const __m256d b_vec = _mm256_loadu_pd(b_base + p * b_ld + col);
        c0 = _mm256_fmadd_pd(_mm256_set1_pd(alpha * load_small_k_a(a, a_base, row, p)), b_vec, c0);
        c1 = _mm256_fmadd_pd(_mm256_set1_pd(alpha * load_small_k_a(a, a_base, row + 1, p)), b_vec, c1);
        c2 = _mm256_fmadd_pd(_mm256_set1_pd(alpha * load_small_k_a(a, a_base, row + 2, p)), b_vec, c2);
        c3 = _mm256_fmadd_pd(_mm256_set1_pd(alpha * load_small_k_a(a, a_base, row + 3, p)), b_vec, c3);
    }

    _mm256_storeu_pd(c_base + row * c_ld + col, c0);
    _mm256_storeu_pd(c_base + (row + 1) * c_ld + col, c1);
    _mm256_storeu_pd(c_base + (row + 2) * c_ld + col, c2);
    _mm256_storeu_pd(c_base + (row + 3) * c_ld + col, c3);
}
#endif

#if defined(__AVX512F__)
inline void small_k_update_tile_4x8_avx512(const MatrixDescriptor& a, const double* a_base,
                                           const double* b_base, double* c_base,
                                           int row, int col, int k, double alpha,
                                           int b_ld, int c_ld) {
    __m512d c0 = _mm512_loadu_pd(c_base + row * c_ld + col);
    __m512d c1 = _mm512_loadu_pd(c_base + (row + 1) * c_ld + col);
    __m512d c2 = _mm512_loadu_pd(c_base + (row + 2) * c_ld + col);
    __m512d c3 = _mm512_loadu_pd(c_base + (row + 3) * c_ld + col);

    for (int p = 0; p < k; ++p) {
        const __m512d b_vec = _mm512_loadu_pd(b_base + p * b_ld + col);
        c0 = _mm512_fmadd_pd(_mm512_set1_pd(alpha * load_small_k_a(a, a_base, row, p)), b_vec, c0);
        c1 = _mm512_fmadd_pd(_mm512_set1_pd(alpha * load_small_k_a(a, a_base, row + 1, p)), b_vec, c1);
        c2 = _mm512_fmadd_pd(_mm512_set1_pd(alpha * load_small_k_a(a, a_base, row + 2, p)), b_vec, c2);
        c3 = _mm512_fmadd_pd(_mm512_set1_pd(alpha * load_small_k_a(a, a_base, row + 3, p)), b_vec, c3);
    }

    _mm512_storeu_pd(c_base + row * c_ld + col, c0);
    _mm512_storeu_pd(c_base + (row + 1) * c_ld + col, c1);
    _mm512_storeu_pd(c_base + (row + 2) * c_ld + col, c2);
    _mm512_storeu_pd(c_base + (row + 3) * c_ld + col, c3);
}
#endif

void small_k_update_gemm(const MatrixDescriptor& a, const MatrixDescriptor& b, const MatrixDescriptor& c,
                         int m, int k, int n, double alpha, bool profile_enabled,
                         ProfileAccumulator& profile) {
    const double* a_base = data_ptr(a);
    const double* b_base = data_ptr(b);
    double* c_base = mutable_data_ptr(c);
    const ProfileClock::time_point kernel_start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};

    int row = 0;
    for (; row + 4 <= m; row += 4) {
        int col = 0;
#if defined(__AVX512F__)
        for (; col + 8 <= n; col += 8) {
            small_k_update_tile_4x8_avx512(a, a_base, b_base, c_base, row, col, k, alpha, b.ld, c.ld);
            if (profile_enabled) {
                ++profile.microtile_calls;
            }
        }
#elif defined(__AVX2__)
        for (; col + 4 <= n; col += 4) {
            small_k_update_tile_4x4_avx2(a, a_base, b_base, c_base, row, col, k, alpha, b.ld, c.ld);
            if (profile_enabled) {
                ++profile.microtile_calls;
            }
        }
#endif
        for (; col < n; ++col) {
            for (int r = 0; r < 4; ++r) {
                double sum = 0.0;
                for (int p = 0; p < k; ++p) {
                    sum = std::fma(load_small_k_a(a, a_base, row + r, p), b_base[p * b.ld + col], sum);
                }
                c_base[(row + r) * c.ld + col] = std::fma(alpha, sum, c_base[(row + r) * c.ld + col]);
            }
            if (profile_enabled) {
                ++profile.microtile_calls;
            }
        }
    }

    for (; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                sum = std::fma(load_small_k_a(a, a_base, row, p), b_base[p * b.ld + col], sum);
            }
            c_base[row * c.ld + col] = std::fma(alpha, sum, c_base[row * c.ld + col]);
        }
    }

    if (profile_enabled) {
        profile.kernel_ns += elapsed_ns(kernel_start);
    }
}

std::string shared_a_cache_key(int cpu) {
#if defined(__linux__)
    const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache";
    std::error_code error;
    std::filesystem::directory_iterator entry(path, error), end;
    while (!error && entry != end) {
        int level = 0;
        std::string type, shared;
        std::ifstream(entry->path() / "level") >> level;
        std::ifstream(entry->path() / "type") >> type;
        if (level == 3 && type == "Unified") {
            std::ifstream(entry->path() / "shared_cpu_list") >> shared;
            if (!shared.empty()) return shared;
        }
        entry.increment(error);
    }
#endif
    // Missing topology never combines unrelated CPUs into an assumed domain.
    return "cpu:" + std::to_string(cpu);
}

PackedASlab::~PackedASlab() { aligned_release(data); }

void PackedASlab::ensure(std::size_t elements) {
        if (elements <= capacity) return;
#if defined(JLC_NATIVE_TEST_A_PACKING_FAULT_INJECTION)
        const char* fault = std::getenv("JLC_NATIVE_TEST_A_FAIL");
        if (fault != nullptr && std::strcmp(fault, "allocate") == 0) throw std::bad_alloc();
#endif
        auto* next = static_cast<double*>(aligned_allocate(64, elements * sizeof(double)));
        if (next == nullptr) throw std::bad_alloc();
        aligned_release(data);
        data = next;
        capacity = elements;
    }

void SharedAStorage::prepare(const std::vector<int>& cpus, std::size_t elements) {
        if (selected_cpus != cpus) {
            SharedAStorage next;
            next.selected_cpus = cpus;
            for (std::size_t w = 0; w < cpus.size(); ++w) {
                const std::string key = shared_a_cache_key(cpus[w]);
                std::size_t d = 0;
                while (d < next.domains.size() && next.domains[d]->cache_key != key) ++d;
                if (d == next.domains.size()) {
                    auto domain = std::make_unique<Domain>();
                    domain->cache_key = key;
                    next.domains.push_back(std::move(domain));
                }
                next.worker_domain.push_back(static_cast<int>(d));
                next.worker_rank.push_back(static_cast<int>(next.domains[d]->workers.size()));
                next.domains[d]->workers.push_back(static_cast<int>(w));
            }
            *this = std::move(next);
        }
        for (auto& domain : domains)
            if (domain->workers.size() > 1) domain->slab.ensure(elements);
    }

void SharedASchedule::prepare(SharedAStorage& backing, const std::vector<int>& cpus,
                 std::size_t elements, bool diagnostics) {
        backing.prepare(cpus, elements);
        storage = &backing;
        diagnostic = diagnostics;
        for (std::size_t d = 0; d < backing.domains.size(); ++d)
            barriers.push_back(std::make_unique<Barrier>());
    }

void SharedASchedule::cancel() {
        for (auto& barrier : barriers) barrier->cancel();
    }

void report_a_packing(const SharedASchedule& shared, const char* fallback) {
    if (shared.storage == nullptr) {
        std::fprintf(stderr, "JLC_A_PACKING effective=private reason=%s copies=0 slab_publications=0 barrier_events=0 worker_arrivals=0\n", fallback);
        return;
    }
    std::uint64_t copies = 0, slabs = 0, events = 0, arrivals = 0;
    for (std::size_t d = 0; d < shared.storage->domains.size(); ++d) {
        const auto& domain = *shared.storage->domains[d];
        const auto& barrier = *shared.barriers[d];
        if (domain.workers.size() <= 1) continue;
        ++copies;
        slabs += barrier.slabs;
        events += barrier.events;
        arrivals += barrier.events * domain.workers.size();
        std::fprintf(stderr, "JLC_A_DOMAIN domain=%zu shared_cpus=%s consumers=%zu capacity_bytes=%zu slab_publications=%llu barrier_events=%llu workers=",
            d, domain.cache_key.c_str(), domain.workers.size(), domain.slab.capacity * sizeof(double),
            static_cast<unsigned long long>(barrier.slabs), static_cast<unsigned long long>(barrier.events));
        for (std::size_t rank = 0; rank < domain.workers.size(); ++rank)
            std::fprintf(stderr, "%s%d", rank ? "," : "", domain.workers[rank]);
        std::fprintf(stderr, "\n");
    }
    std::fprintf(stderr, "JLC_A_PACKING effective=%s copies=%llu slab_publications=%llu barrier_events=%llu worker_arrivals=%llu\n",
        copies ? "l3" : "private", static_cast<unsigned long long>(copies),
        static_cast<unsigned long long>(slabs), static_cast<unsigned long long>(events),
        static_cast<unsigned long long>(arrivals));
}

// Keep each worker's N panels and ascending K-block order. Only the A lifetime
// changes: a domain cooperatively writes one MC*kb slab, then reads it immutably.
void process_l3_column_panels(const ThreadPoolJob& job, int worker, int full_n,
                             Scratch& scratch, ProfileAccumulator& profile) {
    auto& shared = *job.shared_a;
    const auto& storage = *shared.storage;
    const int domain_id = storage.worker_domain[worker];
    const auto& domain = *storage.domains[domain_id];
    const int members = static_cast<int>(domain.workers.size());
    const auto& blocks = job.blocks;
    if (members == 1) {
        process_column_panels(*job.a, *job.b, *job.c, job.m, job.k, full_n, job.alpha,
                              blocks, worker * blocks.nc, job.panel_stride,
                              scratch, job.profile_enabled, profile);
        return;
    }
    const int rank = storage.worker_rank[worker];
    auto& barrier = *shared.barriers[domain_id];
    double* c_base = mutable_data_ptr(*job.c);
    const bool profile_enabled = job.profile_enabled;
    for (int wave = 0; wave < full_n; wave += job.panel_stride) {
        // All domain members take the same path even in an incomplete N wave.
        if (wave + domain.workers.front() * blocks.nc >= full_n) continue;
        const int jj = wave + worker * blocks.nc;
        const int n_panel = std::max(0, std::min(blocks.nc, full_n - jj));
        const int packed_n = round_up(n_panel, blocks.nr);
        for (int kk = 0; kk < job.k; kk += blocks.kc) {
            const int kb = std::min(blocks.kc, job.k - kk);
            double* b_pack = nullptr;
            if (n_panel > 0) {
                b_pack = scratch.ensure_b(packed_elements(kb, packed_n));
                ScopedProfileTimer timer(profile_enabled, profile.pack_b_ns);
                pack_b_selected(blocks, *job.b, kk, kb, jj, n_panel, packed_n, b_pack);
                if (profile_enabled) {
                    ++profile.pack_b_calls;
                    profile.pack_b_bytes += static_cast<std::uint64_t>(kb) * packed_n * sizeof(double);
                }
            }
            for (int ii = 0; ii < job.m; ii += blocks.mc) {
                const int rows = std::min(blocks.mc, job.m - ii);
                const int tiles = (rows + blocks.mr - 1) / blocks.mr;
                const int first = static_cast<int>(static_cast<long long>(tiles) * rank / members);
                const int last = static_cast<int>(static_cast<long long>(tiles) * (rank + 1) / members);
#if defined(JLC_NATIVE_TEST_A_PACKING_FAULT_INJECTION)
                const char* fault = std::getenv("JLC_NATIVE_TEST_A_FAIL");
                if (worker == 1 && wave == 0 && kk == 0 && ii == 0 && fault != nullptr
                    && std::strcmp(fault, "pack") == 0) throw std::bad_alloc();
#endif
                {
                    ScopedProfileTimer timer(profile_enabled, profile.pack_a_ns);
                    for (int tile = first; tile < last; ++tile) {
                        const int row = tile * blocks.mr;
                        const int mb = std::min(blocks.mr, rows - row);
                        pack_a_selected(blocks, *job.a, ii + row, mb, kk, kb, job.alpha,
                               domain.slab.data + static_cast<std::size_t>(row) * kb);
                        if (profile_enabled) {
                            ++profile.pack_a_calls;
                            profile.pack_a_bytes += static_cast<std::uint64_t>(mb) * kb * sizeof(double);
                        }
                    }
                }
                if (!barrier.wait(members, shared.diagnostic)) return; // publish A
                if (shared.diagnostic && rank == 0) ++barrier.slabs;
#if defined(JLC_NATIVE_TEST_A_PACKING_FAULT_INJECTION)
                if (worker == 1 && wave == 0 && kk == 0 && ii == 0 && fault != nullptr
                    && std::strcmp(fault, "consume") == 0) throw std::runtime_error("injected A consumer failure");
#endif
                if (n_panel > 0) {
                    for (int row = 0; row < rows; row += blocks.mr) {
                        const int mb = std::min(blocks.mr, rows - row);
                        const double* a_pack = domain.slab.data + static_cast<std::size_t>(row) * kb;
                        compute_packed_row(blocks, mb, kb, n_panel, packed_n, a_pack, b_pack,
                            c_base + (ii + row) * job.c->ld + jj, job.c->ld, profile_enabled, profile);
                    }
                }
                if (!barrier.wait(members, shared.diagnostic)) return; // release A before overwrite
            }
        }
    }
}

} // namespace jlc_gemm
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
