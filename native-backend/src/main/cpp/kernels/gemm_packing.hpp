#pragma once

#include "gemm_internal.hpp"
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {

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

// Representation is resolved once; only actual M/N tails remain dynamic.
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
void pack_a_experimental(const ResolvedGemmPlan&, const MatrixDescriptor&, int, int, int, int, double, double*);
void pack_b_experimental(const ResolvedGemmPlan&, const MatrixDescriptor&, int, int, int, int, int, double*);
#endif
inline void pack_a_selected(const ResolvedGemmPlan& plan, const MatrixDescriptor& a,
                            int row, int rows, int col, int kb, double alpha, double* dst) {
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    if (plan.kmajor6() && rows == 6) {
        pack_a_experimental(plan, a, row, rows, col, kb, alpha, dst);
        return;
    }
#endif
    pack_a(a, row, rows, col, kb, alpha, dst);
}
inline void pack_b_selected(const ResolvedGemmPlan& plan, const MatrixDescriptor& b,
                            int row, int kb, int col, int cols, int packed_cols, double* dst) {
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    if (plan.nr8()) {
        pack_b_experimental(plan, b, row, kb, col, cols, packed_cols, dst);
        return;
    }
#endif
    pack_b(b, row, kb, col, cols, packed_cols, dst);
}
inline const double* b_micro_panel(const ResolvedGemmPlan& plan, const double* b, int j, int kb) {
#if defined(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    if (plan.nr8()) return b + static_cast<std::size_t>(j) * kb;
#endif
    return b + j;
}

} // namespace jlc_gemm
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
