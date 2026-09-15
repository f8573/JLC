#pragma once

#include "gemm_packing.hpp"
#include "gemm_compute.hpp"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {

// C already contains beta*C; packed A already contains alpha*A. The same
// column/tail contract serves private panels, shared tails and experimental A.
inline void compute_packed_row(const ResolvedGemmPlan& blocks, int m_block, int k_block,
                               int n_panel, int packed_n, const double* a_pack,
                               const double* b_pack, double* c_row, int ldc,
                               bool profile_enabled, ProfileAccumulator& profile) {
    const auto start = profile_enabled ? ProfileClock::now() : ProfileClock::time_point{};
    int j = 0;
    for (; j + blocks.nr <= n_panel; j += blocks.nr) {
        compute_microtile(blocks, m_block, k_block, blocks.nr, packed_n,
                          a_pack, b_micro_panel(blocks, b_pack, j, k_block), c_row + j, ldc);
        if (profile_enabled) ++profile.microtile_calls;
    }
    if (j < n_panel) {
        compute_microtile(blocks, m_block, k_block, n_panel - j, packed_n,
                          a_pack, b_micro_panel(blocks, b_pack, j, k_block), c_row + j, ldc);
        if (profile_enabled) ++profile.microtile_calls;
    }
    if (profile_enabled) profile.kernel_ns += elapsed_ns(start);
}

// Shared tails retain the original MC/MR tile boundaries and reduction order.
inline void process_shared_private_rows(const MatrixDescriptor& a, const MatrixDescriptor& c,
                                 int m, int jj, int kk, int k_block, int n_panel, int packed_n,
                                 double alpha, const ResolvedGemmPlan& blocks, const double* b_pack,
                                 int first_tile, int last_tile, Scratch& scratch,
                                 bool profile_enabled, ProfileAccumulator& profile) {
    double* c_base = mutable_data_ptr(c);
    for (int ii = 0; ii < m; ii += blocks.mc) {
        const int row_end = std::min(ii + blocks.mc, m);
        for (int i = ii; i < row_end; i += blocks.mr) {
            {
                const int tile_index = (ii / blocks.mc) * ((blocks.mc + blocks.mr - 1) / blocks.mr)
                    + (i - ii) / blocks.mr;
                if (tile_index < first_tile || tile_index >= last_tile) continue;
            }
            const int m_block = std::min(blocks.mr, row_end - i);
            double* a_pack = scratch.ensure_a(packed_elements(m_block, k_block));
            {
                ScopedProfileTimer timer(profile_enabled, profile.pack_a_ns);
                pack_a_selected(blocks, a, i, m_block, kk, k_block, alpha, a_pack);
            }
            if (profile_enabled) {
                ++profile.pack_a_calls;
                profile.pack_a_bytes += packed_elements(m_block, k_block) * sizeof(double);
            }
            compute_packed_row(blocks, m_block, k_block, n_panel, packed_n, a_pack, b_pack,
                               c_base + i * c.ld + jj, c.ld, profile_enabled, profile);
        }
    }
}

} // namespace jlc_gemm
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
