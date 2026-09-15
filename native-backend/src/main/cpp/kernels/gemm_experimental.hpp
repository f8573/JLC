#pragma once

#include "gemm_internal.hpp"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {

// Opt-in shared-A storage is separate from worker-private B scratch. No caller
// initialization: packing workers first-touch their own L3 domain's slab.
struct PackedASlab {
    double* data = nullptr;
    std::size_t capacity = 0;
    PackedASlab() = default;
    PackedASlab(const PackedASlab&) = delete;
    PackedASlab& operator=(const PackedASlab&) = delete;
    ~PackedASlab();
    void ensure(std::size_t elements);
};

struct SharedAStorage {
    struct Domain {
        std::string cache_key;
        std::vector<int> workers;
        PackedASlab slab;
    };
    std::vector<int> selected_cpus, worker_domain, worker_rank;
    std::vector<std::unique_ptr<Domain>> domains;

    void prepare(const std::vector<int>& cpus, std::size_t elements);
};

struct SharedASchedule {
    struct Barrier : CancellableBarrier {
        std::uint64_t events = 0, slabs = 0;
        bool wait(int members, bool diagnostic) {
            return barrier(members, diagnostic ? &events : nullptr);
        }
    };
    SharedAStorage* storage = nullptr;
    std::vector<std::unique_ptr<Barrier>> barriers;
    bool diagnostic = false;

    void prepare(SharedAStorage& backing, const std::vector<int>& cpus,
                 std::size_t elements, bool diagnostics);
    void cancel();
};

void* aligned_allocate(std::size_t, std::size_t);
void aligned_release(void*);
void resolve_experimental_layout(ResolvedGemmPlan&, GemmKernelOption);
bool small_k_update_gemm_enabled();
bool supports_small_k_update_gemm(const MatrixDescriptor&, const MatrixDescriptor&, const MatrixDescriptor&, int, int, int, bool);
void small_k_update_gemm(const MatrixDescriptor&, const MatrixDescriptor&, const MatrixDescriptor&, int, int, int, double, bool, ProfileAccumulator&);
void report_a_packing(const SharedASchedule&, const char*);
void process_l3_column_panels(const ThreadPoolJob&, int, int, Scratch&, ProfileAccumulator&);
} // namespace jlc_gemm
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
