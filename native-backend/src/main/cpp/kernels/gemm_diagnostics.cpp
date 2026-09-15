#include "gemm_internal.hpp"
#include <cstdio>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif
namespace jlc_gemm {

void report_scheduler_work(const std::vector<SchedulerWork>& work, bool shared) {
    std::uint64_t max_elapsed = 0;
    for (const auto& item : work) max_elapsed = std::max(max_elapsed, item.elapsed_ns);
    for (std::size_t w = 0; w < work.size(); ++w) {
        const auto& s = work[w];
        std::fprintf(stderr, "JLC_SCHEDULER mode=%s worker=%zu tasks=%llu rows=%llu microtiles=%llu panels=%llu flops=%llu elapsed_ns=%llu group_wait_ns=%llu finish_gap_ns=%llu\n",
            shared ? "shared-tail" : "panel", w,
            static_cast<unsigned long long>(s.tasks), static_cast<unsigned long long>(s.rows),
            static_cast<unsigned long long>(s.microtiles), static_cast<unsigned long long>(s.panels),
            static_cast<unsigned long long>(s.flops), static_cast<unsigned long long>(s.elapsed_ns),
            static_cast<unsigned long long>(s.wait_ns), static_cast<unsigned long long>(max_elapsed - s.elapsed_ns));
    }
}

void report_worker_affinity(const std::vector<WorkerAffinityRecord>& records) {
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        std::fprintf(stderr,
            "JLC_WORKER_AFFINITY workers=%zu worker=%zu tid=%ld intended_cpu=%d affinity_mask=%s observed_cpu=%d status=%d\n",
            records.size(), index, record.tid, record.intended_cpu,
            record.affinity_mask.c_str(), record.observed_cpu, record.status);
    }
}

} // namespace jlc_gemm
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
