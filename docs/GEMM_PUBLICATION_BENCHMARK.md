# GEMM publication benchmark

The current public throughput statement is a same-host comparison, not a
portable performance guarantee:

> At 2048×2048 FP64 GEMM with 16 physical workers, the built-in JLC native
> backend measured 520.959 GFLOP/s versus 583.35 GFLOP/s for AOCL-BLIS, or
> 89.3% of that BLIS reference.

| Item | Recorded value |
|---|---|
| CPU | AMD Ryzen 9 3950X, 16 physical cores / 32 logical CPUs, AVX2/FMA |
| OS | Ubuntu 26.04.1 LTS, Linux 7.0.0-31-generic, x86-64 |
| Workload | square 2048×2048×2048 FP64 GEMM |
| Threads | 16 physical workers; affinity restricted to CPUs 0–15 |
| JLC implementation | built-in C++ native provider through the JLC runtime/JNI path |
| BLIS implementation | AOCL-BLIS, 16 physical workers |
| Warmups | 2 |
| Timed calls | 5 |
| Statistic | median GFLOP/s |
| JLC | 520.959 GFLOP/s |
| AOCL-BLIS | 583.35 GFLOP/s |
| JLC / BLIS | 89.3% |

The comparison is the post-cleanup control recorded in
[`PERFORMANCE_SUMMARY.md`](PERFORMANCE_SUMMARY.md). It preserves the selected
build flags, affinity policy, panel schedule, and provider configuration; it
does not claim that JLC is vendor-competitive for every size or thread count.
The raw machine-specific reports live under `build/reports/` when the local
benchmark evidence bundle is present and are intentionally not required
runtime inputs or committed build outputs.
