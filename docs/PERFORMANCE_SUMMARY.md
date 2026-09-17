# Native GEMM performance summary

This is a concise handoff summary, not a replacement for the raw benchmark
reports.

## Host and workload

- AMD Ryzen 9 3950X, 16 physical Zen 2 cores / 32 logical CPUs, AVX2/FMA.
- FP64 square GEMM, 2048³, built-in native provider, 2 warmups and 5 timed calls.
- Controlled runs used CPUs 0–15, physical worker affinity, panel scheduling,
  private A packing, intrinsic 6x8, and MC/KC/NC = 2048/256/128.

## Cleanup control

The fresh pre-cleanup and post-cleanup controls used the same build flags,
selectors, command shape, and 1/8/12/16 worker sweep:

| Requested workers | Pre-cleanup median GFLOP/s | Post-cleanup median GFLOP/s |
| ---: | ---: | ---: |
| 1 | 44.278 | 44.954 |
| 8 | 310.175 | 315.605 |
| 12 | 311.702 | 305.187 |
| 16 | 500.938 | 520.959 |

The 12-worker difference is within the observed run-to-run spread. The 16-worker
post-cleanup result is higher than the fresh control and sits inside the
preserved healthy range; there is no evidence of a cleanup-induced regression.

## Same-host reference

The preserved corrected-affinity campaign measured 583.35 GFLOP/s for AOCL-BLIS
at 2048³/16 physical workers. The post-cleanup JLC median is approximately
89.3% of that reference. This is a same-host measured comparison, not a generic
claim of parity or superiority; thermal state, build, and campaign timing can
vary.

Raw cleanup measurements and exact commands are in
[`build/reports/jlc-cleanup-pass1-20260914/`](../build/reports/jlc-cleanup-pass1-20260914/).
The exact public claim and its measurement inputs are summarized in
[`GEMM_PUBLICATION_BENCHMARK.md`](GEMM_PUBLICATION_BENCHMARK.md). The broader
machine-specific BLAS evidence is retained in the local `build/reports/`
bundle when available and is not a runtime dependency.
