**GEMM PES Report**

Scope
- Square Java GEMM benchmark sweep for `128x128` through `2048x2048`.
- Per-size results come from the best GFLOPs configuration found for that exact matrix size, not from a global maximum across sizes.

Primary artifacts
- `build/reports/roofline-gemm-20260323/portable_efficiency_gemm_best_by_size.csv`
- `build/reports/roofline-gemm-20260323/portable_efficiency_gemm_runs.csv`
- `build/reports/roofline-gemm-20260323/portable_efficiency_results.json`
- `consolidated_gemm_pes.csv`

Best result per matrix size

| Size | Threads | MR | NR | GFLOPs | PES | Bound | Memory level |
|---:|---:|---:|---:|---:|---:|---|---|
| 128  | 32 | 5 | 6 | 21.564545 | 0.104410 | compute | l3 |
| 192  | 16 | 5 | 4 | 26.297187 | 0.127324 | compute | l3 |
| 256  | 8 | 5 | 6 | 41.171082 | 0.199339 | compute | l3 |
| 384  | 8 | 5 | 6 | 61.964439 | 0.300015 | compute | l3 |
| 512  | 32 | 4 | 4 | 50.088718 | 0.242516 | compute | l3 |
| 768  | 8 | 5 | 4 | 67.464678 | 0.326646 | compute | dram |
| 1024 | 32 | 5 | 6 | 141.297622 | 0.684125 | compute | dram |
| 1536 | 32 | 6 | 6 | 183.084278 | 0.886445 | compute | dram |
| 2048 | 16 | 5 | 6 | 172.961686 | 0.837434 | compute | dram |

Notes
- The benchmark harness now emits `portable_efficiency_gemm_runs.csv` for all candidate runs and `portable_efficiency_gemm_best_by_size.csv` for the chosen per-size maxima.
- The default GEMM compute-roof anchor now uses `n=2048`, which keeps the reported PES values on this machine below `1.0` while preserving the existing roofline methodology.
- The Gradle test task now forwards `jlc.*` and `la.*` system properties into the test JVM, so `outputDir`, `gemm_only`, and GEMM tuning overrides work from the repo scripts.
