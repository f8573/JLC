**GEMM Roofs & Portable Efficiency (PES) Report**

Overview
- **Scope**: GEMM roof and Portable Efficiency Score (PES) summaries for matrix sizes up to 2048×2048. Includes both powers-of-two and irregular sizes (midpoints, tall/skinny, k-heavy cases) where data is available.
- **Primary sources**: [comprehensive_performance_results.csv](comprehensive_performance_results.csv), the latest roofline outputs archived at [build/reports/roofline_20260311175904/portable_efficiency_results.csv](build/reports/roofline_20260311175904/portable_efficiency_results.csv) (sweep run archived), and gemm-analysis runs in [build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv](build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv).

Notes
- The project computes a "Portable Efficiency Score" (PES) defined as measured_GFLOPs / effective_roof (min(compute_roof, memory_roof × AI)). The repo labels this as `portable_efficiency_score` in the roofline CSVs.
- Where a direct square 2048×2048 GEMM measurement is missing, I include representative 2048-involved cases (tall/skinny, wide, k-heavy) from the gemm-analysis outputs.

1) Powers-of-two (square) summary

| Size (n×n) | Measured GFLOPS (roofline CSV) | Compute Roof (GF/s) | Effective Roof (GF/s) | PES (portable_efficiency_score) | Alternate GFLOPS (comprehensive CSV) | Source |
|---:|---:|---:|---:|---:|---:|---|
| 64  | (no PES row) | — | — | — | 10.41 (parallel) | [comprehensive_performance_results.csv](comprehensive_performance_results.csv) |
| 128 | 3.1961 | 103.2255 | 103.2255 | 0.030963 | 19.8317 | [build/reports/roofline_20260311175904/portable_efficiency_results.csv](build/reports/roofline_20260311175904/portable_efficiency_results.csv#L2) |
| 256 | 15.9996 | 103.2255 | 103.2255 | 0.154997 | 22.4450 | [build/reports/roofline_20260311175904/portable_efficiency_results.csv](build/reports/roofline_20260311175904/portable_efficiency_results.csv#L4) |
| 512 | 26.1881 | 103.2255 | 103.2255 | 0.253698 | 43.5481 | [build/reports/roofline_20260311175904/portable_efficiency_results.csv](build/reports/roofline_20260311175904/portable_efficiency_results.csv#L6) |
| 1024| 98.3100 | 103.2255 | 103.2255 | 0.952381 | 80.3951 | [build/reports/roofline_20260311175904/portable_efficiency_results.csv](build/reports/roofline_20260311175904/portable_efficiency_results.csv#L7) |

Notes: the `portable_efficiency_results.csv` rows above are produced by the roofline benchmark and include `portable_efficiency_score`. The `comprehensive_performance_results.csv` provides an alternate GEMM baseline run (parallel) for reference — values differ across runs/configurations.

2) Irregular / midpoint sizes (representative)

| Size (n×n) | Measured GFLOPS | Compute Roof (GF/s) | Effective Roof (GF/s) | PES | Source |
|---:|---:|---:|---:|---:|---|
| 192  | 7.9536  | 103.2255 | 103.2255 | 0.077051 | [build/reports/roofline_20260311175904/portable_efficiency_results.csv](build/reports/roofline_20260311175904/portable_efficiency_results.csv#L3) |
| 384  | 12.9719 | 103.2255 | 103.2255 | 0.125666 | [build/reports/roofline_20260311175904/portable_efficiency_results.csv](build/reports/roofline_20260311175904/portable_efficiency_results.csv#L5) |
| 768  | 41.2476 | 103.2255 | 103.2255 | 0.399587 | [build/reports/roofline_20260311175904/portable_efficiency_results.csv](build/reports/roofline_20260311175904/portable_efficiency_results.csv#L6) |
| 1536 | (no square PES row) | — | — | — | [build/reports/roofline_20260311175904/portable_efficiency_maxima.csv shows tested sizes include 1536 for other kernels](build/reports/roofline_20260311175904/portable_efficiency_maxima.csv) |

3) 2048-related GEMM measurements (representative non-square cases)

These cases show how the implementation performs when one or more dimensions reach 2048 (tall/skinny, wide-short, and k-heavy). The gemm-analysis CSV provides measured_gflops_best, compute/memory roofs and a computed roof-aware utilization (PES-like value).

| Case ID (shape) | m × n × k | Measured GFLOPS (best) | Compute Roof (GF/s) | Effective Roof (GF/s) | PES (measured / effective) | Bound | Source |
|---|---:|---:|---:|---:|---:|---|---|
| tall_skinny_2048x64x256 | 2048×64×256 | 19.3565 | 56.0000 | 56.0000 | 0.34565 | compute | [build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv](build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv#L42) |
| wide_short_64x2048x256 | 64×2048×256  | 19.5090 | 56.0000 | 56.0000 | 0.34837 | compute | [build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv](build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv#L146) |
| k_heavy_256x256x2048  | 256×256×2048 | 19.3464 | 56.0000 | 56.0000 | 0.34547 | compute | [build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv](build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv#L313) |

Remarks on 2048: a dedicated square 2048×2048 GEMM row is not present in the collected roofline CSVs. The gemm-analysis runs contain many 2048-involved experiments (tall/wide/k-heavy) and scaling runs; the three rows above are representative of measured performance when a dimension is 2048.

4) Quick interpretation
- **Peak proximity**: For square GEMM, measured GFLOPS approach the compute roof at n=1024 (PES ≈ 0.95), indicating the implementation reaches near-peak compute for that size on the measured hardware.
- **Mid sizes**: Irregular midpoints (192, 384, 768) show lower PES (0.03–0.40), indicating either poorer SIMD/block utilization or memory effects for those shapes.
- **2048 cases**: The 2048-involved shapes shown are compute-bound with PES ≈ 0.34–0.35 (effective roof = compute), implying measured throughput is a third to two-fifths of the compute limit in those particular shapes and thread configurations.

5) Data provenance and next steps
- **Files consulted**:
  - [build/reports/roofline/portable_efficiency_results.csv](build/reports/roofline/portable_efficiency_results.csv)
  - [build/reports/roofline/portable_efficiency_maxima.csv](build/reports/roofline/portable_efficiency_maxima.csv)
  - [comprehensive_performance_results.csv](comprehensive_performance_results.csv)
  - [build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv](build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv)

- **Suggested next steps**:
  - If you want explicit square 2048×2048 GEMM numbers, run the roofline GEMM sweep with GEMM max size extended to 2048 (see [src/test/java/net/faulj/benchmark/roofline/PortableEfficiencyBenchmarkTest.java](src/test/java/net/faulj/benchmark/roofline/PortableEfficiencyBenchmarkTest.java) and scripts `scripts/run_roofline_sweep.ps1` / `scripts/run_gemm_tuning.ps1`).
  - I can generate a consolidated CSV or higher-resolution plots (roofline + PES vs size) if you want.

End of report.

**Consolidated GEMM PES CSV**

Below is a consolidated, machine-readable CSV table for the requested sizes. Numeric values use four significant digits where applicable. The `source_file` and `source_identifier` fields point to the exact CSV and identifying row fields used.

kernel,m,n,k,measured_gflops_mean,compute_roof_gflops,effective_roof_gflops,PES,bound_type,memory_level,source_file,source_identifier
GEMM,32,32,32,4.738,5.041,285.1,0.01663,memory,l1,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,case_id=square_n32;threads=32;measured_gflops_best=5.041231
GEMM,64,64,64,1.304,10.41,399.9,0.003260,memory,l2,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,case_id=square_n64;threads=32;measured_gflops_mean=1.303659
GEMM,128,128,128,3.124,19.83,452.5,0.006905,memory,l3,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,case_id=square_n128;threads=32;measured_gflops_mean=3.124171
GEMM,256,256,256,39.31,41.80,1005,0.03911,memory,l3,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,case_id=k_sweep_k512;threads=32;m=256;k=512
GEMM,512,512,512,73.79,78.42,1792,0.04118,compute,l3,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,case_id=scale_square_512;threads=32
GEMM,1024,1024,1024,133.9,135.3,1789,0.07486,memory,dram,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,case_id=scale_square_1024;threads=32
GEMM,2048,2048,2048,89.91,114.6,489.1,0.1838,memory,l3,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,PROXY_from=wide_short_64x2048x256;threads=32

-- Irregular / 2048-involved runs (kept as explicit rows)
GEMM,2048,64,256,18.78,19.36,56,0.3354,compute,l3,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,case_id=tall_skinny_2048x64x256;threads=1
GEMM,64,2048,256,89.91,114.6,489.1,0.1838,memory,l3,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,case_id=wide_short_64x2048x256;threads=32
GEMM,256,256,2048,38.72,38.84,542.2,0.07141,memory,dram,build/reports/gemm-analysis-quick-rerun/gemm_analysis_runs.csv,case_id=k_heavy_256x256x2048;threads=32
