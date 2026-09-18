# GEMM Performance Analysis System

This repository includes a dedicated GEMM performance analysis harness focused on
roofline placement, cache-regime transitions, SIMD behavior, and thread scaling.

## Entry Points

- Benchmark harness: `src/test/java/net/faulj/benchmark/roofline/GemmKernelAnalysisBenchmarkTest.java`
- Run script: `scripts/run_gemm_kernel_analysis.ps1`
- Plot script: `tools/gemm_analysis/plot_gemm_analysis.py`

## Run

```powershell
.\scripts\run_gemm_kernel_analysis.ps1 -Profile standard
```

Profiles:

- `quick`: short smoke/perf sanity sweep
- `standard`: default engineering workflow
- `full`: extended sweep for deeper regression analysis

## Output Artifacts

Generated under `build/reports/gemm-analysis`:

- `gemm_analysis_runs.csv`
  - per-observation metrics: geometry, arithmetic intensity, cache regime,
    roofline utilization, SIMD utilization proxies, scaling efficiency, stability
- `gemm_analysis_summary.json`
  - hardware roofline, aggregate results, cache transitions, optimization opportunities
- `gemm_analysis_suite.json`
  - benchmark suite manifest (cases and thread sweeps)
- `gemm_analysis_schema.json`
  - schema of metrics emitted per run
- `PERFORMANCE.md`
  - narrative summary suitable for review packets and hiring loops
- `opportunities.md`
  - prioritized optimization opportunities derived from observed bottlenecks
- Plots (from Python script):
  - `roofline.png`
  - `scaling.png`
  - `cache_transitions.png`
  - `simd_utilization.png`
  - `stability.png`

## Existing Low-Level Tuning Knobs

- `la.gemm.mr`
- `la.gemm.nr`
- `la.gemm.kunroll`
- `la.gemm.parallelThreshold`

These remain valid and can be applied while using the analysis harness.

## Hardware Counter Notes

The schema includes fields for cache miss rates and instruction metrics:

- `l1_miss_rate`, `l2_miss_rate`, `l3_miss_rate`
- `instructions_per_cycle`
- `simd_instruction_ratio`

Default runs mark these as `NaN` with `counter_source=not-collected` unless
external PMU tooling is integrated for the target platform.
