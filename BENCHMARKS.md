# GEMM Benchmarks

This repository includes a repeatable GEMM backend comparison harness for the current production GEMM surfaces:

- `BLAS3Kernels.gemm`
- `OptimizedBLAS3.gemm`
- `CudaGemm.gemm` when CUDA is genuinely usable

The harness lives in test sources so it can use the independent `GemmReference` oracle and package-private CUDA helpers without changing production behavior.

## How To Run

Default run on Linux/macOS:

```bash
./gradlew runGemmBackendComparison
```

Default run on Windows PowerShell:

```powershell
.\gradlew.bat --% runGemmBackendComparison
```

Quick run on Linux/macOS:

```bash
./gradlew runGemmBackendComparison -Dgemm.quick=true
```

Quick run on Windows PowerShell:

```powershell
.\gradlew.bat --% runGemmBackendComparison -Dgemm.quick=true
```

Recommended verification command on Windows PowerShell:

```powershell
.\gradlew.bat --% runGemmBackendComparison -Dgemm.quick=true --no-daemon
```

## Quick Mode

`-Dgemm.quick=true` sets:

- sizes: `128,256`
- warmups: `2`
- iterations: `5`

This is intended for a fast developer-machine smoke run, not for publishing stable benchmark numbers.

## Properties

| Property | Default | Meaning |
| --- | --- | --- |
| `gemm.quick` | `false` | Enables quick mode defaults |
| `gemm.sizes` | `128,256,512` | Comma-separated square GEMM sizes |
| `gemm.warmups` | `8` | Warmup iterations before measurement |
| `gemm.iters` | `25` | Measured iterations |
| `gemm.backends` | `blas3-naive,blas3-simd-1t,blas3-parallel,opt-1t,opt-parallel,cuda` | Backend subset and order |
| `gemm.seed` | `20260604` | Deterministic data seed |
| `gemm.naiveMaxSize` | `512` | Maximum size for `blas3-naive` timing |
| `gemm.independentOracleMaxSize` | `512` | Maximum size for `GemmReference` oracle |
| `gemm.cuda` | `auto` | `auto`, `on`, or `off` |

## Backend Labels

- `blas3-naive`: `BLAS3Kernels.gemm` forced to scalar/naive through `DispatchPolicy`. This is a production scalar baseline, not the oracle.
- `blas3-simd-1t`: `BLAS3Kernels.gemm` forced to the single-thread BLAS3 code path. When Vector API is available, this path uses the packed SIMD kernel; otherwise it falls back to scalar execution and the notes field says so.
- `blas3-parallel`: `BLAS3Kernels.gemm` forced to the parallel path. In the current codebase this is ForkJoin plus `IntStream`-style scalar blocked work, not SIMD parallel.
- `opt-1t`: `OptimizedBLAS3.gemm` with `parallelism=1` and CUDA disabled.
- `opt-parallel`: `OptimizedBLAS3.gemm` with available processor count and CUDA disabled. This is the main production CPU optimized path.
- `cuda`: `CudaGemm.gemm`. It only runs when CUDA is genuinely usable. If unavailable or `CudaGemm.gemm` returns `false`, the row is emitted as skipped and the Gradle task still succeeds.

## Parity Policy

- Every backend is parity-checked before timing.
- For sizes up to `gemm.independentOracleMaxSize`, parity uses the independent `GemmReference` oracle.
- For larger sizes, the harness uses `opt-1t` as `reference_backend=opt-1t` and records `oracle_type=backend_reference`.
- If parity fails, the row is emitted with `valid=false` and `parity_status=FAIL`, and no GFLOPs or speedup is reported.
- Invalid or skipped CUDA rows are still emitted.

## Timing And Statistics

- Warmups run before measurement.
- Measured iterations store per-call nanoseconds.
- Small problems use an internal repeat count to stay above timer granularity.
- The harness reports median, mean, sample standard deviation, min, p10, p25, p75, p90, median-based GFLOPs, and median-based speedup.
- Headline speedup is never based on best-of timing.

## Output Artifacts

Artifacts are written only under `build/reports/gemm/`:

- `gemm-backend-comparison.csv`
- `gemm-backend-comparison-<UTC timestamp>.csv`
- `gemm-backend-comparison.json`

## Caveats

- Numbers are host-specific.
- `blas3-parallel` is not necessarily SIMD.
- The CUDA path may include allocation, transfer, and launch overhead.
- The naive baseline is size-capped.
- The independent oracle is size-capped by default.
- JIT and thermal noise still exist even with warmups and medians.
- The known ignored `OptimizedBLAS3` tiny `4x4x4` issue is outside the benchmark sizes.
