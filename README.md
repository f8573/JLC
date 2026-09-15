# JLC

JLC is a Java/C++ dense linear-algebra system combining a hand-optimized native FP64 GEMM backend with an opt-in matrix-program compiler that performs whole-expression planning, bounded affine/dependence analysis, legality-checked scheduling, and CPU lowering.

## Highlights

| Result | Measured or verified value |
| --- | ---: |
| Native FP64 GEMM, 2048³, controlled median | **520.959 GFLOP/s** |
| Controlled same-host AOCL-BLIS reference | 583.348 GFLOP/s |
| JLC / controlled reference | **89.3%** |
| JLC / nominal Zen 2 FP64 peak (896 GFLOP/s) | **58.1%** |
| Matrix compiler | M1–M4 complete |
| Full Java test suite | 441 recorded; 431 passed, 10 skipped, 0 failed/error |
| Focused compiler correctness tests | 114 passed |
| Native-focused Java tests | 14 passed |

The controlled GEMM result used 16 physical workers with physical-core affinity on an AMD Ryzen 9 3950X. The AOCL-BLIS figure is a **controlled same-host reference**, not a current run. Compiler timings below are a separate fresh validation run on that host.

## Architecture

```text
MatrixExpr DAG
    ↓
graph optimization / matrix-chain planning
    ↓
ExecutionPlan
    ↓
AffineProgram → DependenceGraph
    ↓
SchedulePlan
    ↓
CpuExecutionPlan
    ├─ MatMul → existing Java/native GEMM facade
    └─ elementwise regions → explicit or fused CPU loops
```

The compiler plans across an expression, but matrix multiplication remains an opaque optimized kernel at execution. The eager Matrix API and GEMM backend are still available independently of the compiler.

## Native GEMM

The built-in native FP64 path uses AVX2/FMA, a 6×8 microkernel with twelve vector accumulator chains, cache blocking and packing, and a persistent multithreaded worker pool. Physical-core affinity is optional; the controlled 2048³ result explicitly enabled it. The 520.959 GFLOP/s median is 89.3% of the historical controlled same-host AOCL-BLIS reference and about 58.1% of the host's nominal 896 GFLOP/s FP64 peak. See the [performance summary](docs/PERFORMANCE_SUMMARY.md) and [experimental path manifest](docs/EXPERIMENTAL_GEMM_PATHS.md) for setup and limits of the comparison.

## Matrix compiler

| Milestone | Delivered layer |
| --- | --- |
| M1 | Typed expression DAG, shape and semantic checks, shared-node planning, matrix-chain optimization. |
| M2 | Logical buffers, bounded affine accesses and domains, alias and dependence analysis. |
| M3 | Legality-checked interchange, tiling and fusion, with parallel/vector eligibility metadata. |
| M4 | Executable CPU plans, opaque GEMM lowering, and narrow elementwise fusion. |

The compiler project is intentionally bounded at M4. GPU backends and autotuning are separate future projects rather than unfinished compiler milestones. The [compiler design and validation document](docs/MATRIX_COMPILER.md) gives the exact supported IR and execution contract.

### Matrix-chain example

For A (1000×10), B (10×1000), and C (1000×10), STRICT preserves (A×B)×C at a planner cost of 20,000,000 scalar multiplications. RELAXED chooses A×(B×C) at 200,000, a **100× arithmetic reduction**. In the fresh fixed benchmark, execution medians were **5.921 ms** and **5.094 ms**, respectively, a **1.162× observed runtime ratio**. The STRICT products selected [native, native]; the RELAXED products selected [java, native]. Planner arithmetic reduction does not imply the same runtime speedup: dispatch, allocation, and operand shape also matter.

### Schedule analysis

For `C[i,j] += A[i,k] * B[k,j]`, the affine layer records READ A[i,k], READ B[k,j], and REDUCTION C[i,j]. Schedule queries classify transformations as LEGAL, ILLEGAL, or UNKNOWN; UNKNOWN rejects a transformation. STRICT preserves compiler-visible expression association and order constraints, while optimized GEMM owns its internal reduction order. This is bounded affine/polyhedral-style schedule transformation over JLC's supported matrix IR, not general Presburger solving.

### What executes

MatMul calls the existing optimized Gemm facade. Add, Scale, and Transpose use explicit CPU loops; the supported scale-plus-add family can fuse into one loop and elide a temporary. Parallel and vector eligibility are metadata, with serial/scalar execution as the current fallback. Unsupported executable schedules are rejected before lowering.

## Fresh compiler benchmark snapshot

The fixed four-family M4 suite used deterministic inputs, two warmups, five measured calls, median execution time, and correctness checks before timing. This isolated run used Java 21.0.12 on Linux, 16 physical / 32 logical Ryzen 9 3950X cores, and the active native backend. Compilation and input construction were outside timed regions.

| Case | Fresh observed result |
| --- | --- |
| Matrix chain | STRICT 20,000,000 vs RELAXED 200,000 planner multiplications; **5.921 vs 5.094 ms**, 1.162× ratio; product backends [native,native] vs [java,native]; correctness passed. |
| Scale + add, 256×256 | Eager **2 temporaries / 1,048,576 bytes**, compiled **1 / 524,288 bytes**; **4.403 ms eager vs 15.520 ms fused**. Fusion was slower in this run; correctness passed. |
| Direct GEMM, 192³ | **0.431 ms direct vs 0.419 ms compiled**, −2.66% execution delta, within the benchmark's ±5% measurement-noise band; native selected; correctness passed. |
| Shared DAG, 96×96 | **22.673 ms**, one planned GEMM step and two logical temporary buffers; correctness passed. Runtime GEMM call count was not instrumented. |

These are one-host observations, not portable speedup guarantees. The [compiler benchmark section](docs/MATRIX_COMPILER.md#fixed-m4-benchmark-suite) explains the comparison boundaries.

## Correctness and adversarial validation

Post-M4 adversarial audits exposed and repaired cross-iteration fusion legality, alias/dependence conservatism, tile binding safety, mixed real/complex signed-zero semantics, hidden off-heap intermediate ownership, retained-stage provenance, executable-subset validation, and a benchmark boundary mismatch. Final verification found no remaining publication blockers. The full suite recorded 441 tests with zero failures or errors; the focused compiler correctness groups passed 114/114, and the native-focused groups passed 14/14.

## Build and run

Use a Java 21 JDK and the Gradle wrapper. The JNI backend needs CMake 3.20+ and a C++17 compiler; x86-64 AVX2/FMA is needed for the validated optimized native path. Vendor BLAS/LAPACK libraries are optional. Node.js and npm are needed only for the React diagnostics frontend. The wrapper has mode 100644, so invoke it with bash on Unix-like systems:

```bash
bash ./gradlew build
bash ./gradlew buildNativeBackend
bash ./gradlew test --rerun-tasks
bash ./gradlew testNativeBackend
```

To force the built-in native provider when building, use `bash ./gradlew -Djlc.native.vendor.blas=NONE buildNativeBackend`. On Windows, use `gradlew.bat`. The Spring Boot diagnostics service starts with `bash ./gradlew bootRun`; the optional React frontend starts with `npm run dev` from `frontend/` after `npm install`.

## Repository guide and scope

Start with [compiler design, benchmarks, and validation](docs/MATRIX_COMPILER.md), [native GEMM performance evidence](docs/PERFORMANCE_SUMMARY.md), and [experimental GEMM paths](docs/EXPERIMENTAL_GEMM_PATHS.md). The compiler's bounded IR has no general Presburger solver, arbitrary generated SIMD/parallel code, CUDA/GPU lowering, or autotuner. GEMM lowers through the optimized library boundary. See [LICENSE](LICENSE) for the project license.
