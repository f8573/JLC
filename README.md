# JLC

JLC is a Java/C++ dense linear-algebra system that combines a hand-optimized native FP64 GEMM backend, JNI-backed runtime dispatch, and an opt-in matrix-program compiler with whole-expression optimization, affine/dependence analysis, legality-checked scheduling, and CPU lowering.

> **Project status:** the CPU/native/compiler work described here is complete and published. The matrix compiler is intentionally bounded at M4; GPU/CUDA lowering, autotuning, and broader code generation are separate extensions rather than unfinished milestones.

## Measured results

| Result | Measured or verified value |
| --- | ---: |
| Native FP64 GEMM, 2048³, controlled median | **520.959 GFLOP/s** |
| Controlled same-host AOCL-BLIS reference | 583.348 GFLOP/s |
| JLC / controlled reference | **89.3%** |
| JLC / nominal Zen 2 FP64 peak (896 GFLOP/s) | **58.1%** |
| Matrix-chain planner example | **20,000,000 → 200,000** scalar multiplications |
| 256² scale+add fusion | **0.089060 ms eager → 0.047110 ms compiled** |
| 256² measured allocation | **1,048,672 B eager → 525,072 B compiled** |
| Full Java test suite | **447 / 447 passed** |
| Focused compiler correctness tests | **121 / 121 passed** |
| Native-focused Java tests | **14 / 14 passed** |

The controlled GEMM result used 16 physical workers with physical-core affinity on an AMD Ryzen 9 3950X. The AOCL-BLIS figure is a controlled same-host reference, not a current vendor-library run. Compiler timings are separate JVM measurements on the same host.

## System capabilities

| Layer | What JLC implements |
| --- | --- |
| Java/native runtime | JNI-backed native backend with Java fallback, persistent native context, heap/off-heap handling, and strided GEMM entry points. |
| Backend selection | `auto`, `java`, and `native` runtime modes plus per-algorithm Java/C++ selection. |
| Calibration-driven dispatch | Optional calibration profiles gate native selection on correctness, sample count, shape/size, and measured speedup thresholds. |
| Native GEMM | AVX2/FMA 6×8 FP64 microkernel, twelve vector accumulator chains, cache blocking/packing, multithreaded worker pool, and optional physical-core affinity. |
| Matrix compiler | Expression DAG → planning → affine/access IR → dependence graph → legality-checked schedule → executable CPU plan. |
| Fusion lowering | Legality-checked scale+add fusion with a guarded primitive fast path for canonical real heap matrices and a generic schedule fallback otherwise. |

## Architecture

```text
                         ┌───────────────────────────────┐
                         │          Matrix API           │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                                      Gemm facade
                                         │
                          ┌──────────────┴──────────────┐
                          │ BackendRegistry / dispatch │
                          └──────────────┬──────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                   JavaBackend                     NativeBackend
                                                       │
                                                       ▼
                                                      JNI
                                                       │
                                                       ▼
                                      blocked / packed AVX2 FP64 GEMM

MatrixExpr DAG
    │
    ▼
graph optimization / matrix-chain planning
    │
    ▼
ExecutionPlan
    │
    ▼
AffineProgram → DependenceGraph
    │
    ▼
SchedulePlan
    │
    ▼
CpuExecutionPlan
    ├─ MatMul → Gemm facade → Java/native dispatch
    └─ elementwise regions → primitive fused loop or generic schedule executor
```

The compiler plans across a matrix expression, but optimized matrix multiplication remains an opaque library kernel at execution. This keeps the compiler responsible for program-level optimization while the GEMM backend owns its internal blocking, packing, threading, and reduction order.

## Backend selection and JNI boundary

`BackendRegistry` exposes three runtime preferences:

```text
-Djlc.backend=auto
-Djlc.backend=java
-Djlc.backend=native
```

`auto` is the default. The native backend is JNI-backed and falls back to Java when the native library is unavailable or an operand/layout is unsupported. Native integration includes ordinary heap-backed GEMM, direct/off-heap matrices, and strided entry points.

Algorithm selection is separate from the global backend preference. `AlgorithmDispatch` supports per-algorithm `AUTO`, `JAVA`, and `CPP` choices and can load calibration profiles. A calibrated native choice must have passed correctness validation, have enough samples, and clear the configured speedup threshold; otherwise JLC retains the Java path. Cold-start rules remain conservative when no profile is present.

This boundary is deliberate: JNI calls stay coarse, while backend selection and failure handling stay visible to Java.

## Native GEMM

The built-in native FP64 path uses:

- AVX2/FMA with a 6×8 microkernel;
- twelve vector accumulator chains;
- packed/blocking-based execution;
- a persistent multithreaded worker pool;
- optional physical-core affinity;
- Java fallback for unsupported execution cases.

On the controlled 2048³ run, JLC reached **520.959 GFLOP/s median**, or **89.3%** of the historical controlled same-host AOCL-BLIS reference at **583.348 GFLOP/s**. That is about **58.1%** of the Ryzen 9 3950X host's nominal 896 GFLOP/s FP64 peak.

See [PERFORMANCE_SUMMARY.md](docs/PERFORMANCE_SUMMARY.md) and [EXPERIMENTAL_GEMM_PATHS.md](docs/EXPERIMENTAL_GEMM_PATHS.md) for methodology, rejected experiments, and comparison limits.

## Matrix compiler

| Milestone | Delivered layer |
| --- | --- |
| M1 | Typed expression DAG, shape/semantic checks, shared-node planning, matrix-chain optimization. |
| M2 | Logical buffers, bounded affine accesses/domains, alias analysis, dependence analysis. |
| M3 | Legality-checked interchange, tiling and fusion, with parallel/vector eligibility metadata. |
| M4 | Executable CPU plans, opaque GEMM lowering, and bounded elementwise fusion. |

The compiler is intentionally conservative. Schedule queries classify transformations as **LEGAL**, **ILLEGAL**, or **UNKNOWN**; `UNKNOWN` rejects the transformation. STRICT semantics preserve compiler-visible expression association/order constraints, while RELAXED/FAST modes allow explicitly permitted reassociation.

For a matrix multiplication update such as:

```text
C[i,j] += A[i,k] * B[k,j]
```

the affine layer records reads from `A[i,k]` and `B[k,j]` and a reduction dependence on `C[i,j]`. This is bounded affine/polyhedral-style scheduling over JLC's supported matrix IR, not a general Presburger/polyhedral compiler.

See [MATRIX_COMPILER.md](docs/MATRIX_COMPILER.md) for the exact IR, legality model, lowering contract, audit history, and benchmark methodology.

## Compiler performance

### Matrix-chain planning

For:

```text
A: 1000×10
B:   10×1000
C: 1000×10
```

STRICT preserves `(A×B)×C`, estimated at **20,000,000 scalar multiplications**. RELAXED chooses `A×(B×C)`, estimated at **200,000** — a **100× reduction in planner arithmetic**.

In the fixed benchmark, measured execution medians were **5.921 ms STRICT** and **5.094 ms RELAXED**, a **1.162× observed runtime ratio**. The planner result is therefore not presented as a 100× runtime speedup: backend dispatch, allocations, and intermediate shapes materially affect wall-clock execution.

### Profitable elementwise fusion

M4 can fuse the supported canonical scale+add family into one primitive pass. The first correct implementation routed each element through the generic schedule interpreter; profiling found map binding, affine evaluation, recursive schedule traversal, and transient allocation in the hot path. The repaired lowering proves the canonical identity case once, resolves storage before the loop, and executes:

```java
for (int p = 0; p < count; p++) {
    out[p] = alpha * a[p] + b[p];
}
```

Unsupported, complex, off-heap, shifted, or transformed schedules retain the generic correctness path.

Adequately warmed measurements used independent path warmup, 31 timed calls per run, and medians across three repaired runs:

| Size | Eager | Compiled fused | Eager / fused |
| ---: | ---: | ---: | ---: |
| 64×64 | 0.006160 ms | 0.003470 ms | **1.78×** |
| 128×128 | 0.022600 ms | 0.011460 ms | **1.97×** |
| 256×256 | 0.089060 ms | 0.047110 ms | **1.89×** |
| 512×512 | 0.426281 ms | 0.203941 ms | **2.09×** |
| 1024×1024 | 2.562306 ms | 1.469823 ms | **1.74×** |

At 256², the fused matrix payload is **524,288 bytes** versus **1,048,576 bytes** for eager execution. Measured total allocation was **525,072 bytes compiled** versus **1,048,672 bytes eager**. These are one-host observations, not portable speedup guarantees.

## Correctness and validation

The final CPU/compiler state was validated with:

- **447 / 447** full Java tests;
- **121 / 121** focused compiler correctness tests;
- **14 / 14** native-focused Java tests;
- zero skips, failures, or errors in the final validation;
- `git diff --check` clean;
- `Matrix.java`, `Gemm.java`, and `native-backend/**` unchanged by the bounded fused-lowering repair.

Adversarial compiler review previously exposed and repaired cross-iteration fusion legality, conservative alias/dependence handling, tile binding safety, mixed real/complex signed-zero semantics, off-heap intermediate ownership, retained-stage provenance, executable-subset validation, and benchmark-boundary issues. The later performance investigation isolated the generic per-element interpreter bottleneck without changing compiler semantics.

## Build and run

Requirements for the validated native path:

- Java 21 JDK;
- CMake 3.20+;
- C++17 compiler;
- x86-64 AVX2/FMA CPU.

Vendor BLAS/LAPACK libraries are optional. Node.js/npm are needed only for the React diagnostics frontend.

The Gradle wrapper is mode `100644`, so invoke it with `bash` on Unix-like systems:

```bash
bash ./gradlew build
bash ./gradlew buildNativeBackend
bash ./gradlew test --rerun-tasks
bash ./gradlew testNativeBackend
```

To force the built-in native provider while building:

```bash
bash ./gradlew -Djlc.native.vendor.blas=NONE buildNativeBackend
```

The Spring Boot diagnostics service starts with:

```bash
bash ./gradlew bootRun
```

The optional React frontend starts from `frontend/` after `npm install`:

```bash
npm run dev
```

On Windows, use `gradlew.bat`.

## Repository guide

- [Compiler design, lowering, benchmarks, and validation](docs/MATRIX_COMPILER.md)
- [Native GEMM performance evidence](docs/PERFORMANCE_SUMMARY.md)
- [Experimental GEMM paths and rejected variants](docs/EXPERIMENTAL_GEMM_PATHS.md)
- [License](LICENSE)

## Scope

JLC's current compiler is deliberately bounded. It does **not** claim a general Presburger solver, arbitrary generated SIMD/parallel code, CUDA/GPU lowering, or an autotuner. GEMM lowers through the optimized library boundary, while supported elementwise regions use explicit CPU lowering with conservative fallbacks.

The project is intended to demonstrate end-to-end systems work across Java API/runtime design, JNI/native ownership boundaries, hardware-aware kernel engineering, calibrated backend dispatch, compiler legality analysis, executable lowering, profiling, and measurement-driven optimization.
