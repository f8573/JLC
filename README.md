# LambdaCompute

LambdaCompute is a Java linear algebra library and web application for matrix diagnostics, decomposition, and spectral analysis.

## Technology Stack

- Backend: Java 21, Spring Boot 3.2, Gradle
- Frontend: React 18, Vite 5, Tailwind CSS, KaTeX
- Native/Performance: C++17, CMake, JNI, JCuda, optional BLAS/LAPACK integration
- Testing/Tooling: JUnit 4, Node.js scripts, Python analysis utilities

Live site: https://lambdacompute.org/

## What this project includes

- `src/main/java/net/faulj`: Java core library (matrix/vector types, decompositions, solvers, eigen/spectral routines, condition/accuracy metrics, benchmarking helpers)
- `src/main/java/net/faulj/kernels/gemm`: first-class GEMM nucleus (canonical GEMM facade, dispatch, microkernel, packing, SIMD adapters)
- `src/main/java/net/faulj/nativeblas`: backend registry and JNI bridge for the optional native compute backend
- `src/main/java/net/faulj/web`: Spring Boot API layer (`/api/diagnostics`, `/api/status`, `/api/contact`, benchmark/status streams)
- `native-backend`: C++/JNI shared library sources built through CMake for the optional native compute backend
- `frontend`: React + Vite client for matrix input, analysis views, decompositions, spectral reports, favorites/history, and settings

## Current CPU/compiler/runtime scope

The current checkpoint is the CPU-focused compiler and numerical runtime:

```text
                                  LambdaCompute / JLC
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
          Numerical Algorithm Runtime                 Matrix Compiler
                    │                                         │
          Java implementations                           MatrixExpr
          Native C++ algorithms                              │
          GEMM / decompositions                      M1 — Graph optimization
                    │                                         │
          Algorithm dispatch                       M2 — Affine/dependence IR
             ┌──────┴──────┐                                  │
             │             │                         M3 — Legal scheduling
           Java         C++ / JNI                              │
             │             │                         M4 — CPU lowering
             │             │                                  │
             │             │                         R2 — Generalized fusion
             │             │                                  │
             │             │                  ┌───────────────┴───────────────┐
             │             │                  │                               │
             │             │         R1 — Physical buffer planning   R3 — Verified Kernel IR
             │             │                  │                               │
             │             │                  │                      R4 — Native codegen
             │             │                  │                       ┌───────┴────────┐
             │             │                  │                       │                │
             │             │                  │                  Scalar C++     Generated AVX2
             │             │                  │                       │                │
             │             │                  │                       └───────┬────────┘
             │             │                  │                               │
             │             │                  │                      R5 — Empirical tuning
             │             │                  │                               │
             │             │                  │                    Backend calibration
             │             │                  │                               │
             │             │                  │                    KernelDispatchSelector
             │             │                  │                 ┌────────┼─────────┐
             │             │                  │                 │        │         │
             │             │                  │              R2 Java   Scalar    AVX2
             │             │                  │                 │        │         │
             │             │                  │                 └────────┴─────────┘
             │             │                  │                          │
             └──────┬──────┘                  └────────────┬─────────────┘
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       │
                                  JLC execution
                                       │
                           correctness-preserving fallback


                        ───── Current CPU checkpoint ─────

                                       │
                                       ▼
                               Future compiler backends
                          ┌────────────┼────────────┐
                          │            │            │
                       AVX-512      CUDA/GPU     other ISA
                                       │
                            heterogeneous placement
```

The runtime combines Java numerical implementations, optimized native algorithms,
JNI/native registry execution, correctness-gated generated kernels, and a
correctness-preserving Java fallback. M4 is the terminal matrix-compiler
milestone; R1–R5 are post-M4 research, and typed backend dispatch is the
production integration pass. There is no R6 in this checkpoint.

`R1–R5` names reflect research chronology, not a single linear compiler pipeline.
After R2 determines fusion and materialization, R1 handles physical storage for
executable values while R3–R5 form the generated-kernel path.

**CPU performance checkpoint:** on the documented Ryzen 9 3950X same-host
benchmark, JLC's built-in native FP64 GEMM reached 520.959 GFLOP/s at 2048³
with 16 physical workers, or 89.3% of the measured AOCL-BLIS throughput. See
[`docs/GEMM_PUBLICATION_BENCHMARK.md`](docs/GEMM_PUBLICATION_BENCHMARK.md) for
the methodology and limitations.

Existing or legacy CUDA-capable runtime infrastructure, including JCuda-related
configuration and execution-policy switches, may still exist elsewhere in JLC.
That infrastructure is separate from this checkpoint: the typed Kernel IR
backend path dispatches only R2 Java, scalar native, and generated AVX2. It does
not generate a CUDA backend; a CUDA `BackendChoice` for typed Kernel IR is future
work.

The public design and evidence are documented in
[`docs/COMPILER_RUNTIME_RESEARCH.md`](docs/COMPILER_RUNTIME_RESEARCH.md) and
[`docs/MATRIX_COMPILER.md`](docs/MATRIX_COMPILER.md).

Future work is intentionally outside this CPU checkpoint: CUDA/GPU execution,
AVX-512, heterogeneous CPU/GPU placement, broader shape-family calibration,
thread-count tuning, mixed precision, and runtime JIT.

## Primary use cases

- Run matrix diagnostics from raw matrix input
- Inspect decomposition results (QR, LU, SVD, Schur, Hessenberg, spectral, etc.)
- Evaluate numerical stability and accuracy metadata
- Benchmark selected compute paths via API endpoints

## Requirements

- Java 21 (project toolchain target)
- Node.js 18+
- npm 9+

## Run locally

### 1) Start backend (Spring Boot)

From repository root:

```powershell
.\gradlew.bat bootRun
```

Backend default: `http://localhost:8080`

### 2) Start frontend (Vite)

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend default: `http://localhost:5173`

The Vite dev server proxies `/api` to `http://localhost:8080` via `frontend/vite.config.js`.

## Build

### Backend

```powershell
.\gradlew.bat build
```

### Native Backend

```powershell
.\gradlew.bat buildNativeBackend
.\gradlew.bat testNativeBackend
.\gradlew.bat runGemmBackendComparison
```

### Frontend

```powershell
cd frontend
npm run build
```

## Test

```powershell
.\gradlew.bat test
```

Default `test` is intentionally limited to correctness, dispatch logic, lightweight smoke coverage, and non-crashing unit tests. Heavy benchmark and stress suites are not part of the default test task.

Explicit benchmark / stress entrypoints:

```powershell
.\gradlew.bat benchmarkTest
.\gradlew.bat stressTest
.\gradlew.bat runComprehensivePerfBenchmark
```

If you only want a targeted API smoke test:

```powershell
.\gradlew.bat test --tests net.faulj.web.ApiControllerStatusTest
```

## API at a glance

- `GET /api/ping`
- `GET /api/status`
- `POST /api/diagnostics`
- `GET /api/diagnostics?matrix=...`
- `GET /api/diagnostics/stream` (SSE)
- `GET /api/benchmark/diagnostic`
- `GET /api/benchmark/diagnostic512`
- `POST /api/contact`

## Notes

- Contact form delivery uses `DISCORD_WEBHOOK_URL` from environment variables.
- Large matrices are intentionally limited for synchronous full diagnostics in the API.
- GEMM kernel docs:
  - `src/main/java/net/faulj/kernels/gemm/README.md`
  - `src/main/java/net/faulj/kernels/gemm/PERFORMANCE.md`

## Execution Policy Flags

Runtime policy can be controlled without refactoring the architecture yet.

- `faulj.runtime.profile` / `FAULJ_RUNTIME_PROFILE`:
  - `default` (adaptive)
  - `legacy` (single-thread, scalar-friendly defaults for low-spec hardware)
- `faulj.exec.policy` / `FAULJ_EXEC_POLICY`:
  - `AUTO`
  - `SCALAR_SAFE` (single-thread, no SIMD/BLAS3/CUDA; guaranteed fallback tier)
  - `SCALAR_PARALLEL` (parallel scalar only)
  - `SIMD` (CPU vectorized, no CUDA)
  - `ACCEL` (allow hardware acceleration including CUDA)
- Fine-grained toggles:
  - `faulj.parallel.enabled`, `faulj.parallelism`
  - `faulj.simd.enabled` (or `faulj.vectorization.enabled`)
  - `faulj.blas3.enabled`
  - `faulj.cuda.enabled`

Example:

```powershell
.\gradlew.bat bootRun "-Dfaulj.exec.policy=SCALAR_SAFE"
```

## Backend Selection

JLC routes dense linear algebra through calibrated algorithm dispatch. Java is the correctness baseline. C++ is the performance backend only when the dispatch policy or an explicit `cpp` override selects it for the matching calibration bucket.

- `jlc.algorithm.backend=auto|java|cpp`: global algorithm-dispatch mode; default is `auto`
- `jlc.algorithm.<name>.backend=auto|java|cpp`: per-algorithm override, for example `jlc.algorithm.gemm.backend=cpp`
- `jlc.backend=auto`: default native-library probe mode; algorithm dispatch still decides Java vs C++
- `jlc.backend=java`: disables JNI probing and keeps Java implementations active
- `jlc.backend=native`: probes `jlc_native`, but still routes through the calibrated algorithm policy

Current calibrated native scope:

- GEMM coverage: heap-backed `Matrix` GEMM, compatible strided GEMM variants, and supported direct/off-heap layouts
- Decomposition coverage: optional C++/JNI LU and QR hooks routed by algorithm dispatch; Cholesky has an explicitly enabled native hook
- Stage-level native helpers: guarded C++/JNI Hessenberg and SVD-bidiagonal paths are available for validated shapes; full SVD remains primarily Java, the iterative Schur stage and general eigenvalue path remain Java, and native Hessenberg coverage is bounded by policy and shape rules
- These lower-level helpers are separate from generated Kernel IR backends; the typed compiler path remains limited to R2 Java, scalar native, and generated AVX2
- Java fallback: unsupported shapes/layouts, unavailable native libraries, failed native calls, failed validation, and numerically sensitive or uncalibrated paths
- Diagnostics: `/api/status` and benchmark responses expose requested/effective backend and native load status

Public LAPACK/provider selection is no longer part of runtime routing. Optional vendor linkage may remain a build-time/native implementation detail, but runtime users select only `auto`, `java`, or `cpp`.

Calibration profiles are versioned Java properties files. Bucket keys use:

```text
{algorithm, mode, shape-family, size-band, thread-count}
```

Shape families are `square`, `tall`, and `wide`; size bands are `small`, `medium`, and `large`. A bucket stores Java/C++ timing summaries, sample counts, and C++ correctness status, for example:

```properties
version=1
bucket.gemm.multiply.square.medium.1.java.samples=5
bucket.gemm.multiply.square.medium.1.java.meanNanos=1100000
bucket.gemm.multiply.square.medium.1.cpp.samples=5
bucket.gemm.multiply.square.medium.1.cpp.meanNanos=850000
bucket.gemm.multiply.square.medium.1.cpp.correctness=PASS
```

In `auto`, C++ is selected only when correctness is `PASS`, the C++ sample count meets `jlc.algorithm.calibration.minSamples` (default `5`), and speedup clears `jlc.algorithm.speedupThreshold` (default `1.10`). SVD, Schur, and Polar use `jlc.algorithm.sensitive.speedupThreshold` (default `1.25`) and default to Java on cold start.

Use the QR comparison runner to seed calibration data on a target machine:

```powershell
.\gradlew.bat runQrBackendComparison --args="--mode=decompose --shapes=512x128,1024x128,2048x256"
.\gradlew.bat runQrBackendComparison --args="--mode=factorize --shapes=256x32,512x64,1024x64,2048x64,512x128,1024x256"
```

To persist measured crossover data into a reusable profile:

```powershell
.\gradlew.bat runQrBackendComparison --args="--mode=decompose --shapes=128x128,256x256,512x512 --calibrationOut=build/reports/qr_backend_calibration.properties"
.\gradlew.bat runQrBackendComparison --args="--mode=factorize --shapes=256x32,512x64,1024x64,2048x64,512x128,1024x256 --calibrationOut=build/reports/qr_backend_calibration.properties"
```

Then point runtime at that file:

```powershell
.\gradlew.bat bootRun "-Djlc.backend=auto" "-Djlc.algorithm.calibration.path=build/reports/qr_backend_calibration.properties"
```

Precedence is:

1. `jlc.algorithm.<name>.backend`
2. `jlc.algorithm.backend`
3. matching calibration bucket from `jlc.algorithm.calibration.path`
4. conservative cold-start policy

Current cold-start QR policy:

- `qr.factorize_only`: native is allowed by default
- `qr.decompose_thin` / `qr.decompose_full`:
  - native is allowed for non-tall shapes
  - tall shapes stay on Java unless calibration explicitly proves C++ wins
- This policy is intentionally conservative: native QR is promoted only where measured wins are stable, while tall `thin/full` QR keeps the Java fallback

Current performance status:

- GEMM has two distinct performance references: the smaller `512x512` JNI array-backed regression guard remains a targeted test boundary, while the publication-scale `2048x2048x2048` FP64 run on the Ryzen 9 3950X measured `520.959 GFLOP/s` with 16 physical workers, or `89.3%` of the same-host AOCL-BLIS reference. See [`docs/GEMM_PUBLICATION_BENCHMARK.md`](docs/GEMM_PUBLICATION_BENCHMARK.md) for the exact methodology and scope
- QR is the strongest native subsystem today, with large stable wins for factorize-only and improved square `thin/full` performance after the native direct trailing-update promotion

Default verification boundary:

- `test`: correctness, dispatch, smoke coverage, and non-crashing unit tests
- `benchmarkTest` / `stressTest` / benchmark runners: performance, stress, and large native workloads kept out of the default correctness suite

Gradle run/test tasks auto-wire `jlc.native.lib.path` after `buildNativeBackend`. Outside Gradle, point Java at the built shared library explicitly:

```powershell
"-Djlc.native.lib.path=build/native-backend/lib/jlc_native.dll"
```

Native build knobs for CI or alternate local toolchains:

- `jlc.native.cmake` / `JLC_NATIVE_CMAKE`
- `jlc.native.cmake.generator` / `JLC_NATIVE_CMAKE_GENERATOR`
- `jlc.native.cxx.compiler` / `JLC_NATIVE_CXX_COMPILER`
- `jlc.native.make.program` / `JLC_NATIVE_MAKE_PROGRAM`
- `jlc.native.build.type` / `JLC_NATIVE_BUILD_TYPE`
- `jlc.native.enable.march.native` / `JLC_NATIVE_ENABLE_MARCH_NATIVE`
- `jlc.native.enable.vendor.blas` / `JLC_NATIVE_ENABLE_VENDOR_BLAS`
- `jlc.native.vendor.blas` / `JLC_NATIVE_VENDOR_BLAS` (`AUTO`, `NONE`, `OPENBLAS`, `MKL`)

`AUTO` keeps the C++ backend available when CMake cannot find BLAS/LAPACK. Explicit `OPENBLAS` or `MKL` requests fail configuration if the requested vendor stack is not found. These build knobs do not create public runtime provider selection.

When passing `-D...` values through PowerShell, quote each argument as shown below so the Gradle wrapper receives it intact.

Examples:

```powershell
.\gradlew.bat bootRun "-Djlc.backend=native"
.\gradlew.bat runGemm512Benchmark "-Djlc.backend=auto"
.\gradlew.bat runNativeGemm512Benchmark
.\gradlew.bat runGemmBackendComparison --args="--size=512 --warmup=6 --runs=4"
```

## License

See `LICENSE`.
