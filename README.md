# LambdaCompute

LambdaCompute is a Java linear algebra library and web application for matrix diagnostics, decomposition, and spectral analysis.

Live site: https://lambdacompute.org/

## What this project includes

- `src/main/java/net/faulj`: Java core library (matrix/vector types, decompositions, solvers, eigen/spectral routines, condition/accuracy metrics, benchmarking helpers)
- `src/main/java/net/faulj/kernels/gemm`: first-class GEMM nucleus (canonical GEMM facade, dispatch, microkernel, packing, SIMD adapters)
- `src/main/java/net/faulj/nativeblas`: backend registry and JNI bridge for the optional native compute backend
- `src/main/java/net/faulj/web`: Spring Boot API layer (`/api/diagnostics`, `/api/status`, `/api/contact`, benchmark/status streams)
- `native-backend`: C++/JNI shared library sources built through CMake for the optional native compute backend
- `JNI_CPP_LIBRARY_PROPOSAL.md`: proposed JNI-loaded C++ compute backend design
- `frontend`: React + Vite client for matrix input, analysis views, decompositions, spectral reports, favorites/history, and settings

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

The JNI rollout provides an optional native backend for the hot dense-linear-algebra paths while keeping the Java API and Java fallback path intact.

- `jlc.backend=java`: keep the existing Java/CUDA dispatch path active
- `jlc.backend=native`: load `jlc_native` and route compatible GEMM calls through JNI; unsupported cases still fall back to Java
- `jlc.backend=auto`: prefer native when the shared library is available, otherwise fall back to Java

Current native scope:

- Native GEMM coverage: real heap-backed `Matrix` GEMM, compatible strided GEMM variants, and direct/off-heap GEMM for supported layouts
- Native decomposition coverage: builtin C++ QR, LU, and Cholesky hooks selected by `jlc.native.*` provider/min-size properties
- Optional vendor coverage: BLAS/LAPACK-backed GEMM, QR, LU, Cholesky, and Hessenberg paths when CMake finds compatible libraries
- Java fallback: complex matrices, unsupported direct/off-heap layouts, unavailable native libraries, unavailable vendor LAPACK, and paths explicitly configured for `java`
- Diagnostics: `/api/status` and benchmark responses expose requested/effective backend and native load status

QR backend selection is now shape-aware when `jlc.backend=native` and `jlc.native.qr.provider=auto`:

- `jlc.native.qr.square.decomposeBands` / `factorizeBands`
- `jlc.native.qr.wide.decomposeBands` / `factorizeBands`
- `jlc.native.qr.tall.decomposeBands` / `factorizeBands`
- `jlc.native.qr.tall.factorizeGrid`
- `jlc.native.qr.calibration.path`

`factorizeGrid` uses `short-dimension x long-dimension` rectangular rules, for example:

```powershell
"-Djlc.native.qr.tall.factorizeGrid=1-32x1+:native,33-64x1-1024:native,33-64x1025+:java,65+x1+:java"
```

Use the comparison runner to calibrate those settings on a target machine:

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
.\gradlew.bat bootRun "-Djlc.backend=native" "-Djlc.native.qr.provider=auto" "-Djlc.native.qr.calibration.path=build/reports/qr_backend_calibration.properties"
```

Precedence is:

1. explicit `-Djlc.native.qr.*` system properties
2. file-backed calibration values from `jlc.native.qr.calibration.path`
3. built-in fallback heuristics

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

`AUTO` keeps the builtin backend available when CMake cannot find BLAS/LAPACK. Explicit `OPENBLAS` or `MKL` requests fail configuration if the requested vendor stack is not found.

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
