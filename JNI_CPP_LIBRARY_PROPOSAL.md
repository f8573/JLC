# JNI C++ Compute Library Proposal

## Goal

Provide an alternative dense-linear-algebra backend where Java remains the public API and orchestration layer, while high-throughput numerical kernels run in C++ and are loaded through JNI.

The target is not a full rewrite on day one. The target is a drop-in optional backend for the most performance-sensitive paths:

- GEMM
- batched GEMM
- QR/LU panel kernels
- packing / workspace-heavy kernels
- future GPU interop behind the same native boundary

## Proposed Name

`jlc-native`

- Java facade package: `net.faulj.nativeblas`
- Native library name: `jlc_native`
- Gradle module name: `native-backend`

## High-Level Architecture

### Java layer

- Keeps `Matrix`, `DispatchPolicy`, benchmarking, API endpoints, and algorithm selection.
- Decides whether to use Java kernels or native kernels.
- Owns validation, fallbacks, feature flags, and lifecycle.

### JNI bridge layer

- Small, stable ABI-oriented JNI surface.
- Converts Java-side calls into native handles plus primitive buffers.
- Avoids embedding algorithm policy in JNI glue.

### C++ compute layer

- Implements kernels, packing, thread pools, workspace management, and ISA-specific dispatch.
- Exposes a narrow C ABI internally wrapped by JNI.
- Can link against vendor BLAS later, but should start with self-owned kernels.

## Proposed Repository Layout

```text
native-backend/
  build.gradle
  CMakeLists.txt
  src/
    main/
      java/
        net/faulj/nativeblas/
          NativeBackend.java
          NativeContext.java
          NativeMatrixHandle.java
          NativeStatus.java
      cpp/
        jni/
          jlc_native_Jni.cpp
        api/
          jlc_native.h
          jlc_status.h
        runtime/
          context.cpp
          thread_pool.cpp
          workspace.cpp
        kernels/
          gemm.cpp
          gemm_pack.cpp
          qr_panel.cpp
          lu_panel.cpp
        platform/
          cpu_features.cpp
          aligned_alloc.cpp
        tests/
          gemm_test.cpp
          qr_panel_test.cpp
```

## Java API Shape

```java
public interface ComputeBackend {
    boolean isAvailable();
    void gemm(Matrix a, Matrix b, Matrix c, double alpha, double beta, DispatchPolicy policy);
}
```

```java
public final class NativeBackend implements ComputeBackend {
    static {
        System.loadLibrary("jlc_native");
    }

    public native boolean isAvailable();
    public native void gemm(double[] a, int aRows, int aCols,
                            double[] b, int bRows, int bCols,
                            double[] c, int cRows, int cCols,
                            double alpha, double beta,
                            int threads, int flags);
}
```

## JNI Design Rules

- Pass primitive arrays or direct buffers only.
- Keep JNI methods flat and explicit; do not mirror large Java object graphs.
- Minimize JNI crossings by pushing whole kernels, not inner loops, across the boundary.
- Prefer direct `ByteBuffer` or off-heap storage for long-lived large matrices.
- Never allocate large native workspaces per call; use pooled `NativeContext`.

## Memory Model

### Phase 1

- Use Java `double[]` inputs and JNI critical access carefully for large kernels.
- Copy only when alignment or layout requires it.

### Phase 2

- Add direct/off-heap matrix storage with explicit native ownership.
- Allow Java `Matrix` to wrap a native handle for zero-copy native execution.

### Phase 3

- Support pinned / reusable workspaces for batched and decomposition workloads.

## Native Runtime

### Context

`NativeContext` owns:

- CPU feature detection
- thread-pool configuration
- aligned workspace arenas
- optional logging / perf counters

### Threading

- Use a native thread pool independent from ForkJoin.
- Thread count comes from Java policy and is clamped natively.
- Avoid oversubscription by treating Java-side parallelism as disabled when native kernels are selected.

## Kernel Strategy

### GEMM

- Start with blocked packed GEMM.
- Support AVX2 first, AVX-512 second, scalar fallback always.
- Separate microkernel dispatch from JNI entrypoints.

### Decompositions

- Do not port complete QR/LU/SVD immediately.
- First port the BLAS-3-heavy update kernels used by Java decomposition code.
- Only port full decompositions once the panel/update split is stable.

## Build Integration

### Gradle

- Keep the current Java build intact.
- Add a dedicated Gradle task to configure and build the C++ library through CMake.
- Publish OS/arch-specific artifacts separately.

### CMake

- Build shared library for Windows/Linux/macOS.
- Produce deterministic output name: `jlc_native`.
- Export compile definitions for ISA toggles and diagnostics.

## Loading Strategy

- Default backend remains Java.
- Native backend enabled by property, for example:

```text
-Djlc.backend=native
```

- If load fails, log once and fall back to Java.
- Backend choice should be visible in benchmark output and diagnostics endpoints.

## Safety and Compatibility

- Native code must validate dimensions, nulls, and workspace bounds.
- JNI must convert native failures into Java exceptions with stable status codes.
- No silent native crash path should be accepted as normal failure handling.
- Keep the Java backend as the correctness oracle during rollout.

## Benchmark and Validation Plan

1. Add backend-agnostic GEMM benchmark fixtures.
2. Compare Java vs native for correctness on small, medium, and irregular shapes.
3. Compare throughput for square sizes, tall-skinny, and decomposition update kernels.
4. Gate rollout on both accuracy and speedup thresholds.

## Phased Rollout

### Phase 0

- Add backend abstraction and feature flags.

### Phase 1

- Implement native GEMM only.
- Benchmark against current Java GEMM.

### Phase 2

- Route Java QR/LU trailing updates through native GEMM.

### Phase 3

- Add native packing/workspace APIs and off-heap matrix support.

### Phase 4

- Add optional vendor BLAS / oneMKL / OpenBLAS integration behind the same C++ API.

## Why This Design

- It preserves the existing Java API and dependency graph.
- It keeps JNI thin enough to debug.
- It allows selective acceleration instead of a risky full rewrite.
- It gives a path to vendor BLAS or GPU backends later without exposing those details to most Java code.
