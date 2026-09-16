# Native SVD engineering report

## Implementation

- Branch: `feature/native-svd-baseline`
- Base commit: `77ffc79681719125e27163ecbb8ab08a5374ff05`
- Final implementation commit: `8b3fff6ab386bcda78ff6e01e3e09cb7e384c84d`
- Scope remained confined to the clean native-SVD worktree; the unrelated dirty
  checkout was not changed.

The native SVD now has two selectable algorithms:

1. The existing one-sided Jacobi implementation remains intact and is available
   through `jlc.algorithm.svd.nativeAlgorithm=jacobi`.
2. The production path reuses `jlc_native_bidiagonal_decompose`, then reduces
   the upper bidiagonal matrix with implicit shifted Golub--Kahan QR. Givens
   rotations are accumulated directly into the Householder-produced left and
   right factors, followed by sign normalization and descending selection sort.

The bidiagonalizer gained scaled `hypot`-style reflector norms and finite-value
guards. No Gram matrix is formed. Wide matrices use the shared reducer on the
transpose and exchange the factors back. JNI exposes an internal algorithm
selector while retaining the original entry point.

The native facade performs cheap finite/nonnegative/ordering checks and returns
to the existing Java implementation on linkage, argument, convergence, or
validation failure. With no override, the measured native policy selects
bidiagonal QR (`jacobiMaxDimension=0`); a positive internal cutoff can still be
used for platform-specific tuning. Global backend AUTO behavior was not
changed. `jlc.algorithm.svd.debug=true` emits the selected algorithm and
fallback reason at `FINE` level.

Thin SVD semantics remain compatible with the existing API. The current first
correct implementation still constructs full native factors and crops them in
Java; direct thin-factor construction is a follow-up optimization.

Changed files: `build.gradle`, `native-backend/src/main/cpp/api/jlc_native.h`,
`native-backend/src/main/cpp/jni/jlc_native_Jni.cpp`,
`native-backend/src/main/cpp/kernels/bidiagonal.cpp`,
`native-backend/src/main/cpp/kernels/svd.cpp`,
`src/main/java/net/faulj/nativeblas/NativeBackend.java`,
`src/main/java/net/faulj/nativeblas/NativeBindings.java`,
`src/test/java/net/faulj/nativeblas/NativeSvdProductionTest.java`, and
`src/test/java/net/faulj/nativeblas/NativeSvdBenchmarkRunner.java`.

## Numerical results

The following native QR results use relative Frobenius reconstruction error,
2-norm orthogonality errors, and the maximum singular-value difference versus
OpenBLAS `dgesdd`, normalized by `max(1, sigma[0])`.

| Case | Reconstruction | U orthogonality | V orthogonality | Sigma error vs OpenBLAS |
| --- | ---: | ---: | ---: | ---: |
| Random 64x64 | 2.705e-15 | 4.401e-15 | 5.087e-15 | 2.022e-15 |
| Random 37x11 | 1.404e-15 | 1.614e-15 | 1.462e-15 | 1.139e-15 |
| Random 11x37 | 1.085e-15 | 2.857e-15 | 1.705e-15 | 3.890e-16 |
| Dense rank-one | 4.752e-16 | 4.137e-15 | 6.130e-15 | 1.591e-16 |
| Clustered/repeated | 7.934e-16 | 2.351e-15 | 1.394e-15 | 7.772e-16 |
| Rank deficient | 1.489e-15 | 1.300e-15 | 1.851e-15 | 1.947e-15 |
| Scale disparity | 4.091e-16 | 3.855e-16 | 1.052e-15 | 1.221e-16 |

The adversarial suite also covers zero and identity matrices, sorted and
unsorted diagonal inputs, very small and very large finite entries, nearly
dependent rows and columns, off-heap row-major and column-major inputs, and
full/thin contracts.

## Convergence

- Maximum QR iterations: `max(64, 30 * min(m, n))`.
- Deflation uses relative off-diagonal thresholds tied to `8 * ulp(1)` plus
  `denorm_min`; a corresponding relative test handles zero/tiny diagonals.
- The shift recurrence uses four initial zero-shift steps, a safely scaled
  Wilkinson shift using `hypot`, and an exceptional shift after ten stalled
  iterations.
- No non-convergence occurred in the 21 production tests, an 80-case random /
  adversarial native sweep, or the 512x512 spot-check. Any future failure
  returns a native convergence status and reaches Java fallback.
- The previously capped dense rank-one case now converges and reconstructs
  accurately. Repeated, clustered, and rank-deficient cases preserve ordering,
  signs, and orthogonality.

## Performance

Median milliseconds from one warmup and three measured iterations:

| Shape | Java | Native Jacobi | Native Bidiag-QR | OpenBLAS `dgesdd` |
| --- | ---: | ---: | ---: | ---: |
| 32x32 | 7.443 | 0.697 | 0.263 | 0.152 |
| 64x64 | 13.325 | 5.432 | 1.337 | 0.524 |
| 128x128 | 61.315 | 47.845 | 18.352 | 2.273 |
| 256x256 | 136.017 | 1120.138 | 534.095 | 12.838 |
| 512x512 | 1328.254 | n/a* | 8139.188 | 88.651 |
| 256x64 | 27.351 | 83.057 | 25.222 | 1.857 |
| 64x256 | 27.406 | 81.094 | 25.950 | 2.004 |
| 512x128 | 209.103 | n/a* | 338.469 | 12.876 |
| 128x512 | 208.387 | n/a* | 353.519 | 13.035 |

The Java/native columns come from the JLC JavaExec runner; OpenBLAS was called
through its direct C ABI with one thread, so the timing boundaries are not
identical. The old 256x256 Jacobi result was about 1078--1120 ms on these
runs; QR is about 3.9x Java instead of the old roughly 8x crossover and is
therefore a material improvement. The old 512x512 Jacobi run exceeded two
minutes during exploration and is skipped by default (`*`); QR completed in
about 8.1 seconds. QR remains substantially slower than vendor LAPACK and its
scalar full-factor accumulation is the main remaining performance weakness.

No faster Jacobi region was observed in the measured native sweep: QR was
already faster at 2x2 through 32x32 and in the tested rectangular cases. The
default native AUTO policy therefore uses QR, while Jacobi remains available
for explicit comparison or future platform-specific calibration.

## Tests and build evidence

- New native production SVD tests: **21/21 passed**.
- Existing native SVD baseline tests: **14/14 passed**.
- Native backend tests excluding the unrelated provider-specific GEMM case:
  **47/47 passed**.
- Complete `testNativeBackend`: **48/49 passed**. The one failure is the
  existing `NativeGemmIntegrationTest.nativeBackendUsesAvx2PackedMicrokernelOnWindowsBuild`,
  which requires provider `builtin only` but this Linux/OpenBLAS build reports
  provider `All`; no unrelated GEMM/compiler change was made.
- SVD plus decomposition suites: **50/50 passed**.
- Full project `test`: **481/482 passed**; the same existing provider mismatch
  is the only failure.
- Native CMake/Release build: **passed**, provider reported `All`.
- `git diff --check`: **passed**.

## Remaining weaknesses

- Thin calls still allocate full `m x m` and `n x n` factors before cropping.
- Givens propagation is scalar and single-threaded; vendor LAPACK remains the
  appropriate high-performance reference.
- Production validation is structural rather than a per-call residual check;
  residual and orthogonality diagnostics are exercised in tests and the
  OpenBLAS comparison.
- Singular values below the information content of the FP64 input cannot be
  recovered. For example, a component separated from a `1e12` scale by more
  than roughly `1/epsilon` is numerically unrepresentable; native failure or a
  zero trailing value is preferable to claiming false accuracy.
- Real FP64 is covered by this path; complex matrices continue to use Java.
