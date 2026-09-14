# Native Backend Performance Notes

These notes retain implementation diagnostics and historical measurements for the built-in C++ JNI backend. They are not a current validated performance campaign.

## MSVC AVX2 Build

The Windows MSVC build must compile native targets with `/arch:AVX2`. Without it, MSVC does not define `__AVX2__`, the packed AVX2 GEMM microkernel is compiled out, and GEMM falls back to the scalar path.

Expected runtime diagnostics for the built-in backend on AVX2:

```text
runtimeDescription = jlc_native packed AVX2 GEMM
providerDescription = builtin only
GEMM MR = 6
GEMM NR = 8
```

`NativeGemmIntegrationTest.nativeBackendUsesAvx2PackedMicrokernelOnWindowsBuild` guards this locally by checking the runtime description and profiler-selected tile shape.

The current production GEMM defaults are the intrinsic AVX2 6x8 kernel, row-separated A packing, panel scheduling, private A packing, no worker pinning, and MC/KC/NC = 2048/256/128 for 2048³. Retained tuning selectors are catalogued in [`docs/EXPERIMENTAL_GEMM_PATHS.md`](../docs/EXPERIMENTAL_GEMM_PATHS.md); they are not production recommendations.

## QR Factorization Defaults

Native blocked QR uses a shape-aware default panel size:

```text
max(m, n) < 1536   -> block size 48
max(m, n) >= 1536  -> block size 96
```

`JLC_NATIVE_QR_BLOCK_SIZE` and `nativeQrSetBlockSizeOverride` still override this policy for tuning runs.

## Historical AVX2 Snapshot (not current validation)

Historically measured on a Windows/MSVC built-in backend with one QR GEMM thread. Preserve these values for provenance only; do not treat them as current hardware-relative or scaling evidence:

```text
512x512 factorize   block 48  ~37.7 ms
1024x1024 factorize block 48  ~249 ms
1536x1536 factorize block 96  ~704 ms
2048x2048 factorize block 96  ~1590 ms
```

The important historical comparison is that the pre-AVX2 MSVC scalar fallback measured about `~989 ms` for 1024x1024 factorization. That scalar-era result is not a valid baseline.

## Local Guardrail Commands

These commands are intended as local regression checks, not hard CI gates:

```powershell
.\gradlew.bat testNativeBackend
.\gradlew.bat runQrPerformanceGuardrail --args='--sizes=1024,2048 --warmup=1 --runs=3'
.\gradlew.bat runQrBlockSizeSweep --args='--sizes=512,1024,1536,2048 --blocks=48,96 --warmup=1 --runs=3 --mode=factorize --threads=1'
```

Expected guardrail ranges are intentionally loose to allow workstation variance. Treat a result outside the range as a prompt to inspect `runtimeDescription`, `gemmMr/gemmNr`, and QR profile buckets before changing kernels.
