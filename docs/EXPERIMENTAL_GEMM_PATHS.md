# Native GEMM path manifest

This is the first-pass release-preparation map for the native GEMM backend. A
selector being present does not make that path a production recommendation.
The production path is intentionally small; the remaining selectors are kept
only when they are useful for reproduction, diagnosis, or a later Astra pass.

## Production defaults

With no GEMM tuning environment variables set, the native backend uses:

| Concern | Production behavior |
| --- | --- |
| AVX2 microkernel | Intrinsic 6x8 (`JLC_NATIVE_AVX2_MICROKERNEL` unset or `6x8`) |
| A layout | Row-separated A packing |
| B layout | Regular panel packing |
| Scheduler | Panel (`JLC_NATIVE_GEMM_SCHEDULER` unset or `panel`) |
| A packing | Worker-private (`JLC_NATIVE_A_PACKING` unset or `private`) |
| Worker affinity | None (`JLC_NATIVE_WORKER_AFFINITY` unset or `none`) |
| 2048³ blocking | MC/KC/NC = 2048/256/128; MR/NR = 6/8 |

Physical affinity remains an explicit controlled-run option. It is used by the
release check because it makes the 16-worker comparison repeatable, but it is
not the portable library default.

## Selector classification

| Selector or path | Classification | Evidence and disposition |
| --- | --- | --- |
| Intrinsic AVX2 6x8 | **PRODUCTION** | Replaced the older 5x4 path and is the current default in the microkernel campaign. |
| `6x8-noinline` | **EXPERIMENTAL / OPTIONAL** | Useful for kernel forensics; not promoted as a default. Keep for code-generation comparisons. |
| `6x8-u2` | **REGRESSIVE** | The U2 report measured an 8.25% loss at 2048³. Keep opt-in for historical reproduction only. |
| `6x8-u4` | **EXPERIMENTAL / OPTIONAL** | The U4 report was 0.72% below the noinline control, below the promotion gate. |
| `6x8-u4-asm` | **EXPERIMENTAL / OPTIONAL** | Handwritten assembly was about 1% ahead at 1T but below the promotion gate. Its source is now behind `JLC_NATIVE_ENABLE_EXPERIMENTAL_ASM=ON`. |
| `5x4`, `5x8` | **LEGACY** | Retained as selectable historical controls; neither is the production AVX2 path. |
| `JLC_NATIVE_A_LAYOUT=kmajor6` | **EXPERIMENTAL** | Interleaved/k-major A was useful at 1T but did not improve 16T. Keep out of normal dispatch. |
| `JLC_NATIVE_B_LAYOUT=nr8` | **EXPERIMENTAL / REGRESSIVE** | NR8 was useful in some 1T runs but regressed at 16T. |
| `JLC_NATIVE_NR8_PACKER` | **EXPERIMENTAL / REGRESSIVE** | The fast packer is retained for reproduction; it did not pass the 16T promotion gate. |
| `JLC_NATIVE_NR8_PREFETCH` | **EXPERIMENTAL / REGRESSIVE** | Software-prefetch variants did not improve 16T throughput. |
| `shared-tail` scheduler | **EXPERIMENTAL / OPTIONAL** | Correctness and tail-balancing evidence exists, but results are shape- and worker-sensitive; it remains opt-in. |
| `JLC_NATIVE_A_PACKING=l3` | **REGRESSIVE** | Shared-A/L3 packing reduced logical A copies but lost about 10.7% at 16T in the preserved comparison. It is never selected by default. |
| `JLC_NATIVE_WORKER_AFFINITY=physical` | **SUCCESSFUL OPTIONAL** | Useful for controlled physical-core runs and topology diagnostics. Default remains `none` for portability and compatibility. |
| Automatic 2048/256/128 blocking | **PRODUCTION** | Blocking search found no repeatable promotion above the current defaults. |
| `JLC_NATIVE_GEMM_MC/KC/NC` | **EXPERIMENTAL / OPTIONAL** | Exact overrides are validated once per call for reproducible tuning runs. |
| Legacy `JLC_NATIVE_MR/NR/MC/KC/NC` | **LEGACY / OPTIONAL** | Preserved for existing experiments; the exact `GEMM_*` controls are clearer for new runs. |
| Small-K update GEMM | **EXPERIMENTAL / OPTIONAL** | Explicitly disabled by default; retained for targeted small-K experiments. |
| Scalar, generic-output, and tail fallbacks | **PRODUCTION CORRECTNESS FALLBACKS** | Required for unsupported layouts, tails, and non-AVX2 builds; not tuning experiments. |
| Persistent pool and per-call worker fallback | **PRODUCTION** | Both are correctness-preserving execution modes; the cleanup only consolidated their resolved affinity inputs. |

## Assembly quarantine

The handwritten k-major6 consumers are no longer compiled into a normal Linux
x86-64 build. To reproduce that experiment, configure with
`-DJLC_NATIVE_ENABLE_EXPERIMENTAL_ASM=ON` and select `6x8-u4-asm`. Without the
explicit experimental build support, requesting that selector is rejected
rather than silently selecting the intrinsic production path.
The default build was verified with the option off, and an explicit option-on
build plus the assembly validator was also verified in the cleanup report.

## Configuration rule

Scheduler, A-packing, worker-affinity, and their diagnostic switches are
resolved into one per-call runtime snapshot before dispatch. Hot worker loops
consume the resolved enum/boolean values rather than re-reading those
environment variables. Kernel, layout, blocking, and prefetch selectors are
still resolved with the block/layout selection because they determine the
packed representation.

Detailed outcomes are preserved in
[`build/reports/jlc-cleanup-pass1-20260914/`](../build/reports/jlc-cleanup-pass1-20260914/)
and the earlier experiment reports referenced by the cleanup handoff.
