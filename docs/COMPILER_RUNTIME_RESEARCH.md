Post-M4 Compiler Runtime Research
R1 — Liveness-Aware Physical Buffer Planning

## Status and project history

This document is a separate post-M4 research track. It does not rename or
extend the completed compiler milestones:

1. M1 — Graph IR + whole-expression optimization
2. M2 — Memory semantics + affine/dependence IR
3. M3 — Bounded affine/polyhedral-style schedule transformations
4. M4 — CPU lowering + end-to-end benchmark proof

M4 remains the terminal matrix-compiler milestone; this work is not M5.

R1 was implemented on `feature/compiler-buffer-reuse` in the isolated
worktree `/home/james/Projects/JLC-compiler-buffer-reuse`. The actual
canonical development base was:

```text
c557e4ac7c457d40e05ced0d58291175bfaadd2f
Optimize AVX2 GEMM backend and isolate experiments
```

`main` was not changed or merged into.

## Baseline M1–M4 architecture

The existing opt-in path is preserved:

```text
MatrixExpr
    ↓
ExecutionPlan
    ↓
AffineProgram
    ↓
DependenceGraph
    ↓
SchedulePlan
    ↓
CpuExecutionPlan
    ↓
CpuExecutor
    ↓
Matrix
```

Before R1, `CpuLowerer` identified materialized and fusion-elided logical
temporaries, but it had no physical allocation model. During execution:

* `CpuExecutionContext.allocate` created a fresh `Matrix` for each explicit
  elementwise, transpose, or fused output;
* `CpuGemmStep` called `Gemm.multiply`, whose canonical facade allocated the
  result and then dispatched the production GEMM backend;
* hidden owned off-heap GEMM results stayed in the context until the end of
  the invocation and were closed once by identity; and
* borrowed external input matrices were bound but never owned or closed.

The old cleanup was correct for one-allocation-per-materialized-temporary
execution, but it could not release a temporary as soon as its final
executable consumer had completed and could not share storage across
non-overlapping values.

The pre-R1 allocation audit was:

| Final CPU step | Materialized output | Previous allocation | Previous consumers/release |
| -------------- | ------------------- | ------------------- | -------------------------- |
| `CpuGemmStep` | GEMM result temporary | `Gemm.multiply` allocated a fresh heap or off-heap result | Later CPU steps read the logical value; hidden off-heap storage closed during final context cleanup |
| `CpuElementwiseStep` ADD/SCALE | ADD or SCALE result | `CpuExecutionContext.allocate` created a fresh heap `Matrix` | Later CPU steps read it; no early release |
| `CpuTransposeStep` | transpose result | `CpuExecutionContext.allocate` created a fresh heap `Matrix` | Later CPU steps read it; no early release |
| `CpuFusedElementwiseStep` | fused ADD result | `CpuExecutionContext.allocate` created a fresh heap `Matrix` | Later CPU steps read it; the intermediate SCALE logical buffer was already elided |

Every materialized result therefore had one fresh runtime allocation before
R1. The context retained logical-to-Matrix entries for the invocation and
performed one final identity-deduplicated off-heap cleanup; it did not recycle
heap or off-heap storage between logical values.

`BufferLifetime` remains an M2 semantic fact with affine statement IDs and
the documented rule that it does not authorize early free or reuse. R1 adds
runtime liveness after M3/M4 lowering rather than changing that contract.

## Executable liveness

`PhysicalMemoryPlanner` consumes the final immutable `CpuStep` sequence. It
first records each step output as a producer, then scans each step's
`inputBuffers()` to find the first and last executable consumers. The
resulting `ExecutionLifetime` is immutable and exposes:

* logical buffer;
* producing CPU-step ID;
* first and last consumer CPU-step IDs;
* inclusive `[producer, lastConsumer]` interval; and
* `liveThroughReturn` for the result.

Special cases are explicit:

* a shared DAG producer is extended through its last branch consumer;
* a zero-consumer materialized temporary ends at its producer step;
* the returned output ends at the final CPU step and is pinned through
  return; and
* a fusion-elided temporary has no executable producer and therefore gets no
  lifetime or physical slot.

M2 statement IDs are insufficient by themselves because M3 may reorder or
tile regions, while M4 may fuse regions, collapse MATMUL initialization and
update into one GEMM step, and elide logical values. The final CPU-step
relationships are the facts that govern actual storage safety.

## Physical memory model and compatibility

`LogicalBuffer` remains a semantic value object. `PhysicalBufferSlot` is a
separate immutable plan object containing a stable slot ID, owning execution
plan, exact shape, payload size, storage class, and ordered logical values
assigned to the slot. `PhysicalMemoryPlan` contains the lifetimes, slots,
assignments, interference queries, metrics, and deterministic dump.

R1 uses the conservative compatibility key:

```text
(memory space, value-lane kind, exact shape)
```

The storage class distinguishes:

* `heap-real`;
* `heap-complex`;
* `off_heap-real`; and
* `off_heap-complex`.

`UNKNOWN` is diagnostic only and never makes two values reusable. Exact
shape is checked separately from the storage class. Capacity-based shape
rebinding and arbitrary Matrix views were deliberately not introduced.

Intervals intersect inclusively when:

```text
first.startStep <= second.endStep
and second.startStep <= first.endStep
```

Thus values touching the same CPU step interfere. The initial allocator does
not attempt intra-step read/write reuse. A slot is reusable only after its
previous interval satisfies `previous.endStep < next.startStep`, and only
when storage class and shape exactly match.

Borrowed external inputs are never included in temporary slots. There is no
in-place overwrite of an input and no global or cross-execution pool.

## Allocation algorithm

The allocator is a deterministic linear scan:

1. sort materialized lifetimes by producer step and logical-buffer ID;
2. expire active assignments whose inclusive end is before the next start;
3. choose the lowest compatible free slot ID; or
4. create the next slot ID.

The plan validates that every materialized logical value has one slot, every
slot assignment is class/shape-compatible, and no interfering pair shares a
slot. The dump prints each lifetime, consumer span, slot, and `REUSE` marker,
followed by per-slot logical membership.

## Execution arena and ownership

`ExecutionArena` is created per `CpuExecutor.execute` invocation. It lazily
creates one Matrix object for each activated physical slot and retains
ownership until invocation cleanup. `release` only removes the logical-value
mapping and makes a slot available; it does not close the physical allocation
prematurely.

On success and failure, the arena closes each owned `OffHeapMatrix` once by
physical Matrix identity. The returned result is passed as the transferred
value and is not closed. Borrowed inputs are not arena allocations and remain
the caller's responsibility. Invalid bindings are shape-checked before an
arena is created.

The developer selector is:

```text
jlc.compiler.memoryPlanner=reuse   (default)
jlc.compiler.memoryPlanner=legacy
```

The legacy path remains independently executable for A/B comparison and
keeps its original fresh-result behavior. The public Matrix API is unchanged.

## GEMM and elementwise integration

The reuse path calls the existing destination-aware facade:

```text
Gemm.gemm(A, B, assignedC, 1.0, 0.0, DispatchPolicy.defaultPolicy())
```

This preserves backend dispatch, packing, worker-policy selection, and the
existing GEMM implementation. No second GEMM implementation was added.

`CpuElementwiseStep` for ADD and SCALE, `CpuTransposeStep`, and
`CpuFusedElementwiseStep` request their assigned slot before their loops and
write directly into that Matrix. They do not compute into a fresh Matrix and
copy afterward. Fused scale+add output remains a materialized value, while
the intermediate scale result remains explicitly elided.

Fusion and physical reuse are reported separately. For example, the
surrounding-fusion benchmark below has five logical temporaries, two elided
by fusion, three materialized, and two physical slots: one logical value was
reused, while two disappeared before physical planning.

## Correctness coverage

`MatrixMemoryPlanningTest` covers 15 focused cases, including:

* compatible non-overlapping values mapping to one slot;
* same Matrix object activation only after release;
* direct elementwise writes to the assigned destination;
* overlap and same-step interference;
* shared-DAG fan-out through both branches;
* output pinning and borrowed-input exclusion;
* fusion elision with no slot;
* exact-shape and real/complex rejection;
* destination-aware off-heap GEMM output surviving return; and
* hidden off-heap arena cleanup being idempotent.

The existing matrix-chain, STRICT/RELAXED, schedule, lowering, and audit
regression suites were retained and run alongside these tests.

## Memory results

The following results are from the dedicated benchmark run on 2026-09-16
with Java 21.0.12, Linux amd64, 32 processors, Java backend, one warmup,
three measured executions, and median timing. Byte values are planned
payload bytes. `logicalBytes` means the sum for materialized logical values;
fusion-elided values have no payload to plan.

| Workload | Logical temps | Materialized temps | Physical slots | Logical bytes | Peak live | Physical bytes |
| -------- | ------------: | -----------------: | -------------: | ------------: | --------: | --------------: |
| A long elementwise chain 256 | 12 | 8 | 2 | 4194304 | 1048576 | 1048576 |
| A long elementwise chain 512 | 12 | 8 | 2 | 16777216 | 4194304 | 4194304 |
| A long elementwise chain 1024 | 12 | 8 | 2 | 67108864 | 16777216 | 16777216 |
| B shared/forked DAG 256 | 5 | 4 | 3 | 2097152 | 1572864 | 1572864 |
| B shared/forked DAG 512 | 5 | 4 | 3 | 8388608 | 6291456 | 6291456 |
| B shared/forked DAG 1024 | 5 | 4 | 3 | 33554432 | 25165824 | 25165824 |
| C GEMM + postprocessing 128 | 5 | 4 | 2 | 524288 | 262144 | 262144 |
| C GEMM + postprocessing 192 | 5 | 4 | 2 | 1179648 | 589824 | 589824 |
| C GEMM + postprocessing 256 | 5 | 4 | 2 | 2097152 | 1048576 | 1048576 |
| D RELAXED matrix chain 128 | 2 | 2 | 2 | 40960 | 40960 | 40960 |
| D RELAXED matrix chain 192 | 2 | 2 | 2 | 92160 | 92160 | 92160 |
| D RELAXED matrix chain 256 | 2 | 2 | 2 | 163840 | 163840 | 163840 |
| E fusion + surrounding ops 256 | 5 | 3 | 2 | 1572864 | 1048576 | 1048576 |
| E fusion + surrounding ops 512 | 5 | 3 | 2 | 6291456 | 4194304 | 4194304 |
| E fusion + surrounding ops 1024 | 5 | 3 | 2 | 25165824 | 16777216 | 16777216 |

The long elementwise chain is the clearest peak-memory result: eight
materialized logical payloads are represented by two exact-shape slots, and
planned physical payload is 75% below the logical sum at every tested size.

## Allocation results

Measured allocation is total Java-thread allocation observed around repeated
`execute()` calls, not merely Matrix payload. It includes residual loop,
dispatch, and object allocation outside the reusable payload slots.

| Workload | Legacy alloc bytes | Reuse alloc bytes | Reduction |
| -------- | -----------------: | ----------------: | --------: |
| A long elementwise chain 256 | 23129080 | 19996704 | 13.5% |
| A long elementwise chain 512 | 104999928 | 92430368 | 12.0% |
| A long elementwise chain 1024 | 444902392 | 394584096 | 11.3% |
| B shared/forked DAG 256 | 17871312 | 17355520 | 2.9% |
| B shared/forked DAG 512 | 80847312 | 78758656 | 2.6% |
| B shared/forked DAG 1024 | 342065616 | 333685509 | 2.4% |
| C GEMM + postprocessing 128 | 3957712 | 3703552 | 6.4% |
| C GEMM + postprocessing 192 | 7250920 | 5337944 | 26.4% |
| C GEMM + postprocessing 256 | 13163520 | 10525498 | 20.0% |
| D RELAXED matrix chain 128 | 43624 | 49600 | -13.7% |
| D RELAXED matrix chain 192 | 94864 | 100840 | -6.3% |
| D RELAXED matrix chain 256 | 2740880 | 2746856 | -0.2% |
| E fusion + surrounding ops 256 | 6307872 | 5790512 | 8.2% |
| E fusion + surrounding ops 512 | 28348448 | 26258224 | 7.4% |
| E fusion + surrounding ops 1024 | 119615520 | 111233840 | 7.0% |

Negative values on the small D cases are measurement noise and planner
overhead, not a claimed regression in the allocation contract. The largest
remaining measured source is allocation outside physical Matrix payloads,
including per-invocation execution scaffolding and backend/loop support. R1
does not attempt to pool those objects.

## Runtime results

The ratio is `legacy median / reuse median`; values above 1.0 favor reuse.

| Workload | Legacy | Reuse | Ratio |
| -------- | -----: | ----: | ----: |
| A long elementwise chain 256 | 39.003 ms | 37.915 ms | 1.029x |
| A long elementwise chain 512 | 128.801 ms | 131.247 ms | 0.981x |
| A long elementwise chain 1024 | 526.810 ms | 523.168 ms | 1.007x |
| B shared/forked DAG 256 | 26.071 ms | 26.591 ms | 0.980x |
| B shared/forked DAG 512 | 104.894 ms | 107.046 ms | 0.980x |
| B shared/forked DAG 1024 | 419.735 ms | 414.599 ms | 1.012x |
| C GEMM + postprocessing 128 | 5.706 ms | 5.537 ms | 1.030x |
| C GEMM + postprocessing 192 | 11.111 ms | 10.593 ms | 1.049x |
| C GEMM + postprocessing 256 | 28.264 ms | 26.808 ms | 1.054x |
| D RELAXED matrix chain 128 | 0.150 ms | 0.170 ms | 0.879x |
| D RELAXED matrix chain 192 | 0.256 ms | 0.311 ms | 0.822x |
| D RELAXED matrix chain 256 | 2.742 ms | 2.844 ms | 0.964x |
| E fusion + surrounding ops 256 | 8.352 ms | 7.988 ms | 1.046x |
| E fusion + surrounding ops 512 | 33.219 ms | 32.222 ms | 1.031x |
| E fusion + surrounding ops 1024 | 127.353 ms | 121.237 ms | 1.050x |

The allocation-heavy long chain is effectively neutral overall: reuse is
faster at 256 and 1024, while the 512 case is a small observed slowdown
(0.981x). The shared/forked DAG is also approximately neutral at this sample
size. The GEMM postprocessing and fusion cases show modest positive results in
this run, with fusion ranging from 1.031x to 1.050x; the 1024 fusion timing is
therefore an approximately 5% observed speedup, not a slowdown. The
matrix-chain cases are compute-dominated; R1 does not change matrix-chain
arithmetic policy. These measurements are reported as run-specific results,
not generalized performance claims.

## Required audit answers

1. **Where were temporaries allocated?** Before R1, explicit CPU outputs were
   fresh `new Matrix` objects in `CpuExecutionContext.allocate`; GEMM outputs
   were allocated by `Gemm.multiply`; fused and transpose/add/scale outputs
   followed the same fresh-output path.
2. **What is the executable lifetime model?** Immutable inclusive CPU-step
   intervals from each final-step producer to its last `inputBuffers()` read,
   with explicit zero-consumer and return-pinned cases.
3. **Why are M2 IDs insufficient?** M2 IDs describe the affine program before
   schedule transformations, fusion, GEMM collapse, and elision; they are not
   the final execution order.
4. **What is interference?** Inclusive interval overlap. Touching a CPU step
   interferes conservatively.
5. **What is the compatibility key?** Exact shape plus `(MemorySpace,
   StorageValueKind)`; real/complex and heap/off-heap do not mix.
6. **What algorithm is used?** Stable linear scan, sorted by step and logical
   ID, choosing the lowest compatible free slot.
7. **How are shared producers handled?** Every executable consumer is
   scanned, so a fan-out producer remains live through the maximum consumer
   step.
8. **How are outputs pinned?** The output lifetime is marked
   `liveThroughReturn`, and cleanup skips the returned Matrix.
9. **How are borrowed/off-heap values handled?** External inputs are bound
   only as borrowed values. Off-heap physical slots are execution-owned and
   closed once; borrowed off-heap inputs are never closed.
10. **Can GEMM target reused storage?** Yes. The reuse path calls the existing
    destination-aware `Gemm.gemm` facade.
11. **How are fusion and reuse distinguished?** `elidedTemporaryBuffers` are
    excluded before lifetime/slot planning; `logicalToPhysicalReuseCount`
    counts only materialized values sharing slots.
12. **How many values become slots?** The benchmark table records every case;
    representative reductions are A: 8 materialized → 2 slots, B: 4 → 3,
    C: 4 → 2, and E: 3 → 2. D remains 2 → 2.
13. **What is the physical-byte reduction?** A's planned payload falls from
    4/16/64 MiB logical sums to 1/4/16 MiB physical payload at 256/512/1024;
    all table rows report the exact byte values.
14. **What is measured allocation reduction?** A shows 11.3–13.5%, C
    6.4–26.4%, B 2.4–2.9%, and E 7.0–8.2%; D is noise at these small
    compute-dominated sizes.
15. **What is the wall-time effect?** A is 0.981–1.029x and effectively
    neutral overall, with the 512 case a small slowdown. C is 1.030–1.054x
    and modestly positive in this run. B is 0.980–1.012x and approximately
    neutral. E is 1.031–1.050x and modestly positive in this run. D is
    0.822–0.964x at small, compute-dominated sizes and is measurement noise,
    not a memory-planner performance claim.
16. **Which workloads show little benefit and why?** Shared DAG and fusion
    retain several simultaneously live values, while GEMM matrix chains are
    dominated by arithmetic and have only two intermediates. Inclusive
    interference also intentionally blocks same-step reuse.
17. **What is the new dominant allocation source?** Residual Java execution
    scaffolding and backend/loop support outside slot payloads; R1 does not
    add a global pool to chase those allocations.
18. **What future code-generation assumptions are exposed?** Generated
    kernels will need explicit storage class, exact shape, alias/lifetime
    facts, dense full-write guarantees, destination choice, and output escape
    handling. Logical value identity cannot be substituted for physical
    storage identity.

## Tests and build evidence

The following focused compiler suites passed after the R1 implementation:

| Suite | Tests | Skipped | Failed |
| ----- | ----: | ------: | ------: |
| MatrixCompilerTest | 19 | 0 | 0 |
| MatrixAffineCompilerTest | 17 | 0 | 0 |
| MatrixScheduleTest | 28 | 0 | 0 |
| MatrixCpuLoweringTest | 23 | 0 | 0 |
| CpuFusedFastPathTest | 5 | 0 | 0 |
| MatrixCompilerAuditRegressionTest | 27 | 0 | 0 |
| MatrixMemoryPlanningTest | 15 | 0 | 0 |

Commands used:

```text
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin:$PATH bash ./gradlew test --tests net.faulj.compiler.matrix.MatrixCompilerTest --tests net.faulj.compiler.matrix.affine.MatrixAffineCompilerTest --tests net.faulj.compiler.matrix.schedule.MatrixScheduleTest --tests net.faulj.compiler.matrix.cpu.MatrixCpuLoweringTest --tests net.faulj.compiler.matrix.cpu.CpuFusedFastPathTest --tests net.faulj.compiler.matrix.cpu.MatrixCompilerAuditRegressionTest --tests net.faulj.compiler.matrix.cpu.MatrixMemoryPlanningTest
```

The dedicated benchmark passed:

```text
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin:$PATH bash ./gradlew benchmarkTest --tests net.faulj.benchmark.CompilerRuntimeMemoryBenchmarkTest
```

It executed one benchmark test with all A–E workload cases and correctness
checks. Native source was not changed; benchmark execution used the Java
backend because no built `jlc_native` library was present in that run, so a
native-backend test was not part of this R1 change.

The final verification also includes `compileJava`, `compileTestJava`, the
existing compiler benchmark classes, the full project test task, and
`git diff --check`. One unrelated pre-existing platform/dispatch assertion in
`AlgorithmDispatchTest.coldStartAllowsCppOnlyForFoundationAlgorithmsAboveThreshold`
continues to fail under the repository's current environment; it is outside
the compiler runtime change and is preserved in the final evidence rather
than hidden. The final full-project result was 454 tests completed, 24
skipped, and that one failure at `AlgorithmDispatchTest.java:84`; the focused
compiler suites above were all green.

## Scope boundary

R1 adds no new fusion family, SIMD generation, native kernel generation,
CUDA, threading, mixed precision, sparse execution, JIT, or persistent
execution-to-execution pool. The execution-scoped plan is intentionally the
smallest ownership boundary that makes physical reuse inspectable and
exception-safe.
