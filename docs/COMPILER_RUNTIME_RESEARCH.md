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

The frozen CPU/runtime path is therefore:

```text
MatrixExpr -> M1 -> M2 -> M3 -> M4 CPU lowering
           -> R1 -> R2 -> R3 -> R4 -> R5
           -> typed backend calibration -> KernelDispatchSelector
                |-> R2 Java
                |-> scalar native
                `-> generated AVX2
```

R1–R5 are post-M4 research layers. The typed backend boundary is a narrow
production integration pass; it does not add a compiler milestone, and there
is no R6 in this checkpoint.

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

# R2 — Generalized Legality-Aware Fusion

## Status and base

R2 is a separate post-M4 compiler/runtime research milestone. M4 remains the
terminal historical matrix-compiler milestone; this section does not rename,
rewrite, or extend the M1–M4 history above.

Implementation branch and worktree:

```text
feature/compiler-generalized-fusion
/home/james/Projects/JLC-r2
```

The actual clean R2 base was the R1 branch HEAD:

```text
315a21f28e970fe8f52e24dfc46366427b440242
Add post-M4 physical buffer planning
```

`main` was not changed or merged.

## Motivation

R1 and R2 remove different costs:

| Research stage | What it removes |
| -------------- | ---------------- |
| R1 | Repeated physical storage by reusing slots after a logical value exists |
| R2 | The runtime Matrix materialization of an internal logical value when its scalar value can be composed safely |

R2 therefore runs before CPU-step liveness and physical-memory planning:

```text
final SchedulePlan
    ↓
generalized legality-aware fusion
    ↓
materialization decisions
    ↓
CpuExecutionPlan
    ↓
ExecutionLifetime / PhysicalMemoryPlan
    ↓
CpuExecutor
```

An R2-eliminated temporary never enters R1's executable lifetime or slot
assignment. A nonfusible temporary remains visible and can still benefit from
R1 reuse.

## Baseline special-case audit

The old `CpuLowerer.fusableRegion` and `FusedSpec` assumed exactly two
statements in one region, with a `SCALE` first and an `ADD` second. They stored
two operand buffers, one scale factor, one eliminated buffer, and one boolean
for add-operand order. `CpuFusedElementwiseStep` then recognized only a
two-loop, zero-based, unit-step identity traversal for its raw real fast path;
all other cases used `CpuLoopExecutor`, which created binding maps and index
arrays at scalar points. The old generic branch also contained the explicit
complex-lane behavior that must be preserved.

R2 keeps that class independently executable for the `legacy` selector and
for historical tests. It does not use that pair-specific representation for
generalized regions. `ScheduleLegality.checkFusion` remains the M3 legality
gate for M3 transformations and existing explicit two-region schedules; the
post-M4 planner applies a stronger multi-statement check to its own candidate
regions.

The developer selector is:

```text
jlc.compiler.fusion=off          # no new fusion
jlc.compiler.fusion=legacy       # original M4 scale/add pair
jlc.compiler.fusion=generalized  # R2 planner
```

The default remains `legacy` so the original M4 baseline is reproducible.
Callers can also pass `FusionStrategy` directly to the compiler/lowerer APIs.

## Fusion model

The generalized representation consists of:

* `FusedRegionPlan`: one final output buffer, final output access, schedule
  band, iteration domain, source statements, leaf buffers, internal values,
  eliminated temporaries, legality evidence, and profitability text;
* `ScalarFusionProgram`: an immutable topologically ordered scalar SSA-like
  list with `LOAD`, `SCALE`, `ADD`, and semantic `TRANSPOSE` nodes; and
* `CpuFusedRegionStep`: the executable step that exposes only leaf inputs to
  R1 and materializes only the final output.

The region is single-output by design. Every internal producer must be
consumed entirely inside the candidate. If it escapes, the candidate is
rejected and the producer remains materialized. This avoids inventing a
multi-output kernel or silently recomputing a value for a second branch.

Candidate discovery is deterministic and bounded:

1. scan schedule regions in their existing order;
2. grow across adjacent eligible `ADD`/`SCALE`/`TRANSPOSE` regions only when
   the next region consumes a value already produced by the candidate;
3. validate the maximal candidate, trying shorter connected prefixes only as
   a deterministic fallback;
4. reject unsafe candidates and leave their statements materialized; or
5. emit one region step and advance past all accepted regions.

There is no ML search, autotuning, algebraic rewrite, broadcasting, or
combinatorial candidate enumeration.

## Scalar SSA and numerical ordering

The scalar builder recursively composes each final-output access through its
internal producer. A scalar value is cached by `(source statement,
composed affine indices)`, so a shared DAG producer such as `2*A[p]` is
defined once and can have multiple SSA users. `sharedProducerCount` reports
internal definitions with more than one scalar use.

The builder preserves source read order and statement order. Thus
`scale(add(A,B), alpha)` is represented as `LOAD A`, `LOAD B`, `ADD`,
`SCALE`; it is not reassociated into two scaled loads. STRICT remains STRICT,
and R2 adds no reassociation permission. No FMA node or FMA substitution is
present.

## Legality and affine maps

The planner requires:

* only one-write, non-reduction `ADD`, `SCALE`, or `TRANSPOSE` statement per
  supported source form;
* proven dependence order for all touched graph relationships;
* no `UNKNOWN` dependence or alias fact touching a candidate;
* a complete two-dimensional schedule binding for the final domain;
* unit-coefficient identity/permutation output maps and compatible extents;
* no internal temporary consumer outside the region; and
* a temporary final output and at least one eliminated internal value.

The `UNKNOWN` state is never treated as independence. Reversed in-region
dependences, unsupported access ranks/maps, mismatched domains, and escaping
values reject fusion with a deterministic decision string. GEMM initialization
and reduction update are non-eligible boundaries. A GEMM result may be a leaf
of a later fused elementwise region, but R2 does not fuse into the GEMM.

The canonical iteration space is the final statement's output domain. For a
transpose producer, the producer write map is inverted to bind its source
variables to the consumer's requested output indices; those indices are then
substituted into the producer's reads. For example, a rectangular
`A:MxN -> transpose(A):NxM` chain produces loads `A[j,i]` while storing the
`NxM` output at `[i,j]`. Repeated transpose maps compose through this same
identity/permutation mechanism. General affine expressions outside this
proven subset are rejected.

## CPU lowering paths

`CpuFusedRegionStep` compiles all schedule and scalar access expressions once.
The generic schedule path uses one reusable binding array, validity array,
and scalar real/imaginary arrays per invocation; it performs no map, stream,
lambda-capture, or index-array allocation per matrix point. Complex execution
tracks imaginary-lane presence separately from the numeric value, preserving
real/complex add combinations, complex scale, transpose, and signed zero in
an existing lane.

For ordinary heap `Matrix` values with real lanes, owned output, no alias
hazard, and a two-loop identity/permutation traversal, the fast path uses raw
real arrays and a compact scalar-node array. Its inner loop does not call
`Matrix.get`, `Matrix.set`, or perform map lookups. It remains scalar Java
execution: no SIMD, Kernel IR, FMA, JIT, or generated pseudokernel was added.
The step records `FAST_REAL_DENSE` or `GENERIC_SCHEDULE` for benchmark
diagnostics. The legacy step reports its original fast/generic path separately.

## Profitability rule and metrics

The current deterministic rule is intentionally small: fuse only a connected
legal candidate with at least two statements and at least one internal
materialization to eliminate. The planner reports the number of eliminated
full-buffer writes/reads, collapsed passes, scalar-node counts, and a note
when transpose mapping is present. It does not claim a hardware traffic
measurement. `FusionMetrics` exposes the requested candidate, acceptance,
elision, estimated logical traffic, pass, region-size, and scalar-sharing
fields.

## Materialization results

The following rows are from the opt-in R2 benchmark with R1 `reuse` enabled,
one measured size per workload, Java 25.0.4, Linux amd64, 32 processors,
two warmups, five timed executions, and three allocation samples. Payload
bytes are logical estimates; compilation is outside timing.

| Workload | Logical temps | Fusion-elided | Materialized | R1 slots |
| -------- | ------------: | ------------: | -----------: | -------: |
| A long chain 256 | 11 | 10 | 1 | 1 |
| B shared DAG 256 | 5 | 4 | 1 | 1 |
| C transpose 512x128 | 4 | 3 | 1 | 1 |
| C transpose 128x512 | 4 | 3 | 1 | 1 |
| C transpose 512x512 | 4 | 3 | 1 | 1 |
| D GEMM + epilogue 128 | 5 | 3 | 2 | 2 |
| E fusion + R1 256 | 8 | 3 | 4 | 2 |
| F escaping fan-out 256 | 4 | 0 | 4 | 3 |

The F row is intentionally conservative: the shared producer escapes to a
GEMM branch, so generalized fusion reports the boundary and creates no fused
region.

## Pass and logical-traffic results

`passes` means estimated full-matrix CPU steps before/after fusion. Read/write
values are estimated real-lane payload bytes, not DRAM-counter values.

| Workload | Passes before | Passes after | Logical read before | Logical read after | Logical write before | Logical write after |
| -------- | ------------: | -----------: | ------------------: | -----------------: | -------------------: | ------------------: |
| A long chain 256 | 11 | 1 | 7864320 | 2621440 | 5767168 | 524288 |
| B shared DAG 256 | 5 | 1 | 3670016 | 1048576 | 2621440 | 524288 |
| C transpose 512x128 | 4 | 1 | 2621440 | 1048576 | 2097152 | 524288 |
| C transpose 128x512 | 4 | 1 | 2621440 | 1048576 | 2097152 | 524288 |
| C transpose 512x512 | 4 | 1 | 10485760 | 4194304 | 8388608 | 2097152 |
| D GEMM + epilogue 128 | 5 | 2 | 1048576 | 655360 | 655360 | 262144 |
| E fusion + R1 256 | 7 | 4 | 5767168 | 4194304 | 3670016 | 2097152 |
| F escaping fan-out 256 | 4 | 4 | 3670016 | 3670016 | 2097152 | 2097152 |

## Required fusion-mode performance table

`runtime` is the median execution time from the same benchmark invocation.
`path` identifies the first fused region when present; `MATERIALIZED` means
the mode did not emit a fused step.

| Workload | Fusion mode | Statements | Elided temps | Materialized | Physical slots | Passes | Runtime ms | Path |
| -------- | ----------- | ---------: | -----------: | -----------: | -------------: | -----: | ----------: | ---- |
| A 256 | off | 11 | 0 | 11 | 2 | 11/11 | 104.944775 | MATERIALIZED |
| A 256 | legacy | 11 | 4 | 7 | 2 | 11/7 | 22.085754 | LEGACY_FAST_REAL_IDENTITY |
| A 256 | generalized | 11 | 10 | 1 | 1 | 11/1 | 3.558079 | FAST_REAL_DENSE |
| B 256 | off | 5 | 0 | 5 | 3 | 5/5 | 41.794082 | MATERIALIZED |
| B 256 | legacy | 5 | 1 | 4 | 3 | 5/4 | 24.206869 | LEGACY_FAST_REAL_IDENTITY |
| B 256 | generalized | 5 | 4 | 1 | 1 | 5/1 | 1.280874 | FAST_REAL_DENSE |
| C 512x128 | off | 4 | 0 | 4 | 2 | 4/4 | 31.600547 | MATERIALIZED |
| C 512x128 | legacy | 4 | 1 | 3 | 2 | 4/3 | 14.249785 | LEGACY_FAST_REAL_IDENTITY |
| C 512x128 | generalized | 4 | 3 | 1 | 1 | 4/1 | 1.278313 | FAST_REAL_DENSE |
| C 128x512 | off | 4 | 0 | 4 | 2 | 4/4 | 30.887565 | MATERIALIZED |
| C 128x512 | legacy | 4 | 1 | 3 | 2 | 4/3 | 14.580075 | LEGACY_FAST_REAL_IDENTITY |
| C 128x512 | generalized | 4 | 3 | 1 | 1 | 4/1 | 1.252573 | FAST_REAL_DENSE |
| C 512x512 | off | 4 | 0 | 4 | 2 | 4/4 | 124.281342 | MATERIALIZED |
| C 512x512 | legacy | 4 | 1 | 3 | 2 | 4/3 | 58.643033 | LEGACY_FAST_REAL_IDENTITY |
| C 512x512 | generalized | 4 | 3 | 1 | 1 | 4/1 | 6.761787 | FAST_REAL_DENSE |
| D 128 | off | 6 | 0 | 5 | 2 | 5/5 | 9.964444 | MATERIALIZED |
| D 128 | legacy | 6 | 2 | 3 | 2 | 5/3 | 1.438774 | LEGACY_FAST_REAL_IDENTITY |
| D 128 | generalized | 6 | 3 | 2 | 2 | 5/2 | 1.748415 | FAST_REAL_DENSE |
| E 256 | off | 8 | 0 | 7 | 2 | 7/7 | 55.168755 | MATERIALIZED |
| E 256 | legacy | 8 | 2 | 5 | 2 | 7/5 | 17.937344 | LEGACY_FAST_REAL_IDENTITY |
| E 256 | generalized | 8 | 3 | 4 | 2 | 7/4 | 10.944436 | FAST_REAL_DENSE |
| F 256 | off | 5 | 0 | 4 | 3 | 4/4 | 28.524669 | MATERIALIZED |
| F 256 | legacy | 5 | 0 | 4 | 3 | 4/4 | 27.265426 | MATERIALIZED |
| F 256 | generalized | 5 | 0 | 4 | 3 | 4/4 | 27.590708 | MATERIALIZED |

Relative to `off` in this run, generalized execution was approximately
29.50x faster for A, 32.63x for B, 24.72x/24.66x/18.38x for the three C
shapes, 5.70x for D, and 5.04x for E. F remained essentially break-even by
design because it rejected fusion. D's generalized epilogue was slightly
slower than the legacy pair specialization in this small GEMM sample; the
R2 claim is the reduced passes and materializations, not a universal speedup.

## Allocation and R1 interaction

The benchmark uses R1 reuse for all three fusion modes, so the allocation
comparison isolates fusion mode while keeping the physical-storage policy
constant. `off` is the no-fusion baseline; `legacy` is the old pair; and
`generalized` is R2.

| Workload | Off alloc bytes | Legacy alloc bytes | Generalized alloc bytes |
| -------- | --------------: | ------------------: | ----------------------: |
| A 256 | 59425349 | 15261952 | 531256 |
| B 256 | 28395936 | 17355520 | 530992 |
| C 512x128 | 22340064 | 9747392 | 530984 |
| C 128x512 | 17682912 | 9747392 | 530984 |
| C 512x512 | 98716128 | 48313280 | 2103848 |
| D 128 | 6070136 | 2118496 | 2117680 |
| E 256 | 34188760 | 12099416 | 5792328 |
| F 256 | 18928928 | 18928928 | 18928928 |

R2's larger allocation reduction in long elementwise cases is primarily the
absence of intermediate Matrix payloads. R1 still handles the remaining
materialized values: E has four materialized logical temporaries and two
physical slots, while D has two and two. Fusion-elided values have no
`ExecutionLifetime`, no `PhysicalBufferSlot`, and are absent from the arena.

## Numerical validation

The dedicated generalized suite has coverage for:

* real/real, real/complex, complex/real, and complex/complex addition;
* complex scaling and transpose with explicit imaginary-lane presence;
* NaN, positive infinity, negative zero, and one-by-one values;
* rectangular `2x3`, `3x2`, `37x11`, and `11x37` transpose pipelines;
* identity and transpose-composed loads;
* 7+ operation chains and shared scalar DAG producers;
* tiled generic schedule execution and zero-sized matrices;
* selector `off`, `legacy`, and `generalized` A/B behavior;
* GEMM/reduction boundaries; and
* an escaping fan-out that records a rejection rather than eliminating a
  shared producer.

Strict arithmetic association is preserved by comparing the fused result to
the existing eager evaluator without loosening the comparison for avoidable
differences. The real fast path uses the same scalar operation sequence as
the generic path, and the complex path copies a lone imaginary lane instead
of adding an artificial `+0.0`, matching the current Matrix API.

## Required audit answers

1. **What made the old fusion special?** It required exactly SCALE then ADD,
   two equal domains, one scale result consumed once, and one recovered scalar
   factor; its fast path required identity traversal and three ordinary Matrix
   instances.
2. **What is new?** `FusedRegionPlan` plus `ScalarFusionProgram` plus
   `CpuFusedRegionStep`, with leaves, internal values, maps, statements,
   schedule, legality, and materialization evidence.
3. **How is SSA represented?** Immutable contiguous scalar nodes in
   topological order; loads carry composed affine indices, arithmetic nodes
   carry earlier IDs, and scale nodes carry a factor.
4. **How is sharing preserved?** The scalar builder caches by statement and
   composed index, so one scalar definition can have multiple uses and the
   program reports those uses.
5. **How are escapes detected?** Every internal producer buffer is scanned
   against all program consumers; any read by a statement outside the region
   rejects elimination.
6. **Why single output?** It keeps ownership and R1's one-output step contract
   explicit and avoids multi-output kernels or hidden recomputation.
7. **How are transpose maps composed?** Invert each unit permutation write
   map, substitute the requested indices into its reads, and validate extents
   against the final domain and leaf shape.
8. **What legality facts are required?** Proven dependence order, supported
   pointwise maps, complete compatible domains, no escaping internal values,
   and one temporary output.
9. **Which UNKNOWN cases reject?** Every UNKNOWN alias/dependence touching a
   candidate rejects; it is never downgraded to independence.
10. **How is STRICT ordering preserved?** R2 composes storage and loop
    boundaries only; it retains statement/read order and does no reassociation.
11. **How are complex values executed?** Real and imaginary arrays are
    evaluated in parallel, with a boolean lane-presence value per scalar
    node and eager-compatible output lane allocation.
12. **How are edge values validated?** Fused outputs are compared with eager
    results for NaN, infinity, signed zero, tiny/ordinary values, and complex
    lane variants using zero-tolerance assertions where applicable.
13. **What growth algorithm is used?** Ordered greedy maximal connected growth,
    bounded shorter-prefix retries, then one step per accepted region.
14. **What is profitability?** At least two legal statements and at least one
    eliminated full-matrix value; metrics record pass and traffic estimates,
    scalar size, and transpose complexity.
15. **How many passes are eliminated?** A: 11→1, B: 5→1, C: 4→1, D: 5→2,
    E: 7→4, and F: 4→4 because its candidate is rejected.
16. **How many materializations are eliminated?** The generalized rows show
    A 10, B 4, C 3, D 3, E 3, and F 0.
17. **What does R1 do afterward?** It recomputes executable lifetimes from
    the final R2 steps and linearly assigns only remaining materialized values
    to exact-shape compatible slots.
18. **What benefit is fusion versus reuse?** The mode table holds R1 reuse
    constant: generalized drops A from 11 to 1 materialized value and from
    59.4 MB to 0.53 MB sampled allocation; R1 separately maps E's four values
    to two slots.
19. **Where does fusion lose or break even?** The escaping fan-out is
    intentionally unchanged; the small D generalized epilogue is slower than
    the legacy pair in this run; complex/generic and tiny cases pay evaluator
    overhead.
20. **What should R3 consume?** A fused region's final domain/band, leaf
    storage descriptors, affine load maps, ordered scalar SSA, output store,
    alias/escape facts, and materialization/lifetime evidence.

## Remaining weaknesses

R2 deliberately has no reduction fusion, GEMM epilogues, broadcasting, SIMD,
Kernel IR, generated pseudokernels, JIT, multi-output regions, or empirical
fusion autotuning. The generic path is slower than the canonical raw path;
transpose maps are limited to proven unit permutations; unknown alias facts
remain conservative rejections; and R1 continues to manage remaining
materializations independently.

## Tests and build evidence

The final focused counts and full-project result are recorded below after the
R2 changes were compiled and tested:

| Suite | Tests | Skipped | Failed |
| ----- | ----: | ------: | ------: |
| New `GeneralizedFusionTest` | 17 | 0 | 0 |
| New `R2GeneralizedFusionBenchmarkTest` | 1 | 0 | 0 |
| MatrixCompilerTest | 19 | 0 | 0 |
| MatrixAffineCompilerTest | 17 | 0 | 0 |
| MatrixScheduleTest | 28 | 0 | 0 |
| MatrixCpuLoweringTest | 23 | 0 | 0 |
| CpuFusedFastPathTest | 5 | 0 | 0 |
| MatrixCompilerAuditRegressionTest | 27 | 0 | 0 |
| MatrixMemoryPlanningTest | 15 | 0 | 0 |
| Full project test | 472 | 24 | 1 unrelated pre-existing native BLAS dispatch failure |

The focused compiler/CPU run passed all 154 selected tests (including the
17-test generalized suite and the opt-in benchmark smoke test). The full
project run was not memory-limited; its single failure was reproduced when
`AlgorithmDispatchTest` was run alone:
`coldStartAllowsCppOnlyForFoundationAlgorithmsAboveThreshold`.
It is outside the matrix compiler changes and is retained as an explicit
environment/platform caveat rather than hidden.

The reproducible R2 benchmark command is:

```text
bash ./gradlew test --tests net.faulj.compiler.matrix.cpu.R2GeneralizedFusionBenchmarkTest --no-daemon -q -Djlc.compiler.r2.benchmark=true
```

The opt-in benchmark test defaults to the requested A/B/C/D/E/F size sets;
`jlc.compiler.r2.benchmark.sizes` and
`jlc.compiler.r2.gemm-sizes` can select a bounded smoke run, and
`jlc.compiler.r2.benchmark.memory=legacy` separates R2 mode behavior from
R1 reuse when desired. `git diff --check` is part of final verification.

# R3 — Portable Kernel IR

## Status and branch provenance

R3 is a separate post-M4 research milestone. It does not rename M4, does not
create M5, and does not rewrite the M1–M4 history. The historical and research
sequences remain:

```text
M1 Graph IR + whole-expression optimization
M2 Memory semantics + affine/dependence IR
M3 Bounded affine/polyhedral-style scheduling
M4 CPU lowering + end-to-end benchmark proof

R1 Liveness-Aware Physical Buffer Planning
R2 Generalized Legality-Aware Fusion
R3 Portable Kernel IR
R4 Generated SIMD Pseudokernels
R5 Empirical Kernel/Schedule Autotuning
```

The implementation was developed in this isolated worktree:

```text
branch:       feature/compiler-kernel-ir
worktree:     /tmp/jlc-compiler-kernel-ir
R3 base SHA:  45fdd7aca34ccd31783416631989c19bf43deec6
implementation SHA: e3af92eb7a5f38654e24eb891a21fe4cc4b05749
main:         untouched; no merge performed
```

The base is the actual clean HEAD of `feature/compiler-generalized-fusion`,
including the completed R1 and R2 work. The primary checkout had unrelated
uncommitted changes and was not used for R3 edits.

## Motivation

R2 and R3 intentionally answer different questions:

```text
R2: semantic fused scalar computation
R3: portable explicit kernel computation
```

`FusedRegionPlan` and `ScalarFusionProgram` already prove what scalar value is
needed at each output point. R3 adds the explicit machine-oriented boundary:
which logical buffers are read and written, which affine addresses are used,
which loops execute, which scalar values are defined, and which effects are
observable. That representation can be consumed later by scalar Java, the
Java Vector API, generated C++, AVX2/AVX-512, NEON/SVE, or CUDA backends
without rebuilding expression semantics.

R3 contains no vector register names, ISA instructions, tile search, FMA
selection, unrolling policy, or generated native code. It is portable kernel
IR, not AVX IR.

## IR model

The public immutable model is in
`net.faulj.compiler.matrix.kernel`:

| Object | Responsibility |
| ------ | -------------- |
| `KernelProgram` | Immutable container of kernel functions. |
| `KernelFunction` | Stable function ID/name, buffers, rectangular loops, flat body, SSA values, alias facts, and R2 provenance. |
| `KernelBuffer` | Logical buffer identity, M2 storage class, role, shape, storage/compute/accumulator types, layout, memory space, and ownership. |
| `KernelLoop` | One explicit loop range and semantic induction binding. |
| `KernelAccess` | A buffer plus explicit affine index expressions. |
| `KernelValue` | Stable SSA-like scalar value ID, name, and type. |
| `KernelOp` | One ordered scalar/load/store operation and source provenance. |
| `KernelBlock` | Immutable ordered operation list. |
| `KernelAliasFact` | Conservative pairwise alias fact. |
| `KernelProvenance` | Fused-region ID, source statements, source scalar nodes, eliminated buffers, and R2 legality/profitability evidence. |
| `KernelBinding` | Runtime-only map to `Matrix` objects and optional R1 physical slots. |

`KernelProgram` and all contained descriptors copy their collections and have
no mutators. A `KernelBuffer` retains its M2 `LogicalBuffer` source but never
retains a mutable `Matrix`; allocations stay in R1's execution arena. R3 v1
uses one output per function, matching the R2 `FusedRegionPlan` contract.

The flat body is deliberately more explicit than an expression tree: every
load, constant, arithmetic definition, and output store is ordered inside an
explicit loop nest. It is still above backend choices such as vector width,
register allocation, and instruction selection.

The exact v1 opcode set is:

| Opcode | Result | Effect |
| ------ | ------ | ------ |
| `LOAD` | one `FP64` SSA value | reads one buffer through a `KernelAccess` |
| `CONSTANT` | one `FP64` SSA value | pure |
| `ADD` | one `FP64` SSA value | pure |
| `MUL` | one `FP64` SSA value | pure |
| `STORE` | none | writes one output buffer through a `KernelAccess` |

An R2 `SCALE` node becomes `CONSTANT alpha` followed by `MUL`. The lowerer
places the constant before the multiplication and uses `(alpha, value)` as
the multiplication operand order to retain R2's scalar operation ordering.
There is no R3 `TRANSPOSE` opcode.

## Type model

The type architecture has separate `storageType`, `computeType`, and
`accumulatorType` fields on every `KernelBuffer`, plus a type on every
`KernelValue`. R3 currently emits and verifies only:

```text
storage type:    FP64
compute type:    FP64
accumulator:     FP64
```

`COMPLEX_FP64` is named in `KernelValueType` as an explicit future extension,
but complex arithmetic is deliberately not eligible for the initial R3
reference backend. Current JLC `Matrix` values can acquire an optional
imaginary lane dynamically, so the runtime binding audit rejects a complex
leaf with the deterministic reason `complex FP64 storage is unsupported by
R3; fallback to R2`. R2 remains in control and preserves all complex lanes,
including mixed real/complex cases, NaNs, infinities, and signed zero. No
imaginary lane is silently dropped.

FP32, FP16, BF16, mixed precision, and accumulator promotion are not claimed
implemented. Their future addition is localized to the type descriptors and
typed backend rules rather than requiring a new expression-level IR.

## Memory model

Each kernel buffer records:

* the original logical identity and name;
* its M2 logical storage class (`EXTERNAL_INPUT`, `TEMPORARY`, or `SYMBOLIC`);
* `INPUT` or `OUTPUT` role;
* shape;
* storage, compute, and accumulator value types;
* `KernelLayout`;
* `MemorySpace` (`HEAP`, `OFF_HEAP`, or `UNKNOWN`); and
* logical ownership.

For the current `Matrix` and `OffHeapMatrix` contract, non-symbolic buffers
are row-major dense; symbolic buffers retain `UNKNOWN` layout. R3 does not
invent column-major views, zero-copy transformations, arbitrary strides, or
allocation ownership. `KernelLayout` still has `COLUMN_MAJOR_DENSE`,
`STRIDED`, and `UNKNOWN` extension points, but the lowerer only claims what
the current runtime contract proves.

`KernelOp.effect()` makes effects explicit: loads read, stores write, and
constants/arithmetic are pure. The verifier requires input loads to be
read-only and output stores to target the declared execution-owned output.

Alias facts are copied from the existing conservative M2 analysis. Distinct
external logical buffers are `MAY_ALIAS`; a temporary versus an external
buffer is `NO_ALIAS`; and the same logical identity is `MUST_ALIAS`. R3 never
turns an absent or `MAY_ALIAS` fact into `NO_ALIAS`. The verifier rejects an
alias fact that is stronger than its logical provenance permits.

## Iteration model

R3 v1 represents a flat ordered body under a canonical two-dimensional static
rectangular loop nest:

```text
for i = 0 .. rows, step 1
for j = 0 .. columns, step 1
```

Loop variables are explicit `AffineVariable` objects. Each v1 loop has an
identity semantic binding, zero lower bound, unit step, and bounds equal to
the R2 final iteration domain. There is no unrestricted control-flow graph,
branch, while loop, recursion, exception, or dynamic allocation in a kernel.

R2 tiled/guarded schedules remain valid R2 inputs but are conservatively
ineligible for this initial lowerer. The lowerer rejects them before creating
an executable Kernel IR rather than dropping the tail guard or claiming
rectangular coverage. This is the safe extension point for a future structured
tail predicate.

## Operations and SSA

`KernelValue` IDs are stable non-negative integers assigned in deterministic
operation order. Every value-producing operation defines exactly one ID;
operations can refer only to earlier definitions. `KernelVerifier` checks
that every declared value is defined once, operands exist, operands dominate
uses, and scalar types agree.

R2's cached `(source statement, composed affine indices)` sharing is preserved
by lowering each R2 scalar node once. If a shared R2 producer has three users,
the R3 producer has one SSA ID and three operand references; no computation is
duplicated. R3 performs no algebraic reassociation, `x + 0`, `x * 1`, FMA,
constant folding, or other optimization rewrite.

## R2 lowering

`KernelLowerer.lower(FusedRegionPlan)` is a dedicated compile-time pass. It
consumes:

* the region's final iteration domain and schedule band;
* its leaf buffers and final output;
* `ScalarFusionProgram` nodes and their order;
* R2 composed load maps;
* eliminated/internal-value evidence; and
* region statement IDs and legality evidence.

The pass creates one `KernelBuffer` for every leaf and the final output. It
creates one `KernelOp.LOAD` for each R2 load, one constant plus one multiply
for each scale, one add for each add, and one final store. A semantic R2
transpose node is mapped to its already-composed input value; it creates no
runtime operation. The result is verified before the lowering result is
returned.

The pass is never called from the per-element executor. When the optional
selector is enabled, lowering and verification happen while
`CpuFusedRegionStep` is constructed. Runtime execution only creates a
`KernelBinding` from already-existing matrices and R1 slots.

## Transpose and affine indexing

Transpose is an index transformation, not a scalar computation. For example,
the R2 chain

```text
T = transpose(A)
Z = scale(T, 2)
```

lowers to a load such as:

```text
k0 = load %A[j,i]
k1 = const 2.0
k2 = mul k1, k0
store %Z[i,j], k2
```

The R2 builder has already inverted producer write maps and composed them into
the load. A second transpose composes back to an identity load. The R3
verifier accepts only the proven v1 unit-variable affine subset: each access
dimension is one bound variable with coefficient `+1` and zero constant, and
the variable's extent matches that buffer dimension. Rank, variable binding,
injectivity, and output coverage are checked; arbitrary coefficients,
nonlinear expressions, and unproven maps are rejected.

## Verification

`KernelVerifier.verify` returns an immutable `KernelVerificationResult` with
deterministically ordered diagnostics. `requireValid` raises
`KernelVerificationException` with the same diagnostics. Before execution or
future code generation it verifies:

1. stable function/program IDs and unique function IDs;
2. unique buffer IDs and buffer membership;
3. logical-buffer provenance and shape agreement;
4. declared input/output roles, ownership, and one-output v1 contract;
5. explicit FP64 type compatibility;
6. valid two-dimensional canonical loop bounds and domain coverage;
7. no undefined loop variables in loop bindings or accesses;
8. access rank equals matrix shape rank;
9. supported unit-variable affine access maps and matching extents;
10. unique SSA value IDs and exactly-once definitions;
11. operand references are defined earlier in the block;
12. supported opcode arity, access, immediate, and result-shape rules;
13. stores target the writable declared output, never an input;
14. exactly one output store provides a bijective output coverage map;
15. alias facts are complete, pairwise unique, and not illegally strengthened;
16. all operation and buffer source provenance points back to R2.

Examples of deterministic rejection text include:

```text
undefined or non-dominating SSA operand: 99
duplicate SSA value ID: 0
undefined loop variable in access: k
STORE target is not a writable OUTPUT buffer
access rank 1 does not match matrix rank 2
unsupported affine access expression at dimension 0: 2*i
illegal alias strengthening for %0<->%1: claimed=NO_ALIAS, proven=MAY_ALIAS
```

## R1 integration

R3 describes computation and logical/storage facts; it does not plan or own
allocations. `KernelBinding` is the explicit boundary:

```text
KernelBuffer -> Matrix
KernelBuffer -> PhysicalBufferSlot (optional R1 fact)
```

The R1 `PhysicalMemoryPlan` still decides which non-overlapping logical
temporary lifetimes share a physical slot. Borrowed leaves normally have no
slot; the fused output maps to its R1 slot. The binding is created per
execution and is not stored in the immutable `KernelProgram`. This keeps
`logical value != physical storage` intact while giving a future backend the
slot/layout/alias facts it needs.

## Reference executor

`KernelReferenceExecutor` is a correctness backend. It executes verified real
FP64 kernels with nested scalar loops, `Matrix.get`, and `Matrix.set`. It
compiles all affine expressions into reusable coefficient/slot arrays and
allocates one scalar value array per invocation, not per matrix element. The
executor has no vector width, register, FMA, JIT, native, GPU, or tiling
policy.

The developer selector is opt-in and defaults to the R2 path:

```text
jlc.compiler.kernelIr=off      # existing R2 execution
jlc.compiler.kernelIr=verify   # lower + verify, execute R2
jlc.compiler.kernelIr=execute  # lower + verify, execute R3 reference when real FP64
```

If runtime lanes are complex or the structural lowerer returns
`INELIGIBLE`/`UNKNOWN`, `execute` falls back to R2. The R2 fast path remains
available and is still the production/default baseline.

## Canonical dump and diagnostics

`KernelProgram.dump()` is stable across repeated lowerings. A representative
shape is:

```text
kernel-program kernel-program-F0
function F0 fp64
source:
  fused-region=F0
  statements=[S0, S1]
  scalar-nodes=[0, 1, 2]
  eliminated=[%2]
buffers:
  %0 A input class=EXTERNAL_INPUT storage=FP64 compute=FP64 accumulator=FP64 memory=heap layout=row_major_dense ownership=borrowed
  %1 B input class=EXTERNAL_INPUT storage=FP64 compute=FP64 accumulator=FP64 memory=heap layout=row_major_dense ownership=borrowed
  %2 C output class=TEMPORARY storage=FP64 compute=FP64 accumulator=FP64 memory=unknown layout=row_major_dense ownership=owned
loops:
  for i = 0..M step 1
  for j = 0..N step 1
body:
  k0 = load %0[i,j]
  k1 = const 2.0
  k2 = mul k1, k0
  k3 = load %1[i,j]
  k4 = add k2, k3
  store %2[i,j], k4
```

The actual dump uses concrete bounds and IDs. There is no transpose operation
in the body. A compact lowering diagnostic reports eligibility, buffer/loop/
operation/load/store counts, and verification status, for example:

```text
kernel IR: eligible=eligible, buffers=3, loops=2, scalarOps=6,
loads=2, stores=1, verification=PASS, reason=verified Kernel IR
```

The aggregate `KernelIrMetrics` exposes:

```text
kernelCount, eligibleRegionCount, rejectedRegionCount
bufferCount, loopCount, scalarValueCount, operationCount
loadCount, storeCount, arithmeticOpCount, sharedValueCount
loweringTimeNanos, verificationTimeNanos
```

No normal execution path prints diagnostics.

## Numerical validation

`KernelIrTest` covers direct ADD and SCALE kernels plus R2 lowering for scale,
add, long chains, shared producers with three uses, transpose and double
transpose composition, both `37x11` and `11x37`, zero-size dimensions, `1x1`,
NaN, `+Inf`, `-Inf`, `+0.0`, `-0.0`, strict ordering, complex rejection/R2
fallback, malformed SSA/loop/access/store/alias cases, R1 slot binding,
provenance, escaping R2 candidates, opaque GEMM boundaries, and the optional
selector. Real R3 results are compared to eager results and R2 results with
zero-tolerance real-array checks; bit patterns are checked for edge values.

The complex cases intentionally validate fallback rather than pretending the
real-only kernel has complex semantics. Every valid R2 generalized-fusion
case that fits the R3 canonical subset is cross-checked as:

```text
eager result == R2 result == R3 reference result
```

## Compiler-scaling benchmark

`R3KernelIrBenchmarkTest` is opt-in and measures only R2-to-R3 lowering and
verification. It uses valid scalar scale chains with 2, 10, 25, 50, and 100
source operations. The following median internal timings are from the smoke
run on the R3 worktree; small absolute values are naturally noisy, while the
IR structure is exactly linear:

| Ops | Lowering µs | Verification µs | IR values | IR operations |
| --: | ----------: | ---------------: | --------: | -------------: |
| 2 | 183 | 159 | 5 | 6 |
| 10 | 127 | 200 | 21 | 22 |
| 25 | 129 | 207 | 51 | 52 |
| 50 | 128 | 282 | 101 | 102 |
| 100 | 166 | 450 | 201 | 202 |

Each scale contributes one constant and one multiply; the single load and
single store are fixed. Therefore values and operations grow linearly, and
the verifier's work follows the ordered flat lists rather than a quadratic
pairwise graph algorithm. The benchmark command is:

```text
bash ./gradlew test --tests net.faulj.compiler.matrix.kernel.R3KernelIrBenchmarkTest \
  --no-daemon -q -Djlc.compiler.r3.benchmark=true
```

## Execution comparison

The execution benchmark is informational only. It compares the existing R2
specialized raw-array path with the R3 scalar reference and includes the
different reference/backend setup costs. It makes no hardware-performance
claim and does not define R3 success:

| Workload | R2 fast path µs | R3 scalar reference µs | Ratio | Correct |
| -------- | ---------------: | ---------------------: | ----: | :------ |
| scale-add 37x11 | 324.2 | 157.9 | 0.49 | yes |
| transpose 37x11 | 332.7 | 112.1 | 0.34 | yes |
| shared producer 64x64 | 203.9 | 531.4 | 2.61 | yes |

The execution benchmark is opt-in:

```text
bash ./gradlew test --tests net.faulj.compiler.matrix.kernel.R3KernelIrExecutionBenchmarkTest \
  --no-daemon -q -Djlc.compiler.r3.executionBenchmark=true
```

The shared-DAG row shows the expected interpreter overhead; later generated
backend work in R4 is responsible for vector/code-generation performance.

## Rejections and fallback

R3 does not need to lower every R2 region. Safe fallback is part of the
contract. Current rejections include:

* complex runtime storage for the real-only v1 executor;
* tiled or guarded schedules outside the canonical rectangular subset;
* unsupported affine coefficients, ranks, or non-bijective maps;
* reductions and generic control flow;
* escaping internal R2 producers;
* GEMM initialization/update bodies; and
* incomplete or unknown runtime storage facts when runtime eligibility is
  requested.

An ineligible or unknown result never becomes an executable kernel by
optimistic assumption. R2 remains the fallback. GEMM remains an opaque
`CpuGemmStep`; R3 may describe a future epilogue boundary but emits no GEMM
body or microkernel.

## Architecture audit answers

1. **What does R3 add beyond `ScalarFusionProgram`?** Explicit loops,
   explicit LOAD/CONSTANT/MUL/ADD/STORE operations, output effects, buffer
   layout/memory/ownership/type facts, alias facts, verification, bindings,
   and a backend-neutral executable boundary.
2. **Why is it not another expression tree?** The body is an ordered flat SSA
   block under explicit loops with explicit memory addresses and stores; it
   has no tree-only implicit traversal or materialization semantics.
3. **What are the opcodes?** `LOAD`, `CONSTANT`, `ADD`, `MUL`, and `STORE`.
4. **How are values represented?** Immutable `KernelValue` declarations with
   stable IDs; operands reference earlier IDs, preserving sharing.
5. **How are loops represented?** Ordered immutable `KernelLoop` descriptors
   with induction variable, semantic variable, static half-open bounds, and
   step.
6. **How are affine accesses represented?** `KernelAccess` pairs a buffer
   with immutable `AffineExpr` index lists.
7. **How is transpose represented?** R2 composes its permutation into LOAD
   maps such as `%A[j,i]`; no runtime transpose opcode exists.
8. **How is layout represented?** `KernelLayout`, currently proven
   row-major dense for current concrete Matrix storage and unknown for
   symbolic storage.
9. **How is memory space represented?** Existing M2 `MemorySpace` is carried
   on every buffer: heap, off-heap, or unknown.
10. **How are alias facts represented?** Pairwise `KernelAliasFact` values
    preserve `MUST_ALIAS`, `NO_ALIAS`, or `MAY_ALIAS` from M2 analysis.
11. **How is R1 storage separate?** `KernelBinding` optionally maps buffers to
    R1 `PhysicalBufferSlot` objects; the immutable IR owns no allocations.
12. **How is provenance retained?** `KernelProvenance` stores fused-region,
    statement, source-scalar, and eliminated-buffer IDs plus R2
    legality/profitability evidence; each op retains source IDs.
13. **How does verification work?** `KernelVerifier` walks functions,
    buffers, loops, alias facts, and the ordered body in a fixed order and
    returns all deterministic diagnostics.
14. **What malformed kernels are rejected?** Duplicate IDs, undefined or
    forward SSA operands, unknown buffers, rank/map/domain failures, illegal
    stores, missing output coverage, type mismatches, invalid loop bounds,
    unsupported opcodes/arity, illegal alias strengthening, and missing R2
    provenance.
15. **Is complex FP64 supported?** No: it is explicitly ineligible in v1;
    complex runtime regions fall back to R2, which preserves their lanes.
16. **How is strict ordering preserved?** R2 node/read order is retained;
    scale expands to constant-then-multiply with factor-first operands; no
    reassociation or algebraic rewrite is performed.
17. **What is lowering complexity?** For a valid flat R2 scalar program,
    lowering is O(V + A + O), where V is scalar nodes, A is affine terms, and
    O is emitted operations; each R2 node is visited once.
18. **What is verifier complexity?** The main validation pass is O(B + L +
    O + A + F), with B buffers, L loops, O operations, A affine terms, and F
    alias facts. Alias coverage adds the explicit O(B²) pair check for the
    small kernel boundary; it is not proportional to matrix elements.
19. **How large are realistic R2 regions?** The benchmark emits 5, 21, 51,
    101, and 201 SSA values for 2, 10, 25, 50, and 100 scale operations;
    typical matrix fused regions are tens of scalar operations and three to
    dozens of boundary buffers.
20. **What should R4 consume?** A verified `KernelProgram`/`KernelFunction`,
    its ordered `KernelBlock`, `KernelValue` graph, `KernelLoop` nest,
    `KernelAccess` maps, buffer type/layout/memory/ownership facts, alias
    facts, output effect, and R2 provenance. R4 should not need to revisit
    MatrixExpr semantics or R2 legality.

## Tests and build evidence

The focused R3 run added 21 `KernelIrTest` cases and one benchmark smoke test;
the execution benchmark is also one opt-in test. Final exact counts for all
required regression suites are recorded here after the full validation run:

| Suite | Tests | Skipped | Failed |
| ----- | ----: | ------: | ------: |
| New `KernelIrTest` | 21 | 0 | 0 |
| New `R3KernelIrBenchmarkTest` smoke | 1 | 0 | 0 |
| New `R3KernelIrExecutionBenchmarkTest` smoke | 1 | 0 | 0 |
| R2 `GeneralizedFusionTest` | 17 | 0 | 0 |
| R2 benchmark smoke test | 1 | 0 | 0 |
| R1 `MatrixMemoryPlanningTest` | 15 | 0 | 0 |
| CPU lowering tests (`MatrixCpuLoweringTest` + `CpuFusedFastPathTest`) | 28 | 0 | 0 |
| Affine/schedule tests (`MatrixAffineCompilerTest` + `MatrixScheduleTest`) | 45 | 0 | 0 |
| Compiler audit regression tests | 27 | 0 | 0 |
| Full project test | 495 | 24 | 1 pre-existing |

The final run must also include:

```text
git diff --check: PASS (no whitespace errors)
```

The one full-suite failure is the pre-existing
`net.faulj.nativeblas.AlgorithmDispatchTest.coldStartAllowsCppOnlyForFoundationAlgorithmsAboveThreshold`
assertion at line 84. The same isolated test fails on the clean R2 baseline
(`feature/compiler-generalized-fusion`, `45fdd7aca34ccd31783416631989c19bf43deec6`),
so it is not attributed to R3.

## Remaining weaknesses

These are deliberate post-R3 boundaries, not failures:

* no vector code generation;
* no reductions;
* no GEMM body lowering;
* no GPU backend;
* no FMA selection;
* no ISA selection;
* no unrolling;
* no tile search;
* no autotuning; and
* the reference executor may be slower than R2's specialized fast path.

R3 establishes the intended final separation:

```text
R1: logical value != physical storage
R2: logical matrix intermediate != mandatory runtime Matrix
R3: fused semantic computation != backend-specific implementation
```

## R4 — Generated SIMD pseudokernels

R4 is implemented on branch `feature/compiler-generated-simd` in the clean
worktree `/home/james/Projects/JLC-r4`. It is based on the R3 checkpoint
`ae4bd0b5043b4fb3143a12d722aa23a7e0b166ab`; the implementation checkpoint is
`a66864d` (`Implement R4 generated SIMD pseudokernels`). The primary checkout
and `main` were not modified. This milestone does not rename the project or
introduce an M5 compatibility layer.

### Scope and phase boundary

R4 consumes only verified R3 `KernelProgram`/`KernelFunction` objects. R1
continues to describe storage, R2 continues to choose and lower generalized
fusion regions, and R3 remains the semantic IR and scalar reference oracle.
R4 adds a backend-facing plan, a deterministic signature, strict scalar C++
emission, strict AVX2 C++ intrinsic emission, and optional native dispatch.
The existing GEMM implementation remains opaque and is not lowered into this
path.

The normal default is unchanged:

```text
jlc.compiler.kernelBackend = r2       -> existing R2 execution
jlc.compiler.kernelBackend = scalar_cpp -> generated scalar C++ when registered
jlc.compiler.kernelBackend = avx2      -> generated AVX2 when eligible and supported
```

The generated modes are opt-in and a registry miss, verifier rejection,
binding mismatch, or ISA failure returns to the existing R2 path. No runtime
compiler or C++ toolchain is needed for ordinary execution; source compilation
is a build/deployment step.

### Phase A audit: R3 inputs and the native convention

The implementation uses the following R3 facts directly: ordered
`KernelOp` values and operands, `KernelLoop` bounds, `KernelAccess` affine
maps, `KernelBuffer` roles/shapes/types/layout/memory/ownership,
`KernelAliasFact` relations, output effects, and R2 provenance. The verifier is
called before planning or emission, so R4 does not re-interpret
`MatrixExpr`, redo R2 legality, or infer missing alias facts.

The native audit found the existing JNI library, CMake object-library layout,
and whole-array `double[]` convention. R4 therefore adds a separate
`jlc_native_codegen` object library and registry rather than changing the
historical GEMM target. Existing GEMM compilation may retain its historical
optimization policy; R4 generated objects use `-O2 -fno-fast-math
-ffp-contract=off` (or the strict MSVC equivalent). An optional
`JLC_R4_GENERATED_SOURCE_DIR` CMake cache path links emitted `.cpp` files into
the native library.

### Deterministic identity and shape policy

`KernelSignature` emits inspectable canonical text containing:

* a version marker and `shape-specialized=true`;
* function identity, leaf/output counts, and exact loop bounds/steps;
* all buffer facts and sorted alias facts; and
* the ordered opcode stream, SSA IDs, operands, affine accesses, and raw
  constant bits.

The canonical UTF-8 text is hashed with SHA-256. The generated symbol is
`jlc_pk_<first-12-hex-digits>`. No Java object identity, source-expression
text, or pointer address participates in the key. Rows and columns are part of
the signature and are also checked by the generated function and native
registry descriptor. This is deliberate shape specialization: one artifact
cannot silently execute a different shape.

### Eligibility and plan

`SimdEligibility` is separate from emission. The initial scalar/AVX2 subset
requires one real FP64 output, FP64 inputs, row-major dense buffers in the
initial heap-storage subset, the canonical two-loop rectangular domain, and
only the verified R3 opcodes `LOAD`, `CONSTANT`, `ADD`, `MUL`, and `STORE`.
AVX2 additionally requires identity `[outer, inner]` accesses (unit inner
stride) and a proven `NO_ALIAS` relation from the output to every input.
Unknown or unsupported storage/layout facts are ineligible; non-unit-stride
regions may use scalar C++ when their storage facts are otherwise safe, and
transpose/non-dense cases fall back to R2. Complex FP64 is never promoted.

`PseudokernelPlan` retains the source function and signature and records vector
width 4, element count, vector iterations, scalar-tail elements, load/store
counts, arithmetic and constant counts, estimated bytes per element, and a
conservative maximum live SSA-vector count. The plan is diagnostic and
inspectable; it does not replace R3 or mutate it. Shared producers remain one
SSA value and are emitted once per vector iteration.

### Scalar and AVX2 lowering

The scalar emitter prints the verified operation order inside shape-checked
`i,j` loops. It uses ordinary `double` loads, stores, `+`, and `*`; affine
indices remain explicit, including non-unit-stride scalar cases. The AVX2
emitter uses `_mm256_loadu_pd`, `_mm256_set1_pd`, `_mm256_mul_pd`,
`_mm256_add_pd`, and `_mm256_storeu_pd`. Constants are hoisted into vector
values before the loop, and the exact same ordered operation graph is emitted
for the scalar remainder. Width is fixed at four FP64 values. Remainders use
`p + 4 <= rows*cols` followed by a scalar tail, so columns 1–5 and other
non-multiples are covered by the same artifact.

There is no FMA selection, reassociation, fast-math flag, or contraction
permission in R4. Generated sources include `#pragma STDC FP_CONTRACT OFF`;
the CMake/test compile-smoke uses `-fno-fast-math -ffp-contract=off`. NaNs,
infinities, signed zero, and ordinary IEEE ordering therefore remain in the
strict semantic contract, subject to the host compiler honoring those strict
flags.

### Registry, JNI, and runtime gate

The Java registry key is the exact `KernelVariantSignature`; the older
`(KernelSignature, CodegenBackend)` lookup remains only as a compatibility
helper for baseline callers. A native entry contains the canonical signature,
generated symbol, backend, value type, width, and exact rows/columns. The C++ registry is function-local-static and mutex
protected, which avoids cross-translation-unit initialization-order hazards.
Generated registration adapters expose one whole-region function:

```text
Java double[][] inputs + double[] output
    -> pin all input arrays once and output once
    -> one jlc_generated_execute call for the complete region
    -> release inputs without copy-back; commit output
```

The Java executor checks exact base `Matrix` objects, real storage, output/input
identity, structural AVX2 eligibility, and a conservative runtime AVX2 probe.
The native registry repeats the shape and AVX2 checks. If any check or lookup
fails, the caller returns false and `CpuFusedRegionStep` continues through R2
fast/generic execution. The source-emission tests exercise both standalone
driver dispatch and native registry registration; the normal Gradle native
build also compiles the JNI bridge and registry.

### Validation evidence

Focused R4 validation completed on the implementation checkpoint:

| Check | Result |
| ----- | ------ |
| deterministic signature/source, strict flags, no FMA, shared SSA, tails, zero-size plan, alias/transpose policy, opt-in fallback | 6 tests pass |
| eager/R2/R3 baseline agreement for the generated fixture | pass |
| scalar C++ compile-smoke and registry execution | pass |
| AVX2 intrinsic compile-smoke and registry execution | pass on AVX2 host |
| NaN, infinity, signed-zero, and 3x5 vector-plus-tail correctness | pass |
| opt-in JNI whole-region benchmark, including empty-shape bridge probe | pass |
| AddressSanitizer generated-source run | pass |
| native CMake build with vendor BLAS disabled | pass |
| `git diff --check` on the final branch | pass |

The full project test task completed with 504 tests, 26 skips, and one
failure: the pre-existing
`net.faulj.nativeblas.AlgorithmDispatchTest.coldStartAllowsCppOnlyForFoundationAlgorithmsAboveThreshold`
assertion. Running the generic suite without the generated native library
leaves that same single failure and skips the native-library integration tests;
the extra native profile failure observed when reusing the R4 vendor-disabled
build is therefore an environment/configuration condition, not an R4 code
failure.

The 3x5 native driver checks the scalar tail after a 4-wide vector prefix and
checks NaN, positive infinity, and signed-zero propagation. A generated 0x3
scalar artifact is also compiled and invoked through the registry with empty
vectors, proving the no-work path. Existing R3 tests continue to cover
zero-sized lowering and verification; the R4 native registry rejects shape
mismatches and allows null data pointers only for a zero-element invocation.
The current R2 lowerer does not create a generated fused region for every
empty expression, so those cases remain a deliberate R2/R3 boundary rather
than being optimistically promoted.

Inspectable artifacts from the deterministic test are written under
`build/reports/r4/generated/`. For the representative 4x5 scale-plus-add
kernel, the scalar and AVX2 source sizes were 792 and 1,154 bytes and the
signature/plan report was 766 bytes. The plan reported width 4, five vector
iterations, zero tail elements, 24 estimated bytes per element, and a maximum
live-vector count of 3.

### Execution benchmark

The opt-in benchmark generated and compiled the same registry-backed scalar
and AVX2 artifacts for 2, 5, 10, and 25-operation scale chains. It warmed the
R2 Java path and timed 21 steady-state native samples; code generation and C++
compilation were excluded from the execution numbers. For shapes through
256x256 it also timed five warmed R3 reference samples separately. Values below are
nanoseconds per element on this host, with R2 Java allocation/setup included
in its path. The native numbers are standalone registry calls, so they do not
claim to be a DRAM-bandwidth model and do not include JNI overhead.

| Kernel | Shape | R2 Java | Scalar C++ | AVX2 | AVX2 vs R2 | Correct |
| ------ | ----- | ------: | ---------: | ---: | ----------: | :-----: |
| 2 ops | 256x256 | 16.9 | 0.3 | 0.3 | 51.35x | yes |
| 10 ops | 256x256 | 44.6 | 0.6 | 0.4 | 124.30x | yes |
| 25 ops | 256x256 | 101.8 | 2.7 | 1.4 | 72.39x | yes |
| 2 ops | 1024x1024 | 15.8 | 0.9 | 0.9 | 17.42x | yes |
| 25 ops | 1024x1024 | 95.5 | 2.8 | 1.5 | 64.22x | yes |

The 2-operation rows are memory-bound and show little scalar-versus-AVX2
separation in the standalone native loop. Arithmetic-heavy chains expose the
expected AVX2 benefit. Representative source generation took 80–2,886 µs
(the first scalar sample was a cold JVM case); C++ compilation took roughly
0.69 s for scalar and 1.13 s for AVX2 artifacts. Those costs are build/cache
costs, not steady-state execution costs.

The R3 reference context from the same harness was 74.9, 185.6, and 408.2
ns/element for the 2-, 10-, and 25-operation 256x256 cases, respectively.
Those interpreter/reference numbers include its per-call output setup and are
reported to show the lowering gap, not as a production backend target.

The separate opt-in JNI benchmark built a temporary CMake native library with
the generated AVX2 source linked in. A 256x256 whole-region JNI call measured
21,750 ns median (0.3319 ns/element) on the same host. An empty 0x3 generated
call measured 1,470 ns median; this is a useful bridge/pin/registry baseline
because it has no element work, but it is not subtracted from the non-empty
kernel number.

### R4 architecture audit answers

1. **What exactly is consumed from R3?** Verified `KernelFunction` buffers,
   loops, ordered SSA operations, accesses, types/layouts, memory/ownership,
   aliases, effects, and provenance.
2. **What is the separate eligibility layer?** `SimdEligibility` returns
   `AVX2_CONTIGUOUS`, `SCALAR_CPP`, or `INELIGIBLE` with deterministic reasons.
3. **What is the backend-neutral representation?** `PseudokernelPlan` keeps
   R3 as the semantic source and adds shape/vector/pressure metrics.
4. **How is identity made deterministic?** Canonical inspectable text plus
   SHA-256; no object identity or addresses.
5. **Is shape policy explicit?** Yes: rows, columns, loops, and buffer shapes
   are signature fields and runtime descriptor checks.
6. **What vector width is selected?** Four FP64 lanes for AVX2.
7. **How is shared SSA preserved?** One R3 value produces one emitted vector
   definition per iteration; the shared-producer test verifies one MUL.
8. **How are constants handled?** Scalar constants use exact literals/raw
   bits; AVX2 constants are hoisted with `_mm256_set1_pd`.
9. **How is ordering preserved?** Emission walks the verified flat operation
   list in order without reassociation.
10. **How are tails handled?** A four-lane loop is followed by scalar `p<count`
    code from the same operation graph.
11. **How is aliasing handled?** Output/input must be proven `NO_ALIAS` for
    AVX2; `MAY_ALIAS` is never promoted.
12. **What happens to transpose/non-unit stride?** Non-unit stride is scalar
    C++ eligible when safe; non-dense/unknown layouts fall back to R2.
13. **What happens without AVX2?** The runtime gate returns false; R2 remains
    the fallback. Scalar C++ has no AVX2 execution requirement.
14. **Are generated sources compiled at runtime?** No. They are build/deploy
    artifacts; the runtime only looks up a registered function.
15. **How is registration done?** Static generated registration calls the
    mutex-protected native registry with a whole-region adapter.
16. **How is JNI crossed?** Inputs and output are pinned once around exactly
    one native whole-region call; no per-element JNI call exists.
17. **What JNI overhead was measured?** The opt-in bridge benchmark measured a
    1,470 ns median empty 0x3 whole-region call, including JNI array handling,
    registry lookup, and the no-work generated entry. A 256x256 AVX2 call was
    21,750 ns median; the execution table intentionally keeps that JNI total
    separate from standalone native kernel timing.
18. **Which kernels benefit most?** Real FP64, dense, contiguous, alias-safe
    elementwise regions with enough arithmetic per element.
19. **Which kernels are bandwidth-bound?** Short scale/add chains and other
    low-arithmetic kernels; scalar and AVX2 native rows are close there.
20. **What is live-vector pressure?** It is exposed by the plan; the
    representative shared scale/add plan measured three live vector values.
21. **Is performance tied to correctness evidence?** Yes: every reported row
    is marked correct by the benchmark driver's expected-value check; the
    generated fixture also agrees with eager, R2, and R3 Java baselines, while
    compile-smoke/ASAN remains separate evidence.
22. **What belongs in R5?** Width/unroll variants, FMA policy experiments,
    layout-aware kernels, broader zero-shape coverage, JNI pin/copy tuning, live-range
    pressure studies, cache/roofline measurement, and registration/cache
    lifecycle tuning.

### Definition-of-done mapping

R4 now has a verified R3 consumer, explicit scalar and AVX2 emitters, fixed
width/tail policy, deterministic shape-specialized signatures, conservative
alias and ISA gates, a native registry, one-call JNI plumbing, strict compile
flags, opt-in backend selection, safe R2 fallback, correctness/compile/ASAN
evidence, benchmark output, a measured JNI bridge probe, and this audit. The
remaining scope boundary is that shapes the current lowerer does not represent
as fused regions remain on the R2/R3 side of its contract.

## R5 — Empirical Kernel and Schedule Autotuning

R5 is the empirical selection layer after R4; it is not M5 and it does not
rewrite R1–R4 history:

```text
R1 storage planning
  -> R2 legality-aware fusion
  -> R3 verified Kernel IR
  -> R4 generated scalar/AVX2 artifacts
  -> R5 correctness-gated measurement
  -> typed post-R5 backend calibration and profile dispatch
```

### Base and environment

The R5 work was performed in an isolated worktree and was not merged into
`main`.

| Item | Value |
|---|---|
| R5 base SHA | `10a29053dbc63ddc02518e572692eb783b75cb65` |
| implementation SHA | final checkpoint commit (`git HEAD` on this branch) |
| branch | `feature/compiler-autotuning` |
| worktree | `/home/james/Projects/JLC-r5` |
| date | 2026-09-17 |
| CPU | AMD Ryzen 9 3950X 16-Core Processor |
| CPU identity | AuthenticAMD, family 23, model 113 |
| ISA used | x86-64, AVX2 and FMA present |
| OS/kernel | Linux 7.0.0-31-generic |
| native compiler | Ubuntu GNU `c++` 15.2.0 |
| JVM | OpenJDK 21.0.12 |
| generated strict flags | `-O2 -fno-fast-math -ffp-contract=off` |
| direct benchmark flags | `-std=c++17 -O3 -fno-fast-math -ffp-contract=off -mavx2` |

The exact host facts are recorded by `KernelMachineIdentity`; the build key
also records the JLC SHA, clean/dirty state, verified-identity flag, JLC
version, KernelSignature/KernelVariantSignature versions, codegen ABI, Java
compiler/runtime, native compiler family/version/vendor, and strict
floating-point flags. The current profile path is explicit
(`jlc.compiler.autotune.profile`); normal matrix execution does not write a
profile under the user's home directory.

### Candidate model

R5 consumes only a verified `PseudokernelPlan`. It does not mutate
`MatrixExpr`, redo R2 legality, or invent alias facts. The bounded initial
search space is:

| dimension | candidates |
|---|---|
| backend | `SCALAR_CPP`, `AVX2` |
| AVX2 width | exactly 4 FP64 lanes |
| AVX2 unroll | 1, 2, 4, 8 |
| loop form | `FLAT` and `NESTED` where contiguous legality permits both |
| tail | scalar |
| semantic mode | `STRICT` by default |

The default enumeration is at most ten candidates: scalar nested/flat plus
AVX2 u1/u2/u4/u8 in flat/nested form. The fixed R4 u1 flat AVX2 variant is
named `BASELINE_AVX2` and is always retained when the AVX2 plan is eligible,
even when a caller sets a smaller candidate cap. `unroll=4` means four
independent `__m256d` instances, or 16 FP64 elements per loop body; it is not
a 16-lane vector.

`KernelVariantSignature` canonically includes the semantic KernelSignature,
backend, ISA, vector width, unroll, loop form, tail policy, semantic mode, and
FMA policy. Its SHA-256 is used only for a deterministic symbol suffix; timing
never participates in code identity. The registry key is now
`(KernelSignature, KernelVariantSignature)`, so all exact variants can coexist.

Register pressure is an explanatory structural gate, not a claim about the
compiler allocator:

```text
estimatedLiveVectorRegisters =
    maxLiveVectorValues * unroll + constantCount + 2
```

Candidates over 32 estimated vector registers, unavailable ISAs, or an
unroll larger than the available vector iterations are retained as `PRUNED`
evidence with a reason. This keeps the search bounded without pretending that
the estimate predicts every spill.

The initial R5 implementation deliberately does not include tiled transpose
variants, software prefetch, thread-count tuning, GPU backends, AVX-512, or
ML/Bayesian search. The layout-aware family remains a documented follow-up.

### Correctness gate

The order is fixed:

```text
generate -> compile -> deterministic validation corpus
         -> PASS -> benchmark
         -> FAIL -> quarantine and persist diagnostics
```

`KernelCorrectnessGate` uses the verified R3 `KernelReferenceExecutor` as its
oracle. It copies inputs and output for every case, so a candidate cannot
poison the next case. The corpus contains deterministic finite data and IEEE
edge data including NaN, both infinities, +0.0, -0.0, tiny values, and huge
finite values. Strict comparisons use raw double bits, while accepting NaN
payload differences as NaN results; no strict MUL+ADD is contracted.

The native coexistence test exercises all eligible scalar/AVX2 variants for a
3x17 tail-sensitive shape in one registry and invokes every exact variant.
The benchmark driver checks every output before reporting timing. Zero-size
and small/tail shapes remain guarded by the exact shape descriptor and scalar
tail path. A failed candidate is never passed to the timing runner.

### Benchmark methodology

The reusable `KernelBenchmark` defaults are five warmups, nine measured
samples, at most ten candidates, a 10% relative-MAD noise threshold, and a 3%
minimum promotion threshold. The property-based sample count is clamped to at
least seven. The opt-in direct-native experiment uses ten warmups and eleven
samples per candidate so its output is easy to inspect.

For each candidate, R5 records:

* raw samples, median, minimum, maximum, MAD, and relative MAD;
* sample count and an explicit `PASS`/`NOISY` outcome;
* source-generation, compilation, and benchmark wall time; and
* generated source bytes plus pressure and pruning evidence.

Median plus MAD is used without arbitrary outlier deletion. A candidate is
stable only when relative MAD is at most 0.10. The selector chooses the
lowest stable median only after correctness and the 3% promotion threshold.
Equal medians use the stable variant ID as a deterministic tie-break. If the
trusted generated baseline is not stable or no candidate clears the threshold,
the baseline is retained. A bounded `maxBenchmarkMillis` can terminate a
session, but it gives the baseline a chance to establish the comparison point.

Compilation is outside the timed kernel region. The direct-native report
(`build/reports/r5/variant-benchmark.txt`) recorded approximately 1.2–7.2 ms
of source generation and 4.69–5.89 s to compile one translation unit
containing the ten registered candidates; these are calibration costs, not
steady-state costs. The JNI crossover harness measures Java and native
execution separately from profile parsing and selector lookup.

### Calibration profile

`KernelCalibrationProfile` keeps the R5 schema version at 1 and uses
`dispatchSchemaVersion: 3` for typed backend decisions with explicit winner
semantics. It stores
machine/build identity, timestamp, exact `KernelSignature`, workload bucket,
typed baseline and winner, measured speedup, decision reason, every backend
candidate outcome, raw samples, correctness/stability status, and the existing
R5 generated-variant evidence. `R2_JAVA`, scalar-native, and generated AVX2
are represented as real `BackendChoice` values; Java is no longer an implicit
only-at-the-end fallback in calibration data.

The profile is deterministic, pretty JSON using the project's existing
Jackson dependency. A sanitized excerpt has this shape:

```json
{
  "schemaVersion" : 1,
  "dispatchSchemaVersion" : 3,
  "methodologyVersion" : "r5-median-mad-v1",
  "machine" : {
    "key" : "<sha256>",
    "cpuVendor" : "AuthenticAMD",
    "cpuFamily" : "23",
    "cpuModel" : "113",
    "isaFeatures" : "avx,avx2,fma,sse4_1,sse4_2"
  },
  "build" : {
    "key" : "<sha256>",
    "jlcVersion" : "1.0-SNAPSHOT",
    "gitSha" : "<40-hex-sha>",
    "gitDirty" : false,
    "identityVerified" : true,
    "kernelSignatureVersion" : "jlc-kernel-signature-v1",
    "variantSignatureVersion" : "jlc-kernel-variant-v1",
    "codegenAbiVersion" : "jlc-r5-codegen-abi-v1",
    "compilerIdentity" : "c++",
    "javaCompiler" : "javac 21.0.12",
    "javaRuntime" : "21.0.12|Ubuntu|OpenJDK 64-Bit Server VM",
    "nativeCompilerVersion" : "c++ (GNU ...) ...",
    "nativeVendor" : "NONE",
    "strictFlags" : "-O2,-fno-fast-math,-ffp-contract=off"
  },
  "entries" : [ {
    "kernelSha256" : "<sha256>",
    "workloadBucket" : "64x64/ops=10",
    "baselineChoice" : {
      "kind" : "GENERATED_AVX2",
      "variantId" : "baseline_avx2",
      "variantSha256" : "<sha256>",
      "variantSignature" : "<canonical variant signature>"
    },
    "selectionStatus" : "CALIBRATED",
    "stable" : true,
    "winnerChoice" : {
      "kind" : "R2_JAVA"
    },
    "baselineVariant" : "<canonical BASELINE_AVX2 signature>",
    "winnerVariant" : null,
    "winnerSpeedup" : 1.08,
    "candidates" : [ {
      "variantId" : "avx2_u8_flat",
      "outcome" : "PASS",
      "correctnessPassed" : true,
      "stable" : true,
      "medianNanos" : 880.0,
      "madNanos" : 0.0,
      "rawSamplesNanos" : [ 880, 881, 879 ]
    } ],
    "backendCandidates" : [ {
      "choice" : { "kind" : "R2_JAVA" },
      "outcome" : "PASS",
      "correctnessPassed" : true,
      "stable" : true,
      "medianNanos" : 910.0,
      "madNanos" : 2.0,
      "rawSamplesNanos" : [ 908, 910, 912 ]
    }, {
      "choice" : { "kind" : "GENERATED_AVX2",
        "variantId" : "avx2_u8_flat",
        "variantSha256" : "<sha256>",
        "variantSignature" : "<canonical variant signature>" },
      "outcome" : "PASS",
      "correctnessPassed" : true,
      "stable" : true,
      "medianNanos" : 880.0,
      "madNanos" : 0.0,
      "rawSamplesNanos" : [ 880, 881, 879 ]
    } ],
    "trustedBaselines" : [ {
      "name" : "R2_JAVA_FUSED",
      "backend" : "java",
      "correctnessPassed" : true
    } ]
  } ]
}
```

The angle-bracket values in this excerpt are intentionally sanitized; the
writer emits the full canonical signatures and full hashes. A corrupt,
partial, old-R5-only, schema-mismatched, machine-mismatched, build-mismatched,
missing, unregistered, or ISA-incompatible entry is ignored and falls back
safely.

#### Winner semantics

`selectionStatus=CALIBRATED` is the only state with an empirical winner:
`stable=true`, `winnerChoice` is non-null, and `winnerSpeedup` is finite and
positive. A stable baseline may be the calibrated winner when it is the best
correctness-gated choice within the promotion policy.

`selectionStatus=NO_STABLE_WINNER` means that no correctness-gated candidate
provided usable stability evidence. Its invariant is:

```json
{
  "baselineChoice" : {
    "kind" : "GENERATED_AVX2",
    "variantId" : "baseline_avx2",
    "variantSha256" : "<sha256>",
    "variantSignature" : "<canonical variant signature>"
  },
  "winnerChoice" : null,
  "selectionStatus" : "NO_STABLE_WINNER",
  "stable" : false,
  "winnerSpeedup" : 0.0,
  "decisionReason" : "no stable correctness-gated backend; use fallback baseline"
}
```

Runtime treats that entry as a miss and applies the normal baseline-AVX2,
scalar-native, then R2-Java fallback hierarchy; it does not claim that the
baseline won empirically. The dispatch schema was bumped from 2 to 3 because
older typed profiles cannot distinguish this state from a retained calibrated
baseline and are therefore rejected conservatively.

### Runtime selector and modes

`KernelDispatchSelector` is the single runtime authority. It loads and validates
a profile once, captures `RuntimeEnvironment` once, and caches resolved typed
choices by exact `KernelSignature` in a concurrent map. The hot path does not
parse JSON, compile, benchmark, enumerate variants, or acquire a tuning lock.
A native profile hit requires an exact signature, matching build/machine
metadata, an exact registered variant, a loaded native runtime, available ISA,
correctness-passed evidence, and stable evidence. The explain surface reports
the typed choice, exact variant when present, profile/machine match, median,
speedup, and fallback reason.

The explicit operating modes are:

| property | behavior |
|---|---|
| `jlc.compiler.autotune=off` | unchanged R2 execution by default |
| `...=profile` | profile/calibration tooling mode; no implicit application-thread tuning |
| `...=tune` | explicit `KernelVariantTuner` mode; compilation and measurement are caller-controlled |
| `...=use` | load the configured profile and dispatch only to already-registered variants |

Cold start and invalid-profile behavior is an explicit hierarchy:

```text
calibrated BackendChoice
  -> baseline AVX2, if exact artifact/native runtime/ISA are valid
  -> scalar-native, if its exact artifact/native runtime are valid
  -> R2 Java
```

`CpuFusedRegionStep` asks the selector for a typed choice in `use` mode. A
calibrated native choice is passed directly to `GeneratedKernelExecutor`, so
there is no second R5 variant selection. A Java choice continues through the
existing R2 fused fast/generic implementation. Native execution exceptions are
not silently classified as calibration misses; only established availability
misses return to the Java path. Ordinary execution never starts a compiler or
benchmark.

### Post-R5 production backend integration

This is a narrow post-M4/R5 runtime integration, not a new compiler milestone
and not R6. The type boundaries are:

```text
KernelSignature          semantic kernel/workload identity
KernelVariantSignature   one generated implementation identity
BackendChoice            runtime execution decision
  R2JavaBackend()
  ScalarNativeBackend(variant)
  GeneratedAvx2Backend(variant)
```

`BackendCalibrationRunner` reuses R5's median/MAD policy but times a
`BackendInvocation` for the complete execution boundary. The CLI task
`bash gradlew calibrateBackends` discovers exact-shape workloads, emits and
builds temporary registry-backed sources, correctness-gates native choices,
times R2 Java and JNI calls with the same warmup/sample policy, and atomically
merges the result at `jlc.compiler.autotune.profile` (or the explicit
`--profile` path). The profile keeps unrelated exact kernels when metadata
matches; incompatible old profiles are deliberately not migrated.

The JNI crossover is therefore a selector input rather than a detached
benchmark claim: small workloads may select Java or scalar-native, while a
larger workload may select a particular AVX2 variant. No compiler pass chooses
hardware, and no CUDA, AVX-512, JIT, or dynamic recompilation is added.

### Public numerical algorithm status

The CPU checkpoint distinguishes an implemented and validated path from a
claim of vendor-library competitiveness:

| algorithm | Java implementation | native implementation | production dispatch | fallback | validation status | known performance limitation |
|---|---|---|---|---|---|---|
| GEMM | canonical `Gemm` facade and Java kernels | built-in C++/JNI GEMM, including guarded layouts | backend registry plus algorithm policy; native provider is optional | Java GEMM | Java/native correctness and JNI integration coverage | 2048² FP64 control measured 89.3% of same-host AOCL-BLIS; not a universal parity claim |
| SVD | Golub–Kahan QR and divide-and-conquer Java solvers | optional guarded native bidiagonal stage; remaining SVD stages are Java | sensitivity-critical policy defaults to Java until calibrated or explicitly selected | Java SVD | SVD reconstruction, orthogonality, rank-deficient, and edge-case tests | no claim of vendor-LAPACK throughput; full SVD remains primarily Java |
| QR | Householder QR, thin/full and factorization-only modes | C++/JNI QR paths with shape/mode guards | calibrated policy and conservative cold-start rules | Java QR | QR reconstruction/orthogonality and native phase tests | tall thin/full cold-start paths stay Java; coverage is guarded rather than universal |
| Hessenberg | Java Householder reduction | optional C++/JNI reduction/decomposition for validated square sizes | algorithm policy, including Schur/SVD stage overrides | Java Hessenberg | Hessenberg and native decomposition integration tests | native size/shape coverage is bounded; no vendor-competitive claim |
| Schur | implicit Francis double-shift QR | no separate native Schur iteration; may use native Hessenberg stage when explicitly/calibrated | sensitivity-critical policy defaults to Java | Java Schur | Schur reconstruction/eigenvalue tests | the iterative Schur stage remains Java and is not benchmarked as a vendor replacement |
| eigenvalue path | symmetric eigen decomposition through real Schur | may inherit an allowed native Hessenberg stage through Schur | same conservative Schur policy | Java Schur/eigen path | symmetric eigenvalue/eigenvector and spectral tests | no general native eigensolver or vendor-performance claim |

This table is a runtime-status statement. It does not promote an algorithm to
the generated Kernel IR path merely because a lower-level native helper exists.

A fresh three-bucket calibration on the local AMD Ryzen 9 3950X used two
warmups and seven measured samples. Each row is a complete R2 Java or JNI
region invocation. The value in parentheses is MAD in nanoseconds; `stable`
means relative MAD ≤ 0.10. Noisy candidates remain in the JSON evidence but
are not promoted.

| workload | R2 Java median (MAD; stable) ns | scalar-native median (MAD; stable) ns | best stable AVX2 median (MAD; stable) ns | persisted winner |
|---|---:|---:|---:|---|
| 1x1 / 2 ops | 71,661 (5,549; yes) | 53,441 (10,471; no) | 32,580 (3,180; yes; `baseline_avx2`) | `CALIBRATED:GENERATED_AVX2(baseline_avx2)` |
| 16x16 / 2 ops | 226,281 (12,971; yes) | 43,760 (5,439; no) | 24,460 (740; yes; `avx2_u4_flat`) | `NO_STABLE_WINNER` |
| 64x64 / 2 ops | 271,471 (13,511; yes) | 31,380 (2,840; yes) | 25,030 (1,490; yes; `baseline_avx2`) | `CALIBRATED:GENERATED_AVX2(baseline_avx2)` |

The run reports all peers and does not hard-code backend diversity. At 1×1,
the trusted AVX2 baseline was stable and persisted as the calibrated winner. At
16×16, the trusted AVX2 baseline was noisy, so the profile intentionally records
no empirical winner even though other AVX2 candidates were stable. At 64×64 the
AVX2 baseline is retained as a stable calibrated winner. The profile records
exact variant signatures, all raw samples, and losing/noisy evidence at
`build/profiles/cpu-checkpoint-avx-enabled.json` in the local evidence bundle.

With the runtime AVX2 gate disabled, the 1×1 smoke calibration omitted all
AVX2 candidates and recorded Java at 83,820 ns (MAD 7,259; stable) and
scalar-native at 57,560 ns (MAD 10,280; noisy). It persisted
`NO_STABLE_WINNER` with `baselineChoice` set to the scalar choice and no
`winnerChoice`. This verifies the no-AVX2 path; the separate focused test
`noStableWinnerRoundTripsWithoutReconstructingBaselineAsWinner` demonstrates
that a profile with no stable peer instead persists `NO_STABLE_WINNER` and a
null `winnerChoice`.

### AVX2 unroll experiment

The direct-native experiment used 64x64 and 256x256 exact shapes with 2, 10,
and 25 arithmetic operations per element. The following medians are
nanoseconds per complete kernel call; they are intentionally kernel-only and
exclude JNI. The full report retains min/max/MAD and every eligible variant.

| workload | shape | baseline u1 flat | representative u2 | representative u4 | representative u8 | observation |
|---|---:|---:|---:|---:|---:|---|
| 2-op | 64² | 1040 | u2 flat 1000 | u4 nested 990 | u8 flat/nested 880 | u8 clears 3% promotion margin |
| 10-op | 64² | 1580 | u2 nested 1520 | u4 flat 1540 | u8 flat 1510 | u8 is fastest in this run |
| 25-op | 64² | 6000 | u2 nested 5940 | u4 flat 5950 | u8 pruned | gain is below 3%; retain baseline |
| 2-op | 256² | 15600 | u2 flat 15540 | u4 flat 15440 | u8 flat 15630 | bandwidth-sensitive differences are small |
| 10-op | 256² | 22340 | u2 flat 22411 | u4 flat 22771 | u8 flat 22840 | no clear promotion over baseline |
| 25-op | 256² | 92721 | u2 flat 93081 | u4 flat 93121 | u8 pruned | u8 pressure estimate is 39; retained as pruned evidence |

The 25-op plan has three live vector values and 13 constants, so the pressure
estimate is `3*U+13+2`: u1=18, u2=21, u4=27, and u8=39. This is an
explicit example of the pressure gate preventing a large fused DAG from
blindly compiling u8. Results move with system noise; the table is evidence
from this host, not a hard-coded universal choice.

### Backend crossover including JNI

`R5GeneratedJniCrossoverTest` builds one temporary JNI library containing all
eligible variants for five shapes (1, 4, 16, 64, and 256 square) and the
2/10/25-operation graphs. It times the existing R2 Java program, the R4
generated scalar variant, and the R4 baseline AVX2 variant through one
whole-region JNI call. Compilation is outside every timing loop.

| kernel | shape | R2 Java ns | scalar JNI ns | AVX2 JNI ns | selected |
|---|---:|---:|---:|---:|---|
| 2-op | 1² | 36090 | 1900 | 1910 | scalar JNI |
| 10-op | 1² | 37450 | 2510 | 2450 | AVX2 JNI |
| 25-op | 4² | 35630 | 3610 | 3530 | AVX2 JNI |
| 2-op | 16² | 37920 | 2310 | 1790 | AVX2 JNI |
| 10-op | 64² | 359931 | 7760 | 3960 | AVX2 JNI |
| 25-op | 64² | 431011 | 29320 | 9350 | AVX2 JNI |
| 2-op | 256² | 1060983 | 28380 | 21390 | AVX2 JNI |
| 10-op | 256² | 2650448 | 85580 | 26850 | AVX2 JNI |
| 25-op | 256² | 6466589 | 361191 | 96630 | AVX2 JNI |

On this host, the R2 Java implementation did not win the sampled points, but
the native floor is visible: scalar and AVX2 are effectively tied at 1–4
elements, and AVX2 separates from scalar at 16 elements and above. The prior
R4 bridge probe measured approximately 1.47 µs for an empty whole-region JNI
call and 21.75 µs for a 256x256 AVX2 call. These are end-to-end bridge
measurements, not a cost subtracted from kernel timings. The crossover harness
is therefore the evidence used when a calibration client supplies trusted
R2/scalar/AVX2 baselines to the tuner; standalone native ns/element is not
presented as a Java/native dispatch rule.

### Arithmetic intensity

`KernelWorkloadAnalysis` derives FP operations per element and estimated
loaded/stored bytes per element from verified R3 operations:

| graph | FP ops/element | bytes/element | intensity | class |
|---|---:|---:|---:|---|
| 2-op | 2 | 24 | 0.0833 | bandwidth-sensitive |
| 10-op | 10 | 56 | 0.1786 | mixed |
| 25-op | 25 | 112 | 0.2232 | mixed |

These thresholds are an explanation aid, not a formal roofline model. The
low-intensity 2-op rows show small schedule differences and a useful u8 win
at 64², while the 10-op rows expose more ILP variation. The 25-op rows expose
pressure/noise and the u8 structural prune. No thread-count tuning is mixed
into these results.

### Register pressure

Pressure is recorded per candidate and serialized even when the candidate is
pruned. For the 25-op plan, `maxLiveVectors=3`, `constantCount=13`, and the
candidate values are:

| unroll | estimated live vector registers | result |
|---:|---:|---|
| 1 | 18 | eligible |
| 2 | 21 | eligible |
| 4 | 27 | eligible |
| 8 | 39 | pruned before compilation |

The implementation does not claim that every eligible u4/u8 loss is a spill
without assembly or hardware-counter evidence. It preserves the losing
timings, code size, and noise so a later assembly/perf pass can distinguish
spills from frequency, cache, or measurement effects.

### Layout and schedule experiment

The implemented schedule family is backend traversal only: flat contiguous
address progression versus nested row/column loops, with the same verified
SSA operation order and scalar tail. It does not alter polyhedral legality.
Non-contiguous/transpose-like layouts remain outside the AVX2 candidate family;
R4's scalar/R2 fallback remains authoritative. The requested tiled transpose
family (8/16/32/64) was not forced into this milestone, so there is no
unmeasured layout claim.

### FMA experiment

The default generator is strictly non-FMA. `KernelVariantSignature` rejects
explicit FMA under `STRICT`, and the default tuner always enumerates
`OptimizationSemantics.STRICT` with `FMA_CONTRACTION=OFF`. An opt-in
`enumerateWithFma` API exists only for an explicit `RELAXED` or `FAST` caller,
requires the `avx2+fma` ISA, and emits `_mm256_fmadd_pd`/`std::fma`. No FMA
candidate was used in the R5 strict benchmark, and no semantic permission was
inferred from expression shape or absent metadata.

### Roofline/perf analysis

The intensity calculation above is deliberately lightweight. R5 does not
pretend to have a calibrated DRAM roofline, does not depend on `perf`, and
does not make correctness depend on hardware counters. No perf-counter run or
assembly golden test is part of this checkpoint; that is a remaining evidence
gap rather than a fabricated spill claim. The source audit confirms strict
compile flags, scalar tails, constants hoisted for AVX2, and no `fmadd` in
strict generated sources.

### Negative results

The negative evidence is retained instead of only reporting the fastest row:

* 2-op 256² u4's approximately 1% kernel-only improvement is below the 3%
  promotion threshold.
* 25-op 64² u2/u4 variants are close to baseline and do not earn promotion.
* 25-op u8 is rejected before compilation at estimated pressure 39.
* Several nested/flat differences are inside the 10% relative-MAD policy;
  they are not converted into universal schedule rules.
* Scalar JNI ties or beats AVX2 at the smallest crossover points.
* The initial implementation does not claim a tiled-transpose or prefetch win.

### Regression evidence

The focused R4/R5/post-R5 suite passed, including `R4CodegenTest`,
`R4GeneratedNativeExecutionTest`, `R5AutotuningTest`, native variant
coexistence, the direct-native benchmark, and the opt-in JNI crossover
harness. A full `bash gradlew test` run executed 519 tests with five skips and
two failures. Both failures reproduce on the clean R4 base
`feature/compiler-generated-simd` at `10a2905`: the environment-sensitive
`AlgorithmDispatchTest.coldStartAllowsCppOnlyForFoundationAlgorithmsAboveThreshold`
assertion and the vendor-BLAS expectation in
`NativeGemmIntegrationTest.nativeBackendUsesAvx2PackedMicrokernelOnWindowsBuild`.
They are unrelated to the R5 files and are retained as explicitly documented
pre-existing/environment failures.

### Existing JLC calibration patterns

R5 follows the existing JLC calibration ideas in
`src/main/java/net/faulj/autotune/persist/MachineFingerprint.java`,
`ProfileValidator`, and `ProfileStore`: explicit fingerprints, versioned
profiles, correctness/sanity validation, and conservative fallback. It keeps a
separate profile type because exact generated KernelSignatures, variant
availability, native ISA, and raw candidate samples are different contracts.
Unlike the existing GEMM convenience store, R5 does not write to a home
directory during normal execution.

### Required audit questions

1. **What dimensions are tuned?** Backend, AVX2 u1/u2/u4/u8, flat/nested
   traversal, scalar tail, ISA, and semantic/FMA policy.
2. **How many candidates exist?** Ten by default per exact KernelSignature;
   the caller can lower the cap and explicit families remain bounded.
3. **How are they identified?** Canonical `KernelVariantSignature` text and
   SHA-256; timing is absent from identity.
4. **How are illegal candidates pruned?** Verified plan eligibility, ISA,
   available vector iterations, and the documented pressure estimate.
5. **How is correctness gated?** Every compiled candidate runs against the
   R3 reference on finite and IEEE-edge fixtures before timing.
6. **What statistic chooses?** Lowest stable median, with raw samples and
   min/max/MAD retained.
7. **What noise threshold is used?** Relative MAD at most 10%.
8. **What promotion margin is required?** At least 3% over the trusted
   generated baseline.
9. **How are ties resolved?** Median first, then stable variant ID.
10. **What identifies a machine?** Architecture, vendor, family/model/model
    name, ISA bits, OS, JVM, native compiler, and processor count.
11. **What identifies a build?** JLC version/SHA, signature versions, codegen
    ABI, compiler identity, and strict FP flags.
12. **What invalidates a profile?** Schema, machine, build, exact kernel
    signature, registered symbol, correctness/stability, or ISA mismatch.
13. **How are profiles persisted?** Human-readable deterministic JSON at an
    explicit configured path.
14. **How is parsing kept out of the hot path?** The selector loads once and
    caches resolved exact-signature selections.
15. **What is lookup overhead?** `measureLookupNanos` exists on the selector;
    the final full-suite probe reported a microsecond-scale cached lookup in this
    JVM run; this is a diagnostic, host-sensitive measurement.
    Lookup is an in-memory map operation and never benchmarks a kernel; the
    crossover report keeps it separate from kernel timing.
16. **What is the Java/native JNI crossover?** In this sample, native is
    already faster at 1–4 elements, scalar ties at the floor, and AVX2 wins
    consistently from 16 elements upward; the measured result is host-specific.
17. **Which unroll wins for low intensity?** u8 flat/nested was 880 ns at 64²
    for the 2-op direct-native case; at 256² the approximately 1% u4 result
    was below promotion margin.
18. **Which wins for arithmetic-heavy kernels?** 25-op u2/u4 remained close;
    25-op u8 was pressure-pruned, so R5 does not invent a heavy-kernel win.
19. **Where does pressure reverse gains?** The first clear structural boundary
    in the measured 25-op plan is u8: estimate 39 exceeds the 32 gate.
20. **Does flat versus nested matter?** Sometimes by a few percent, but many
    differences fall inside noise; the report retains both.
21. **Which ideas were negative?** High unroll on the 25-op DAG, below-threshold
    bandwidth wins, and unmeasured transpose/prefetch assumptions.
22. **Is FMA implemented?** Only explicit RELAXED/FAST `avx2+fma`; never default
    and never STRICT.
23. **How do winners generalize across shapes?** This evidence is exact-shape;
    the 64² and 256² winners differ, so no family heuristic is installed.
24. **What justifies future family heuristics?** Repeated wins across 64, 128,
    256, 512, and 1024 shapes for the same graph, with stable margins and
    end-to-end JNI evidence, would justify an analysis-only family study.
25. **What is the production integration?** `BackendChoice` and
    `KernelDispatchSelector` make R2 Java, scalar-native, and exact generated
    AVX2 variants peers; `calibrateBackends` records the comparable evidence.

### Remaining weaknesses

The honest post-R5 boundaries are exact-shape profiles, AVX2-only generated
native code, finite candidate space, no thread-count tuning, no production
runtime JIT, limited layout-aware scheduling, no GPU/distributed tuning, no
mixed precision, no optional perf counters, and no assembly-backed spill
diagnosis. These are explicit non-goals of this integration; the typed choice
boundary is ready for later backends without changing semantic signatures.

### Definition-of-done mapping

R5 retains multiple registered variants per semantic kernel, deterministic
bounded generation, structural pruning, correctness-before-timing, repeated
median/min/max/MAD statistics, explicit noise and promotion policies, losing
candidate evidence, and strict non-FMA semantics. The post-R5 integration adds
typed `BackendChoice` persistence, comparable end-to-end Java/JNI calibration,
cached exact-choice dispatch, atomic profile updates, and the
baseline-AVX2-to-scalar-to-R2-Java fallback. The opt-in native tests prove
variant coexistence and exact invocation; ordinary execution remains R2-safe
and does not compile or benchmark. This remains outside the M1–M4 milestone
path and does not begin R6.
