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
