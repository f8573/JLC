# Matrix Compiler M1–M4

JLC now has a small, opt-in matrix-expression compiler in
`net.faulj.compiler.matrix`. It implements bounded affine/polyhedral-style
schedule transformations over JLC's supported matrix IR without changing the eager
`net.faulj.matrix.Matrix` API or the existing GEMM backend.

## Matrix and MatrixExpr

`Matrix` remains mutable and eager. `MatrixExpr` is an immutable, typed DAG
whose runtime `Input` nodes hold references to existing matrices. The M1
contract is that input matrices must not be mutated concurrently with graph
evaluation. `MatrixExpr.symbolicInput` is shape-only and is useful for planner
tests; symbolic graphs cannot be evaluated.

Every node exposes a `MatrixShape` without executing:

- `Input`: the referenced matrix shape;
- `MatMul`: `(m x k) * (k x n) -> (m x n)`;
- `Add`: identical operand shapes;
- `Scale`: its operand shape;
- `Transpose`: `(m x n) -> (n x m)`.

Invalid shapes fail when the node is built. `transpose(transpose(x))` and the
exact identity `scale(1.0, x)` are the only M1 canonicalizations. Scale
composition and algebraic identities are intentionally not inferred.

## Optimization semantics

`OptimizationSemantics.STRICT` is the default and preserves user-specified
MatMul association and compiler-visible expression ordering constraints.
Individual GEMM calls still use the production GEMM implementation's internal
reduction order; `STRICT` does not promise Java scalar or bitwise execution.
`RELAXED` allows matrix-chain reassociation and documents that rounding can
differ. `FAST` currently has the same permissions as `RELAXED` and does not
authorize unsafe M1 rewrites.

## Planning and cost

`MatrixCompiler.compile` returns an immutable `ExecutionPlan`, separate from
the source expression graph. It exposes a nested expression form through
`expression()` and an SSA-like `dump()` that makes shared producers visible.
The planner uses identity maps, so a shared node is planned and evaluated once
rather than duplicated.

For `RELAXED` and `FAST`, maximal pure MatMul chains are selected with the
classical dynamic-programming matrix-chain algorithm. `FlopCostModel` uses
`m * k * n` scalar multiplications for `(m x k) * (k x n)`, plus a
saturation-safe estimate of dense output bytes (`rows * columns * 8`).
Arithmetic uses `long` values and saturates at `Long.MAX_VALUE` on overflow.
The cost-model interface leaves room for measured GEMM throughput, temporary
allocation, cache traffic, backend availability, and CPU/GPU placement later.

## Execution

`MatrixCompiler.evaluate(expr)` is an opt-in interpreter for a plan. MatMul
nodes call the existing canonical `net.faulj.kernels.gemm.Gemm` facade. Add,
scale, and transpose reuse the existing `Matrix` operations. Intermediates
are materialized in M1, with identity-based memoization for shared DAG
producers. Compiler-owned hidden `OffHeapMatrix` intermediates are closed once
at the end of evaluation on success or failure. Borrowed inputs are never
closed, and an off-heap result returned to the caller remains the caller's
responsibility.

## Explicitly out of scope for M1

M1 does not add polyhedral integer sets, ISL, affine dependence analysis,
loop IR, loop transformations, fusion, generated C++/AVX2/CUDA, autotuning,
dynamic compilation, MLIR, expression-template changes to `Matrix`, GEMM
dispatch replacement, or whole-program interception.

## M2: memory semantics and affine/dependence IR

M2 answers what computation a selected M1 `ExecutionPlan` implies. The public
entry point is `MatrixAffineCompiler.lower(plan)`. Lowering walks the selected
plan in deterministic producer-before-consumer order and does not bypass the
execution plan or execute the resulting IR.

### Logical buffers

Every value visible to the affine program has one stable `LogicalBuffer` ID.
Buffers are compiler facts, not replacement runtime allocations:

- runtime `Input`: `EXTERNAL_INPUT`, `BORROWED`, with `HEAP` or `OFF_HEAP`
  classification when its public Java type makes that reliable;
- compiler-produced values: `TEMPORARY`, `OWNED`, with abstract `UNKNOWN`
  placement;
- `SymbolicInput`: `SYMBOLIC`, `NONE`, and no runtime lifetime.

Distinct external objects remain conservatively `MAY_ALIAS`, since wrappers
and shared backing storage can make Java-object identity insufficient to prove
disjointness. Shared runtime matrix identity is one logical external buffer;
distinct temporaries are `NO_ALIAS`; a temporary and an external input are
`NO_ALIAS`.

Graph inputs are borrowed. JLC does not snapshot their contents, and
evaluation observes the values supplied at evaluation time. Callers must not
mutate an input concurrently with evaluation; caller-owned off-heap storage
must remain valid for the whole evaluation; and the compiler never closes or
frees caller-owned off-heap memory. M2 adds no deep copies, allocator, buffer
reuse, or memory planning.

Each temporary records its producer, first use, and last use when those facts
exist. External inputs are marked live for the evaluation span. M2 only
reports these facts; it does not free or reuse storage.

### Bounded affine IR

The `net.faulj.compiler.matrix.affine` package contains immutable
`AffineVariable`, `AffineExpr`, `IterationDomain`, `AffineAccess`,
`AffineStatement`, and `AffineProgram` objects. `AffineExpr` is an integer
linear form: constants and terms such as `i`, `i + 4`, or `2*i + j` are
representable, while nonlinear products are not. Domains are static,
half-open rectangular ranges such as `0 <= i < M` and `0 <= j < N`; M2 does
not include a Presburger solver, floor/modulo machinery, ISL, LLVM, or MLIR.

### Supported lowering

`Input` and `SymbolicInput` create buffers but no compute statements. The
remaining M1 operations lower as follows:

- `Add`: `C[i,j] = A[i,j] + B[i,j]` over the result rectangle;
- `Scale`: `C[i,j] = alpha * A[i,j]` over the operand rectangle;
- `Transpose`: read `A[i,j]` and write `C[j,i]` over the source rectangle;
- `MatMul`: an initialization statement `C[i,j] = 0` followed by an update
  statement `C[i,j] += A[i,k] * B[k,j]` over `i`, `j`, and reduction `k`.

Every statement exposes its domain and `READ`, `WRITE`, `READ_WRITE`, or
`REDUCTION` accesses. The separate `DependenceGraph` reports `RAW`, `WAR`,
`WAW`, and `REDUCTION` relationships. Queries return
`PROVEN_DEPENDENCE`, `PROVEN_NONE`, or `UNKNOWN`; an ambiguous alias is never
silently treated as independence. Only `READ`/`READ` participation is harmless
under `MAY_ALIAS`; any possible write keeps the relationship `UNKNOWN`.
Ordinary self RAW/WAR/WAW effects are analyzed independently of reduction
metadata, and unresolved access forms stay conservative.

MatMul initialization-to-update and update-to-consumer relationships are
derived for the canonical accesses emitted by M2. The update statement also
records its `k`-carried reduction relationship. `STRICT` records an ordered,
conservative reduction; `RELAXED` and `FAST` mark reassociation as eligible
for bounded legality checks. A carried reduction dependence is reported only
when the domain contains at least two valid `k` instances; empty and `K=1`
domains do not invent one. M2 performs no schedule transformation and does not
claim bitwise scalar ordering from the existing optimized GEMM backend.

### Inspection example

For a symbolic `A: M x K` and `B: K x N`, an affine dump includes the shape
and access facts in a form like:

```text
buffers:
  %0 A symbolic none unknown shape=MxK
  %1 B symbolic none unknown shape=KxN
  %2 tmp0 temporary owned unknown shape=MxN

statements:
  S0[i,j] : %2[i,j] = 0
  S1[i,j,k] : %2[i,j] += %0[i,k] * %1[k,j]

accesses:
  S0 WRITE %2[i,j]
  S1 READ %0[i,k]
  S1 READ %1[k,j]
  S1 REDUCTION %2[i,j]

dependences:
  S0 -> S1 : RAW [PROVEN_DEPENDENCE]
  S0 -> S1 : WAW [PROVEN_DEPENDENCE]
  S1 -> S1 : REDUCTION [PROVEN_DEPENDENCE] same (i,j); k -> k+1
```

The M2 affine representation has no interpreter. Runtime evaluation remains
the M1 path through `MatrixCompiler` and the existing `Gemm` facade. CPU
lowering from this semantic representation is reserved for M4.

## M3: bounded affine schedule transformations

M3 consumes an `AffineProgram` and its unchanged `DependenceGraph` and
produces a separate immutable `SchedulePlan`. The computation and schedule
remain distinct: schedule transformations reorder or annotate semantic
statements, but M3 never executes a transformed schedule.

### Schedule representation

The `net.faulj.compiler.matrix.schedule` package contains immutable
`ScheduleSequence`, `ScheduleBand`, `ScheduleLoop`, and `ScheduleStatement`
nodes. The initial schedule copies each affine statement's domain order
directly, for example `i -> j -> k` for a MatMul update. Loop bounds, steps,
tile bindings, guards, and annotations are explicit and dumps are
deterministic.

Parallel and vector annotations are eligibility metadata. Relaxed reduction
dimensions use explicit reduction metadata rather than claiming that M3 has
created executable parallel reduction code.

### Centralized legality

`ScheduleLegality` returns `LEGAL`, `ILLEGAL`, or `UNKNOWN` with a deterministic
explanation. M3 derives only the bounded direction facts required by the
canonical M2 accesses, including constant affine offsets and the MatMul
reduction dimension. An unknown alias, direction, or dependence is rejected;
it never becomes optimistic independence.
Identical write accesses are compared across iterations: unique unit
projections can prove independence, while collapsed or unsupported mappings
remain conservative.

### M3 transformation vocabulary

M3 implements exactly these five transformations:

- adjacent loop `INTERCHANGE` with dependence-direction checking;
- constant-positive `STRIP_MINE` / tiling, including explicit remainder
  guards and multidimensional composition; generated tile loops cannot be
  re-strip-mined, and a binder cannot move behind an index that depends on it;
- producer/consumer or sibling `FUSION` only for compatible domains, bands,
  proven statement ordering, and pointwise RAW, WAR, and WAW ordering at
  unique shared locations; unsupported offsets remain `UNKNOWN`;
- `PARALLEL` loop marking only when no blocking loop-carried dependence exists;
- `VECTOR` loop marking only when legality is proven.

Strict MatMul reductions preserve `k` ordering: `k` cannot be interchanged,
parallel-marked, or vector-marked when that would change reduction ordering.
Relaxed and fast semantics may accept explicitly requested reduction
reassociation, parallel-reduction eligibility, or vector metadata, but no
reduction execution is implemented.

### Candidate bound

The structural candidate generator emits the initial plan and at most one-step
legal candidates from the supported vocabulary. It has a hard cap of 32
candidates per generation. Tile sizes remain explicit inputs, and M3 performs
no measured, cache-based, hardware-specific, machine-learning, or open-ended
schedule search.

### Schedule inspection

An inspectable M3 dump has the following shape:

```text
initial schedule:
  region 1:
    for i [0,2) step 1
      for j [0,4) step 1
        for k [0,3) step 1
          S1[i,j,k]
schedule:
  region 1:
    for j [0,4) step 1
      for i [0,2) step 1
        for k [0,3) step 1
          S1[i,j,k]
transformation history:
  interchange(i,j) in S1 -> LEGAL: no dependence carried in the proposed loop order
```

This output claims representation and legality only. M3 has no
`AffineProgram` interpreter, CPU code generator, native-loop generator,
Vector API generation, runtime thread scheduler, or benchmark-driven schedule
choice. Those concerns remain outside M3 and CPU lowering begins in M4.

## M4: CPU lowering and end-to-end execution

M4 is the terminal execution layer. The complete opt-in path is:

```text
MatrixExpr
  -> ExecutionPlan
  -> AffineProgram
  -> DependenceGraph
  -> SchedulePlan
  -> CpuExecutionPlan
  -> Matrix
```

`MatrixCompiler.compileProgram(expression, semantics)` returns an immutable
`CompiledMatrixProgram` containing every inspectable stage. Its `execute()`
method executes the `CpuExecutionPlan`; it does not interpret the affine IR.
The CPU package is `net.faulj.compiler.matrix.cpu` and contains the bounded
`CpuLowerer`, `CpuExecutionPlan`, `CpuExecutor`, `CpuBufferBinding`, and
operation-specific `CpuStep` types.

### Hybrid CPU lowering

M4 deliberately has two lowering boundaries:

- `MATMUL_INIT` plus `MATMUL_UPDATE` become one opaque `CpuGemmStep` that calls
  the canonical `net.faulj.kernels.gemm.Gemm.multiply` facade. The native or
  Java backend, packing, blocking, microkernel dispatch, and worker scheduling
  remain owned by the existing GEMM implementation. Displayed MatMul loop
  interchange, tiling, and annotations are not interpreted by the opaque call.
- `ADD`, `SCALE`, and `TRANSPOSE` become bounded explicit Java loops over the
  M3 schedule band. The loop realization honors loop interchange and
  strip-mining, including generated remainder guards. It uses public
  `Matrix.get`, `getImag`, `set`, and `setComplex` APIs and does not weaken
  Matrix encapsulation.

The initial M4 schedule recognizes one safe M3 fusion family:

```text
T = scale(A, alpha)
Z = add(T, B)
```

The CPU plan realizes this as one loop,
`Z[i,j] = alpha * A[i,j] + B[i,j]`, and elides the logical `T` materialization.
The affine statements and M2 logical buffer remain available for inspection;
only the CPU materialization decision changes. Arbitrary expression fusion,
fusion search, and buffer reuse are not implemented.

M3 `PARALLEL` and `VECTOR` annotations continue to mean “legal to realize,”
not “must realize.” M4 preserves those annotations and uses deterministic
serial scalar fallback. This keeps legality separate from a new thread-pool
or SIMD implementation and leaves native GEMM concurrency unchanged.

Before lowering, M4 validates the executable subset and the retained M1/M2
provenance. It requires exact logical `[0,N)` bounds for executable matrix
dimensions, including tiled bindings, and rejects writes to borrowed buffers,
foreign buffers, incomplete or shifted domains or loop bands, malformed tile
bindings and guards, nested schedule
bodies it cannot reproduce, mismatched MatMul forms, and fusion families other
than the validated scale-then-add case. Rejection is intentional; M4 is not a
general affine interpreter.

### Runtime ownership and symbolic plans

External `Input` buffers reuse M2's `EXTERNAL_INPUT` / `BORROWED` facts and are
bound by reference at execution time. The CPU plan does not snapshot inputs;
callers must avoid concurrent mutation and must keep caller-owned off-heap
storage valid. The compiler never closes or frees external storage. CPU
temporaries are execution-owned. Hidden owned off-heap intermediates are
closed exactly once at the end of execution on success or failure, including
shared-DAG values. A returned off-heap result transfers to the caller and must
be closed by the caller. There is no allocator, pool, lifetime-based reuse, or
early-free policy. `SymbolicInput` buffers remain non-executable and unresolved
symbolic plans fail before any CPU step begins.

### Fixed M4 benchmark suite

The benchmark suite is one JUnit class with exactly four families. Each case
uses deterministic inputs, two warmups, five measured iterations, median
reporting, checksum consumption, and a correctness comparison before timing.
Compilation and input construction are outside every timed region. Fusion
compares eager direct `Matrix` operations with an already-compiled fused
program's `execute()` method. GEMM dispatch is reported per product where the
chain can mix Java and native selection. The measurements below are a fresh,
isolated publication validation run from Java 21.0.12 on Linux/amd64, on an
AMD Ryzen 9 3950X with 16 physical cores / 32 logical processors and the
active native backend. They are not universal performance claims.

| Family | Shapes / comparison | Observed result |
| --- | --- | --- |
| Matrix-chain reassociation | `1000x10 * 10x1000 * 1000x10`; STRICT vs RELAXED | Planner cost `20,000,000` vs `200,000`; product backends `[native,native]` vs `[java,native]`; median `5.921 ms` vs `5.094 ms`; observed ratio `1.162x`; correctness passed. |
| Elementwise fusion | `scale(256x256, 2.5) + 256x256` | Temporary materializations `2` eager vs `1` compiler; estimated temporary bytes `1,048,576` vs `524,288`; median `4.403 ms` eager direct operations vs `15.520 ms` compiled fused execution; fusion was slower; correctness passed. |
| Direct GEMM dispatch boundary | `192x192 * 192x192` | Direct `0.431 ms`; compiled `0.419 ms`; execution delta `-2.66%`, classified as measurement noise under the benchmark's ±5% band; selected backend `native`; correctness passed through the same GEMM facade. |
| Shared DAG | `X = A * B; Y = X + X`, `96x96` | One planned GEMM step, runtime invocation count `not-instrumented`, two logical temporary buffers, median `22.673 ms`; shared plan node represented once; correctness passed. |

These results distinguish planner arithmetic reduction, temporary
materialization reduction, and observed execution timing. No result is used as
a completion gate or as a reason to begin more performance tuning.

### M4 limitations

M4 does not add new schedule transformations, a universal affine interpreter,
automatic parallel execution, arbitrary SIMD synthesis, GEMM replacement,
buffer reuse, allocation planning, JIT/code generation, or a new native
backend. The explicit loop backend is intentionally small and conservative;
opaque GEMM remains the production path for multiplication.

## Fixed matrix-compiler milestone path

1. M1 — Graph IR + whole-expression optimization
2. M2 — Memory semantics + affine/dependence IR
3. M3 — Bounded affine/polyhedral-style schedule transformations
4. M4 — CPU lowering + end-to-end benchmark proof

**M4 IS THE TERMINAL MATRIX-COMPILER MILESTONE. THERE IS NO M5.**

Extensions outside this M1–M4 pathway include CUDA, GPU scheduling,
heterogeneous CPU/GPU placement, autotuning research, ML cost models,
distributed execution, and unrelated compiler research. Those may be separate
future projects, but they must not extend the matrix-compiler milestone path.

## Post-M4 ideas

CUDA/GPU backends, heterogeneous placement, autotuning, ML cost models, more
general polyhedral solving, JIT/code generation, and additional fusion families
are separate future projects, not unfinished M4 work. They do not extend the
M1–M4 matrix-compiler pathway.
