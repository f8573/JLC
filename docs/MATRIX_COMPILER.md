# Matrix Compiler M1–M3

JLC now has a small, opt-in matrix-expression compiler in
`net.faulj.compiler.matrix`. It provides trustworthy infrastructure for
future matrix-program and polyhedral work without changing the eager
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
MatMul association and operation order. `RELAXED` allows matrix-chain
reassociation and documents that rounding can differ. `FAST` currently has
the same permissions as `RELAXED`; it reserves a named home for future
fusion, vectorization, and scheduling permissions without authorizing unsafe
M1 rewrites today.

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
producers; fusion and temporary lifetime optimization are future work.

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
silently treated as independence.

MatMul initialization-to-update and update-to-consumer relationships are
derived for the canonical accesses emitted by M2. The update statement also
records its `k`-carried reduction relationship. `STRICT` records an ordered,
conservative reduction; `RELAXED` and `FAST` mark reassociation as eligible
for a future legality proof. M2 performs no schedule transformation and does
not claim bitwise scalar ordering from the existing optimized GEMM backend.

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

## M3: legal polyhedral schedule transformations

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

### M3 transformation vocabulary

M3 implements exactly these five transformations:

- adjacent loop `INTERCHANGE` with dependence-direction checking;
- constant-positive `STRIP_MINE` / tiling, including explicit remainder
  guards and multidimensional composition;
- producer/consumer or sibling `FUSION` only for compatible domains, bands,
  and proven statement ordering;
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

## Fixed matrix-compiler milestone path

1. M1 — Graph IR + whole-expression optimization
2. M2 — Memory semantics + affine/dependence IR
3. M3 — Legal polyhedral schedule transformations
4. M4 — CPU lowering + end-to-end benchmark proof

**M4 is the terminal milestone. There is no M5 in the matrix-compiler pathway.**

Extensions outside this M1–M4 pathway include CUDA, GPU scheduling,
heterogeneous CPU/GPU placement, autotuning research, ML cost models,
distributed execution, and unrelated compiler research. Those may be separate
future projects, but they must not extend the matrix-compiler milestone path.

M1 and M2 demonstrate inspectable compiler infrastructure, not a real-workload
performance claim. Controlled benchmarks on actual programs are required
before making one.
