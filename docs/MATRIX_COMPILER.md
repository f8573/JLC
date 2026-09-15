# Matrix Compiler M1

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

## Roadmap

1. M1 — graph IR and matrix-chain optimization;
2. M2 — affine loop IR and dependence representation;
3. M3 — schedule transformations and CPU lowering;
4. M4 — CUDA lowering and hardware-aware backend selection.

M1 demonstrates planner capability, not a real-workload performance claim.
Controlled benchmarks on actual programs are required before making one.
