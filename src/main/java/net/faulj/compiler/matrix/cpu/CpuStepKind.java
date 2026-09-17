package net.faulj.compiler.matrix.cpu;

/** The small set of executable operations emitted by the M4 CPU lowerer. */
public enum CpuStepKind {
    GEMM,
    ELEMENTWISE,
    TRANSPOSE,
    FUSED_ELEMENTWISE
}
