package net.faulj.compiler.matrix.codegen;

/** Evidence state for one candidate in a tuning report. */
public enum KernelCandidateOutcome {
    PRUNED,
    COMPILATION_FAILED,
    CORRECTNESS_FAILED,
    BENCHMARK_FAILED,
    NOISY,
    PASS
}
