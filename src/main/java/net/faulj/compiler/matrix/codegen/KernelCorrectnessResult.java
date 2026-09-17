package net.faulj.compiler.matrix.codegen;

/** Result of the correctness gate; failures are evidence and never benchmarked. */
public record KernelCorrectnessResult(boolean passed, int casesChecked, String diagnostic) {
    public KernelCorrectnessResult {
        diagnostic = diagnostic == null ? "" : diagnostic;
    }
}
