package net.faulj.compiler.matrix.codegen;

/** The existing R2 generalized fused Java executor. */
public record R2JavaBackend() implements BackendChoice {
    @Override
    public BackendKind kind() {
        return BackendKind.R2_JAVA;
    }
}
