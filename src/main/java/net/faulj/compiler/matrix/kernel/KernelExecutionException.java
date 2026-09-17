package net.faulj.compiler.matrix.kernel;

/** Deterministic failure for an unresolved or unsupported reference binding. */
public final class KernelExecutionException extends IllegalStateException {
    public KernelExecutionException(String message) {
        super(message);
    }
}
