package net.faulj.compiler.matrix.kernel;

/** Raised when execution or code generation is attempted for invalid IR. */
public final class KernelVerificationException extends IllegalArgumentException {
    private final KernelVerificationResult result;

    KernelVerificationException(KernelVerificationResult result) {
        super("Kernel IR verification failed: " + String.join("; ", result.diagnostics()));
        this.result = result;
    }

    public KernelVerificationResult result() {
        return result;
    }
}
