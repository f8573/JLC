package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable result of deterministic Kernel IR verification. */
public final class KernelVerificationResult {
    private final boolean valid;
    private final List<String> diagnostics;
    private final long verificationTimeNanos;

    KernelVerificationResult(boolean valid,
                             List<String> diagnostics,
                             long verificationTimeNanos) {
        this.valid = valid;
        this.diagnostics = Collections.unmodifiableList(new ArrayList<>(diagnostics));
        this.verificationTimeNanos = verificationTimeNanos;
    }

    public boolean valid() {
        return valid;
    }

    public boolean isValid() {
        return valid;
    }

    public List<String> diagnostics() {
        return diagnostics;
    }

    public String diagnostic() {
        return diagnostics.isEmpty() ? "PASS" : diagnostics.get(0);
    }

    public long verificationTimeNanos() {
        return verificationTimeNanos;
    }

    public long verificationTimeMicros() {
        return verificationTimeNanos / 1_000L;
    }

    @Override
    public String toString() {
        return valid ? "PASS" : "FAIL: " + String.join("; ", diagnostics);
    }
}
