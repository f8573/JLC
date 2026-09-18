package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable result and diagnostics for one R2-to-R3 lowering. */
public final class KernelLoweringResult {
    private final KernelEligibility eligibility;
    private final KernelProgram program;
    private final KernelVerificationResult verification;
    private final String reason;
    private final long loweringTimeNanos;
    private final long verificationTimeNanos;

    KernelLoweringResult(KernelEligibility eligibility,
                         KernelProgram program,
                         KernelVerificationResult verification,
                         String reason,
                         long loweringTimeNanos,
                         long verificationTimeNanos) {
        this.eligibility = eligibility;
        this.program = program;
        this.verification = verification;
        this.reason = reason;
        this.loweringTimeNanos = loweringTimeNanos;
        this.verificationTimeNanos = verificationTimeNanos;
    }

    public KernelEligibility eligibility() {
        return eligibility;
    }

    public boolean isEligible() {
        return eligibility == KernelEligibility.ELIGIBLE
            && program != null
            && verification != null
            && verification.valid();
    }

    public KernelProgram program() {
        return program;
    }

    public KernelProgram kernelProgram() {
        return program;
    }

    public KernelVerificationResult verification() {
        return verification;
    }

    public boolean verified() {
        return verification != null && verification.valid();
    }

    public String reason() {
        return reason;
    }

    public List<String> diagnostics() {
        List<String> result = new ArrayList<>();
        if (reason != null && !reason.isBlank()) {
            result.add(reason);
        }
        if (verification != null && !verification.valid()) {
            result.addAll(verification.diagnostics());
        }
        return Collections.unmodifiableList(result);
    }

    public long loweringTimeNanos() {
        return loweringTimeNanos;
    }

    public long verificationTimeNanos() {
        return verificationTimeNanos;
    }

    public long loweringTimeMicros() {
        return loweringTimeNanos / 1_000L;
    }

    public long verificationTimeMicros() {
        return verificationTimeNanos / 1_000L;
    }

    public KernelIrMetrics metrics() {
        return KernelIrMetrics.from(List.of(this));
    }

    /** Compact diagnostic line suitable for compiler reports. */
    public String diagnostic() {
        int buffers = 0;
        int loops = 0;
        int scalarOps = 0;
        int loads = 0;
        int stores = 0;
        if (program != null) {
            for (KernelFunction function : program.functions()) {
                buffers += function.buffers().size();
                loops += function.loops().size();
                if (function.body() != null) {
                    scalarOps += function.body().operations().size();
                    for (KernelOp operation : function.body().operations()) {
                        if (operation.opcode() == KernelOpcode.LOAD) {
                            loads++;
                        } else if (operation.opcode() == KernelOpcode.STORE) {
                            stores++;
                        }
                    }
                }
            }
        }
        return "kernel IR: eligible=" + eligibility.name().toLowerCase()
            + ", buffers=" + buffers + ", loops=" + loops
            + ", scalarOps=" + scalarOps + ", loads=" + loads
            + ", stores=" + stores + ", verification="
            + (verified() ? "PASS" : "FAIL") + ", reason=" + reason;
    }
}
