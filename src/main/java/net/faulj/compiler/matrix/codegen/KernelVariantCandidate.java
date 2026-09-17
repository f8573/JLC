package net.faulj.compiler.matrix.codegen;

import java.util.Objects;

/** One deterministic search-space member, including candidates pruned before compilation. */
public final class KernelVariantCandidate {
    public enum Status {
        ELIGIBLE,
        PRUNED
    }

    private final KernelVariantSignature signature;
    private final Status status;
    private final int estimatedLiveVectorRegisters;
    private final String reason;

    private KernelVariantCandidate(KernelVariantSignature signature,
                                   Status status,
                                   int estimatedLiveVectorRegisters,
                                   String reason) {
        this.signature = Objects.requireNonNull(signature, "Variant signature");
        this.status = Objects.requireNonNull(status, "Candidate status");
        this.estimatedLiveVectorRegisters = Math.max(0, estimatedLiveVectorRegisters);
        this.reason = reason == null ? "" : reason;
    }

    public static KernelVariantCandidate eligible(KernelVariantSignature signature,
                                                  int estimatedLiveVectorRegisters) {
        return new KernelVariantCandidate(signature, Status.ELIGIBLE,
            estimatedLiveVectorRegisters, "eligible");
    }

    public static KernelVariantCandidate pruned(KernelVariantSignature signature,
                                                int estimatedLiveVectorRegisters,
                                                String reason) {
        return new KernelVariantCandidate(signature, Status.PRUNED,
            estimatedLiveVectorRegisters, reason);
    }

    public KernelVariantSignature signature() {
        return signature;
    }

    public KernelVariantSignature variantSignature() {
        return signature;
    }

    public Status status() {
        return status;
    }

    public boolean eligible() {
        return status == Status.ELIGIBLE;
    }

    public boolean isPruned() {
        return status == Status.PRUNED;
    }

    public int estimatedLiveVectorRegisters() {
        return estimatedLiveVectorRegisters;
    }

    public String reason() {
        return reason;
    }

    @Override
    public String toString() {
        return signature.variantId() + " [" + status + ", pressure="
            + estimatedLiveVectorRegisters + ", reason=" + reason + "]";
    }
}
