package net.faulj.compiler.matrix.schedule;

import java.util.Objects;

/**
 * Deterministic legality status and human-readable explanation.
 */
public record LegalityResult(LegalityStatus status, String explanation) {
    public LegalityResult {
        Objects.requireNonNull(status, "Legality status must not be null");
        if (explanation == null || explanation.isBlank()) {
            throw new IllegalArgumentException("Legality explanation must not be blank");
        }
    }

    public static LegalityResult legal(String explanation) {
        return new LegalityResult(LegalityStatus.LEGAL, explanation);
    }

    public static LegalityResult illegal(String explanation) {
        return new LegalityResult(LegalityStatus.ILLEGAL, explanation);
    }

    public static LegalityResult unknown(String explanation) {
        return new LegalityResult(LegalityStatus.UNKNOWN, explanation);
    }

    public boolean isLegal() {
        return status == LegalityStatus.LEGAL;
    }

    public boolean accepted() {
        return isLegal();
    }

    public boolean isRejected() {
        return !isLegal();
    }

    public String diagnostic() {
        return explanation;
    }

    @Override
    public String toString() {
        return status + ": " + explanation;
    }
}
