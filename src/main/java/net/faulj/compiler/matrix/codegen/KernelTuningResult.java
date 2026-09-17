package net.faulj.compiler.matrix.codegen;

import java.util.List;
import java.util.Objects;

/** Result of one explicit bounded tuning session. */
public final class KernelTuningResult {
    private final KernelSignature kernelSignature;
    private final KernelVariantSignature baseline;
    private final List<KernelCandidateEvidence> candidates;
    private final KernelVariantSignature winner;
    private final double winnerSpeedup;
    private final String decisionReason;
    private final List<KernelTrustedBaseline> trustedBaselines;

    public KernelTuningResult(KernelSignature kernelSignature,
                              KernelVariantSignature baseline,
                              List<KernelCandidateEvidence> candidates,
                              KernelVariantSignature winner,
                              double winnerSpeedup,
                              String decisionReason) {
        this(kernelSignature, baseline, candidates, winner, winnerSpeedup, decisionReason, List.of());
    }

    public KernelTuningResult(KernelSignature kernelSignature,
                              KernelVariantSignature baseline,
                              List<KernelCandidateEvidence> candidates,
                              KernelVariantSignature winner,
                              double winnerSpeedup,
                              String decisionReason,
                              List<KernelTrustedBaseline> trustedBaselines) {
        this.kernelSignature = Objects.requireNonNull(kernelSignature, "Kernel signature");
        this.baseline = Objects.requireNonNull(baseline, "Baseline variant");
        this.candidates = List.copyOf(candidates);
        this.winner = winner;
        this.winnerSpeedup = winnerSpeedup;
        this.decisionReason = decisionReason == null ? "" : decisionReason;
        this.trustedBaselines = trustedBaselines == null ? List.of() : List.copyOf(trustedBaselines);
    }

    public KernelSignature kernelSignature() {
        return kernelSignature;
    }

    public KernelVariantSignature baseline() {
        return baseline;
    }

    public List<KernelCandidateEvidence> candidates() {
        return candidates;
    }

    public KernelVariantSignature winner() {
        return winner;
    }

    public boolean hasWinner() {
        return winner != null;
    }

    public double winnerSpeedup() {
        return winnerSpeedup;
    }

    public String decisionReason() {
        return decisionReason;
    }

    public List<KernelTrustedBaseline> trustedBaselines() {
        return trustedBaselines;
    }

    public KernelCandidateEvidence evidence(KernelVariantSignature signature) {
        return candidates.stream()
            .filter(candidate -> candidate.variantSignature().equals(signature))
            .findFirst().orElse(null);
    }

    public String reportTable() {
        StringBuilder result = new StringBuilder(
            "| Kernel | Variant | Unroll | Median ns/el | MAD | Correct | Selected |\n"
                + "|---|---|---:|---:|---:|:---:|:---:|\n");
        long elements = 1L;
        for (KernelCandidateEvidence evidence : candidates) {
            KernelBenchmarkStatistics stats = evidence.statistics();
            double median = stats == null ? Double.NaN : stats.nsPerElement(elements);
            double mad = stats == null ? Double.NaN : stats.madNanos();
            result.append("| ").append(kernelSignature.shortHash()).append(" | ")
                .append(evidence.variantSignature().variantId()).append(" | ")
                .append(evidence.variantSignature().unroll()).append(" | ")
                .append(median).append(" | ").append(mad).append(" | ")
                .append(evidence.correctnessPassed() ? "PASS" : "NO")
                .append(" | ").append(evidence.variantSignature().equals(winner) ? "yes" : "no")
                .append(" |\n");
        }
        return result.toString();
    }
}
