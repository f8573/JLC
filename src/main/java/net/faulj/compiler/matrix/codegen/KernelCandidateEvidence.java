package net.faulj.compiler.matrix.codegen;

import java.util.Objects;

/** Complete inspectable result for one considered candidate, including losers. */
public final class KernelCandidateEvidence {
    private final KernelVariantCandidate candidate;
    private final KernelCandidateOutcome outcome;
    private final KernelCorrectnessResult correctness;
    private final KernelBenchmarkStatistics statistics;
    private final long sourceGenerationNanos;
    private final long compilationNanos;
    private final long benchmarkNanos;
    private final long compiledTextBytes;
    private final String reason;

    public KernelCandidateEvidence(KernelVariantCandidate candidate,
                                   KernelCandidateOutcome outcome,
                                   KernelCorrectnessResult correctness,
                                   KernelBenchmarkStatistics statistics,
                                   long sourceGenerationNanos,
                                   long compilationNanos,
                                   long benchmarkNanos,
                                   long compiledTextBytes,
                                   String reason) {
        this.candidate = Objects.requireNonNull(candidate, "Candidate");
        this.outcome = Objects.requireNonNull(outcome, "Outcome");
        this.correctness = correctness;
        this.statistics = statistics;
        this.sourceGenerationNanos = Math.max(0L, sourceGenerationNanos);
        this.compilationNanos = Math.max(0L, compilationNanos);
        this.benchmarkNanos = Math.max(0L, benchmarkNanos);
        this.compiledTextBytes = Math.max(0L, compiledTextBytes);
        this.reason = reason == null ? "" : reason;
    }

    public KernelVariantCandidate candidate() {
        return candidate;
    }

    public KernelVariantSignature variantSignature() {
        return candidate.signature();
    }

    public KernelCandidateOutcome outcome() {
        return outcome;
    }

    public KernelCorrectnessResult correctness() {
        return correctness;
    }

    public KernelBenchmarkStatistics statistics() {
        return statistics;
    }

    public boolean correctnessPassed() {
        return correctness != null && correctness.passed();
    }

    public boolean stable() {
        return outcome == KernelCandidateOutcome.PASS && statistics != null;
    }

    public long sourceGenerationNanos() {
        return sourceGenerationNanos;
    }

    public long compilationNanos() {
        return compilationNanos;
    }

    public long benchmarkNanos() {
        return benchmarkNanos;
    }

    public long compiledTextBytes() {
        return compiledTextBytes;
    }

    public String reason() {
        return reason;
    }
}
