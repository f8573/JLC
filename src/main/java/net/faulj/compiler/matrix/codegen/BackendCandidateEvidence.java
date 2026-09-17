package net.faulj.compiler.matrix.codegen;

import java.util.Objects;

/** Comparable end-to-end evidence for one typed backend choice. */
public final class BackendCandidateEvidence {
    private final BackendChoice choice;
    private final KernelCandidateOutcome outcome;
    private final boolean correctnessPassed;
    private final boolean stable;
    private final KernelBenchmarkStatistics statistics;
    private final long benchmarkNanos;
    private final String reason;

    public BackendCandidateEvidence(BackendChoice choice,
                                    KernelCandidateOutcome outcome,
                                    boolean correctnessPassed,
                                    boolean stable,
                                    KernelBenchmarkStatistics statistics,
                                    long benchmarkNanos,
                                    String reason) {
        this.choice = Objects.requireNonNull(choice, "Backend choice");
        this.outcome = Objects.requireNonNull(outcome, "Candidate outcome");
        this.correctnessPassed = correctnessPassed;
        this.stable = stable;
        this.statistics = statistics;
        this.benchmarkNanos = Math.max(0L, benchmarkNanos);
        this.reason = reason == null ? "" : reason;
    }

    public BackendChoice choice() { return choice; }
    public KernelCandidateOutcome outcome() { return outcome; }
    public boolean correctnessPassed() { return correctnessPassed; }
    public boolean stable() { return stable; }
    public KernelBenchmarkStatistics statistics() { return statistics; }
    public long benchmarkNanos() { return benchmarkNanos; }
    public String reason() { return reason; }

    public boolean valid() {
        return correctnessPassed && stable && statistics != null
            && outcome == KernelCandidateOutcome.PASS;
    }
}
