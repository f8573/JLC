package net.faulj.compiler.matrix.codegen;

import java.util.List;
import java.util.Objects;

/** Result of comparing Java, scalar-native, and generated AVX2 choices. */
public final class BackendCalibrationResult {
    private final KernelSignature kernelSignature;
    private final String workloadBucket;
    private final BackendChoice baseline;
    private final List<BackendCandidateEvidence> candidates;
    private final BackendSelectionStatus selectionStatus;
    private final BackendChoice winner;
    private final double winnerSpeedup;
    private final boolean stable;
    private final String decisionReason;

    public BackendCalibrationResult(KernelSignature kernelSignature,
                                    String workloadBucket,
                                    BackendChoice baseline,
                                    List<BackendCandidateEvidence> candidates,
                                    BackendChoice winner,
                                    double winnerSpeedup,
                                    String decisionReason) {
        this(kernelSignature, workloadBucket, baseline, candidates,
            winner == null ? BackendSelectionStatus.NO_STABLE_WINNER
                : BackendSelectionStatus.CALIBRATED,
            winner, winnerSpeedup, winner != null, decisionReason);
    }

    public BackendCalibrationResult(KernelSignature kernelSignature,
                                    String workloadBucket,
                                    BackendChoice baseline,
                                    List<BackendCandidateEvidence> candidates,
                                    BackendSelectionStatus selectionStatus,
                                    BackendChoice winner,
                                    double winnerSpeedup,
                                    boolean stable,
                                    String decisionReason) {
        this.kernelSignature = Objects.requireNonNull(kernelSignature, "Kernel signature");
        this.workloadBucket = workloadBucket == null || workloadBucket.isBlank()
            ? "exact" : workloadBucket.trim();
        this.baseline = Objects.requireNonNull(baseline, "Calibration baseline");
        this.candidates = candidates == null ? List.of() : List.copyOf(candidates);
        this.selectionStatus = Objects.requireNonNull(selectionStatus, "Selection status");
        this.winner = winner;
        this.winnerSpeedup = winnerSpeedup;
        this.stable = stable;
        this.decisionReason = decisionReason == null ? "" : decisionReason;
        validate();
    }

    public KernelSignature kernelSignature() { return kernelSignature; }
    public String workloadBucket() { return workloadBucket; }
    public BackendChoice baseline() { return baseline; }
    public List<BackendCandidateEvidence> candidates() { return candidates; }
    public BackendSelectionStatus selectionStatus() { return selectionStatus; }
    public BackendChoice winner() { return winner; }
    public BackendChoice winnerChoice() { return winner; }
    public double winnerSpeedup() { return winnerSpeedup; }
    public boolean stable() { return stable; }
    public String decisionReason() { return decisionReason; }

    public BackendCandidateEvidence evidence(BackendChoice choice) {
        return candidates.stream().filter(candidate -> candidate.choice().equals(choice))
            .findFirst().orElse(null);
    }

    private void validate() {
        if (selectionStatus == BackendSelectionStatus.NO_STABLE_WINNER) {
            if (winner != null || stable || winnerSpeedup != 0.0) {
                throw new IllegalArgumentException(
                    "No-stable-winner results must not retain an empirical winner");
            }
            return;
        }
        if (winner == null || !stable || !Double.isFinite(winnerSpeedup)
            || winnerSpeedup <= 0.0) {
            throw new IllegalArgumentException(
                "Calibrated results require a stable finite positive winner speedup");
        }
    }
}
