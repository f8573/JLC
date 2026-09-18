package net.faulj.compiler.matrix.codegen;

import java.util.Objects;

/** One candidate offered to the end-to-end backend calibrator. */
public record BackendCalibrationCandidate(BackendChoice choice,
                                          boolean correctnessPassed,
                                          String reason) {
    public BackendCalibrationCandidate {
        Objects.requireNonNull(choice, "Backend choice");
        reason = reason == null ? "" : reason;
    }

    public static BackendCalibrationCandidate valid(BackendChoice choice) {
        return new BackendCalibrationCandidate(choice, true, "correctness-gated");
    }

    public static BackendCalibrationCandidate invalid(BackendChoice choice, String reason) {
        return new BackendCalibrationCandidate(choice, false, reason);
    }
}
