package net.faulj.compiler.matrix.codegen;

/** Persisted meaning of one backend calibration decision. */
public enum BackendSelectionStatus {
    /** A correctness-gated, stable choice was empirically selected. */
    CALIBRATED,
    /** No candidate met the stability policy; runtime must use fallback policy. */
    NO_STABLE_WINNER
}
