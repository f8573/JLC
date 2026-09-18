package net.faulj.autotune.inference;

/**
 * Confidence level for inferred hardware limits.
 * Derived from underlying probe confidence levels.
 */
public enum InferenceConfidence {
    /** All critical probes returned MEASURED data. */
    HIGH,
    /** Some probes returned ESTIMATED or one non-critical probe FAILED. */
    MEDIUM,
    /** Multiple probes FAILED or critical probes ESTIMATED. */
    LOW,
    /** Inference could not produce meaningful bounds. */
    FAILED
}
