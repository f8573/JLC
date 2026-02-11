package net.faulj.autotune.probe;

/**
 * Confidence classification for each probe measurement.
 */
public enum ProbeConfidence {
    /** Value obtained from direct empirical measurement. */
    MEASURED,
    /** Value estimated from partial measurements or architecture inference. */
    ESTIMATED,
    /** Probe failed; value is a conservative default. */
    FAILED
}
