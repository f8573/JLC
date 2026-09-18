package net.faulj.autotune.benchmark;

/** Measurement repetition tiers for adaptive benchmarking. */
public enum MeasurementTier {
    BASE, MID, HIGH;

    public static final int BASE_REPS = 7;
    public static final int MID_REPS = 15;
    public static final int HIGH_REPS = 25;
}
