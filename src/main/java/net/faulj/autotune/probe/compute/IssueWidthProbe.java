package net.faulj.autotune.probe.compute;

import net.faulj.autotune.probe.ProbeConfidence;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Measures the effective FMA issue width (number of FMA execution
 * ports per core) by comparing throughput with different numbers of
 * independent accumulator chains.
 */
public final class IssueWidthProbe {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /** Threshold separating single-issue from dual-issue throughput ratios. */
    private static final double DUAL_ISSUE_THRESHOLD = 1.35;

    private static final int WARMUP = 5;
    private static final int RUNS = 7;

    private static volatile double sink;

    private IssueWidthProbe() {}

    public static IssueWidthResult run() {
        try {
            return doProbe();
        } catch (Throwable t) {
            // Vector API unavailable or probe failure.
            return new IssueWidthResult(1, false, ProbeConfidence.FAILED);
        }
    }

    private static IssueWidthResult doProbe() {
        int vecLen = SPECIES.length();
        if (vecLen < 1) {
            return new IssueWidthResult(1, false, ProbeConfidence.FAILED);
        }

        // Calibrate iteration count so each trial runs ≥ 50ms.
        int iterations = calibrate();

        // Warmup both chain methods
        for (int w = 0; w < WARMUP; w++) {
            sink += fmaChains4(iterations / 10);
            sink += fmaChains8(iterations / 10);
        }

        // Measure 4-chain throughput
        double best4 = Double.MAX_VALUE;
        for (int r = 0; r < RUNS; r++) {
            long start = System.nanoTime();
            sink += fmaChains4(iterations);
            long elapsed = System.nanoTime() - start;
            best4 = Math.min(best4, elapsed);
        }

        // Measure 8-chain throughput
        double best8 = Double.MAX_VALUE;
        for (int r = 0; r < RUNS; r++) {
            long start = System.nanoTime();
            sink += fmaChains8(iterations);
            long elapsed = System.nanoTime() - start;
            best8 = Math.min(best8, elapsed);
        }

        if (best4 <= 0 || best8 <= 0) {
            return new IssueWidthResult(1, false, ProbeConfidence.FAILED);
        }

        // Throughput in FMA-ops per nanosecond.
        double tp4 = (4.0 * iterations) / best4;
        double tp8 = (8.0 * iterations) / best8;

        // FMA is supported if we get meaningful throughput.
        boolean fmaSupported = tp4 > 0.001;

        // Ratio: if dual-issue, 8-chain throughput is ~2× the 4-chain.
        double ratio = tp8 / tp4;

        int issueWidth = ratio >= DUAL_ISSUE_THRESHOLD ? 2 : 1;

        return new IssueWidthResult(issueWidth, fmaSupported, ProbeConfidence.MEASURED);
    }

    private static int calibrate() {
        int trial = 100_000;
        long start = System.nanoTime();
        sink += fmaChains4(trial);
        long elapsed = System.nanoTime() - start;
        double targetNs = 50_000_000.0; // 50ms
        if (elapsed <= 0) return trial * 10;
        return Math.max(trial, (int) (trial * targetNs / elapsed));
    }

    private static double fmaChains4(int iterations) {
        DoubleVector v0 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);

        for (int i = 0; i < iterations; i++) {
            v0 = v0.lanewise(VectorOperators.FMA, x, y);
            v1 = v1.lanewise(VectorOperators.FMA, x, y);
            v2 = v2.lanewise(VectorOperators.FMA, x, y);
            v3 = v3.lanewise(VectorOperators.FMA, x, y);
        }

        return v0.add(v1).add(v2.add(v3)).reduceLanes(VectorOperators.ADD);
    }

    private static double fmaChains8(int iterations) {
        DoubleVector v0 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v4 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v5 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v6 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v7 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);

        for (int i = 0; i < iterations; i++) {
            v0 = v0.lanewise(VectorOperators.FMA, x, y);
            v1 = v1.lanewise(VectorOperators.FMA, x, y);
            v2 = v2.lanewise(VectorOperators.FMA, x, y);
            v3 = v3.lanewise(VectorOperators.FMA, x, y);
            v4 = v4.lanewise(VectorOperators.FMA, x, y);
            v5 = v5.lanewise(VectorOperators.FMA, x, y);
            v6 = v6.lanewise(VectorOperators.FMA, x, y);
            v7 = v7.lanewise(VectorOperators.FMA, x, y);
        }

        DoubleVector s0 = v0.add(v1).add(v2.add(v3));
        DoubleVector s1 = v4.add(v5).add(v6.add(v7));
        return s0.add(s1).reduceLanes(VectorOperators.ADD);
    }

    public static final class IssueWidthResult {
        public final int effectiveIssueWidth;
        public final boolean fmaSupported;
        public final ProbeConfidence confidence;

        public IssueWidthResult(int effectiveIssueWidth, boolean fmaSupported,
                                ProbeConfidence confidence) {
            this.effectiveIssueWidth = effectiveIssueWidth;
            this.fmaSupported = fmaSupported;
            this.confidence = confidence;
        }
    }
}
