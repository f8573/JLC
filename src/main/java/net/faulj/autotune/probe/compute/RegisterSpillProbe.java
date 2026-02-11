package net.faulj.autotune.probe.compute;

import net.faulj.autotune.probe.ProbeConfidence;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Detects the register spill threshold by running synthetic kernels
 * with increasing numbers of live vector accumulators and measuring
 * throughput at each count.
 *
 * <p>When the accumulator count exceeds the physical register file
 * (minus JVM/loop overhead), the JIT spills to L1. Each spill/reload
 * adds two L1 accesses per iteration, visible as a throughput cliff
 * of &gt; 15% per-accumulator throughput drop.</p>
 *
 * <p>Each accumulator count is in its own static method to prevent
 * the JIT from optimizing across count levels. All accumulators are
 * independent (not chained) so the probe is throughput-bound, making
 * spill overhead directly visible.</p>
 *
 * <p>Tested counts: 4, 8, 10, 12, 14, 16, 20, 24, 28.
 * Covers AVX2 (16 YMM → spill ~10–12) and AVX-512 (32 ZMM → spill ~24–28).</p>
 */
public final class RegisterSpillProbe {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private static final int WARMUP = 3;
    private static final int RUNS = 5;
    /** Throughput drop per-accumulator that signals a spill cliff. */
    private static final double SPILL_DROP_THRESHOLD = 0.15;

    private static final int[] CHAIN_COUNTS = {4, 8, 10, 12, 14, 16, 20, 24, 28};

    private static volatile double sink;

    private RegisterSpillProbe() {}

    public static SpillResult run() {
        try {
            return doProbe();
        } catch (Throwable t) {
            return fallback();
        }
    }

    private static SpillResult doProbe() {
        int vecLen = SPECIES.length();
        if (vecLen < 1) return fallback();

        int iterations = calibrate();

        // Measure throughput (FMAs per nanosecond per accumulator) for each count.
        double[] tpPerAcc = new double[CHAIN_COUNTS.length];

        for (int idx = 0; idx < CHAIN_COUNTS.length; idx++) {
            int count = CHAIN_COUNTS[idx];

            // Warmup.
            for (int w = 0; w < WARMUP; w++) {
                sink += dispatch(count, iterations / 10);
            }

            // Measure: best-of-N.
            long bestNs = Long.MAX_VALUE;
            for (int r = 0; r < RUNS; r++) {
                long start = System.nanoTime();
                sink += dispatch(count, iterations);
                long elapsed = System.nanoTime() - start;
                bestNs = Math.min(bestNs, elapsed);
            }

            if (bestNs <= 0) {
                tpPerAcc[idx] = 0;
            } else {
                double totalFmas = (double) count * iterations;
                tpPerAcc[idx] = totalFmas / bestNs / count; // per-accumulator throughput
            }
        }

        // Find spill cliff: first count where per-accumulator throughput drops > 15%
        // relative to the baseline at count=4.
        double baseline = tpPerAcc[0]; // count=4, should be well under spill limit
        if (baseline <= 0) return fallback();

        int threshold = CHAIN_COUNTS[CHAIN_COUNTS.length - 1]; // default: no cliff found
        for (int idx = 1; idx < CHAIN_COUNTS.length; idx++) {
            if (tpPerAcc[idx] <= 0) {
                threshold = CHAIN_COUNTS[idx - 1];
                break;
            }
            double drop = 1.0 - (tpPerAcc[idx] / baseline);
            if (drop > SPILL_DROP_THRESHOLD) {
                threshold = CHAIN_COUNTS[idx - 1];
                break;
            }
        }

        return new SpillResult(threshold, ProbeConfidence.MEASURED);
    }

    private static SpillResult fallback() {
        int vecLen;
        try { vecLen = SPECIES.length(); } catch (Throwable t) { vecLen = 1; }
        // Conservative: assume 16 registers minus overhead.
        int fallbackThreshold = Math.max(4, 16 / Math.max(1, (int) Math.ceil(vecLen / 4.0)));
        return new SpillResult(fallbackThreshold, ProbeConfidence.FAILED);
    }

    private static int calibrate() {
        int trial = 500_000;
        long start = System.nanoTime();
        sink += fma4(trial);
        long elapsed = System.nanoTime() - start;
        double targetNs = 30_000_000.0; // 30ms per measurement
        if (elapsed <= 0) return trial * 10;
        return Math.max(trial, (int) (trial * targetNs / elapsed));
    }

    /** Dispatch to the correct chain-count method. */
    private static double dispatch(int count, int iterations) {
        return switch (count) {
            case 4  -> fma4(iterations);
            case 8  -> fma8(iterations);
            case 10 -> fma10(iterations);
            case 12 -> fma12(iterations);
            case 14 -> fma14(iterations);
            case 16 -> fma16(iterations);
            case 20 -> fma20(iterations);
            case 24 -> fma24(iterations);
            case 28 -> fma28(iterations);
            default -> fma4(iterations);
        };
    }

    // ── Explicit accumulator methods ────────────────────────────────
    // Each method uses N independent accumulators doing FMA.
    // Separate methods ensure independent JIT compilation.

    private static double fma4(int n) {
        DoubleVector v0 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);
        for (int i = 0; i < n; i++) {
            v0 = v0.lanewise(VectorOperators.FMA, x, y);
            v1 = v1.lanewise(VectorOperators.FMA, x, y);
            v2 = v2.lanewise(VectorOperators.FMA, x, y);
            v3 = v3.lanewise(VectorOperators.FMA, x, y);
        }
        return v0.add(v1).add(v2.add(v3)).reduceLanes(VectorOperators.ADD);
    }

    private static double fma8(int n) {
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
        for (int i = 0; i < n; i++) {
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

    private static double fma10(int n) {
        DoubleVector v0 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v4 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v5 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v6 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v7 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v8 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v9 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);
        for (int i = 0; i < n; i++) {
            v0 = v0.lanewise(VectorOperators.FMA, x, y);
            v1 = v1.lanewise(VectorOperators.FMA, x, y);
            v2 = v2.lanewise(VectorOperators.FMA, x, y);
            v3 = v3.lanewise(VectorOperators.FMA, x, y);
            v4 = v4.lanewise(VectorOperators.FMA, x, y);
            v5 = v5.lanewise(VectorOperators.FMA, x, y);
            v6 = v6.lanewise(VectorOperators.FMA, x, y);
            v7 = v7.lanewise(VectorOperators.FMA, x, y);
            v8 = v8.lanewise(VectorOperators.FMA, x, y);
            v9 = v9.lanewise(VectorOperators.FMA, x, y);
        }
        DoubleVector s0 = v0.add(v1).add(v2.add(v3));
        DoubleVector s1 = v4.add(v5).add(v6.add(v7));
        DoubleVector s2 = v8.add(v9);
        return s0.add(s1).add(s2).reduceLanes(VectorOperators.ADD);
    }

    private static double fma12(int n) {
        DoubleVector v0  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v4  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v5  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v6  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v7  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v8  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v9  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v10 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v11 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);
        for (int i = 0; i < n; i++) {
            v0  = v0.lanewise(VectorOperators.FMA, x, y);
            v1  = v1.lanewise(VectorOperators.FMA, x, y);
            v2  = v2.lanewise(VectorOperators.FMA, x, y);
            v3  = v3.lanewise(VectorOperators.FMA, x, y);
            v4  = v4.lanewise(VectorOperators.FMA, x, y);
            v5  = v5.lanewise(VectorOperators.FMA, x, y);
            v6  = v6.lanewise(VectorOperators.FMA, x, y);
            v7  = v7.lanewise(VectorOperators.FMA, x, y);
            v8  = v8.lanewise(VectorOperators.FMA, x, y);
            v9  = v9.lanewise(VectorOperators.FMA, x, y);
            v10 = v10.lanewise(VectorOperators.FMA, x, y);
            v11 = v11.lanewise(VectorOperators.FMA, x, y);
        }
        DoubleVector s0 = v0.add(v1).add(v2.add(v3));
        DoubleVector s1 = v4.add(v5).add(v6.add(v7));
        DoubleVector s2 = v8.add(v9).add(v10.add(v11));
        return s0.add(s1).add(s2).reduceLanes(VectorOperators.ADD);
    }

    private static double fma14(int n) {
        DoubleVector v0  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v4  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v5  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v6  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v7  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v8  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v9  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v10 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v11 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v12 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v13 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);
        for (int i = 0; i < n; i++) {
            v0  = v0.lanewise(VectorOperators.FMA, x, y);
            v1  = v1.lanewise(VectorOperators.FMA, x, y);
            v2  = v2.lanewise(VectorOperators.FMA, x, y);
            v3  = v3.lanewise(VectorOperators.FMA, x, y);
            v4  = v4.lanewise(VectorOperators.FMA, x, y);
            v5  = v5.lanewise(VectorOperators.FMA, x, y);
            v6  = v6.lanewise(VectorOperators.FMA, x, y);
            v7  = v7.lanewise(VectorOperators.FMA, x, y);
            v8  = v8.lanewise(VectorOperators.FMA, x, y);
            v9  = v9.lanewise(VectorOperators.FMA, x, y);
            v10 = v10.lanewise(VectorOperators.FMA, x, y);
            v11 = v11.lanewise(VectorOperators.FMA, x, y);
            v12 = v12.lanewise(VectorOperators.FMA, x, y);
            v13 = v13.lanewise(VectorOperators.FMA, x, y);
        }
        DoubleVector s0 = v0.add(v1).add(v2.add(v3));
        DoubleVector s1 = v4.add(v5).add(v6.add(v7));
        DoubleVector s2 = v8.add(v9).add(v10.add(v11));
        DoubleVector s3 = v12.add(v13);
        return s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD);
    }

    private static double fma16(int n) {
        DoubleVector v0  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v4  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v5  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v6  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v7  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v8  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v9  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v10 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v11 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v12 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v13 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v14 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v15 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);
        for (int i = 0; i < n; i++) {
            v0  = v0.lanewise(VectorOperators.FMA, x, y);
            v1  = v1.lanewise(VectorOperators.FMA, x, y);
            v2  = v2.lanewise(VectorOperators.FMA, x, y);
            v3  = v3.lanewise(VectorOperators.FMA, x, y);
            v4  = v4.lanewise(VectorOperators.FMA, x, y);
            v5  = v5.lanewise(VectorOperators.FMA, x, y);
            v6  = v6.lanewise(VectorOperators.FMA, x, y);
            v7  = v7.lanewise(VectorOperators.FMA, x, y);
            v8  = v8.lanewise(VectorOperators.FMA, x, y);
            v9  = v9.lanewise(VectorOperators.FMA, x, y);
            v10 = v10.lanewise(VectorOperators.FMA, x, y);
            v11 = v11.lanewise(VectorOperators.FMA, x, y);
            v12 = v12.lanewise(VectorOperators.FMA, x, y);
            v13 = v13.lanewise(VectorOperators.FMA, x, y);
            v14 = v14.lanewise(VectorOperators.FMA, x, y);
            v15 = v15.lanewise(VectorOperators.FMA, x, y);
        }
        DoubleVector s0 = v0.add(v1).add(v2.add(v3));
        DoubleVector s1 = v4.add(v5).add(v6.add(v7));
        DoubleVector s2 = v8.add(v9).add(v10.add(v11));
        DoubleVector s3 = v12.add(v13).add(v14.add(v15));
        return s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD);
    }

    private static double fma20(int n) {
        DoubleVector v0  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v4  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v5  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v6  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v7  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v8  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v9  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v10 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v11 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v12 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v13 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v14 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v15 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v16 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v17 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v18 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v19 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);
        for (int i = 0; i < n; i++) {
            v0  = v0.lanewise(VectorOperators.FMA, x, y);
            v1  = v1.lanewise(VectorOperators.FMA, x, y);
            v2  = v2.lanewise(VectorOperators.FMA, x, y);
            v3  = v3.lanewise(VectorOperators.FMA, x, y);
            v4  = v4.lanewise(VectorOperators.FMA, x, y);
            v5  = v5.lanewise(VectorOperators.FMA, x, y);
            v6  = v6.lanewise(VectorOperators.FMA, x, y);
            v7  = v7.lanewise(VectorOperators.FMA, x, y);
            v8  = v8.lanewise(VectorOperators.FMA, x, y);
            v9  = v9.lanewise(VectorOperators.FMA, x, y);
            v10 = v10.lanewise(VectorOperators.FMA, x, y);
            v11 = v11.lanewise(VectorOperators.FMA, x, y);
            v12 = v12.lanewise(VectorOperators.FMA, x, y);
            v13 = v13.lanewise(VectorOperators.FMA, x, y);
            v14 = v14.lanewise(VectorOperators.FMA, x, y);
            v15 = v15.lanewise(VectorOperators.FMA, x, y);
            v16 = v16.lanewise(VectorOperators.FMA, x, y);
            v17 = v17.lanewise(VectorOperators.FMA, x, y);
            v18 = v18.lanewise(VectorOperators.FMA, x, y);
            v19 = v19.lanewise(VectorOperators.FMA, x, y);
        }
        DoubleVector s0 = v0.add(v1).add(v2.add(v3));
        DoubleVector s1 = v4.add(v5).add(v6.add(v7));
        DoubleVector s2 = v8.add(v9).add(v10.add(v11));
        DoubleVector s3 = v12.add(v13).add(v14.add(v15));
        DoubleVector s4 = v16.add(v17).add(v18.add(v19));
        return s0.add(s1).add(s2).add(s3).add(s4).reduceLanes(VectorOperators.ADD);
    }

    private static double fma24(int n) {
        DoubleVector v0  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v4  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v5  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v6  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v7  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v8  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v9  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v10 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v11 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v12 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v13 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v14 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v15 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v16 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v17 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v18 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v19 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v20 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v21 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v22 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v23 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);
        for (int i = 0; i < n; i++) {
            v0  = v0.lanewise(VectorOperators.FMA, x, y);
            v1  = v1.lanewise(VectorOperators.FMA, x, y);
            v2  = v2.lanewise(VectorOperators.FMA, x, y);
            v3  = v3.lanewise(VectorOperators.FMA, x, y);
            v4  = v4.lanewise(VectorOperators.FMA, x, y);
            v5  = v5.lanewise(VectorOperators.FMA, x, y);
            v6  = v6.lanewise(VectorOperators.FMA, x, y);
            v7  = v7.lanewise(VectorOperators.FMA, x, y);
            v8  = v8.lanewise(VectorOperators.FMA, x, y);
            v9  = v9.lanewise(VectorOperators.FMA, x, y);
            v10 = v10.lanewise(VectorOperators.FMA, x, y);
            v11 = v11.lanewise(VectorOperators.FMA, x, y);
            v12 = v12.lanewise(VectorOperators.FMA, x, y);
            v13 = v13.lanewise(VectorOperators.FMA, x, y);
            v14 = v14.lanewise(VectorOperators.FMA, x, y);
            v15 = v15.lanewise(VectorOperators.FMA, x, y);
            v16 = v16.lanewise(VectorOperators.FMA, x, y);
            v17 = v17.lanewise(VectorOperators.FMA, x, y);
            v18 = v18.lanewise(VectorOperators.FMA, x, y);
            v19 = v19.lanewise(VectorOperators.FMA, x, y);
            v20 = v20.lanewise(VectorOperators.FMA, x, y);
            v21 = v21.lanewise(VectorOperators.FMA, x, y);
            v22 = v22.lanewise(VectorOperators.FMA, x, y);
            v23 = v23.lanewise(VectorOperators.FMA, x, y);
        }
        DoubleVector s0 = v0.add(v1).add(v2.add(v3));
        DoubleVector s1 = v4.add(v5).add(v6.add(v7));
        DoubleVector s2 = v8.add(v9).add(v10.add(v11));
        DoubleVector s3 = v12.add(v13).add(v14.add(v15));
        DoubleVector s4 = v16.add(v17).add(v18.add(v19));
        DoubleVector s5 = v20.add(v21).add(v22.add(v23));
        return s0.add(s1).add(s2).add(s3).add(s4).add(s5)
                .reduceLanes(VectorOperators.ADD);
    }

    private static double fma28(int n) {
        DoubleVector v0  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v1  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v2  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v3  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v4  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v5  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v6  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v7  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v8  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v9  = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v10 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v11 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v12 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v13 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v14 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v15 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v16 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v17 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v18 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v19 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v20 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v21 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v22 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v23 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v24 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v25 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v26 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector v27 = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);
        for (int i = 0; i < n; i++) {
            v0  = v0.lanewise(VectorOperators.FMA, x, y);
            v1  = v1.lanewise(VectorOperators.FMA, x, y);
            v2  = v2.lanewise(VectorOperators.FMA, x, y);
            v3  = v3.lanewise(VectorOperators.FMA, x, y);
            v4  = v4.lanewise(VectorOperators.FMA, x, y);
            v5  = v5.lanewise(VectorOperators.FMA, x, y);
            v6  = v6.lanewise(VectorOperators.FMA, x, y);
            v7  = v7.lanewise(VectorOperators.FMA, x, y);
            v8  = v8.lanewise(VectorOperators.FMA, x, y);
            v9  = v9.lanewise(VectorOperators.FMA, x, y);
            v10 = v10.lanewise(VectorOperators.FMA, x, y);
            v11 = v11.lanewise(VectorOperators.FMA, x, y);
            v12 = v12.lanewise(VectorOperators.FMA, x, y);
            v13 = v13.lanewise(VectorOperators.FMA, x, y);
            v14 = v14.lanewise(VectorOperators.FMA, x, y);
            v15 = v15.lanewise(VectorOperators.FMA, x, y);
            v16 = v16.lanewise(VectorOperators.FMA, x, y);
            v17 = v17.lanewise(VectorOperators.FMA, x, y);
            v18 = v18.lanewise(VectorOperators.FMA, x, y);
            v19 = v19.lanewise(VectorOperators.FMA, x, y);
            v20 = v20.lanewise(VectorOperators.FMA, x, y);
            v21 = v21.lanewise(VectorOperators.FMA, x, y);
            v22 = v22.lanewise(VectorOperators.FMA, x, y);
            v23 = v23.lanewise(VectorOperators.FMA, x, y);
            v24 = v24.lanewise(VectorOperators.FMA, x, y);
            v25 = v25.lanewise(VectorOperators.FMA, x, y);
            v26 = v26.lanewise(VectorOperators.FMA, x, y);
            v27 = v27.lanewise(VectorOperators.FMA, x, y);
        }
        DoubleVector s0 = v0.add(v1).add(v2.add(v3));
        DoubleVector s1 = v4.add(v5).add(v6.add(v7));
        DoubleVector s2 = v8.add(v9).add(v10.add(v11));
        DoubleVector s3 = v12.add(v13).add(v14.add(v15));
        DoubleVector s4 = v16.add(v17).add(v18.add(v19));
        DoubleVector s5 = v20.add(v21).add(v22.add(v23));
        DoubleVector s6 = v24.add(v25).add(v26.add(v27));
        return s0.add(s1).add(s2).add(s3).add(s4).add(s5).add(s6)
                .reduceLanes(VectorOperators.ADD);
    }

    // ── Result ──────────────────────────────────────────────────────

    public static final class SpillResult {
        /** Last accumulator count before throughput cliff. */
        public final int spillThreshold;
        public final ProbeConfidence confidence;

        public SpillResult(int spillThreshold, ProbeConfidence confidence) {
            this.spillThreshold = spillThreshold;
            this.confidence = confidence;
        }
    }
}
