package net.faulj.autotune.probe.simd;

import net.faulj.autotune.probe.ProbeConfidence;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Measures Vector API FMA lowering quality by comparing the
 * instruction throughput of vector FMA against scalar FMA.
 */
public final class SimdQualityProbe {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private static final int WARMUP = 5;
    private static final int RUNS = 7;

    private static volatile double sink;

    private SimdQualityProbe() {}

    public static SimdQualityResult run() {
        try {
            return doProbe();
        } catch (Throwable t) {
            return new SimdQualityResult(0.0, 1, ProbeConfidence.FAILED);
        }
    }

    private static SimdQualityResult doProbe() {
        int vecLen = SPECIES.length();
        if (vecLen <= 1) return new SimdQualityResult(0.0, vecLen, ProbeConfidence.FAILED);

        int iterations = calibrate();

        for (int w = 0; w < WARMUP; w++) {
            sink += scalarFma8(iterations / 10);
            sink += vectorFma8(iterations / 10);
        }

        long bestScalar = Long.MAX_VALUE;
        for (int r = 0; r < RUNS; r++) {
            long start = System.nanoTime();
            sink += scalarFma8(iterations);
            long elapsed = System.nanoTime() - start;
            bestScalar = Math.min(bestScalar, elapsed);
        }

        long bestVector = Long.MAX_VALUE;
        for (int r = 0; r < RUNS; r++) {
            long start = System.nanoTime();
            sink += vectorFma8(iterations);
            long elapsed = System.nanoTime() - start;
            bestVector = Math.min(bestVector, elapsed);
        }

        if (bestScalar <= 0 || bestVector <= 0) return new SimdQualityResult(0.0, vecLen, ProbeConfidence.FAILED);

        double quality = (double) bestScalar / bestVector;
        quality = Math.min(1.5, Math.max(0.0, quality));

        return new SimdQualityResult(quality, vecLen, ProbeConfidence.MEASURED);
    }

    private static int calibrate() {
        int trial = 500_000;
        long start = System.nanoTime();
        sink += scalarFma8(trial);
        long elapsed = System.nanoTime() - start;
        double targetNs = 50_000_000.0;
        if (elapsed <= 0) return trial * 10;
        return Math.max(trial, (int) (trial * targetNs / elapsed));
    }

    private static double scalarFma8(int iterations) {
        double a0 = 1.0, a1 = 1.0, a2 = 1.0, a3 = 1.0;
        double a4 = 1.0, a5 = 1.0, a6 = 1.0, a7 = 1.0;
        double x = 1.0000001, y = 0.9999999;
        for (int i = 0; i < iterations; i++) {
            a0 = Math.fma(a0, x, y);
            a1 = Math.fma(a1, x, y);
            a2 = Math.fma(a2, x, y);
            a3 = Math.fma(a3, x, y);
            a4 = Math.fma(a4, x, y);
            a5 = Math.fma(a5, x, y);
            a6 = Math.fma(a6, x, y);
            a7 = Math.fma(a7, x, y);
        }
        return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }

    private static double vectorFma8(int iterations) {
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

    public static final class SimdQualityResult {
        public final double quality;
        public final int vectorLength;
        public final ProbeConfidence confidence;

        public SimdQualityResult(double quality, int vectorLength,
                                 ProbeConfidence confidence) {
            this.quality = quality;
            this.vectorLength = vectorLength;
            this.confidence = confidence;
        }
    }
}
