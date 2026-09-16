package net.faulj.nativeblas;

import net.faulj.decomposition.result.SVDResult;
import net.faulj.matrix.Matrix;
import net.faulj.svd.SVDecomposition;

import java.util.Arrays;
import java.util.Random;

/**
 * Repeatable developer benchmark for the Java and native SVD implementations.
 *
 * <p>This is a main-class runner rather than a JUnit test so it is not part of
 * the normal verification suite. Run it with {@code benchmarkNativeSvd}.</p>
 */
public final class NativeSvdBenchmarkRunner {
    private static final int WARMUPS = Integer.getInteger("jlc.benchmark.svd.warmups", 1);
    private static final int MEASURED = Integer.getInteger("jlc.benchmark.svd.measured", 3);
    private static final int[][] SHAPES = {
        {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512},
        {256, 64}, {64, 256}, {512, 128}, {128, 512}
    };

    private NativeSvdBenchmarkRunner() {
    }

    public static void main(String[] args) {
        System.setProperty("jlc.backend", "native");
        System.setProperty("jlc.algorithm.svd.backend", "cpp");
        BackendRegistry.resetForTests();
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        if (!snapshot.nativeContext().isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: "
                + snapshot.nativeContext().getMessage());
        }

        System.out.println("Native SVD benchmark: warmups=" + WARMUPS
            + ", measured=" + MEASURED + ", statistic=median milliseconds");
        System.out.println("shape       Java       Jacobi       Bidiag-QR");
        System.out.println("---------------------------------------------");

        for (int shapeIndex = 0; shapeIndex < SHAPES.length; shapeIndex++) {
            int rows = SHAPES[shapeIndex][0];
            int cols = SHAPES[shapeIndex][1];
            Matrix input = randomMatrix(rows, cols, 0x5eedL + shapeIndex);
            double javaMillis = measure(input, "java", null);
            double jacobiMillis = shouldSkipSlowJacobi(rows, cols)
                ? Double.NaN : measure(input, "native", "jacobi");
            double bidiagMillis = measure(input, "native", "bidiag_qr");
            System.out.printf("%4dx%-4d %9s %11s %12s%n",
                rows, cols, formatMillis(javaMillis), formatMillis(jacobiMillis),
                formatMillis(bidiagMillis));
        }

        System.clearProperty("jlc.algorithm.svd.nativeAlgorithm");
        System.clearProperty("jlc.algorithm.svd.backend");
        System.clearProperty("jlc.backend");
        BackendRegistry.resetForTests();
    }

    private static double measure(Matrix input, String backend, String nativeAlgorithm) {
        System.setProperty("jlc.backend", backend);
        if (nativeAlgorithm == null) {
            System.clearProperty("jlc.algorithm.svd.nativeAlgorithm");
        } else {
            System.setProperty("jlc.algorithm.svd.nativeAlgorithm", nativeAlgorithm);
        }
        BackendRegistry.resetForTests();
        SVDecomposition svd = new SVDecomposition();

        for (int i = 0; i < WARMUPS; i++) {
            consume(svd.decompose(input));
        }
        double[] samples = new double[MEASURED];
        for (int i = 0; i < MEASURED; i++) {
            long start = System.nanoTime();
            consume(svd.decompose(input));
            samples[i] = (System.nanoTime() - start) / 1_000_000.0;
        }
        Arrays.sort(samples);
        return samples[samples.length / 2];
    }

    private static boolean shouldSkipSlowJacobi(int rows, int cols) {
        return !Boolean.getBoolean("jlc.benchmark.svd.includeSlowJacobi")
            && Math.max(rows, cols) >= 512;
    }

    private static String formatMillis(double millis) {
        return Double.isNaN(millis) ? "n/a" : String.format("%.3f", millis);
    }

    private static volatile double sink;

    private static void consume(SVDResult result) {
        sink = result.getSingularValues()[0];
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[][] data = new double[rows][cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                data[row][col] = 2.0 * random.nextDouble() - 1.0;
            }
        }
        return new Matrix(data);
    }
}
