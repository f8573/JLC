package net.faulj.bench;

import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.matrix.Matrix;
import org.junit.Test;

import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Benchmarks to discover the maximum achieved FLOPs for Householder QR and
 * Hessenberg reduction by increasing matrix sizes until GFLOPs stabilize.
 *
 * This test prints results to stdout; it intentionally does not assert a
 * numeric limit — it's a measurement utility.
 */
public class FlopScalingTest {
    private static final double QR_COEFF_SQUARE = 4.0 / 3.0; // ~1.333 n^3 for square QR
    private static final double HESSENBERG_COEFF = 10.0 / 3.0; // ~3.333 n^3 for reduction

    @Test
    public void findPeakFlops() {
        System.out.println("Starting FLOPs scaling benchmark...");

        // sizes to try (will double until condition met or max reached)
        int envMax = 1024;
        String env = System.getenv("BENCH_MAX_SIZE");
        if (env != null) {
            try { envMax = Integer.parseInt(env); } catch (NumberFormatException ignored) { }
        }

        // Determine a safe maximum based on available heap to avoid OOME.
        long maxHeapBytes = Runtime.getRuntime().maxMemory();
        // Estimate bytes per double matrix: 8 bytes per entry. Use safety factor
        // to account for additional temporaries and copies inside algorithms.
        double safetyFactor = 6.0; // conservative multiplier
        int maxNByMemory = (int) Math.floor(Math.sqrt(maxHeapBytes / (8.0 * safetyFactor)));

        if (maxNByMemory < 128) {
            System.out.printf("Warning: available heap (%.1f MB) is too small for benchmarks; reducing sizes.\n", maxHeapBytes / 1024.0 / 1024.0);
        }

        int effectiveMax = Math.min(envMax, Math.max(128, maxNByMemory));
        System.out.printf("Heap limit: %.1f MB, computed max N: %d, using max size: %d\n", maxHeapBytes / 1024.0 / 1024.0, maxNByMemory, effectiveMax);

        List<Integer> sizeList = new ArrayList<>();
        for (int s = 128; s <= effectiveMax; s *= 2) sizeList.add(s);
        int[] sizes = sizeList.stream().mapToInt(Integer::intValue).toArray();

        System.out.println("=== Householder QR ===");
        double maxQr = runBenchmarkQR(sizes);

        System.out.println("=== Hessenberg Reduction ===");
        double maxHess = runBenchmarkHessenberg(sizes);

        // write results
        String out = String.format("HouseholderQR_max_GFLOPs=%.6f\nHessenberg_max_GFLOPs=%.6f\n", maxQr, maxHess);
        System.out.println("Writing results to Algorithm_FLOP_results");
        try (FileWriter fw = new FileWriter("Algorithm_FLOP_results")) {
            fw.write(out);
        } catch (IOException e) {
            System.err.println("Failed to write results: " + e.getMessage());
        }

        assert true;
    }

    private double runBenchmarkQR(int[] sizes) {
        double prevGflops = -1.0;
        int stableCount = 0;
        final double relThreshold = 0.01; // 1% relative change
        double maxG = 0.0;

        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 42);

            // warmup
            for (int i = 0; i < 2; i++) {
                HouseholderQR.decompose(A);
            }

            int runs = (n >= 1024) ? 1 : 3;
            long t0 = System.nanoTime();
            for (int i = 0; i < runs; i++) {
                HouseholderQR.decompose(A);
            }
            long t1 = System.nanoTime();

            double elapsedSec = (t1 - t0) / 1e9 / runs;
            double flops = QR_COEFF_SQUARE * n * (double) n * (double) n; // square-case estimate
            double gflops = flops / elapsedSec / 1e9;

            System.out.printf("n=%d: time=%.6fs, GFLOPs=%.3f\n", n, elapsedSec, gflops);
            if (gflops > maxG) maxG = gflops;

            if (prevGflops > 0) {
                double rel = Math.abs(gflops - prevGflops) / prevGflops;
                if (rel < relThreshold) {
                    stableCount++;
                } else {
                    stableCount = 0;
                }
            }
            prevGflops = gflops;

            if (stableCount >= 2) {
                System.out.println("GFLOPs stabilized; stopping QR scaling.");
                break;
            }
        }
        return maxG;
    }

    private double runBenchmarkHessenberg(int[] sizes) {
        double prevGflops = -1.0;
        int stableCount = 0;
        final double relThreshold = 0.01; // 1% relative change
        double maxG = 0.0;

        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 123);

            // warmup
            for (int i = 0; i < 1; i++) {
                HessenbergReduction.decompose(A);
            }

            int runs = (n >= 1024) ? 1 : 2; // heavier op: fewer runs
            long t0 = System.nanoTime();
            for (int i = 0; i < runs; i++) {
                HessenbergReduction.decompose(A);
            }
            long t1 = System.nanoTime();

            double elapsedSec = (t1 - t0) / 1e9 / runs;
            double flops = HESSENBERG_COEFF * n * (double) n * (double) n;
            double gflops = flops / elapsedSec / 1e9;

            System.out.printf("n=%d: time=%.6fs, GFLOPs=%.3f\n", n, elapsedSec, gflops);
            if (gflops > maxG) maxG = gflops;

            if (prevGflops > 0) {
                double rel = Math.abs(gflops - prevGflops) / prevGflops;
                if (rel < relThreshold) {
                    stableCount++;
                } else {
                    stableCount = 0;
                }
            }
            prevGflops = gflops;

            if (stableCount >= 2) {
                System.out.println("GFLOPs stabilized; stopping Hessenberg scaling.");
                break;
            }
        }
        return maxG;
    }

    private Matrix randomMatrix(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        double[] a = new double[rows * cols];
        for (int i = 0; i < a.length; i++) a[i] = rnd.nextDouble() - 0.5;
        return Matrix.wrap(a, rows, cols);
    }
}
