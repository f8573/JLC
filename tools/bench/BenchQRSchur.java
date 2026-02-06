package net.faulj.bench;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.matrix.Matrix;

import java.util.Arrays;
import java.util.Random;

/**
 * Benchmark runner for QR and Schur routines.
 *
 * For each matrix size this runs 10 iterations, prints per-iteration ms and estimated FLOPs,
 * then discards the fastest and slowest and averages the middle 8 runs.
 */
public class BenchQRSchur {

    public static void main(String[] args) {
        int[] sizes = new int[]{2,3,4,8,16,32,64,128,256};
        int iterations = 10;
        long seedBase = 0xC0FFEE;

        System.out.println("Benchmarking QR (Householder) and Schur (ExplicitQRIteration)");
        System.out.println("Each size: 10 iterations; discard min/max and average middle 8");
        System.out.println();

        for (int n : sizes) {
            System.out.println("Size n = " + n);
            double[] timesQR = new double[iterations];
            double[] timesSchur = new double[iterations];

            // FLOP estimates
            double flopsQR = (2.0/3.0) * n * (double)n * (double)n; // classical Householder QR
            // Estimate Schur cost: reduction to Hessenberg (~10/3 n^3) + QR iterations (~2 n^3) => ~16/3 n^3
            double flopsSchur = (16.0/3.0) * n * (double)n * (double)n;

            for (int it = 0; it < iterations; it++) {
                long seed = seedBase + n * 100 + it;
                Matrix A = randomDense(n, seed);

                // QR timing
                long t0 = System.nanoTime();
                QRResult qr = HouseholderQR.decompose(A);
                long t1 = System.nanoTime();
                double msQR = (t1 - t0) / 1e6;
                timesQR[it] = msQR;
                double gflopsQR = (flopsQR / (t1 - t0)) * 1e-9; // flops / ns -> GFLOP/s

                System.out.printf("  QR iter %2d: %8.3f ms, est FLOPs=%.3e, GFLOPS=%.3f\n", it+1, msQR, flopsQR, gflopsQR);

                // Schur timing (use fresh matrix)
                Matrix B = randomDense(n, seed + 0x1234);
                long s0 = System.nanoTime();
                Matrix[] schur = ExplicitQRIteration.decompose(B);
                long s1 = System.nanoTime();
                double msSchur = (s1 - s0) / 1e6;
                timesSchur[it] = msSchur;
                double gflopsSchur = (flopsSchur / (s1 - s0)) * 1e-9;

                System.out.printf("  Schur iter %2d: %8.3f ms, est FLOPs=%.3e, GFLOPS=%.3f\n", it+1, msSchur, flopsSchur, gflopsSchur);
            }

            // Aggregate: discard fastest and slowest, average middle 8
            double avgQR = averageMiddle(timesQR);
            double avgSchur = averageMiddle(timesSchur);

            double avgGflopsQR = (flopsQR / (avgQR * 1e6)) * 1e-9; // flops / ms -> GFLOP/s
            double avgGflopsSchur = (flopsSchur / (avgSchur * 1e6)) * 1e-9;

            System.out.println("  -> QR  : avg (middle 8) = " + String.format("%.3f ms", avgQR)
                    + String.format(", GFLOPS=%.3f", avgGflopsQR)
                    + String.format(", est FLOPs=%.3e", flopsQR));
            System.out.println("  -> Schur: avg (middle 8) = " + String.format("%.3f ms", avgSchur)
                    + String.format(", GFLOPS=%.3f", avgGflopsSchur)
                    + String.format(", est FLOPs=%.3e", flopsSchur));

            System.out.println();
        }
    }

    private static double averageMiddle(double[] times) {
        double[] copy = Arrays.copyOf(times, times.length);
        Arrays.sort(copy);
        // discard index 0 and last
        double sum = 0.0;
        for (int i = 1; i < copy.length - 1; i++) sum += copy[i];
        return sum / (copy.length - 2);
    }

    private static Matrix randomDense(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        return new Matrix(a);
    }
}
