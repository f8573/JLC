package net.faulj.stress;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.result.HessenbergResult;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * Benchmark test for Hessenberg reduction FLOPs performance.
 */
public class HessenbergBenchmark {
    
    @Test
    public void benchmarkFLOPs() {
        int n = 2000;
        System.out.println("=== Hessenberg Reduction FLOP Benchmark ===");
        System.out.println("Matrix size: " + n + "x" + n);
        
        // Generate random matrix
        Random rnd = new Random(42);
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        Matrix A = new Matrix(data);
        
        // Warm-up runs
        System.out.println("\nWarm-up runs...");
        for (int warmup = 0; warmup < 2; warmup++) {
            HessenbergResult result = HessenbergReduction.decompose(A);
        }
        
        // Timed runs
        int numRuns = 3;
        long[] times = new long[numRuns];
        HessenbergResult[] results = new HessenbergResult[numRuns];
        
        System.out.println("\nTimed runs:");
        for (int run = 0; run < numRuns; run++) {
            long start = System.nanoTime();
            results[run] = HessenbergReduction.decompose(A);
            long end = System.nanoTime();
            times[run] = end - start;
            
            double seconds = times[run] / 1e9;
            System.out.printf("  Run %d: %.3f seconds%n", run + 1, seconds);
        }
        
        // Calculate statistics
        long minTime = Long.MAX_VALUE;
        long totalTime = 0;
        for (long t : times) {
            minTime = Math.min(minTime, t);
            totalTime += t;
        }
        double avgTime = totalTime / (double) numRuns;
        
        // FLOPs calculation for Hessenberg reduction:
        // H only: ~(10/3)n^3 flops
        // With Q: ~(14/3)n^3 flops  (approximately)
        // Using more conservative estimate of 10n^3/3 for the reduction itself
        double nCubed = (double) n * n * n;
        double flops = (10.0 / 3.0) * nCubed;  // Hessenberg with Q accumulation
        
        double avgTimeSeconds = avgTime / 1e9;
        double minTimeSeconds = minTime / 1e9;
        double avgGFLOPs = flops / avgTime;  // GFLOP/s
        double peakGFLOPs = flops / minTime; // GFLOP/s (best run)
        
        System.out.println("\n=== Results ===");
        System.out.printf("Matrix size: %d x %d%n", n, n);
        System.out.printf("Average time: %.3f seconds%n", avgTimeSeconds);
        System.out.printf("Best time: %.3f seconds%n", minTimeSeconds);
        System.out.printf("Estimated FLOPs: %.3e%n", flops);
        System.out.printf("Average GFLOP/s: %.2f%n", avgGFLOPs);
        System.out.printf("Peak GFLOP/s: %.2f%n", peakGFLOPs);
        
        // Verify basic correctness: check Hessenberg structure
        Matrix H = results[0].getH();
        double[] hData = H.getRawData();
        double maxSubdiag = 0.0;
        for (int col = 0; col < n - 2; col++) {
            for (int row = col + 2; row < n; row++) {
                maxSubdiag = Math.max(maxSubdiag, Math.abs(hData[row * n + col]));
            }
        }
        System.out.printf("Max below subdiagonal: %.2e%n", maxSubdiag);
        assertTrue("H is not Hessenberg: " + maxSubdiag, maxSubdiag < 1e-12);
        
        // Check orthogonality of Q
        Matrix Q = results[0].getQ();
        double[] qData = Q.getRawData();
        double orthError = computeOrthogonalityError(qData, n);
        System.out.printf("Orthogonality error: %.2e%n", orthError);
        assertTrue("Q is not orthogonal: " + orthError, orthError < 1e-10);
        
        // Check performance threshold (>1 GFLOP/s is reasonable for modern CPUs)
        assertTrue("Performance too low: " + avgGFLOPs + " GFLOP/s (expected > 0.5)", avgGFLOPs > 0.5);
        
        System.out.println("\nBenchmark PASSED!");
    }
    
    private static double computeOrthogonalityError(double[] qData, int n) {
        // Compute Q^T * Q and check it equals identity
        double maxError = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double dot = 0.0;
                for (int k = 0; k < n; k++) {
                    dot += qData[k * n + i] * qData[k * n + j];
                }
                double expected = (i == j) ? 1.0 : 0.0;
                maxError = Math.max(maxError, Math.abs(dot - expected));
            }
        }
        return maxError;
    }
    
    @Test
    public void testCorrectnessSmall() {
        int n = 100;
        System.out.println("\n=== Small Matrix Correctness Test ===");
        System.out.println("Matrix size: " + n + "x" + n);
        
        Random rnd = new Random(12345);
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        Matrix A = new Matrix(data);
        double[] aData = A.getRawData().clone();
        
        HessenbergResult result = HessenbergReduction.decompose(A);
        Matrix H = result.getH();
        Matrix Q = result.getQ();
        
        // Check Hessenberg structure
        double[] hData = H.getRawData();
        double maxSubdiag = 0.0;
        for (int col = 0; col < n - 2; col++) {
            for (int row = col + 2; row < n; row++) {
                maxSubdiag = Math.max(maxSubdiag, Math.abs(hData[row * n + col]));
            }
        }
        System.out.printf("Max below subdiagonal: %.2e%n", maxSubdiag);
        assertTrue("H is not Hessenberg: " + maxSubdiag, maxSubdiag < 1e-12);
        
        // Check orthogonality of Q
        double[] qData = Q.getRawData();
        double orthError = computeOrthogonalityError(qData, n);
        System.out.printf("Orthogonality error: %.2e%n", orthError);
        assertTrue("Q is not orthogonal: " + orthError, orthError < 1e-10);
        
        // Check reconstruction: A = Q * H * Q^T
        double[] reconstructed = matMulQHQt(qData, hData, n);
        double residual = 0.0;
        double normA = 0.0;
        for (int i = 0; i < n * n; i++) {
            double diff = reconstructed[i] - aData[i];
            residual += diff * diff;
            normA += aData[i] * aData[i];
        }
        residual = Math.sqrt(residual) / Math.sqrt(normA);
        System.out.printf("Relative residual: %.2e%n", residual);
        assertTrue("Reconstruction error too large: " + residual, residual < 1e-10);
        
        System.out.println("Correctness test PASSED!");
    }
    
    private static double[] matMulQHQt(double[] Q, double[] H, int n) {
        // Compute Q * H
        double[] QH = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += Q[i * n + k] * H[k * n + j];
                }
                QH[i * n + j] = sum;
            }
        }
        
        // Compute (Q*H) * Q^T
        double[] result = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += QH[i * n + k] * Q[j * n + k];  // Q^T[k,j] = Q[j,k]
                }
                result[i * n + j] = sum;
            }
        }
        
        return result;
    }
}
