package net.faulj.bench;

import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.RuntimeProfile;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import jdk.incubator.vector.DoubleVector;

import java.util.Arrays;
import java.util.Random;

public final class Gemm512BenchmarkRunner {
    private Gemm512BenchmarkRunner() {
    }

    public static void main(String[] args) {
        RuntimeProfile.applyConfiguredProfile();

        int n = 512;
        int warmupRuns = 12;
        int measuredRuns = 9;

        for (String arg : args) {
            if (arg == null) {
                continue;
            }
            if (arg.startsWith("--size=")) {
                n = parsePositiveInt(arg.substring("--size=".length()), n);
            } else if (arg.startsWith("--warmup=")) {
                warmupRuns = parsePositiveInt(arg.substring("--warmup=".length()), warmupRuns);
            } else if (arg.startsWith("--runs=")) {
                measuredRuns = parsePositiveInt(arg.substring("--runs=".length()), measuredRuns);
            }
        }

        Matrix a = randomMatrix(n, n, 7101L);
        Matrix b = randomMatrix(n, n, 7102L);
        Matrix c = new Matrix(n, n);
        DispatchPolicy policy = DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(true)
            .parallelism(Math.max(1, Runtime.getRuntime().availableProcessors()))
            .enableBlas3(true)
            .enableSimd(true)
            .build();

        for (int i = 0; i < warmupRuns; i++) {
            Gemm.gemm(a, b, c, 1.0, 0.0, policy);
        }

        double flops = 2.0 * n * (double) n * n;
        double[] seconds = new double[measuredRuns];
        for (int i = 0; i < measuredRuns; i++) {
            long start = System.nanoTime();
            Gemm.gemm(a, b, c, 1.0, 0.0, policy);
            seconds[i] = (System.nanoTime() - start) / 1e9;
        }

        double[] sorted = seconds.clone();
        Arrays.sort(sorted);
        double best = sorted[0];
        double median = sorted[sorted.length / 2];
        double mean = 0.0;
        for (double sec : seconds) {
            mean += sec;
        }
        mean /= seconds.length;

        System.out.println("GEMM_512_BENCHMARK");
        System.out.println("size=" + n);
        System.out.println("vector_lanes=" + DoubleVector.SPECIES_PREFERRED.length());
        System.out.println("parallelism=" + policy.getParallelism());
        System.out.println("warmupRuns=" + warmupRuns);
        System.out.println("measuredRuns=" + measuredRuns);
        System.out.printf("best_ms=%.6f%n", best * 1000.0);
        System.out.printf("median_ms=%.6f%n", median * 1000.0);
        System.out.printf("mean_ms=%.6f%n", mean * 1000.0);
        System.out.printf("best_gflops=%.6f%n", flops / best / 1e9);
        System.out.printf("median_gflops=%.6f%n", flops / median / 1e9);
        System.out.printf("mean_gflops=%.6f%n", flops / mean / 1e9);
        for (int i = 0; i < seconds.length; i++) {
            System.out.printf("run_%d_ms=%.6f run_%d_gflops=%.6f%n",
                i + 1, seconds[i] * 1000.0, i + 1, flops / seconds[i] / 1e9);
        }
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return Matrix.wrap(data, rows, cols);
    }

    private static int parsePositiveInt(String value, int fallback) {
        try {
            int parsed = Integer.parseInt(value.trim());
            return parsed > 0 ? parsed : fallback;
        } catch (Exception ignored) {
            return fallback;
        }
    }
}