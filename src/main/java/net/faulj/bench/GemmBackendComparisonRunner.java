package net.faulj.bench;

import net.faulj.compute.DispatchPolicy;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.BackendRegistry;
import net.faulj.nativeblas.BackendSnapshot;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Compare the current Java GEMM path with the JNI native backend on the same workload.
 */
public final class GemmBackendComparisonRunner {
    private GemmBackendComparisonRunner() {
    }

    public static void main(String[] args) {
        int size = 512;
        int warmupRuns = 8;
        int measuredRuns = 6;

        for (String arg : args) {
            if (arg == null) {
                continue;
            }
            if (arg.startsWith("--size=")) {
                size = parsePositiveInt(arg.substring("--size=".length()), size);
            } else if (arg.startsWith("--warmup=")) {
                warmupRuns = parsePositiveInt(arg.substring("--warmup=".length()), warmupRuns);
            } else if (arg.startsWith("--runs=")) {
                measuredRuns = parsePositiveInt(arg.substring("--runs=".length()), measuredRuns);
            }
        }

        Matrix a = randomMatrix(size, size, 991L);
        Matrix b = randomMatrix(size, size, 992L);
        DispatchPolicy policy = DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(false)
            .parallelism(1)
            .enableBlas3(true)
            .enableSimd(true)
            .build();

        Map<String, Result> results = new LinkedHashMap<>();
        results.put("java", runBackend("java", a, b, policy, warmupRuns, measuredRuns));
        results.put("native", runBackend("native", a, b, policy, warmupRuns, measuredRuns));

        Result javaResult = results.get("java");
        Result nativeResult = results.get("native");
        double speedup = nativeResult.bestSeconds > 0.0 ? javaResult.bestSeconds / nativeResult.bestSeconds : 0.0;

        System.out.println("GEMM_BACKEND_COMPARISON");
        System.out.println("size=" + size);
        System.out.println("warmupRuns=" + warmupRuns);
        System.out.println("measuredRuns=" + measuredRuns);
        printResult("java", javaResult);
        printResult("native", nativeResult);
        System.out.printf(Locale.ROOT, "native_vs_java_best_speedup=%.6f%n", speedup);
    }

    private static Result runBackend(String backend, Matrix a, Matrix b, DispatchPolicy policy,
                                     int warmupRuns, int measuredRuns) {
        System.setProperty("jlc.backend", backend);
        Matrix c = new Matrix(a.getRowCount(), b.getColumnCount());

        for (int i = 0; i < warmupRuns; i++) {
            Gemm.gemm(a, b, c, 1.0, 0.0, policy);
        }

        double flops = 2.0 * a.getRowCount() * (double) b.getColumnCount() * a.getColumnCount();
        double best = Double.POSITIVE_INFINITY;
        double total = 0.0;
        for (int i = 0; i < measuredRuns; i++) {
            long start = System.nanoTime();
            Gemm.gemm(a, b, c, 1.0, 0.0, policy);
            double seconds = (System.nanoTime() - start) / 1e9;
            best = Math.min(best, seconds);
            total += seconds;
        }

        BackendSnapshot snapshot = BackendRegistry.snapshot();
        return new Result(snapshot, best, total / measuredRuns, flops);
    }

    private static void printResult(String label, Result result) {
        System.out.println(label + "_requested=" + result.snapshot.requestedBackend().id());
        System.out.println(label + "_active=" + result.snapshot.activeBackend());
        System.out.println(label + "_native_status=" + result.snapshot.nativeContext().getStatus());
        System.out.println(label + "_native_provider=" + result.snapshot.nativeContext().getProviderDescription());
        System.out.println(label + "_native_workspace_handle=" + result.snapshot.nativeContext().getWorkspaceHandle().address());
        System.out.printf(Locale.ROOT, "%s_best_ms=%.6f%n", label, result.bestSeconds * 1000.0);
        System.out.printf(Locale.ROOT, "%s_mean_ms=%.6f%n", label, result.meanSeconds * 1000.0);
        System.out.printf(Locale.ROOT, "%s_best_gflops=%.6f%n", label, result.flops / result.bestSeconds / 1e9);
        System.out.printf(Locale.ROOT, "%s_mean_gflops=%.6f%n", label, result.flops / result.meanSeconds / 1e9);
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

    private record Result(BackendSnapshot snapshot, double bestSeconds, double meanSeconds, double flops) {
    }
}
