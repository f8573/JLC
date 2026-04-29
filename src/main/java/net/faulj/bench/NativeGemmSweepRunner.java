package net.faulj.bench;

import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.RuntimeProfile;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.BackendRegistry;
import net.faulj.nativeblas.BackendSnapshot;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

/**
 * Sweep square GEMM sizes through the JNI native backend.
 */
public final class NativeGemmSweepRunner {
    private static final int[] DEFAULT_SIZES = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048};

    private NativeGemmSweepRunner() {
    }

    public static void main(String[] args) {
        RuntimeProfile.applyConfiguredProfile();

        int[] sizes = DEFAULT_SIZES;
        int warmupRuns = 6;
        int measuredRuns = 6;
        int threads = Math.max(1, Runtime.getRuntime().availableProcessors());

        for (String arg : args) {
            if (arg == null) {
                continue;
            }
            if (arg.startsWith("--sizes=")) {
                sizes = parseSizes(arg.substring("--sizes=".length()), sizes);
            } else if (arg.startsWith("--warmup=")) {
                warmupRuns = parsePositiveInt(arg.substring("--warmup=".length()), warmupRuns);
            } else if (arg.startsWith("--runs=")) {
                measuredRuns = parsePositiveInt(arg.substring("--runs=".length()), measuredRuns);
            } else if (arg.startsWith("--threads=")) {
                threads = parsePositiveInt(arg.substring("--threads=".length()), threads);
            }
        }

        System.setProperty("jlc.backend", "native");
        if (System.getProperty("jlc.native.gemm.provider") == null) {
            System.setProperty("jlc.native.gemm.provider", "auto");
        }

        DispatchPolicy policy = DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(threads > 1)
            .parallelism(threads)
            .enableBlas3(true)
            .enableSimd(true)
            .build();

        System.out.println("JNI_NATIVE_GEMM_SWEEP");
        System.out.println("sizes=" + Arrays.toString(sizes));
        System.out.println("warmupRuns=" + warmupRuns);
        System.out.println("measuredRuns=" + measuredRuns);
        System.out.println("threads=" + threads);
        System.out.println("providerMode=" + System.getProperty("jlc.native.gemm.provider"));
        System.out.println("csv=size,best_ms,mean_ms,best_gflops,mean_gflops,active_backend,native_status,native_provider,workspace_handle");

        for (int size : sizes) {
            Result result = runSize(size, warmupRuns, measuredRuns, policy);
            BackendSnapshot snapshot = result.snapshot();
            System.out.printf(
                Locale.ROOT,
                "size=%d best_ms=%.6f mean_ms=%.6f best_gflops=%.6f mean_gflops=%.6f active=%s provider=%s%n",
                size,
                result.bestSeconds() * 1000.0,
                result.meanSeconds() * 1000.0,
                result.bestGflops(),
                result.meanGflops(),
                snapshot.activeBackend(),
                snapshot.nativeContext().getProviderDescription()
            );
            System.out.printf(
                Locale.ROOT,
                "csv_row=%d,%.6f,%.6f,%.6f,%.6f,%s,%s,%s,%d%n",
                size,
                result.bestSeconds() * 1000.0,
                result.meanSeconds() * 1000.0,
                result.bestGflops(),
                result.meanGflops(),
                snapshot.activeBackend(),
                snapshot.nativeContext().getStatus(),
                csvEscape(snapshot.nativeContext().getProviderDescription()),
                snapshot.nativeContext().getWorkspaceHandle().address()
            );
        }
    }

    private static Result runSize(int size, int warmupRuns, int measuredRuns, DispatchPolicy policy) {
        Matrix a = randomMatrix(size, size, 17_000L + size);
        Matrix b = randomMatrix(size, size, 29_000L + size);
        Matrix c = new Matrix(size, size);

        for (int i = 0; i < warmupRuns; i++) {
            Gemm.gemm(a, b, c, 1.0, 0.0, policy);
        }

        double flops = 2.0 * size * (double) size * size;
        double best = Double.POSITIVE_INFINITY;
        double total = 0.0;
        for (int i = 0; i < measuredRuns; i++) {
            long start = System.nanoTime();
            Gemm.gemm(a, b, c, 1.0, 0.0, policy);
            double seconds = (System.nanoTime() - start) / 1e9;
            best = Math.min(best, seconds);
            total += seconds;
        }
        double mean = total / measuredRuns;
        return new Result(BackendRegistry.snapshot(), best, mean, flops / best / 1e9, flops / mean / 1e9);
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

    private static int[] parseSizes(String value, int[] fallback) {
        try {
            String[] parts = value.split(",");
            int[] parsed = new int[parts.length];
            for (int i = 0; i < parts.length; i++) {
                parsed[i] = parsePositiveInt(parts[i], -1);
                if (parsed[i] <= 0) {
                    return fallback;
                }
            }
            return parsed.length == 0 ? fallback : parsed;
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static String csvEscape(String value) {
        if (value == null) {
            return "";
        }
        if (value.indexOf(',') < 0 && value.indexOf('"') < 0) {
            return value;
        }
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    private record Result(BackendSnapshot snapshot, double bestSeconds, double meanSeconds,
                          double bestGflops, double meanGflops) {
    }
}
