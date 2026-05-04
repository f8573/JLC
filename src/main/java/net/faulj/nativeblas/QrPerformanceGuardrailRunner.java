package net.faulj.nativeblas;

import net.faulj.matrix.Matrix;

import java.util.Locale;
import java.util.Random;

public final class QrPerformanceGuardrailRunner {
    private QrPerformanceGuardrailRunner() {
    }

    public static void main(String[] args) {
        int[] sizes = {1024, 2048};
        int warmup = 1;
        int runs = 3;

        for (String arg : args) {
            if (arg.startsWith("--sizes=")) {
                sizes = parseInts(arg.substring("--sizes=".length()), sizes);
            } else if (arg.startsWith("--warmup=")) {
                warmup = Math.max(0, Integer.parseInt(arg.substring("--warmup=".length())));
            } else if (arg.startsWith("--runs=")) {
                runs = Math.max(1, Integer.parseInt(arg.substring("--runs=".length())));
            }
        }

        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        if (!"native".equals(snapshot.activeBackend())) {
            throw new IllegalStateException("Native backend unavailable: " + snapshot.nativeContext().getMessage());
        }

        System.out.println("QR_PERFORMANCE_GUARDRAIL");
        System.out.println("runtime=" + snapshot.nativeContext().getRuntimeDescription());
        System.out.println("provider=" + snapshot.nativeContext().getProviderDescription());
        System.out.println("expected_runtime_contains=AVX2");
        System.out.println("expected_provider=builtin only");
        System.out.println("expected_gemm_tile=MR5_NR4");
        System.out.println("csv,size,bestMs,expectedMinMs,expectedMaxMs,status,gemmMr,gemmNr,qrTrailingGemmMs,qrPanelMs,qrTBuildMs,qrTrailingPackMs,qrTrailingUnpackMs");

        NativeProfiling.setEnabled(true);
        NativeProfiling.setQrEnabled(true);
        for (int size : sizes) {
            Result result = measure(size, warmup, runs);
            Range range = expectedRange(size);
            String status = result.bestMs >= range.minMs && result.bestMs <= range.maxMs ? "OK" : "CHECK";
            System.out.printf(Locale.ROOT,
                "csv,%d,%.3f,%.3f,%.3f,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f%n",
                size, result.bestMs, range.minMs, range.maxMs, status,
                result.gemmProfile.lastMr(), result.gemmProfile.lastNr(),
                result.qrProfile.trailingGemmNanos() / 1e6,
                result.qrProfile.panelNanos() / 1e6,
                result.qrProfile.tBuildNanos() / 1e6,
                result.qrProfile.trailingPackNanos() / 1e6,
                result.qrProfile.trailingUnpackNanos() / 1e6
            );
        }
        NativeProfiling.setEnabled(false);
        NativeProfiling.setQrEnabled(false);
    }

    private static Result measure(int size, int warmup, int runs) {
        Matrix input = randomMatrix(size, 90_000L + size);
        for (int i = 0; i < warmup; i++) {
            executeFactorize(input);
        }

        double bestMs = Double.POSITIVE_INFINITY;
        NativeQrProfile bestQr = NativeQrProfile.EMPTY;
        NativeGemmProfile bestGemm = NativeGemmProfile.EMPTY;
        for (int run = 0; run < runs; run++) {
            NativeProfiling.reset();
            NativeProfiling.resetQr();
            long start = System.nanoTime();
            executeFactorize(input);
            double ms = (System.nanoTime() - start) / 1e6;
            if (ms < bestMs) {
                bestMs = ms;
                bestQr = NativeProfiling.qrSnapshot().orElse(NativeQrProfile.EMPTY);
                bestGemm = NativeProfiling.snapshot().orElse(NativeGemmProfile.EMPTY);
            }
        }
        return new Result(bestMs, bestQr, bestGemm);
    }

    private static void executeFactorize(Matrix input) {
        double[] a = input.getRawData().clone();
        NativeBindings.nativeQrFactorizeOnly(a, input.getRowCount(), input.getColumnCount());
    }

    private static Range expectedRange(int size) {
        if (size <= 1024) {
            return new Range(150.0, 450.0);
        }
        if (size <= 2048) {
            return new Range(1100.0, 2600.0);
        }
        return new Range(0.0, Double.POSITIVE_INFINITY);
    }

    private static Matrix randomMatrix(int size, long seed) {
        Random random = new Random(seed);
        double[] data = new double[size * size];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return Matrix.wrap(data, size, size);
    }

    private static int[] parseInts(String value, int[] fallback) {
        try {
            String[] parts = value.split(",");
            int[] out = new int[parts.length];
            for (int i = 0; i < parts.length; i++) {
                out[i] = Integer.parseInt(parts[i].trim());
            }
            return out;
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    private record Range(double minMs, double maxMs) {
    }

    private record Result(double bestMs, NativeQrProfile qrProfile, NativeGemmProfile gemmProfile) {
    }
}
