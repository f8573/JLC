package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;
import net.faulj.matrix.Matrix;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.DoubleBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Low-perturbation 512 comparator for JNI-vs-standalone policy selection.
 *
 * No profiling is enabled in the timed region. JNI array/direct paths reuse
 * buffers across warmup and measurement runs, and standalone candidates are
 * launched from the same orchestrating JVM with identical seeds and overrides.
 */
public final class NativeGemmLowPerturbationComparatorRunner {
    private static final int LOOP_DEFAULT = 0;
    private static final int LOOP_JC_PC_IC = 1;

    private NativeGemmLowPerturbationComparatorRunner() {
    }

    public static void main(String[] args) throws Exception {
        int rows = 512;
        int inner = 512;
        int cols = 512;
        int warmup = 5;
        int runs = 15;
        int threads = 1;
        long seed = 17_512L;

        for (String arg : args) {
            if (arg.startsWith("--rows=")) rows = Integer.parseInt(arg.substring("--rows=".length()));
            else if (arg.startsWith("--inner=")) inner = Integer.parseInt(arg.substring("--inner=".length()));
            else if (arg.startsWith("--cols=")) cols = Integer.parseInt(arg.substring("--cols=".length()));
            else if (arg.startsWith("--warmup=")) warmup = Integer.parseInt(arg.substring("--warmup=".length()));
            else if (arg.startsWith("--runs=")) runs = Integer.parseInt(arg.substring("--runs=".length()));
            else if (arg.startsWith("--threads=")) threads = Integer.parseInt(arg.substring("--threads=".length()));
            else if (arg.startsWith("--seed=")) seed = Long.parseLong(arg.substring("--seed=".length()));
        }

        double[] aSeed = randomData(rows, inner, seed);
        double[] bSeed = randomData(inner, cols, seed + 1);
        double flops = 2.0 * rows * cols * inner;

        DispatchPolicy javaPolicy = DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(threads > 1)
            .parallelism(threads)
            .enableBlas3(true)
            .enableSimd(true)
            .build();

        Timing javaTiming = measureJava(aSeed, bSeed, rows, inner, cols, warmup, runs, javaPolicy);

        List<Variant> variants = List.of(
            new Variant("current", 0, 0, 0, false, LOOP_DEFAULT),
            new Variant("candidate_nc48", 128, 512, 48, true, LOOP_JC_PC_IC),
            new Variant("candidate_nc32", 128, 512, 32, true, LOOP_JC_PC_IC)
        );

        List<Row> rowsOut = new ArrayList<>();
        for (Variant variant : variants) {
            Timing jniArray = measureJniArray(aSeed, bSeed, rows, inner, cols, warmup, runs, threads, variant);
            Timing jniDirect = measureJniDirect(aSeed, bSeed, rows, inner, cols, warmup, runs, threads, variant);
            Timing standalone = measureStandalone(rows, inner, cols, warmup, runs, threads, seed, variant);
            rowsOut.add(new Row(variant, jniArray, jniDirect, standalone));
        }

        System.out.println("NATIVE_GEMM_LOW_PERTURBATION_COMPARATOR");
        System.out.printf(Locale.ROOT, "shape=%dx%d * %dx%d warmup=%d runs=%d threads=%d%n",
            rows, inner, inner, cols, warmup, runs, threads);
        System.out.println();
        System.out.printf(Locale.ROOT,
            "Java baseline: median=%.3f ms p25=%.3f p75=%.3f min=%.3f max=%.3f mean=%.3f gflops(median)=%.3f%n",
            javaTiming.medianMs(), javaTiming.p25Ms(), javaTiming.p75Ms(), javaTiming.minMs(),
            javaTiming.maxMs(), javaTiming.meanMs(), gflops(flops, javaTiming.medianMs()));
        System.out.println();
        System.out.println("| Variant | Path | Median ms | P25 ms | P75 ms | Min ms | Max ms | Mean ms | GFLOPs(median) | Ratio vs Java |");
        System.out.println("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|");
        for (Row row : rowsOut) {
            printRow("JNI array", row.jniArray(), javaTiming, flops, row.variant().label());
            printRow("JNI direct", row.jniDirect(), javaTiming, flops, row.variant().label());
            printRow("Standalone", row.standalone(), javaTiming, flops, row.variant().label());
        }
    }

    private static void printRow(String path, Timing timing, Timing javaTiming, double flops, String label) {
        System.out.printf(Locale.ROOT,
            "| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |%n",
            label,
            path,
            timing.medianMs(),
            timing.p25Ms(),
            timing.p75Ms(),
            timing.minMs(),
            timing.maxMs(),
            timing.meanMs(),
            gflops(flops, timing.medianMs()),
            timing.medianMs() / javaTiming.medianMs()
        );
    }

    private static Timing measureJava(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                      int warmup, int runs, DispatchPolicy policy) {
        JavaBackend backend = new JavaBackend();
        double[] a = aSeed;
        double[] b = bSeed;
        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < warmup + runs; i++) {
            double[] c = new double[rows * cols];
            long start = System.nanoTime();
            backend.gemm(Matrix.wrap(a, rows, inner), Matrix.wrap(b, inner, cols), Matrix.wrap(c, rows, cols), 1.0, 0.0, policy);
            if (i >= warmup) {
                samples.add((System.nanoTime() - start) / 1e6);
            }
        }
        return summarize(samples);
    }

    private static Timing measureJniArray(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                          int warmup, int runs, int threads, Variant variant) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }

        double[] a = aSeed.clone();
        double[] b = bSeed.clone();
        double[] c = new double[rows * cols];

        applyVariant(variant);
        try {
            return measureLoop(warmup, runs, () -> {
                java.util.Arrays.fill(c, 0.0);
                NativeBindings.nativeGemm(a, rows, inner, b, inner, cols, c, rows, cols, 1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN);
            });
        } finally {
            clearVariant();
        }
    }

    private static Timing measureJniDirect(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                           int warmup, int runs, int threads, Variant variant) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }

        try (NativeMatrix a = NativeMatrix.allocate(rows, inner);
             NativeMatrix b = NativeMatrix.allocate(inner, cols);
             NativeMatrix c = NativeMatrix.allocate(rows, cols)) {
            fillNativeMatrix(a, aSeed);
            fillNativeMatrix(b, bSeed);

            applyVariant(variant);
            try {
                return measureLoop(warmup, runs, () -> {
                    zeroNativeMatrix(c);
                    NativeBindings.nativeGemmDirect(
                        a.buffer(), 0L, a.ld(), rows, inner, 0,
                        b.buffer(), 0L, b.ld(), inner, cols, 0,
                        c.buffer(), 0L, c.ld(), rows, cols, 0,
                        1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN
                    );
                });
            } finally {
                clearVariant();
            }
        }
    }

    private static Timing measureStandalone(int rows, int inner, int cols, int warmup, int runs, int threads,
                                            long seed, Variant variant) throws Exception {
        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }

        List<String> command = List.of(
            executable,
            "--algorithm=gemm",
            "--rows=" + rows,
            "--cols=" + cols,
            "--inner=" + inner,
            "--warmup=" + warmup,
            "--runs=" + runs,
            "--threads=" + threads,
            "--seed=" + seed
        );

        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        Map<String, String> env = builder.environment();
        if (variant.disableSquareTuning()) {
            env.put("JLC_NATIVE_DISABLE_SQUARE_TUNING", "1");
            env.put("JLC_NATIVE_MC", Integer.toString(variant.mc()));
            env.put("JLC_NATIVE_KC", Integer.toString(variant.kc()));
            env.put("JLC_NATIVE_NC", Integer.toString(variant.nc()));
            env.put("JLC_NATIVE_LOOP_ORDER", variant.loopOrder() == LOOP_JC_PC_IC ? "jc_pc_ic" : "ic_pc_jc");
        } else {
            env.remove("JLC_NATIVE_DISABLE_SQUARE_TUNING");
            env.remove("JLC_NATIVE_MC");
            env.remove("JLC_NATIVE_KC");
            env.remove("JLC_NATIVE_NC");
            env.remove("JLC_NATIVE_LOOP_ORDER");
        }

        Process process = builder.start();
        Map<String, String> values = new LinkedHashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                int idx = line.indexOf('=');
                if (idx > 0) {
                    values.put(line.substring(0, idx), line.substring(idx + 1));
                }
            }
        }
        int exit = process.waitFor();
        if (exit != 0) {
            throw new IllegalStateException("Native standalone GEMM exited with code " + exit);
        }
        return new Timing(
            Double.parseDouble(values.getOrDefault("best_ms", "NaN")),
            Double.parseDouble(values.getOrDefault("max_ms", "NaN")),
            Double.parseDouble(values.getOrDefault("mean_ms", "NaN")),
            Double.parseDouble(values.getOrDefault("median_ms", "NaN")),
            percentileFallback(values, "p25_ms"),
            percentileFallback(values, "p75_ms")
        );
    }

    private static double percentileFallback(Map<String, String> values, String key) {
        String direct = values.get(key);
        if (direct != null) {
            return Double.parseDouble(direct);
        }
        return Double.parseDouble(values.getOrDefault("median_ms", "NaN"));
    }

    private static Timing measureLoop(int warmup, int runs, ThrowingRunnable action) {
        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < warmup + runs; i++) {
            long start = System.nanoTime();
            try {
                action.run();
            } catch (Exception e) {
                throw new IllegalStateException("Low-perturbation run failed", e);
            }
            if (i >= warmup) {
                samples.add((System.nanoTime() - start) / 1e6);
            }
        }
        return summarize(samples);
    }

    private static void applyVariant(Variant variant) {
        if (variant.disableSquareTuning()) {
            NativeBindings.nativeGemmSetRuntimeOverrides(
                variant.mc(), variant.kc(), variant.nc(),
                0, 0,
                true,
                variant.loopOrder()
            );
        } else {
            NativeBindings.nativeGemmClearRuntimeOverrides();
        }
    }

    private static void clearVariant() {
        NativeBindings.nativeGemmClearRuntimeOverrides();
    }

    private static void fillNativeMatrix(NativeMatrix matrix, double[] values) {
        DoubleBuffer buffer = matrix.asDoubleBuffer();
        buffer.position(0);
        buffer.put(values);
        buffer.position(0);
    }

    private static void zeroNativeMatrix(NativeMatrix matrix) {
        DoubleBuffer buffer = matrix.asDoubleBuffer();
        buffer.position(0);
        for (int i = 0; i < matrix.rows() * matrix.cols(); i++) {
            buffer.put(0.0);
        }
        buffer.position(0);
    }

    private static Timing summarize(List<Double> samples) {
        List<Double> sorted = new ArrayList<>(samples);
        Collections.sort(sorted);
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double total = 0.0;
        for (double sample : sorted) {
            min = Math.min(min, sample);
            max = Math.max(max, sample);
            total += sample;
        }
        int middle = sorted.size() / 2;
        double median = (sorted.size() % 2 == 0)
            ? 0.5 * (sorted.get(middle - 1) + sorted.get(middle))
            : sorted.get(middle);
        int q1Index = Math.max(0, (int) Math.floor(0.25 * (sorted.size() - 1)));
        int q3Index = Math.max(0, (int) Math.floor(0.75 * (sorted.size() - 1)));
        return new Timing(
            min,
            max,
            total / sorted.size(),
            median,
            sorted.get(q1Index),
            sorted.get(q3Index)
        );
    }

    private static double[] randomData(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return data;
    }

    private static double gflops(double flops, double ms) {
        return ms == 0.0 ? 0.0 : flops / (ms / 1e3) / 1e9;
    }

    private record Variant(String label, int mc, int kc, int nc, boolean disableSquareTuning, int loopOrder) {
    }

    private record Timing(double minMs, double maxMs, double meanMs, double medianMs, double p25Ms, double p75Ms) {
    }

    private record Row(Variant variant, Timing jniArray, Timing jniDirect, Timing standalone) {
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
