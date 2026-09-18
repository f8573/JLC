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
import java.util.Optional;
import java.util.Random;

/**
 * Reconciles JNI-vs-standalone native GEMM at one shape with shared seeds and same-process JNI variants.
 */
public final class NativeGemmJniStandaloneReconciliationRunner {
    private static final int LOOP_DEFAULT = 0;
    private static final int LOOP_JC_PC_IC = 1;

    private NativeGemmJniStandaloneReconciliationRunner() {
    }

    public static void main(String[] args) throws Exception {
        int rows = 512;
        int inner = 512;
        int cols = 512;
        int warmup = 3;
        int runs = 7;
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

        List<ResultRow> rowsOut = new ArrayList<>();
        for (Variant variant : variants) {
            JniRun jniArrayNoProfile = measureJniVariant(aSeed, bSeed, rows, inner, cols, warmup, runs, threads, variant, false, false);
            JniRun jniArrayProfile = measureJniVariant(aSeed, bSeed, rows, inner, cols, warmup, runs, threads, variant, false, true);
            JniRun jniDirectNoProfile = measureJniVariant(aSeed, bSeed, rows, inner, cols, warmup, runs, threads, variant, true, false);
            JniRun jniDirectProfile = measureJniVariant(aSeed, bSeed, rows, inner, cols, warmup, runs, threads, variant, true, true);
            CppRun cppNoDiag = measureCppVariant(rows, inner, cols, warmup, runs, threads, seed, variant, false);
            CppRun cppDiag = measureCppVariant(rows, inner, cols, warmup, runs, threads, seed, variant, true);
            rowsOut.add(new ResultRow(variant, jniArrayNoProfile, jniArrayProfile, jniDirectNoProfile, jniDirectProfile, cppNoDiag, cppDiag));
        }

        System.out.println("NATIVE_GEMM_JNI_STANDALONE_RECONCILIATION");
        System.out.printf(Locale.ROOT, "shape=%dx%d * %dx%d warmup=%d runs=%d threads=%d%n",
            rows, inner, inner, cols, warmup, runs, threads);
        System.out.println();
        System.out.printf(Locale.ROOT,
            "Java median: %.3f ms (min %.3f / max %.3f / mean %.3f, %.3f GFLOPs)%n",
            javaTiming.medianMs(), javaTiming.minMs(), javaTiming.maxMs(), javaTiming.meanMs(), gflops(flops, javaTiming.medianMs()));
        System.out.println();
        System.out.println("| Variant | JNI array no-profile | JNI array profile | JNI direct no-profile | JNI direct profile | C++ no-diag | C++ diag |");
        System.out.println("|---|---:|---:|---:|---:|---:|---:|");
        for (ResultRow row : rowsOut) {
            System.out.printf(Locale.ROOT,
                "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |%n",
                row.variant.label,
                row.jniArrayNoProfile.timing.medianMs(),
                row.jniArrayProfile.timing.medianMs(),
                row.jniDirectNoProfile.timing.medianMs(),
                row.jniDirectProfile.timing.medianMs(),
                row.cppNoDiag.timing.medianMs(),
                row.cppDiag.timing.medianMs()
            );
        }
        System.out.println();
        System.out.println("| Variant | Path | Profile | Median ms | Native-core avg ms | Overhead avg ms | Alloc ms | PackA ms | PackB ms | Kernel ms | Other ms | Ratio vs Java |");
        System.out.println("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for (ResultRow row : rowsOut) {
            printBreakdownRow(javaTiming, row.variant.label, "array", "on", row.jniArrayProfile);
            printBreakdownRow(javaTiming, row.variant.label, "direct", "on", row.jniDirectProfile);
        }
    }

    private static Timing measureJava(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                      int warmup, int runs, DispatchPolicy policy) {
        JavaBackend backend = new JavaBackend();
        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < warmup + runs; i++) {
            double[] c = new double[rows * cols];
            long start = System.nanoTime();
            backend.gemm(Matrix.wrap(aSeed.clone(), rows, inner), Matrix.wrap(bSeed.clone(), inner, cols), Matrix.wrap(c, rows, cols), 1.0, 0.0, policy);
            double ms = (System.nanoTime() - start) / 1e6;
            if (i >= warmup) {
                samples.add(ms);
            }
        }
        return summarize(samples);
    }

    private static JniRun measureJniVariant(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                            int warmup, int runs, int threads, Variant variant,
                                            boolean directBuffers, boolean profileEnabled) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }

        applyVariant(variant);
        try {
            for (int i = 0; i < warmup; i++) {
                runJniOnce(aSeed, bSeed, rows, inner, cols, threads, directBuffers);
            }

            if (profileEnabled) {
                NativeProfiling.setEnabled(true);
                NativeProfiling.reset();
            }
            List<Double> samples = new ArrayList<>();
            for (int i = 0; i < runs; i++) {
                long start = System.nanoTime();
                runJniOnce(aSeed, bSeed, rows, inner, cols, threads, directBuffers);
                samples.add((System.nanoTime() - start) / 1e6);
            }
            Optional<NativeGemmProfile> profile = profileEnabled ? NativeProfiling.snapshot() : Optional.empty();
            if (profileEnabled) {
                NativeProfiling.setEnabled(false);
            }
            Timing timing = summarize(samples);
            NativeGemmProfile snapshot = profile.orElse(NativeGemmProfile.EMPTY);
            return new JniRun(timing, profileEnabled ? breakdown(snapshot, runs) : null);
        } finally {
            clearVariant();
            NativeProfiling.setEnabled(false);
        }
    }

    private static CppRun measureCppVariant(int rows, int inner, int cols, int warmup, int runs, int threads,
                                            long seed, Variant variant, boolean diagnostic) throws Exception {
        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }

        List<String> baseCommand = List.of(
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
        List<String> command = new ArrayList<>(baseCommand);
        if (diagnostic) {
            command.add("--diag");
        }
        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        Map<String, String> env = builder.environment();
        if (variant.disableSquareTuning) {
            env.put("JLC_NATIVE_DISABLE_SQUARE_TUNING", "1");
            env.put("JLC_NATIVE_MC", Integer.toString(variant.mc));
            env.put("JLC_NATIVE_KC", Integer.toString(variant.kc));
            env.put("JLC_NATIVE_NC", Integer.toString(variant.nc));
            env.put("JLC_NATIVE_LOOP_ORDER", variant.loopOrder == LOOP_JC_PC_IC ? "jc_pc_ic" : "ic_pc_jc");
        } else {
            env.remove("JLC_NATIVE_DISABLE_SQUARE_TUNING");
            env.remove("JLC_NATIVE_MC");
            env.remove("JLC_NATIVE_KC");
            env.remove("JLC_NATIVE_NC");
            env.remove("JLC_NATIVE_LOOP_ORDER");
        }

        Process process = builder.start();
        Map<String, String> diag = new LinkedHashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                int idx = line.indexOf('=');
                if (idx > 0) {
                    diag.put(line.substring(0, idx), line.substring(idx + 1));
                }
            }
        }
        int exit = process.waitFor();
        if (exit != 0) {
            throw new IllegalStateException("Native GEMM diagnostic exited with code " + exit);
        }
        return new CppRun(new Timing(
            Double.parseDouble(diag.getOrDefault("best_ms", "NaN")),
            Double.parseDouble(diag.getOrDefault("median_ms", "NaN")),
            Double.parseDouble(diag.getOrDefault("max_ms", "NaN")),
            Double.parseDouble(diag.getOrDefault("mean_ms", "NaN"))
        ));
    }

    private static void runJniOnce(double[] aSeed, double[] bSeed, int rows, int inner, int cols, int threads, boolean directBuffers) {
        if (!directBuffers) {
            double[] c = new double[rows * cols];
            NativeBindings.nativeGemm(aSeed.clone(), rows, inner, bSeed.clone(), inner, cols, c, rows, cols, 1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN);
            return;
        }

        try (NativeMatrix a = NativeMatrix.allocate(rows, inner);
             NativeMatrix b = NativeMatrix.allocate(inner, cols);
             NativeMatrix c = NativeMatrix.allocate(rows, cols)) {
            fillNativeMatrix(a, aSeed);
            fillNativeMatrix(b, bSeed);
            zeroNativeMatrix(c);
            NativeBindings.nativeGemmDirect(
                a.buffer(), 0L, a.ld(), rows, inner, 0,
                b.buffer(), 0L, b.ld(), inner, cols, 0,
                c.buffer(), 0L, c.ld(), rows, cols, 0,
                1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN
            );
        }
    }

    private static void applyVariant(Variant variant) {
        if (variant.disableSquareTuning) {
            NativeBindings.nativeGemmSetRuntimeOverrides(
                variant.mc, variant.kc, variant.nc,
                0, 0,
                true,
                variant.loopOrder
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

    private static JniBreakdown breakdown(NativeGemmProfile profile, int runs) {
        double nativeCoreAvgMs = nanosToMs(profile.wallNanos()) / runs;
        double allocAvgMs = nanosToMs(profile.allocNanos()) / runs;
        double packAAvgMs = nanosToMs(profile.packANanos()) / runs;
        double packBAvgMs = nanosToMs(profile.packBNanos()) / runs;
        double kernelAvgMs = nanosToMs(profile.kernelNanos()) / runs;
        double scaleAvgMs = nanosToMs(profile.scaleCNanos()) / runs;
        double threadingAvgMs = nanosToMs(profile.threadLaunchNanos() + profile.threadJoinNanos()) / runs;
        double otherAvgMs = Math.max(0.0, nativeCoreAvgMs - allocAvgMs - packAAvgMs - packBAvgMs - kernelAvgMs - scaleAvgMs - threadingAvgMs);
        return new JniBreakdown(nativeCoreAvgMs, allocAvgMs, packAAvgMs, packBAvgMs, kernelAvgMs, scaleAvgMs, threadingAvgMs, otherAvgMs);
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
        return new Timing(min, median, max, total / sorted.size());
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

    private static double nanosToMs(long nanos) {
        return nanos / 1e6;
    }

    private static void printBreakdownRow(Timing javaTiming, String label, String path, String profile, JniRun run) {
        JniBreakdown b = run.breakdown;
        double overhead = Math.max(0.0, run.timing.meanMs() - b.nativeCoreAvgMs());
        System.out.printf(Locale.ROOT,
            "| %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |%n",
            label,
            path,
            profile,
            run.timing.medianMs(),
            b.nativeCoreAvgMs(),
            overhead,
            b.allocAvgMs(),
            b.packAAvgMs(),
            b.packBAvgMs(),
            b.kernelAvgMs(),
            b.otherAvgMs(),
            run.timing.medianMs() / javaTiming.medianMs()
        );
    }

    private record Variant(String label, int mc, int kc, int nc, boolean disableSquareTuning, int loopOrder) {
    }

    private record Timing(double minMs, double medianMs, double maxMs, double meanMs) {
    }

    private record JniBreakdown(double nativeCoreAvgMs, double allocAvgMs, double packAAvgMs, double packBAvgMs,
                                double kernelAvgMs, double scaleAvgMs, double threadingAvgMs, double otherAvgMs) {
    }

    private record JniRun(Timing timing, JniBreakdown breakdown) {
    }

    private record CppRun(Timing timing) {
    }

    private record ResultRow(Variant variant, JniRun jniArrayNoProfile, JniRun jniArrayProfile,
                             JniRun jniDirectNoProfile, JniRun jniDirectProfile,
                             CppRun cppNoDiag, CppRun cppDiag) {
    }
}
