package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;
import net.faulj.matrix.Matrix;

import java.io.BufferedReader;
import java.io.InputStreamReader;
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
 * Focused native GEMM regression diagnostic for one shape.
 */
public final class NativeGemmRegressionDiagnosticRunner {
    private NativeGemmRegressionDiagnosticRunner() {
    }

    public static void main(String[] args) throws Exception {
        int rows = 512;
        int inner = 512;
        int cols = 512;
        int warmup = 2;
        int runs = 3;
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

        DispatchPolicy javaPolicy = DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(threads > 1)
            .parallelism(threads)
            .enableBlas3(true)
            .enableSimd(true)
            .build();

        Timing javaTiming = measureJava(aSeed, bSeed, rows, inner, cols, warmup, runs, javaPolicy);
        NativeRun jniRun = measureJni(aSeed, bSeed, rows, inner, cols, warmup, runs, threads);
        CppRun cppRun = measureCpp(rows, inner, cols, warmup, runs, threads, seed);

        double flops = 2.0 * rows * cols * inner;
        System.out.println("NATIVE_GEMM_REGRESSION_DIAGNOSTIC");
        System.out.printf(Locale.ROOT, "shape=%dx%d * %dx%d%n", rows, inner, inner, cols);
        System.out.println("threads=" + threads);
        System.out.println();
        System.out.println("| Backend | Min ms | Median ms | Max ms | Mean ms | GFLOPs(median) | Notes |");
        System.out.println("|---|---:|---:|---:|---:|---:|---|");
        System.out.printf(Locale.ROOT, "| Java | %.3f | %.3f | %.3f | %.3f | %.3f | SIMD/Java path |%n",
            javaTiming.minMs(), javaTiming.medianMs(), javaTiming.maxMs(), javaTiming.meanMs(),
            gflops(flops, javaTiming.medianMs()));
        System.out.printf(Locale.ROOT, "| JNI-C++ | %.3f | %.3f | %.3f | %.3f | %.3f | provider=%s, kernel=%s |%n",
            jniRun.minMs(), jniRun.medianMs(), jniRun.maxMs(), jniRun.meanMs(), gflops(flops, jniRun.medianMs()),
            jniRun.provider(), jniRun.profile().map(NativeGemmRegressionDiagnosticRunner::kernelName).orElse("unknown"));
        System.out.printf(Locale.ROOT, "| C++ | %.3f | %.3f | %.3f | %.3f | %.3f | kernel=%s |%n",
            cppRun.minMs(), cppRun.medianMs(), cppRun.maxMs(), cppRun.meanMs(), gflops(flops, cppRun.medianMs()),
            cppRun.diag().getOrDefault("selected_kernel", "unknown"));
        System.out.println();

        System.out.println("JNI native profile:");
        System.out.println("backend selected: jni builtin");
        System.out.println("provider: " + jniRun.provider());
        printJniProfile(jniRun.profile(), flops);
        System.out.println();

        System.out.println("Standalone C++ profile:");
        printCppProfile(cppRun.diag(), flops);
    }

    private static Timing measureJava(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                      int warmup, int runs, DispatchPolicy policy) {
        JavaBackend backend = new JavaBackend();
        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < warmup + runs; i++) {
            double[] a = aSeed.clone();
            double[] b = bSeed.clone();
            double[] c = new double[rows * cols];
            long start = System.nanoTime();
            backend.gemm(Matrix.wrap(a, rows, inner), Matrix.wrap(b, inner, cols), Matrix.wrap(c, rows, cols), 1.0, 0.0, policy);
            double ms = (System.nanoTime() - start) / 1e6;
            if (i >= warmup) {
                samples.add(ms);
            }
        }
        return summarize(samples);
    }

    private static NativeRun measureJni(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                        int warmup, int runs, int threads) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }
        NativeProfiling.setEnabled(true);
        NativeProfiling.reset();
        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < warmup + runs; i++) {
            double[] a = aSeed.clone();
            double[] b = bSeed.clone();
            double[] c = new double[rows * cols];
            long start = System.nanoTime();
            NativeBindings.nativeGemm(a, rows, inner, b, inner, cols, c, rows, cols, 1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN);
            double ms = (System.nanoTime() - start) / 1e6;
            if (i >= warmup) {
                samples.add(ms);
            }
        }
        Optional<NativeGemmProfile> profile = NativeProfiling.snapshot();
        NativeProfiling.setEnabled(false);
        Timing timing = summarize(samples);
        return new NativeRun(timing.minMs(), timing.medianMs(), timing.maxMs(), timing.meanMs(), context.getProviderDescription(), profile);
    }

    private static CppRun measureCpp(int rows, int inner, int cols, int warmup, int runs, int threads, long seed) throws Exception {
        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }
        Process process = new ProcessBuilder(
            executable,
            "--algorithm=gemm",
            "--rows=" + rows,
            "--cols=" + cols,
            "--inner=" + inner,
            "--warmup=" + warmup,
            "--runs=" + runs,
            "--threads=" + threads,
            "--seed=" + seed,
            "--diag"
        ).redirectErrorStream(true).start();

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
        return new CppRun(
            Double.parseDouble(diag.getOrDefault("best_ms", "NaN")),
            Double.parseDouble(diag.getOrDefault("median_ms", "NaN")),
            Double.parseDouble(diag.getOrDefault("max_ms", "NaN")),
            Double.parseDouble(diag.getOrDefault("mean_ms", "NaN")),
            diag
        );
    }

    private static void printJniProfile(Optional<NativeGemmProfile> profileOpt, double flops) {
        if (profileOpt.isEmpty()) {
            System.out.println("profile unavailable");
            return;
        }
        NativeGemmProfile profile = profileOpt.get();
        System.out.println("avx2 enabled: " + kernelName(profile).startsWith("avx2_"));
        System.out.println("thread count: requested=" + profile.lastRequestedThreads() + ", actual=" + profile.lastActualThreads());
        System.out.println("MR/NR/MC/NC/KC: " + profile.lastMr() + "/" + profile.lastNr() + "/" + profile.lastMc() + "/" + profile.lastNc() + "/" + profile.lastKc());
        System.out.println("packed kernel selected: " + kernelName(profile));
        System.out.println("fallback/scalar selected: " + ("scalar".equals(kernelName(profile))));
        System.out.printf(Locale.ROOT, "packing time vs compute time: %.3f ms packA, %.3f ms packB, %.3f ms kernel%n",
            nanosToMs(profile.packANanos()), nanosToMs(profile.packBNanos()), nanosToMs(profile.kernelNanos()));
        System.out.printf(Locale.ROOT, "GFLOPs(profile wall): %.3f%n", profile.wallSeconds() == 0.0 ? 0.0 : flops / profile.wallSeconds() / 1e9);
        System.out.println("microtile calls: " + profile.microtileCalls());
        System.out.println("vendor calls: " + profile.vendorCalls());
    }

    private static void printCppProfile(Map<String, String> diag, double flops) {
        System.out.println("backend selected: standalone builtin");
        System.out.println("compiler flags: " + diag.getOrDefault("compile_flags", "unknown"));
        System.out.println("compiler family: " + diag.getOrDefault("compiler_family", "unknown"));
        System.out.println("AVX2 enabled?: " + diag.getOrDefault("compiled_avx2", "unknown"));
        System.out.println("thread count: requested=" + diag.getOrDefault("requested_threads", "?")
            + ", actual=" + diag.getOrDefault("actual_threads", "?"));
        System.out.println("MR/NR/MC/NC/KC: " + diag.getOrDefault("mr", "?") + "/"
            + diag.getOrDefault("nr", "?") + "/"
            + diag.getOrDefault("mc", "?") + "/"
            + diag.getOrDefault("nc", "?") + "/"
            + diag.getOrDefault("kc", "?"));
        System.out.println("packed kernel selected: " + diag.getOrDefault("selected_kernel", "unknown"));
        System.out.println("fallback/scalar selected?: " + diag.getOrDefault("scalar_fallback", "unknown"));
        System.out.printf(Locale.ROOT, "packing time vs compute time: %s ms packA, %s ms packB, %s ms kernel%n",
            nanosStringToMs(diag.get("pack_a_ns")), nanosStringToMs(diag.get("pack_b_ns")), nanosStringToMs(diag.get("kernel_ns")));
        System.out.printf(Locale.ROOT, "GFLOPs(best): %.3f%n",
            Double.parseDouble(diag.getOrDefault("best_gflops", "0")));
        System.out.println("microtile calls: " + diag.getOrDefault("microtile_calls", "?"));
        System.out.println("vendor calls: " + diag.getOrDefault("vendor_calls", "?"));
    }

    private static String kernelName(NativeGemmProfile profile) {
        if (profile.lastMr() == 4 && profile.lastNr() == 8) return "avx2_4x8";
        if (profile.lastMr() == 4 && profile.lastNr() == 4) return "avx2_4x4";
        if (profile.lastMr() == 6 && profile.lastNr() == 8) return "avx512_6x8";
        if (profile.lastMr() == 5 && profile.lastNr() == 4) return "avx2_5x4";
        return "scalar";
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

    private static String nanosStringToMs(String nanos) {
        if (nanos == null) return "?";
        return String.format(Locale.ROOT, "%.3f", Long.parseLong(nanos) / 1e6);
    }

    private record Timing(double minMs, double medianMs, double maxMs, double meanMs) {
    }

    private record NativeRun(double minMs, double medianMs, double maxMs, double meanMs, String provider, Optional<NativeGemmProfile> profile) {
    }

    private record CppRun(double minMs, double medianMs, double maxMs, double meanMs, Map<String, String> diag) {
    }
}
