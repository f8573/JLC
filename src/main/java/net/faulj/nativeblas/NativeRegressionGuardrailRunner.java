package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.result.CholeskyResult;
import net.faulj.matrix.Matrix;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

/**
 * Regression guardrail for the native correctness/performance boundary.
 *
 * The only expected red check in the current repo state is the documented
 * 512x512 JNI GEMM ratio guard; everything else should remain green.
 */
public final class NativeRegressionGuardrailRunner {
    private static final double[] CHOLESKY_SMOKE = {
        4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
        -16.0, -43.0, 98.0
    };

    private static final double[] CHOLESKY_EXPECTED = {
        2.0, 0.0, 0.0,
        6.0, 1.0, 0.0,
        -8.0, 5.0, 3.0
    };

    private NativeRegressionGuardrailRunner() {
    }

    public static void main(String[] args) throws Exception {
        int warmup = 3;
        int runs = 7;
        int threads = 1;
        double nativeMustBeatJavaRatio = 1.50;

        for (String arg : args) {
            if (arg.startsWith("--warmup=")) warmup = Integer.parseInt(arg.substring("--warmup=".length()));
            else if (arg.startsWith("--runs=")) runs = Integer.parseInt(arg.substring("--runs=".length()));
            else if (arg.startsWith("--threads=")) threads = Integer.parseInt(arg.substring("--threads=".length()));
            else if (arg.startsWith("--native-java-ratio=")) nativeMustBeatJavaRatio = Double.parseDouble(arg.substring("--native-java-ratio=".length()));
        }

        List<GuardResult> results = new ArrayList<>();
        GemmGuardResult gemm = checkGemm512(warmup, runs, threads, nativeMustBeatJavaRatio);
        results.add(new GuardResult("compiled_avx2_on_avx2_machine", gemm.runtimeAvx2() ? gemm.compiledAvx2() : true,
            gemm.runtimeAvx2() ? "compiled_avx2=" + gemm.compiledAvx2() : "runtime_avx2=false"));
        results.add(new GuardResult("scalar_fallback_not_selected_for_512", gemm.runtimeAvx2() ? !gemm.scalarFallback() : true,
            "selected_kernel=" + gemm.selectedKernel() + ", scalar_fallback=" + gemm.scalarFallback()));
        results.add(new GuardResult("native_gemm_512_not_catastrophically_slower_than_java",
            gemm.nativeMedianMs() <= gemm.javaMedianMs() * nativeMustBeatJavaRatio,
            String.format(Locale.ROOT,
                "java median=%.3fms p25=%.3f p75=%.3f native median=%.3fms p25=%.3f p75=%.3f ratio=%.3f threshold=%.3f",
                gemm.javaMedianMs(), gemm.javaP25Ms(), gemm.javaP75Ms(),
                gemm.nativeMedianMs(), gemm.nativeP25Ms(), gemm.nativeP75Ms(),
                gemm.nativeMedianMs() / gemm.javaMedianMs(), nativeMustBeatJavaRatio)));

        CholeskyGuardResult cholesky = checkCholeskySmoke();
        results.add(new GuardResult("cholesky_java_smoke_residual_zero", cholesky.javaResidual() <= 1e-12,
            String.format(Locale.ROOT, "javaResidual=%.3e mismatch=%s", cholesky.javaResidual(), cholesky.javaMismatch())));
        results.add(new GuardResult("cholesky_jni_smoke_residual_zero", cholesky.jniResidual() <= 1e-12 && cholesky.jniInfo() == 0,
            String.format(Locale.ROOT, "jniResidual=%.3e info=%d mismatch=%s", cholesky.jniResidual(), cholesky.jniInfo(), cholesky.jniMismatch())));
        results.add(new GuardResult("cholesky_cpp_smoke_residual_zero", cholesky.cppResidual() <= 1e-12 && cholesky.cppInfo() == 0,
            String.format(Locale.ROOT, "cppResidual=%.3e info=%d mismatch=%s", cholesky.cppResidual(), cholesky.cppInfo(), cholesky.cppMismatch())));

        HessenbergGuardResult hessenberg = checkHessenberg512();
        results.add(new GuardResult("hessenberg_512_succeeds", hessenberg.success(),
            String.format(Locale.ROOT, "elapsed=%.3fms detail=%s", hessenberg.elapsedMs(), hessenberg.detail())));

        boolean allOk = results.stream().allMatch(GuardResult::passed);
        System.out.println("NATIVE_REGRESSION_GUARDRAIL");
        System.out.printf(Locale.ROOT, "gemm_512_java_median_ms=%.3f%n", gemm.javaMedianMs());
        System.out.printf(Locale.ROOT, "gemm_512_java_p25_ms=%.3f%n", gemm.javaP25Ms());
        System.out.printf(Locale.ROOT, "gemm_512_java_p75_ms=%.3f%n", gemm.javaP75Ms());
        System.out.printf(Locale.ROOT, "gemm_512_native_median_ms=%.3f%n", gemm.nativeMedianMs());
        System.out.printf(Locale.ROOT, "gemm_512_native_p25_ms=%.3f%n", gemm.nativeP25Ms());
        System.out.printf(Locale.ROOT, "gemm_512_native_p75_ms=%.3f%n", gemm.nativeP75Ms());
        System.out.println("| Guard | Status | Detail |");
        System.out.println("|---|---|---|");
        for (GuardResult result : results) {
            System.out.printf(Locale.ROOT, "| %s | %s | %s |%n",
                result.name(), result.passed() ? "PASS" : "FAIL", result.detail().replace("|", "/"));
        }

        if (!allOk) {
            throw new IllegalStateException("One or more native regression guards failed");
        }
    }

    private static GemmGuardResult checkGemm512(int warmup, int runs, int threads) throws Exception {
        return checkGemm512(warmup, runs, threads, 1.50);
    }

    private static GemmGuardResult checkGemm512(int warmup, int runs, int threads, double ratio) throws Exception {
        int rows = 512;
        int inner = 512;
        int cols = 512;
        long seed = 17_512L;
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
        NativeGemmProfile nativeProfile;
        Timing nativeTiming;
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }
        nativeTiming = measureJni(aSeed, bSeed, rows, inner, cols, warmup, runs, threads);
        NativeProfiling.setEnabled(true);
        NativeProfiling.reset();
        double[] profileA = aSeed.clone();
        double[] profileB = bSeed.clone();
        double[] profileC = new double[rows * cols];
        NativeBindings.nativeGemm(profileA, rows, inner, profileB, inner, cols, profileC, rows, cols, 1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN);
        nativeProfile = NativeProfiling.snapshot().orElse(NativeGemmProfile.EMPTY);
        NativeProfiling.setEnabled(false);

        Map<String, String> cppDiag = runCppGemmDiag(rows, inner, cols, warmup, runs, threads, seed);
        return new GemmGuardResult(
            javaTiming.medianMs(),
            javaTiming.p25Ms(),
            javaTiming.p75Ms(),
            nativeTiming.medianMs(),
            nativeTiming.p25Ms(),
            nativeTiming.p75Ms(),
            Boolean.parseBoolean(cppDiag.getOrDefault("runtime_avx2", "false")),
            Boolean.parseBoolean(cppDiag.getOrDefault("compiled_avx2", "false")),
            "scalar".equals(kernelName(nativeProfile)),
            kernelName(nativeProfile),
            nativeProfile
        );
    }

    private static CholeskyGuardResult checkCholeskySmoke() throws Exception {
        String previousBackend = System.getProperty("jlc.backend");
        System.setProperty("jlc.backend", "java");
        double javaResidual;
        String javaMismatch;
        try {
            CholeskyResult result = new CholeskyDecomposition().decompose(Matrix.wrap(CHOLESKY_SMOKE.clone(), 3, 3));
            double[] factor = result.getL().getRawData().clone();
            javaResidual = residual(CHOLESKY_SMOKE, factor, 3);
            javaMismatch = mismatch(factor, CHOLESKY_EXPECTED);
        } finally {
            if (previousBackend == null) System.clearProperty("jlc.backend");
            else System.setProperty("jlc.backend", previousBackend);
        }

        double[] jni = CHOLESKY_SMOKE.clone();
        int jniInfo = NativeBindings.nativeCholeskyDecompose(jni, 3);
        double jniResidual = residual(CHOLESKY_SMOKE, jni, 3);
        String jniMismatch = mismatch(jni, CHOLESKY_EXPECTED);

        Map<String, String> cppDiag = runCppCholeskySmoke();
        double cppResidual = Double.parseDouble(cppDiag.getOrDefault("residual", "Infinity"));
        int cppInfo = Integer.parseInt(cppDiag.getOrDefault("info", "-1"));
        double[] cppFactor = parseThreeByThree(cppDiag);
        String cppMismatch = mismatch(cppFactor, CHOLESKY_EXPECTED);

        return new CholeskyGuardResult(javaResidual, javaMismatch, jniResidual, jniInfo, jniMismatch, cppResidual, cppInfo, cppMismatch);
    }

    private static HessenbergGuardResult checkHessenberg512() {
        double[] matrix = randomData(512, 512, 288004099791341793L);
        long start = System.nanoTime();
        try {
            String previousBackend = System.getProperty("jlc.backend");
            System.setProperty("jlc.backend", "java");
            try {
                HessenbergReduction.decompose(Matrix.wrap(matrix, 512, 512));
            } finally {
                if (previousBackend == null) System.clearProperty("jlc.backend");
                else System.setProperty("jlc.backend", previousBackend);
            }
            return new HessenbergGuardResult(true, (System.nanoTime() - start) / 1e6, "ok");
        } catch (Throwable t) {
            return new HessenbergGuardResult(false, (System.nanoTime() - start) / 1e6, t.getClass().getSimpleName() + ": " + t.getMessage());
        }
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

    private static Timing measureJni(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                     int warmup, int runs, int threads) {
        double[] a = aSeed.clone();
        double[] b = bSeed.clone();
        double[] c = new double[rows * cols];
        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < warmup + runs; i++) {
            java.util.Arrays.fill(c, 0.0);
            long start = System.nanoTime();
            NativeBindings.nativeGemm(a, rows, inner, b, inner, cols, c, rows, cols, 1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN);
            if (i >= warmup) {
                samples.add((System.nanoTime() - start) / 1e6);
            }
        }
        return summarize(samples);
    }

    private static Map<String, String> runCppGemmDiag(int rows, int inner, int cols, int warmup, int runs, int threads, long seed) throws Exception {
        return runNativeExecutable(List.of(
            "--algorithm=gemm",
            "--rows=" + rows,
            "--cols=" + cols,
            "--inner=" + inner,
            "--warmup=" + warmup,
            "--runs=" + runs,
            "--threads=" + threads,
            "--seed=" + seed,
            "--diag"
        ));
    }

    private static Map<String, String> runCppCholeskySmoke() throws Exception {
        return runNativeExecutable(List.of(
            "--algorithm=cholesky",
            "--matrix=smoke3",
            "--diag",
            "--threads=1"
        ));
    }

    private static Map<String, String> runNativeExecutable(List<String> args) throws Exception {
        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }
        List<String> command = new ArrayList<>();
        command.add(executable);
        command.addAll(args);
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
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
            throw new IllegalStateException("Native executable failed with code " + exit + " for args " + args);
        }
        return values;
    }

    private static double[] parseThreeByThree(Map<String, String> values) {
        double[] out = new double[9];
        for (int row = 0; row < 3; row++) {
            String line = values.get("block_row_" + row);
            if (line == null) {
                continue;
            }
            String[] parts = line.split(",");
            for (int col = 0; col < Math.min(parts.length, 3); col++) {
                out[row * 3 + col] = Double.parseDouble(parts[col]);
            }
        }
        return out;
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
        sorted.sort(Double::compareTo);
        double best = Double.POSITIVE_INFINITY;
        double worst = Double.NEGATIVE_INFINITY;
        double total = 0.0;
        for (double sample : sorted) {
            best = Math.min(best, sample);
            worst = Math.max(worst, sample);
            total += sample;
        }
        int middle = sorted.size() / 2;
        double median = (sorted.size() % 2 == 0)
            ? 0.5 * (sorted.get(middle - 1) + sorted.get(middle))
            : sorted.get(middle);
        double p25 = percentile(sorted, 0.25);
        double p75 = percentile(sorted, 0.75);
        return new Timing(best, worst, total / sorted.size(), median, p25, p75);
    }

    private static double percentile(List<Double> sorted, double p) {
        if (sorted.isEmpty()) {
            return Double.NaN;
        }
        double index = p * (sorted.size() - 1);
        int lo = (int) Math.floor(index);
        int hi = (int) Math.ceil(index);
        if (lo == hi) {
            return sorted.get(lo);
        }
        double weight = index - lo;
        return sorted.get(lo) * (1.0 - weight) + sorted.get(hi) * weight;
    }

    private static double[] randomData(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return data;
    }

    private static double residual(double[] original, double[] factor, int n) {
        double[] reconstructed = new double[n * n];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                double sum = 0.0;
                for (int k = 0; k <= Math.min(row, col); k++) {
                    sum = Math.fma(factor[row * n + k], factor[col * n + k], sum);
                }
                reconstructed[row * n + col] = sum;
            }
        }
        double delta = 0.0;
        double base = 0.0;
        for (int i = 0; i < original.length; i++) {
            double diff = original[i] - reconstructed[i];
            delta = Math.fma(diff, diff, delta);
            base = Math.fma(original[i], original[i], base);
        }
        double denom = Math.sqrt(base);
        return denom == 0.0 ? Math.sqrt(delta) : Math.sqrt(delta) / denom;
    }

    private static String mismatch(double[] actual, double[] expected) {
        for (int i = 0; i < expected.length; i++) {
            if (Math.abs(actual[i] - expected[i]) > 1e-9) {
                return "row=" + (i / 3) + ",col=" + (i % 3) + ",expected=" + expected[i] + ",actual=" + actual[i];
            }
        }
        return "none";
    }

    private record GuardResult(String name, boolean passed, String detail) {
    }

    private record Timing(double bestMs, double maxMs, double meanMs, double medianMs, double p25Ms, double p75Ms) {
    }

    private record GemmGuardResult(double javaMedianMs, double javaP25Ms, double javaP75Ms,
                                   double nativeMedianMs, double nativeP25Ms, double nativeP75Ms,
                                   boolean runtimeAvx2,
                                   boolean compiledAvx2, boolean scalarFallback, String selectedKernel,
                                   NativeGemmProfile profile) {
    }

    private record CholeskyGuardResult(double javaResidual, String javaMismatch, double jniResidual,
                                       int jniInfo, String jniMismatch, double cppResidual, int cppInfo,
                                       String cppMismatch) {
    }

    private record HessenbergGuardResult(boolean success, double elapsedMs, String detail) {
    }
}
