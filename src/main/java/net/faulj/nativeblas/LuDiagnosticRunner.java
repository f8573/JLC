package net.faulj.nativeblas;

import net.faulj.core.PermutationVector;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Dedicated LU correctness and phase diagnostic runner.
 */
public final class LuDiagnosticRunner {
    private static final double[] SMOKE3 = {
        2.0, 1.0, 1.0,
        4.0, -6.0, 0.0,
        -2.0, 7.0, 2.0
    };

    private static final double[] SMOKE4 = {
        0.0, 2.0, 1.0, 3.0,
        4.0, 8.0, 2.0, 6.0,
        2.0, 1.0, 3.0, 1.0,
        6.0, 4.0, 5.0, 9.0
    };

    private LuDiagnosticRunner() {
    }

    public static void main(String[] args) throws Exception {
        int warmup = 0;
        int runs = 1;
        int progressInterval = 32;
        String sizesArg = "128,256,512,1024";
        String blockSizesArg = "16,32,64,96,128";
        boolean runPerf = true;
        boolean nativeOnly = false;
        boolean jniBoundaryOnly = false;
        boolean jniBoundaryWorker = false;
        boolean jniGemmWorker = false;
        boolean jniGemmSequenceWorker = false;
        boolean copyBack = true;
        boolean aliasing = true;
        int workerSize = 0;
        double medianGainThreshold = 0.04;

        for (String arg : args) {
            if (arg.startsWith("--warmup=")) {
                warmup = Integer.parseInt(arg.substring("--warmup=".length()));
            } else if (arg.startsWith("--runs=")) {
                runs = Integer.parseInt(arg.substring("--runs=".length()));
            } else if (arg.startsWith("--sizes=")) {
                sizesArg = arg.substring("--sizes=".length());
            } else if (arg.startsWith("--progressInterval=")) {
                progressInterval = Integer.parseInt(arg.substring("--progressInterval=".length()));
            } else if (arg.startsWith("--blockSizes=")) {
                blockSizesArg = arg.substring("--blockSizes=".length());
            } else if (arg.startsWith("--medianGainThreshold=")) {
                medianGainThreshold = Double.parseDouble(arg.substring("--medianGainThreshold=".length()));
            } else if (arg.equals("--smoke-only")) {
                runPerf = false;
            } else if (arg.equals("--native-only")) {
                nativeOnly = true;
            } else if (arg.equals("--jni-boundary-only")) {
                jniBoundaryOnly = true;
            } else if (arg.equals("--jni-boundary-worker")) {
                jniBoundaryWorker = true;
            } else if (arg.equals("--jni-gemm-worker")) {
                jniGemmWorker = true;
            } else if (arg.equals("--jni-gemm-sequence-worker")) {
                jniGemmSequenceWorker = true;
            } else if (arg.startsWith("--copyBack=")) {
                copyBack = Boolean.parseBoolean(arg.substring("--copyBack=".length()));
            } else if (arg.startsWith("--aliasing=")) {
                aliasing = Boolean.parseBoolean(arg.substring("--aliasing=".length()));
            } else if (arg.startsWith("--size=")) {
                workerSize = Integer.parseInt(arg.substring("--size=".length()));
            }
        }

        if (jniBoundaryWorker) {
            runJniBoundaryWorker(workerSize, copyBack);
            return;
        }

        if (jniGemmWorker) {
            runJniGemmWorker(workerSize, aliasing);
            return;
        }

        if (jniGemmSequenceWorker) {
            runJniGemmSequenceWorker(workerSize);
            return;
        }

        System.out.println("LU_DIAGNOSTIC");
        System.out.println();
        runSmoke("smoke3", SMOKE3, 3);
        runSmoke("smoke4", SMOKE4, 4);

        if (!runPerf) {
            return;
        }

        if (nativeOnly) {
            int[] sizes = parseSizes(sizesArg);
            System.out.println();
            System.out.println("| Backend | Variant | Size | Best ms | Median ms | Mean ms | Residual | Singular | Notes |");
            System.out.println("|---|---|---:|---:|---:|---:|---:|---|---|");
            for (int n : sizes) {
                NativePerfResult cppDirect = runCppPerfVariant(n, warmup, runs, "direct");
                NativePerfResult cppSafeReuse = runCppPerfVariant(n, warmup, runs, "safe_reuse");
                NativePerfResult cppSafeFresh = runCppPerfVariant(n, warmup, runs, "safe_fresh");
                NativePerfResult cppUnblocked = runCppPerfVariant(n, warmup, runs, "unblocked");
                NativePerfResult jni = runJniPerf(n, warmup, runs);
                System.out.printf(Locale.ROOT,
                    "| C++ | direct | %d | %.3f | %.3f | %.3f | %.3e | %s | %s |%n",
                    n, cppDirect.bestMs(), cppDirect.medianMs(), cppDirect.meanMs(), cppDirect.residual(), cppDirect.singular(), cppDirect.notes());
                System.out.printf(Locale.ROOT,
                    "| C++ | safe_reuse | %d | %.3f | %.3f | %.3f | %.3e | %s | %s |%n",
                    n, cppSafeReuse.bestMs(), cppSafeReuse.medianMs(), cppSafeReuse.meanMs(), cppSafeReuse.residual(), cppSafeReuse.singular(), cppSafeReuse.notes());
                System.out.printf(Locale.ROOT,
                    "| C++ | safe_fresh | %d | %.3f | %.3f | %.3f | %.3e | %s | %s |%n",
                    n, cppSafeFresh.bestMs(), cppSafeFresh.medianMs(), cppSafeFresh.meanMs(), cppSafeFresh.residual(), cppSafeFresh.singular(), cppSafeFresh.notes());
                System.out.printf(Locale.ROOT,
                    "| C++ | unblocked | %d | %.3f | %.3f | %.3f | %.3e | %s | %s |%n",
                    n, cppUnblocked.bestMs(), cppUnblocked.medianMs(), cppUnblocked.meanMs(), cppUnblocked.residual(), cppUnblocked.singular(), cppUnblocked.notes());
                System.out.printf(Locale.ROOT,
                    "| JNI | safe_reuse | %d | %.3f | %.3f | %.3f | %.3e | %s | %s |%n",
                    n, jni.bestMs(), jni.medianMs(), jni.meanMs(), jni.residual(), jni.singular(), jni.notes());
            }
            return;
        }

        if (jniBoundaryOnly) {
            int[] sizes = parseSizes(sizesArg);
            System.out.println();
            System.out.println("| Mode | Size | Status | Detail |");
            System.out.println("|---|---:|---|---|");
            for (int n : sizes) {
                JniBoundaryResult normal = runJniBoundaryCase(n, true);
                JniBoundaryResult noCopyBack = runJniBoundaryCase(n, false);
                System.out.printf(Locale.ROOT, "| copy-back | %d | %s | %s |%n", n, normal.status(), escapePipes(normal.detail()));
                System.out.printf(Locale.ROOT, "| no-copy-back | %d | %s | %s |%n", n, noCopyBack.status(), escapePipes(noCopyBack.detail()));
                if (n > 64) {
                    JniBoundaryResult aliasingGemm = runJniGemmCase(n, true);
                    JniBoundaryResult separatedGemm = runJniGemmCase(n, false);
                    JniBoundaryResult sequenceGemm = runJniGemmSequenceCase(n);
                    System.out.printf(Locale.ROOT, "| gemm-aliasing | %d | %s | %s |%n", n, aliasingGemm.status(), escapePipes(aliasingGemm.detail()));
                    System.out.printf(Locale.ROOT, "| gemm-separated | %d | %s | %s |%n", n, separatedGemm.status(), escapePipes(separatedGemm.detail()));
                    System.out.printf(Locale.ROOT, "| gemm-sequence | %d | %s | %s |%n", n, sequenceGemm.status(), escapePipes(sequenceGemm.detail()));
                }
            }
            return;
        }

        int[] sizes = parseSizes(sizesArg);
        int[] blockSizes = parseSizes(blockSizesArg);
        System.out.println();
        System.out.println("| Variant | Size | Best ms | Median ms | Mean ms | Residual | Singular | Notes |");
        System.out.println("|---|---:|---:|---:|---:|---:|---|---|");
        for (int n : sizes) {
            PerfResult auto = runJavaPerf(n, warmup, runs, null, null, progressInterval);
            PerfResult legacyBlocked = runJavaPerf(n, warmup, runs, 384, 32, progressInterval);
            PerfResult unblocked = runJavaPerf(n, warmup, runs, 999_999, null, progressInterval);
            System.out.printf(Locale.ROOT, "| Java auto policy | %d | %.3f | %.3f | %.3f | %.3e | %s | threshold=%d block=%d |%n",
                n, auto.bestMs(), auto.medianMs(), auto.meanMs(), auto.residual(), auto.singular(),
                auto.blockThreshold(), auto.blockSize());
            System.out.printf(Locale.ROOT, "| Java legacy blocked b=32 | %d | %.3f | %.3f | %.3f | %.3e | %s | threshold=%d block=%d |%n",
                n, legacyBlocked.bestMs(), legacyBlocked.medianMs(), legacyBlocked.meanMs(), legacyBlocked.residual(), legacyBlocked.singular(),
                legacyBlocked.blockThreshold(), legacyBlocked.blockSize());
            System.out.printf(Locale.ROOT, "| Java unblocked | %d | %.3f | %.3f | %.3f | %.3e | %s | threshold=%d |%n",
                n, unblocked.bestMs(), unblocked.medianMs(), unblocked.meanMs(), unblocked.residual(), unblocked.singular(),
                unblocked.blockThreshold());
            PerfResult bestOverride = null;
            for (int blockSize : blockSizes) {
                PerfResult forcedBlocked = runJavaPerf(n, warmup, runs, 1, blockSize, progressInterval);
                System.out.printf(Locale.ROOT, "| Java blocked b=%d | %d | %.3f | %.3f | %.3f | %.3e | %s | threshold=%d block=%d |%n",
                    blockSize, n, forcedBlocked.bestMs(), forcedBlocked.medianMs(), forcedBlocked.meanMs(),
                    forcedBlocked.residual(), forcedBlocked.singular(),
                    forcedBlocked.blockThreshold(), forcedBlocked.blockSize());
                if (bestOverride == null || forcedBlocked.medianMs() < bestOverride.medianMs()) {
                    bestOverride = forcedBlocked;
                }
            }
            if (bestOverride != null) {
                PerfResult baseline = auto;
                double gain = relativeGain(baseline.medianMs(), bestOverride.medianMs());
                String verdict = gain >= medianGainThreshold ? "PASS" : "SKIP";
                System.out.printf(Locale.ROOT,
                    "| Guardrail summary | %d | %.3f | %.3f | %.3f | %.3e | %s | baselineMedian=%.3f candidate=b=%d gain=%.2f%% threshold=%.2f%% verdict=%s |%n",
                    n, bestOverride.bestMs(), bestOverride.medianMs(), bestOverride.meanMs(), bestOverride.residual(),
                    bestOverride.singular(), baseline.medianMs(), bestOverride.blockSize(), gain * 100.0,
                    medianGainThreshold * 100.0, verdict);
            }
        }
    }

    private static void runSmoke(String name, double[] matrix, int n) throws Exception {
        JavaLuResult javaResult = runJavaSmoke(matrix, n);
        NativeLuResult jniResult = runJniSmoke(matrix, n);
        NativeLuResult cppResult = runCppSmoke(name, n);

        System.out.println("SMOKE " + name);
        System.out.println("| Backend | Residual | Singular | Input checksum before | Input checksum after | Pivots |");
        System.out.println("|---|---:|---|---:|---:|---|");
        System.out.printf(Locale.ROOT, "| Java | %.3e | %s | %.3f | %.3f | %s |%n",
            javaResult.residual(), javaResult.singular(), javaResult.inputChecksumBefore(), javaResult.inputChecksumAfter(), javaResult.pivots());
        System.out.printf(Locale.ROOT, "| JNI | %.3e | %s | %.3f | %.3f | %s |%n",
            jniResult.residual(), jniResult.singular(), jniResult.inputChecksumBefore(), jniResult.inputChecksumAfter(), jniResult.pivots());
        System.out.printf(Locale.ROOT, "| C++ | %.3e | %s | %.3f | %.3f | %s |%n",
            cppResult.residual(), cppResult.singular(), cppResult.inputChecksumBefore(), cppResult.inputChecksumAfter(), cppResult.pivots());
        System.out.println();
    }

    private static JavaLuResult runJavaSmoke(double[] input, int n) {
        String previousBackend = System.getProperty("jlc.backend");
        System.setProperty("jlc.backend", "java");
        try {
            double[] original = input.clone();
            Matrix matrix = Matrix.wrap(input.clone(), n, n);
            LUResult result = new LUDecomposition().decompose(matrix);
            String pivots = permutationToString(result.getP());
            return new JavaLuResult(
                result.residualNorm(),
                result.isSingular(),
                checksum(original),
                checksum(result.getL().getRawData()) + checksum(result.getU().getRawData()),
                pivots
            );
        } finally {
            restoreProperty("jlc.backend", previousBackend);
        }
    }

    private static NativeLuResult runJniSmoke(double[] input, int n) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }
        double[] packed = input.clone();
        int[] pivots = new int[n];
        NativeBindings.nativeLuFactor(packed, n, pivots);
        return buildNativeResult(input, packed, pivots, n);
    }

    private static NativeLuResult runCppSmoke(String matrixCase, int n) throws Exception {
        Map<String, String> values = runNativeExecutable(List.of(
            "--algorithm=lu",
            "--matrix=" + matrixCase,
            "--size=" + n,
            "--diag"
        ));
        return new NativeLuResult(
            Double.parseDouble(values.getOrDefault("residual", "NaN")),
            Boolean.parseBoolean(values.getOrDefault("singular", "false")),
            Double.parseDouble(values.getOrDefault("input_checksum_before", "NaN")),
            Double.parseDouble(values.getOrDefault("input_checksum_after", "NaN")),
            values.getOrDefault("pivots", "")
        );
    }

    private static NativePerfResult runCppPerfVariant(int n, int warmup, int runs, String variant) throws Exception {
        List<String> args = new ArrayList<>(List.of(
            "--algorithm=lu",
            "--size=" + n,
            "--warmup=" + warmup,
            "--runs=" + runs
        ));
        Map<String, String> env = new LinkedHashMap<>();
        switch (variant) {
            case "direct" -> env.put("JLC_NATIVE_LU_SAFE_MODE", "direct");
            case "safe_reuse" -> env.put("JLC_NATIVE_LU_SAFE_MODE", "safe_reuse");
            case "safe_fresh" -> env.put("JLC_NATIVE_LU_SAFE_MODE", "safe_fresh");
            case "unblocked" -> {
                env.put("JLC_NATIVE_LU_SAFE_MODE", "direct");
                env.put("JLC_NATIVE_LU_BLOCK_THRESHOLD", "999999");
            }
            default -> throw new IllegalArgumentException("Unknown LU variant: " + variant);
        }
        Map<String, String> values = runNativeExecutable(args, env);
        String notes = String.format(Locale.ROOT,
            "mode=%s block=%s threshold=%s kernel=%s scalarFallback=%s safeUpdates=%s copiedMB=%.2f allocNs=%s copyNs=%s/%s/%s writeNs=%s pivotCount=%s zeroPivot=%s",
            values.getOrDefault("safe_mode", "?"),
            values.getOrDefault("block_size", "?"),
            values.getOrDefault("block_threshold", "?"),
            values.getOrDefault("gemm_kernel", "?"),
            values.getOrDefault("gemm_scalar_fallback", "?"),
            values.getOrDefault("safe_update_count", "0"),
            parseLong(values.getOrDefault("safe_bytes_copied", "0")) / (1024.0 * 1024.0),
            values.getOrDefault("safe_alloc_ns", "0"),
            values.getOrDefault("safe_copy_l21_ns", "0"),
            values.getOrDefault("safe_copy_u12_ns", "0"),
            values.getOrDefault("safe_copy_a22_ns", "0"),
            values.getOrDefault("safe_write_a22_ns", "0"),
            values.getOrDefault("pivot_count", "?"),
            values.getOrDefault("zero_pivot_count", "?"));
        return new NativePerfResult(
            Double.parseDouble(values.getOrDefault("best_ms", "NaN")),
            Double.parseDouble(values.getOrDefault("median_ms", "NaN")),
            Double.parseDouble(values.getOrDefault("mean_ms", "NaN")),
            Double.parseDouble(values.getOrDefault("residual", "NaN")),
            Boolean.parseBoolean(values.getOrDefault("singular", "false")),
            notes
        );
    }

    private static PerfResult runJavaPerf(int n, int warmup, int runs, Integer threshold, Integer blockSize, int progressInterval) {
        String previousBackend = System.getProperty("jlc.backend");
        String previousDiag = System.getProperty("net.faulj.decomposition.lu.diagnostics");
        String previousInterval = System.getProperty("net.faulj.decomposition.lu.progressInterval");
        String previousThreshold = System.getProperty("net.faulj.decomposition.lu.blockThreshold");
        String previousBlockSize = System.getProperty("net.faulj.decomposition.lu.blockSize");
        try {
            System.setProperty("jlc.backend", "java");
            System.setProperty("net.faulj.decomposition.lu.diagnostics", "true");
            System.setProperty("net.faulj.decomposition.lu.progressInterval", Integer.toString(progressInterval));
            setOrClearProperty("net.faulj.decomposition.lu.blockThreshold", threshold);
            setOrClearProperty("net.faulj.decomposition.lu.blockSize", blockSize);

            double[] seed = randomData(n, 17_000L + n);
            Matrix primeMatrix = Matrix.wrap(seed.clone(), n, n);
            new LUDecomposition().decompose(primeMatrix);

            double bestMs = Double.POSITIVE_INFINITY;
            double totalMs = 0.0;
            double[] samples = new double[runs];
            double residual = Double.NaN;
            boolean singular = false;
            for (int i = 0; i < warmup + runs; i++) {
                Matrix matrix = Matrix.wrap(seed.clone(), n, n);
                long start = System.nanoTime();
                LUResult result = new LUDecomposition().decompose(matrix);
                double elapsedMs = (System.nanoTime() - start) / 1e6;
                if (i >= warmup) {
                    bestMs = Math.min(bestMs, elapsedMs);
                    samples[i - warmup] = elapsedMs;
                    totalMs += elapsedMs;
                    residual = result.residualNorm();
                    singular = result.isSingular();
                }
            }
            double medianMs = median(samples);
            double meanMs = totalMs / runs;
            int effectiveThreshold = threshold != null ? threshold : effectiveBlockThreshold();
            int effectiveBlockSize = blockSize != null ? blockSize : effectiveBlockSize(n);
            return new PerfResult(bestMs, medianMs, meanMs, residual, singular, effectiveThreshold, effectiveBlockSize);
        } finally {
            restoreProperty("jlc.backend", previousBackend);
            restoreProperty("net.faulj.decomposition.lu.diagnostics", previousDiag);
            restoreProperty("net.faulj.decomposition.lu.progressInterval", previousInterval);
            restoreProperty("net.faulj.decomposition.lu.blockThreshold", previousThreshold);
            restoreProperty("net.faulj.decomposition.lu.blockSize", previousBlockSize);
        }
    }

    private static NativePerfResult runJniPerf(int n, int warmup, int runs) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }

        double[] seed = randomData(n, 71_000L + n);
        for (int i = 0; i < warmup; i++) {
            double[] lu = seed.clone();
            int[] pivots = new int[n];
            NativeBindings.nativeLuFactor(lu, n, pivots);
        }

        double[] samples = new double[runs];
        double totalMs = 0.0;
        double bestMs = Double.POSITIVE_INFINITY;
        double residual = Double.NaN;
        boolean singular = false;
        for (int i = 0; i < runs; i++) {
            double[] lu = seed.clone();
            int[] pivots = new int[n];
            long start = System.nanoTime();
            NativeBindings.nativeLuFactor(lu, n, pivots);
            double elapsedMs = (System.nanoTime() - start) / 1e6;
            samples[i] = elapsedMs;
            totalMs += elapsedMs;
            bestMs = Math.min(bestMs, elapsedMs);
            residual = luResidual(seed, lu, pivots, n);
            singular = hasZeroPivot(lu, n);
        }
        String notes = "JNI wrapper timing only; detailed LU/GEMM profile taken from standalone C++ path";
        return new NativePerfResult(bestMs, median(samples), totalMs / runs, residual, singular, notes);
    }

    private static void runJniBoundaryWorker(int n, boolean copyBack) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }

        double[] matrix = randomData(n, 91_000L + n);
        int[] pivots = new int[n];
        System.out.println("jni_boundary.before.n=" + n);
        System.out.println("jni_boundary.before.matrixLength=" + matrix.length);
        System.out.println("jni_boundary.before.pivotLength=" + pivots.length);
        System.out.println("jni_boundary.before.expectedElements=" + (n * n));
        System.out.println("jni_boundary.before.checksum=" + checksum(matrix));
        System.out.println("jni_boundary.before.copyBack=" + copyBack);
        try {
            NativeBindings.nativeLuFactorDebug(matrix, n, pivots, copyBack);
            System.out.println("jni_boundary.after.callStatus=success");
            System.out.println("jni_boundary.after.checksum=" + checksum(matrix));
            System.out.println("jni_boundary.after.firstNonFinite=" + firstNonFiniteIndex(matrix));
            System.out.println("jni_boundary.after.pivotMin=" + minValue(pivots));
            System.out.println("jni_boundary.after.pivotMax=" + maxValue(pivots));
            if (copyBack) {
                System.out.println("jni_boundary.after.residual=" + luResidual(randomData(n, 91_000L + n), matrix, pivots, n));
            }
        } catch (Throwable t) {
            System.out.println("jni_boundary.after.callStatus=throw");
            System.out.println("jni_boundary.after.throwType=" + t.getClass().getName());
            System.out.println("jni_boundary.after.throwMessage=" + String.valueOf(t.getMessage()));
            throw t;
        }
    }

    private static void runJniGemmWorker(int n, boolean aliasing) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }
        if (n <= 64) {
            throw new IllegalArgumentException("JNI GEMM worker requires n > 64");
        }

        int block = 64;
        int panelStart = 0;
        int panelEnd = block;
        int trailing = n - panelEnd;
        double[] matrix = randomData(n, 93_000L + n);

        int l21Offset = panelEnd * n + panelStart;
        int u12Offset = panelStart * n + panelEnd;
        int a22Offset = panelEnd * n + panelEnd;
        System.out.println("jni_gemm.before.n=" + n);
        System.out.println("jni_gemm.before.aliasing=" + aliasing);
        System.out.println("jni_gemm.before.block=" + block);
        System.out.println("jni_gemm.before.trailing=" + trailing);
        System.out.println("jni_gemm.before.l21Offset=" + l21Offset);
        System.out.println("jni_gemm.before.u12Offset=" + u12Offset);
        System.out.println("jni_gemm.before.a22Offset=" + a22Offset);

        try {
            if (aliasing) {
                double before = checksum(matrix);
                NativeBindings.nativeGemmStrided(
                    matrix, l21Offset, n, trailing, block, 0,
                    matrix, u12Offset, n, block, trailing, 0,
                    matrix, a22Offset, n, trailing, trailing, 0,
                    -1.0, 1.0, 0, 0
                );
                System.out.println("jni_gemm.after.callStatus=success");
                System.out.println("jni_gemm.after.checksumBefore=" + before);
                System.out.println("jni_gemm.after.checksumAfter=" + checksum(matrix));
                System.out.println("jni_gemm.after.firstNonFinite=" + firstNonFiniteIndex(matrix));
                return;
            }

            double[] a = new double[trailing * block];
            double[] b = new double[block * trailing];
            double[] c = new double[trailing * trailing];
            copyWindow(matrix, n, panelEnd, panelStart, trailing, block, a, block);
            copyWindow(matrix, n, panelStart, panelEnd, block, trailing, b, trailing);
            copyWindow(matrix, n, panelEnd, panelEnd, trailing, trailing, c, trailing);
            double before = checksum(c);
            NativeBindings.nativeGemmStrided(
                a, 0, block, trailing, block, 0,
                b, 0, trailing, block, trailing, 0,
                c, 0, trailing, trailing, trailing, 0,
                -1.0, 1.0, 0, 0
            );
            System.out.println("jni_gemm.after.callStatus=success");
            System.out.println("jni_gemm.after.checksumBefore=" + before);
            System.out.println("jni_gemm.after.checksumAfter=" + checksum(c));
            System.out.println("jni_gemm.after.firstNonFinite=" + firstNonFiniteIndex(c));
        } catch (Throwable t) {
            System.out.println("jni_gemm.after.callStatus=throw");
            System.out.println("jni_gemm.after.throwType=" + t.getClass().getName());
            System.out.println("jni_gemm.after.throwMessage=" + String.valueOf(t.getMessage()));
            throw t;
        }
    }

    private static void runJniGemmSequenceWorker(int n) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }
        if (n <= 128) {
            throw new IllegalArgumentException("JNI GEMM sequence worker requires n > 128");
        }

        int block = 64;
        double[] matrix = randomData(n, 95_000L + n);
        System.out.println("jni_gemm_sequence.before.n=" + n);
        System.out.println("jni_gemm_sequence.before.block=" + block);
        System.out.println("jni_gemm_sequence.before.checksum=" + checksum(matrix));
        try {
            runAliasingUpdate(matrix, n, 0, block, "first");
            runAliasingUpdate(matrix, n, block, block, "second");
            System.out.println("jni_gemm_sequence.after.callStatus=success");
            System.out.println("jni_gemm_sequence.after.checksum=" + checksum(matrix));
            System.out.println("jni_gemm_sequence.after.firstNonFinite=" + firstNonFiniteIndex(matrix));
        } catch (Throwable t) {
            System.out.println("jni_gemm_sequence.after.callStatus=throw");
            System.out.println("jni_gemm_sequence.after.throwType=" + t.getClass().getName());
            System.out.println("jni_gemm_sequence.after.throwMessage=" + String.valueOf(t.getMessage()));
            throw t;
        }
    }

    private static JniBoundaryResult runJniBoundaryCase(int n, boolean copyBack) throws Exception {
        return runWorkerCase(n, List.of("--jni-boundary-worker", "--copyBack=" + copyBack));
    }

    private static JniBoundaryResult runJniGemmCase(int n, boolean aliasing) throws Exception {
        return runWorkerCase(n, List.of("--jni-gemm-worker", "--aliasing=" + aliasing));
    }

    private static JniBoundaryResult runJniGemmSequenceCase(int n) throws Exception {
        return runWorkerCase(n, List.of("--jni-gemm-sequence-worker"));
    }

    private static JniBoundaryResult runWorkerCase(int n, List<String> workerArgs) throws Exception {
        List<String> command = new ArrayList<>();
        command.add(Paths.get(System.getProperty("java.home"), "bin", "java").toString());
        command.add("--add-modules=jdk.incubator.vector");
        command.add("--enable-preview");
        String nativeLibPath = System.getProperty("jlc.native.lib.path");
        if (nativeLibPath != null && !nativeLibPath.isBlank()) {
            command.add("-Djlc.native.lib.path=" + nativeLibPath);
        }
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add(LuDiagnosticRunner.class.getName());
        command.addAll(workerArgs);
        command.add("--size=" + n);

        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        List<String> output = new ArrayList<>();
        Thread readerThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    synchronized (output) {
                        output.add(line);
                    }
                }
            } catch (Exception ignored) {
            }
        });
        readerThread.setDaemon(true);
        readerThread.start();
        boolean finished = process.waitFor(30, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            process.waitFor(5, TimeUnit.SECONDS);
            readerThread.join(2000);
            synchronized (output) {
                return new JniBoundaryResult("TIMEOUT", tail(output));
            }
        }
        readerThread.join(2000);
        int exit = process.exitValue();
        String detail = tail(output);
        return new JniBoundaryResult(exit == 0 ? "OK" : "FAIL(" + exit + ")", detail);
    }

    private static NativeLuResult buildNativeResult(double[] original, double[] packed, int[] pivots, int n) {
        double residual = luResidual(original, packed, pivots, n);
        boolean singular = false;
        for (int i = 0; i < n; i++) {
            singular |= Math.abs(packed[i * n + i]) < 1e-12;
        }
        return new NativeLuResult(
            residual,
            singular,
            checksum(original),
            checksum(packed),
            Arrays.toString(pivots)
        );
    }

    private static Map<String, String> runNativeExecutable(List<String> args) throws Exception {
        return runNativeExecutable(args, Map.of());
    }

    private static Map<String, String> runNativeExecutable(List<String> args, Map<String, String> envOverrides) throws Exception {
        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }
        List<String> command = new ArrayList<>();
        command.add(executable);
        command.addAll(args);
        ProcessBuilder processBuilder = new ProcessBuilder(command).redirectErrorStream(true);
        processBuilder.environment().putAll(envOverrides);
        Process process = processBuilder.start();
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
            throw new IllegalStateException("Native executable failed for args " + args);
        }
        return values;
    }

    private static int[] parseSizes(String raw) {
        String[] parts = raw.split(",");
        int[] sizes = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            sizes[i] = Integer.parseInt(parts[i].trim());
        }
        return sizes;
    }

    private static String permutationToString(PermutationVector permutation) {
        int[] out = new int[permutation.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = permutation.get(i);
        }
        return Arrays.toString(out);
    }

    private static double[] randomData(int n, long seed) {
        Random random = new Random(seed);
        double[] data = new double[n * n];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return data;
    }

    private static double checksum(double[] values) {
        double sum = 0.0;
        for (int i = 0; i < values.length; i++) {
            sum = Math.fma(values[i], i + 1.0, sum);
        }
        return sum;
    }

    private static double luResidual(double[] original, double[] packedLu, int[] pivots, int n) {
        double[] permuted = original.clone();
        for (int i = 0; i < n; i++) {
            int pivot = pivots[i];
            if (pivot >= 0 && pivot < n && pivot != i) {
                swapRows(permuted, n, i, pivot);
            }
        }
        double[] reconstructed = new double[n * n];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                double sum = 0.0;
                int limit = Math.min(row, col);
                for (int k = 0; k <= limit; k++) {
                    double l = row == k ? 1.0 : (row > k ? packedLu[row * n + k] : 0.0);
                    double u = k <= col ? packedLu[k * n + col] : 0.0;
                    sum = Math.fma(l, u, sum);
                }
                reconstructed[row * n + col] = sum;
            }
        }
        double delta = 0.0;
        double base = 0.0;
        for (int i = 0; i < original.length; i++) {
            double diff = permuted[i] - reconstructed[i];
            delta = Math.fma(diff, diff, delta);
            base = Math.fma(permuted[i], permuted[i], base);
        }
        return Math.sqrt(delta) / Math.sqrt(base);
    }

    private static void swapRows(double[] data, int n, int rowA, int rowB) {
        int offsetA = rowA * n;
        int offsetB = rowB * n;
        for (int col = 0; col < n; col++) {
            double tmp = data[offsetA + col];
            data[offsetA + col] = data[offsetB + col];
            data[offsetB + col] = tmp;
        }
    }

    private static void copyWindow(double[] src, int srcLd, int rowStart, int colStart,
                                   int rows, int cols, double[] dst, int dstLd) {
        for (int row = 0; row < rows; row++) {
            int srcBase = (rowStart + row) * srcLd + colStart;
            int dstBase = row * dstLd;
            System.arraycopy(src, srcBase, dst, dstBase, cols);
        }
    }

    private static void runAliasingUpdate(double[] matrix, int n, int panelStart, int block, String label) {
        int panelEnd = panelStart + block;
        int trailing = n - panelEnd;
        int l21Offset = panelEnd * n + panelStart;
        int u12Offset = panelStart * n + panelEnd;
        int a22Offset = panelEnd * n + panelEnd;
        System.out.println("jni_gemm_sequence." + label + ".panelStart=" + panelStart);
        System.out.println("jni_gemm_sequence." + label + ".panelEnd=" + panelEnd);
        System.out.println("jni_gemm_sequence." + label + ".trailing=" + trailing);
        System.out.println("jni_gemm_sequence." + label + ".checksumBefore=" + checksum(matrix));
        NativeBindings.nativeGemmStrided(
            matrix, l21Offset, n, trailing, block, 0,
            matrix, u12Offset, n, block, trailing, 0,
            matrix, a22Offset, n, trailing, trailing, 0,
            -1.0, 1.0, 0, 0
        );
        System.out.println("jni_gemm_sequence." + label + ".checksumAfter=" + checksum(matrix));
        System.out.println("jni_gemm_sequence." + label + ".firstNonFinite=" + firstNonFiniteIndex(matrix));
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    private static void setOrClearProperty(String key, Integer value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, Integer.toString(value));
        }
    }

    private static double median(double[] samples) {
        double[] sorted = samples.clone();
        Arrays.sort(sorted);
        int middle = sorted.length / 2;
        if ((sorted.length & 1) == 1) {
            return sorted[middle];
        }
        return 0.5 * (sorted[middle - 1] + sorted[middle]);
    }

    private static double relativeGain(double baselineMs, double candidateMs) {
        if (!(baselineMs > 0.0) || !(candidateMs > 0.0)) {
            return Double.NaN;
        }
        return (baselineMs - candidateMs) / baselineMs;
    }

    private static int firstNonFiniteIndex(double[] values) {
        for (int i = 0; i < values.length; i++) {
            if (!Double.isFinite(values[i])) {
                return i;
            }
        }
        return -1;
    }

    private static int minValue(int[] values) {
        int min = Integer.MAX_VALUE;
        for (int value : values) {
            min = Math.min(min, value);
        }
        return values.length == 0 ? 0 : min;
    }

    private static int maxValue(int[] values) {
        int max = Integer.MIN_VALUE;
        for (int value : values) {
            max = Math.max(max, value);
        }
        return values.length == 0 ? 0 : max;
    }

    private static String tail(List<String> lines) {
        int start = Math.max(0, lines.size() - 16);
        return String.join(" ; ", lines.subList(start, lines.size()));
    }

    private static String escapePipes(String text) {
        return text.replace("|", "\\|");
    }

    private static boolean hasZeroPivot(double[] packedLu, int n) {
        for (int i = 0; i < n; i++) {
            if (Math.abs(packedLu[i * n + i]) < 1e-12) {
                return true;
            }
        }
        return false;
    }

    private static int effectiveBlockThreshold() {
        String configured = System.getProperty("net.faulj.decomposition.lu.blockThreshold");
        if (configured != null && !configured.isBlank()) {
            return Integer.parseInt(configured.trim());
        }
        return 384;
    }

    private static int effectiveBlockSize(int n) {
        String configured = System.getProperty("net.faulj.decomposition.lu.blockSize");
        if (configured != null && !configured.isBlank()) {
            return Integer.parseInt(configured.trim());
        }
        if (n >= 1024) {
            return 80;
        }
        if (n >= 512) {
            return 64;
        }
        return 32;
    }

    private static long parseLong(String raw) {
        try {
            return Long.parseLong(raw);
        } catch (NumberFormatException ignored) {
            return 0L;
        }
    }

    private record JavaLuResult(double residual, boolean singular, double inputChecksumBefore,
                                double inputChecksumAfter, String pivots) {
    }

    private record NativeLuResult(double residual, boolean singular, double inputChecksumBefore,
                                  double inputChecksumAfter, String pivots) {
    }

    private record PerfResult(double bestMs, double medianMs, double meanMs, double residual,
                              boolean singular, int blockThreshold, int blockSize) {
    }

    private record NativePerfResult(double bestMs, double medianMs, double meanMs, double residual,
                                    boolean singular, String notes) {
    }

    private record JniBoundaryResult(String status, String detail) {
    }
}
