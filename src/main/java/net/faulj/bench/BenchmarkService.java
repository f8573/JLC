package net.faulj.bench;

import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.matrix.Matrix;
import net.faulj.compute.BLAS3Kernels;
import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.OptimizedBLAS3;
import net.faulj.compute.GemmDispatch;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.concurrent.TimeUnit;

@Service
public class BenchmarkService {
    private static final double QR_COEFF = 4.0 / 3.0;
    private static final double HESS_COEFF = 10.0 / 3.0;

    public Map<String, Object> runBenchmark() {
        int[] sizes = new int[] {128, 256, 512, 1024, 2048};

        double maxQr = 0.0;
        double maxHess = 0.0;
        List<Map<String, Object>> iterations = new ArrayList<>();

        // Parameters to mitigate slowdowns caused by JIT/GC and measurement noise
        final int warmupRuns = 1;   // warmup before measured runs to let JIT optimize
        final int measuredRuns = 3; // number of measured repetitions per algorithm/size
        final int totalMeasuredRuns = sizes.length * 2 * measuredRuns;
        int measuredRunCounter = 0;

        for (int n : sizes) {
            double flops = QR_COEFF * n * (double)n * n;
            // Householder QR: warmup + measured runs
            for (int run = 0; run < warmupRuns + measuredRuns; run++) {
                Matrix A = randomMatrix(n, n, 42 + run);
                if (run < warmupRuns && run == 0) {
                    HouseholderQR.decompose(A); // warmup
                    continue;
                }
                long memBefore = usedMemoryBytes();
                long gcCountBefore = getGcCount();
                long gcTimeBefore = getGcTimeMs();
                long t0 = System.nanoTime();
                HouseholderQR.factorize(A);
                long t1 = System.nanoTime();
                long memAfter = usedMemoryBytes();
                long gcCountAfter = getGcCount();
                long gcTimeAfter = getGcTimeMs();
                double elapsed = (t1 - t0) / 1e9;
                double gflops = flops / elapsed / 1e9;
                measuredRunCounter++;
                double progress = measuredRunCounter / (double) totalMeasuredRuns;
                Map<String, Object> qrEntry = new LinkedHashMap<>();
                qrEntry.put("algorithm", "HouseholderQR");
                qrEntry.put("n", n);
                qrEntry.put("runIndex", run - warmupRuns + 1);
                qrEntry.put("elapsedSeconds", elapsed);
                qrEntry.put("flops", flops);
                qrEntry.put("gflops", gflops);
                qrEntry.put("memBeforeBytes", memBefore);
                qrEntry.put("memAfterBytes", memAfter);
                qrEntry.put("memDeltaBytes", memAfter - memBefore);
                qrEntry.put("gcCountDelta", gcCountAfter - gcCountBefore);
                qrEntry.put("gcTimeMsDelta", gcTimeAfter - gcTimeBefore);
                qrEntry.put("iterationIndex", measuredRunCounter);
                qrEntry.put("totalIterations", totalMeasuredRuns);
                qrEntry.put("progress", progress);
                iterations.add(qrEntry);
                System.out.printf(
                    "[BENCH] %d/%d (%.1f%%) QR n=%d run=%d elapsed=%.6fs flops=%.0f gflops=%.3f mem=%.2fMB (Δ%.2fMB) gc=%d/%dms\n",
                    measuredRunCounter, totalMeasuredRuns, progress * 100.0,
                    n, run - warmupRuns + 1, elapsed, flops, gflops,
                    memAfter / (1024.0 * 1024.0),
                    (memAfter - memBefore) / (1024.0 * 1024.0),
                    (gcCountAfter - gcCountBefore),
                    (gcTimeAfter - gcTimeBefore));
                if (gflops > maxQr) maxQr = gflops;
                // Hint GC and briefly sleep to reduce interference accumulation
                try { System.gc(); Thread.sleep(50); } catch (InterruptedException ignored) {}
            }

            flops = HESS_COEFF * n * (double)n * n;
            // Hessenberg: warmup + measured runs
            for (int run = 0; run < warmupRuns + measuredRuns; run++) {
                Matrix B = randomMatrix(n, n, 123 + run);
                if (run < warmupRuns) {
                    HessenbergReduction.decompose(B);
                    continue;
                }
                long memBefore = usedMemoryBytes();
                long gcCountBefore = getGcCount();
                long gcTimeBefore = getGcTimeMs();
                long t0 = System.nanoTime();
                HessenbergReduction.reduceToHessenberg(B);
                long t1 = System.nanoTime();
                long memAfter = usedMemoryBytes();
                long gcCountAfter = getGcCount();
                long gcTimeAfter = getGcTimeMs();
                double elapsed = (t1 - t0) / 1e9;
                double gflopsH = flops / elapsed / 1e9;
                measuredRunCounter++;
                double progress = measuredRunCounter / (double) totalMeasuredRuns;
                Map<String, Object> hessEntry = new LinkedHashMap<>();
                hessEntry.put("algorithm", "HessenbergReduction");
                hessEntry.put("n", n);
                hessEntry.put("runIndex", run - warmupRuns + 1);
                hessEntry.put("elapsedSeconds", elapsed);
                hessEntry.put("flops", flops);
                hessEntry.put("gflops", gflopsH);
                hessEntry.put("memBeforeBytes", memBefore);
                hessEntry.put("memAfterBytes", memAfter);
                hessEntry.put("memDeltaBytes", memAfter - memBefore);
                hessEntry.put("gcCountDelta", gcCountAfter - gcCountBefore);
                hessEntry.put("gcTimeMsDelta", gcTimeAfter - gcTimeBefore);
                hessEntry.put("iterationIndex", measuredRunCounter);
                hessEntry.put("totalIterations", totalMeasuredRuns);
                hessEntry.put("progress", progress);
                iterations.add(hessEntry);
                System.out.printf(
                    "[BENCH] %d/%d (%.1f%%) HESS n=%d run=%d elapsed=%.6fs flops=%.0f gflops=%.3f mem=%.2fMB (Δ%.2fMB) gc=%d/%dms\n",
                    measuredRunCounter, totalMeasuredRuns, progress * 100.0,
                    n, run - warmupRuns + 1, elapsed, flops, gflopsH,
                    memAfter / (1024.0 * 1024.0),
                    (memAfter - memBefore) / (1024.0 * 1024.0),
                    (gcCountAfter - gcCountBefore),
                    (gcTimeAfter - gcTimeBefore));
                if (gflopsH > maxHess) maxHess = gflopsH;
                try { System.gc(); Thread.sleep(50); } catch (InterruptedException ignored) {}
            }
        }

        Map<String, Object> cpu = new HashMap<>();
        cpu.put("name", "LocalBenchmark");
        cpu.put("gflops", Math.max(maxQr, maxHess));
        cpu.put("state", "online");

        Map<String, Object> out = new HashMap<>();
        out.put("status", "ONLINE");
        out.put("cpu", cpu);
        out.put("iterations", iterations);
        return out;
    }

        public Map<String, Object> runGemm(int n, int iterations) {
            final int reps = Math.max(1, iterations);
            try {
                List<Map<String, Object>> results = new ArrayList<>();

                // Prepare a dispatch policy that prefers CPU BLAS3/parallel kernels for peak throughput
                DispatchPolicy policy = DispatchPolicy.builder()
                        .enableCuda(false)
                        .enableBlas3(true)
                        .enableParallel(true)
                        .parallelism(Math.max(1, Runtime.getRuntime().availableProcessors()))
                        .build();

                // Pre-allocate matrices once and operate on raw arrays using BLAS3 direct kernel
                Matrix A = randomMatrix(n, n, 42L);
                Matrix B = randomMatrix(n, n, 4242L);
                Matrix C = Matrix.zero(n, n);

                double[] ad = A.getRawData();
                double[] bd = B.getRawData();
                double[] cd = C.getRawData();

                // Use a direct packed BLAS3 path with a reasonable block size (matches tests)
                final int blockSize = 64;

                // Warmup: more iterations to stabilize JIT and CPU frequency
                final int warmupRuns = 10;
                for (int w = 0; w < warmupRuns; w++) {
                    BLAS3Kernels.gemmStrided(ad, 0, n, bd, 0, n, cd, 0, n, n, n, n, 1.0, 0.0, blockSize);
                }

                // Short pause to let frequency and caches settle
                try { Thread.sleep(50); } catch (InterruptedException ignored) {}

                // Measured runs: batch-time the loop to amortize dispatch/pool overhead
                final int measuredRuns = Math.max(reps, 20);
                long t0 = System.nanoTime();
                for (int i = 0; i < measuredRuns; i++) {
                    // Note: beta=0.0 causes gemmStrided to zero C internally, no need for extra fill
                    BLAS3Kernels.gemmStrided(ad, 0, n, bd, 0, n, cd, 0, n, n, n, n, 1.0, 0.0, blockSize);
                }
                long t1 = System.nanoTime();

                double totalSeconds = (t1 - t0) / 1e9;
                double flops = 2.0 * (double)n * n * n;
                double avgSeconds = totalSeconds / (double) measuredRuns;
                double gflopsAvg = avgSeconds > 0 ? (flops / avgSeconds) / 1e9 : 0.0;

                // Record averaged result entries (replicate per requested iteration count)
                for (int i = 0; i < reps; i++) {
                    Map<String, Object> row = new LinkedHashMap<>();
                    row.put("operation", "GEMM(single-thread)");
                    row.put("n", n);
                    row.put("iteration", i + 1);
                    row.put("ms", (long) Math.round(avgSeconds * 1000.0));
                    row.put("flops", flops);
                    row.put("flopsPerSec", avgSeconds > 0 ? (flops / avgSeconds) : 0.0);
                    results.add(row);
                }

                // Diagnostic metadata
                Map<String, Object> diagInfo = new LinkedHashMap<>();
                try {
                    int vecLen = jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();
                    diagInfo.put("vectorLength", vecLen);
                } catch (Throwable ignored) {}
                GemmDispatch.BlockSizes bs = GemmDispatch.computeBlockSizes();
                diagInfo.put("blockSizes", bs.toString());
                diagInfo.put("requestedBlockSize", blockSize);
                diagInfo.put("measuredRuns", measuredRuns);
                diagInfo.put("warmupRuns", warmupRuns);

                // Also run a parallel measurement using OptimizedBLAS3 (default parallel policy)
                int availableThreads = Math.max(1, Runtime.getRuntime().availableProcessors());
                DispatchPolicy parallelPolicy = DispatchPolicy.builder()
                        .enableCuda(false)
                        .enableBlas3(true)
                        .enableParallel(true)
                        .parallelism(availableThreads)
                        .build();

                // Warmup for parallel path
                for (int w = 0; w < Math.min(5, warmupRuns); w++) {
                    OptimizedBLAS3.gemm(A, B, C, 1.0, 0.0, parallelPolicy);
                }

                // Measured parallel runs (batch timed)
                int measuredParallel = Math.max(reps, 10);
                long tp0 = System.nanoTime();
                for (int i = 0; i < measuredParallel; i++) {
                    OptimizedBLAS3.gemm(A, B, C, 1.0, 0.0, parallelPolicy);
                }
                long tp1 = System.nanoTime();
                double totalSecondsParallel = (tp1 - tp0) / 1e9;
                double avgSecondsParallel = totalSecondsParallel / (double) measuredParallel;
                double gflopsParallel = avgSecondsParallel > 0 ? (flops / avgSecondsParallel) / 1e9 : 0.0;

                for (int i = 0; i < reps; i++) {
                    Map<String, Object> prow = new LinkedHashMap<>();
                    prow.put("operation", "GEMM(parallel)");
                    prow.put("n", n);
                    prow.put("iteration", i + 1);
                    prow.put("ms", (long) Math.round(avgSecondsParallel * 1000.0));
                    prow.put("flops", flops);
                    prow.put("flopsPerSec", avgSecondsParallel > 0 ? (flops / avgSecondsParallel) : 0.0);
                    results.add(prow);
                }

                diagInfo.put("parallelThreads", availableThreads);
                diagInfo.put("gflopsParallel", gflopsParallel);

                Map<String, Object> cpu = new LinkedHashMap<>();
                cpu.put("name", "DiagnosticGEMM");
                cpu.put("gflops", gflopsParallel);
                cpu.put("state", "online");
                cpu.put("queuedJobs", 0);
                cpu.put("benchmark", Map.of(
                    "workload", "GEMM",
                    "size", n,
                    "iterations", reps,
                    "results", results,
                    "info", diagInfo
                ));

                Map<String, Object> out = new LinkedHashMap<>();
                out.put("status", "ONLINE");
                out.put("cpu", cpu);
                out.put("iterations", results);
                return out;
            } catch (Exception ex) {
                throw new RuntimeException("Failed to run GEMM benchmark: " + ex.getMessage(), ex);
            }
        }

    private Matrix randomMatrix(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        double[] a = new double[rows * cols];
        for (int i = 0; i < a.length; i++) a[i] = rnd.nextDouble() - 0.5;
        return Matrix.wrap(a, rows, cols);
    }

    private static long usedMemoryBytes() {
        Runtime rt = Runtime.getRuntime();
        return rt.totalMemory() - rt.freeMemory();
    }

    private static long getGcCount() {
        long count = 0;
        for (GarbageCollectorMXBean bean : ManagementFactory.getGarbageCollectorMXBeans()) {
            long c = bean.getCollectionCount();
            if (c > 0) {
                count += c;
            }
        }
        return count;
    }

    private static long getGcTimeMs() {
        long time = 0;
        for (GarbageCollectorMXBean bean : ManagementFactory.getGarbageCollectorMXBeans()) {
            long t = bean.getCollectionTime();
            if (t > 0) {
                time += t;
            }
        }
        return time;
    }

    public Map<String, Object> runDiagnostic512(int iterations) {
        final int reps = Math.max(1, iterations);
        final int n = 512;
        Path csvPath = null;
        Path logPath = null;
        try {
            csvPath = Files.createTempFile("diagnostic512-", ".csv");
            logPath = Files.createTempFile("diagnostic512-runner-", ".log");

            List<String> cmd = buildDiagnosticRunnerCommand(reps, csvPath);
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            pb.redirectOutput(logPath.toFile());

            Process process = pb.start();

            int expectedLines = (reps * 13) + 1; // header + 13 ops per iteration
            long deadline = System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(20);
            boolean readyByCsv = false;
            while (System.currentTimeMillis() < deadline) {
                if (!process.isAlive()) {
                    break;
                }
                int lineCount = countLines(csvPath);
                if (lineCount >= expectedLines) {
                    readyByCsv = true;
                    break;
                }
                try { Thread.sleep(500); } catch (InterruptedException ignored) {}
            }

            if (readyByCsv && process.isAlive()) {
                // Give the runner a short grace period to exit cleanly after writing all rows.
                process.waitFor(30, TimeUnit.SECONDS);
                if (process.isAlive()) {
                    process.destroy();
                    process.waitFor(5, TimeUnit.SECONDS);
                    if (process.isAlive()) {
                        process.destroyForcibly();
                    }
                }
            }

            if (process.isAlive()) {
                process.destroyForcibly();
                throw new IllegalStateException("Diagnostic512Runner timed out after 20 minutes");
            }
            int exit = process.exitValue();
            String output = readFileSafe(logPath);

            List<Map<String, Object>> rows = parseDiagnosticCsv(csvPath);
            if (exit != 0 && rows.isEmpty()) {
                throw new IllegalStateException("Diagnostic512Runner failed (exit=" + exit + "):\n" + tail(output, 2000));
            }

            double totalFlops = 0.0;
            double totalSeconds = 0.0;
            for (Map<String, Object> row : rows) {
                Object flopsObj = row.get("flops");
                Object msObj = row.get("ms");
                if (!(flopsObj instanceof Number) || !(msObj instanceof Number)) continue;
                double flops = ((Number) flopsObj).doubleValue();
                long ms = ((Number) msObj).longValue();
                if (ms <= 0) continue;
                totalFlops += flops;
                totalSeconds += ms / 1000.0;
            }

            double gflops = totalSeconds > 0 ? (totalFlops / totalSeconds) / 1e9 : 0.0;
            Map<String, Object> cpu = new LinkedHashMap<>();
            cpu.put("name", "Diagnostic512");
            cpu.put("gflops", gflops);
            cpu.put("state", "online");
            cpu.put("queuedJobs", 0);
            cpu.put("benchmark", Map.of(
                    "workload", "Diagnostic512Runner (isolated JVM)",
                    "size", n,
                    "iterations", reps,
                    "results", rows
            ));

            Map<String, Object> out = new LinkedHashMap<>();
            out.put("status", "ONLINE");
            out.put("cpu", cpu);
            out.put("iterations", rows);
            out.put("runnerExitCode", exit);
            return out;
        } catch (Exception ex) {
            throw new RuntimeException("Failed to run Diagnostic512 in isolated process: " + ex.getMessage(), ex);
        } finally {
            if (csvPath != null) {
                try { Files.deleteIfExists(csvPath); } catch (IOException ignored) {}
            }
            if (logPath != null) {
                try { Files.deleteIfExists(logPath); } catch (IOException ignored) {}
            }
        }
    }

    private static List<String> buildDiagnosticRunnerCommand(int iterations, Path csvPath) {
        String javaHome = System.getProperty("java.home");
        String javaBin = javaHome + File.separator + "bin" + File.separator
                + (System.getProperty("os.name", "").toLowerCase().contains("win") ? "java.exe" : "java");
        String classpath = System.getProperty("java.class.path");
        List<String> cmd = new ArrayList<>();
        cmd.add(javaBin);
        cmd.add("--enable-preview");
        cmd.add("--add-modules=jdk.incubator.vector");
        cmd.add("-cp");
        cmd.add(classpath);
        cmd.add("net.faulj.bench.Diagnostic512Runner");
        cmd.add("--iterations=" + iterations);
        cmd.add("--output=" + csvPath.toAbsolutePath());
        return cmd;
    }

    private static List<Map<String, Object>> parseDiagnosticCsv(Path csvPath) throws IOException {
        List<Map<String, Object>> rows = new ArrayList<>();
        if (!Files.exists(csvPath)) return rows;

        List<String> lines = Files.readAllLines(csvPath, StandardCharsets.UTF_8);
        for (String line : lines) {
            if (line == null || line.isBlank()) continue;
            if (line.startsWith("operation,")) continue;
            String[] parts = line.split(",", 7);
            if (parts.length < 6) continue;

            Map<String, Object> row = new LinkedHashMap<>();
            row.put("operation", parts[0]);
            row.put("n", parseInt(parts[1], 0));
            row.put("iteration", parseInt(parts[2], 0));
            row.put("ms", parseLong(parts[3], 0L));
            row.put("flops", parseDouble(parts[4], 0.0));
            row.put("flopsPerSec", parseDouble(parts[5], 0.0));
            row.put("info", parts.length >= 7 ? parts[6] : "");
            rows.add(row);
        }
        return rows;
    }

    private static int countLines(Path p) {
        if (p == null || !Files.exists(p)) return 0;
        try (var lines = Files.lines(p, StandardCharsets.UTF_8)) {
            return (int) lines.count();
        } catch (IOException ex) {
            return 0;
        }
    }

    private static String readFileSafe(Path p) {
        if (p == null || !Files.exists(p)) return "";
        try {
            return Files.readString(p, StandardCharsets.UTF_8);
        } catch (IOException ex) {
            return "";
        }
    }

    private static int parseInt(String s, int fallback) {
        try { return Integer.parseInt(s.trim()); } catch (Exception ex) { return fallback; }
    }

    private static long parseLong(String s, long fallback) {
        try { return Long.parseLong(s.trim()); } catch (Exception ex) { return fallback; }
    }

    private static double parseDouble(String s, double fallback) {
        try { return Double.parseDouble(s.trim()); } catch (Exception ex) { return fallback; }
    }

    private static String tail(String s, int maxChars) {
        if (s == null) return "";
        if (s.length() <= maxChars) return s;
        return s.substring(s.length() - maxChars);
    }

}
