package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.matrix.Matrix;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Orchestrates benchmark rows in isolated child processes so slow or unstable workloads can be time-capped.
 */
public final class AlgorithmTripleComparisonRunner {
    private static final int[] GEMM_SQUARE_SIZES = {128, 512, 1024, 2048};
    private static final int[] LU_SQUARE_SIZES = {128, 1024, 2048};
    private static final int[] QR_FACTOR_SQUARE_SIZES = {128, 1024, 2048};
    private static final int[] QR_DECOMPOSE_SQUARE_SIZES = {128, 512, 1024};
    private static final int[] CHOLESKY_SQUARE_SIZES = {128, 1024, 2048};
    private static final int[] HESSENBERG_SQUARE_SIZES = {128, 512};
    private static final Shape[] QR_RECT_SHAPES = {
        new Shape(2048, 256),
        new Shape(256, 2048)
    };
    private static final GemmShape[] GEMM_RECT_SHAPES = {
        new GemmShape(2048, 256, 128),
        new GemmShape(128, 256, 2048),
        new GemmShape(512, 2048, 256)
    };
    private static final long DEFAULT_TIMEOUT_SECONDS = 300L;
    private static final AtomicReference<WorkerTrace> WORKER_TRACE = new AtomicReference<>();

    private AlgorithmTripleComparisonRunner() {
    }

    public static void main(String[] args) throws Exception {
        CliOptions options = CliOptions.parse(args);
        if (options.workerMode()) {
            runWorker(options);
            return;
        }

        List<Workload> workloads = options.hasSingleWorkload()
            ? List.of(Workload.fromSingleOptions(options))
            : buildWorkloads(options.includes());
        Path cppExecutable = resolveCppExecutable();
        String nativeLibPath = firstNonBlank(
            System.getProperty("jlc.native.lib.path"),
            System.getProperty("faulj.native.lib.path")
        );

        List<Row> rows = new ArrayList<>();
        for (Workload workload : workloads) {
            rows.add(runWorkload(workload, options, cppExecutable, nativeLibPath));
        }

        System.out.println("ALGORITHM_TRIPLE_COMPARISON");
        System.out.println("timeoutPerWorkload=" + options.timeoutSeconds() + "s");
        System.out.println("threads=" + options.threads());
        System.out.println("warmupRuns=" + options.warmupRuns());
        System.out.println("measuredRuns=" + options.measuredRuns());
        System.out.println("cppExecutable=" + cppExecutable.toAbsolutePath());
        System.out.println();
        printMarkdownTable(rows);
        System.out.println();
        printDiagnostics(rows);
    }

    private static Row runWorkload(Workload workload, CliOptions options, Path cppExecutable, String nativeLibPath) {
        long budgetMs = Duration.ofSeconds(options.timeoutSeconds()).toMillis();
        long rowStart = System.nanoTime();
        List<String> diag = new ArrayList<>();

        BackendResult javaResult = null;
        BackendResult jniResult = null;
        BackendResult cppResult = null;

        if (workload.supportsJava()) {
            long remaining = remainingBudgetMs(rowStart, budgetMs);
            javaResult = runJvmWorker("java", workload, options, nativeLibPath, remaining);
            diag.add("java:" + javaResult.status().label() + "(" + javaResult.wallMs() + "ms)");
        }
        if (workload.supportsJni()) {
            long remaining = remainingBudgetMs(rowStart, budgetMs);
            if (remaining > 0) {
                jniResult = runJvmWorker("jni", workload, options, nativeLibPath, remaining);
            } else {
                jniResult = BackendResult.timeout(0, "Row budget exhausted before JNI run", List.of(), "", null);
            }
            diag.add("jni:" + jniResult.status().label() + "(" + jniResult.wallMs() + "ms)");
        }
        if (workload.supportsCpp()) {
            long remaining = remainingBudgetMs(rowStart, budgetMs);
            if (remaining > 0) {
                cppResult = runCppWorker(workload, options, cppExecutable, remaining);
            } else {
                cppResult = BackendResult.timeout(0, "Row budget exhausted before C++ run", List.of(), "", null);
            }
            diag.add("cpp:" + cppResult.status().label() + "(" + cppResult.wallMs() + "ms)");
        }

        return new Row(workload, javaResult, jniResult, cppResult, elapsedMs(rowStart), String.join("; ", diag));
    }

    private static BackendResult runJvmWorker(String backend, Workload workload, CliOptions options,
                                              String nativeLibPath, long timeoutMs) {
        List<String> command = new ArrayList<>();
        command.add(Paths.get(System.getProperty("java.home"), "bin", "java").toString());
        command.add("--add-modules=jdk.incubator.vector");
        command.add("--enable-preview");
        if (nativeLibPath != null) {
            command.add("-Djlc.native.lib.path=" + nativeLibPath);
        }
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add(AlgorithmTripleComparisonRunner.class.getName());
        command.add("--worker");
        command.add("--backend=" + backend);
        command.add("--algorithm=" + workload.algorithmKey());
        command.add("--rows=" + workload.rows());
        command.add("--cols=" + workload.cols());
        command.add("--inner=" + workload.inner());
        command.add("--seed=" + workload.seed());
        command.add("--warmup=" + effectiveWarmupRuns(workload, options));
        command.add("--runs=" + options.measuredRuns());
        command.add("--threads=" + options.threads());
        return runProcess(command, timeoutMs, true, workload, backend);
    }

    private static BackendResult runCppWorker(Workload workload, CliOptions options, Path cppExecutable, long timeoutMs) {
        List<String> command = new ArrayList<>();
        command.add(cppExecutable.toAbsolutePath().toString());
        command.add("--algorithm=" + workload.cppAlgorithmKey());
        command.add("--rows=" + workload.rows());
        command.add("--cols=" + workload.cols());
        command.add("--inner=" + workload.inner());
        command.add("--seed=" + workload.seed());
        command.add("--warmup=" + effectiveWarmupRuns(workload, options));
        command.add("--runs=" + options.measuredRuns());
        command.add("--threads=" + options.threads());
        return runProcess(command, timeoutMs, false, workload, "cpp");
    }

    private static BackendResult runProcess(List<String> command, long timeoutMs, boolean workerOutput,
                                            Workload workload, String backend) {
        long start = System.nanoTime();
        try {
            Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
            List<String> output = new ArrayList<>();
            Thread reader = new Thread(() -> readProcessOutput(process, output));
            reader.setDaemon(true);
            reader.start();

            boolean finished = process.waitFor(timeoutMs, TimeUnit.MILLISECONDS);
            if (!finished) {
                process.destroyForcibly();
                process.waitFor(5, TimeUnit.SECONDS);
                return BackendResult.timeout(
                    elapsedMs(start),
                    "Timed out after " + timeoutMs + "ms",
                    List.copyOf(command),
                    tail(output),
                    failureReport(workload, backend, command, elapsedMs(start), tail(output), "TIMEOUT", null, null, null)
                );
            }
            reader.join(5000);
            if (process.exitValue() != 0) {
                WorkerMeta meta = WorkerMeta.parse(output);
                return BackendResult.failure(
                    elapsedMs(start),
                    "Exit " + process.exitValue(),
                    List.copyOf(command),
                    tail(output),
                    failureReport(workload, backend, command, elapsedMs(start), tail(output), "FAIL",
                        meta.exceptionStackTrace(), meta.currentPhase(), meta.completedPhases())
                );
            }

            DoublePair parsed = workerOutput ? parseWorkerResult(output) : parseCppResult(output);
            if (parsed == null) {
                WorkerMeta meta = WorkerMeta.parse(output);
                return BackendResult.failure(
                    elapsedMs(start),
                    "No parsable result",
                    List.copyOf(command),
                    tail(output),
                    failureReport(workload, backend, command, elapsedMs(start), tail(output), "FAIL",
                        meta.exceptionStackTrace(), meta.currentPhase(), meta.completedPhases())
                );
            }

            return BackendResult.success(parsed.best(), parsed.mean(), elapsedMs(start), tail(output), List.copyOf(command));
        } catch (Exception ex) {
            return BackendResult.failure(
                elapsedMs(start),
                ex.getClass().getSimpleName() + ": " + ex.getMessage(),
                List.copyOf(command),
                ex.getMessage() == null ? "" : ex.getMessage(),
                failureReport(workload, backend, command, elapsedMs(start), ex.toString(), "FAIL", stackTrace(ex), null, null)
            );
        }
    }

    private static void readProcessOutput(Process process, List<String> output) {
        try (BufferedReader bufferedReader = new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                output.add(line);
            }
        } catch (IOException ignored) {
        }
    }

    private static DoublePair parseWorkerResult(List<String> output) {
        for (String line : output) {
            if (line.startsWith("worker_result=")) {
                String[] parts = line.substring("worker_result=".length()).split(",");
                if (parts.length >= 2) {
                    return new DoublePair(Double.parseDouble(parts[0]), Double.parseDouble(parts[1]));
                }
            }
        }
        return null;
    }

    private static DoublePair parseCppResult(List<String> output) {
        for (String line : output) {
            if (line.startsWith("csv_row=")) {
                String[] parts = line.substring("csv_row=".length()).split(",");
                if (parts.length >= 6) {
                    return new DoublePair(Double.parseDouble(parts[4]), Double.parseDouble(parts[5]));
                }
            }
        }
        return null;
    }

    private static void runWorker(CliOptions options) {
        Workload workload = Workload.fromWorkerOptions(options);
        WorkerTrace trace = new WorkerTrace();
        WORKER_TRACE.set(trace);
        printWorkerMeta("workloadId", workload.id());
        printWorkerMeta("algorithm", workload.algorithmKey());
        printWorkerMeta("mode", workload.label());
        printWorkerMeta("shape", workload.shapeLabel());
        printWorkerMeta("seed", Long.toString(workload.seed()));
        printWorkerMeta("backendRequested", options.backend());
        printWorkerMeta("matrixDimensions", "rows=" + workload.rows() + " cols=" + workload.cols() + " inner=" + workload.inner());
        printWorkerMeta("jvmArgs", String.join(" ", ManagementFactory.getRuntimeMXBean().getInputArguments()));

        try {
            BackendResult result = switch (options.backend()) {
                case "java" -> benchmarkJava(workload, options);
                case "jni" -> benchmarkJni(workload, options);
                default -> BackendResult.failure(0, "Unsupported worker backend", List.of(), "", "Unsupported worker backend");
            };
            if (result.status() != Status.OK) {
                printWorkerMeta("currentPhase", trace.currentPhase());
                printWorkerMeta("completedPhases", String.join(" | ", trace.completedPhases()));
                System.exit(2);
                return;
            }
            printWorkerMeta("currentPhase", trace.currentPhase());
            printWorkerMeta("completedPhases", String.join(" | ", trace.completedPhases()));
            System.out.printf(Locale.ROOT, "worker_result=%.6f,%.6f%n", result.bestMs(), result.meanMs());
        } catch (Throwable t) {
            printWorkerMeta("currentPhase", trace.currentPhase());
            printWorkerMeta("completedPhases", String.join(" | ", trace.completedPhases()));
            printWorkerMeta("exceptionType", t.getClass().getName());
            System.out.println("worker_meta.stackTraceBegin");
            System.out.print(stackTrace(t));
            System.out.println("worker_meta.stackTraceEnd");
            System.exit(2);
        }
    }

    private static void printWorkerMeta(String key, String value) {
        System.out.println("worker_meta." + key + "=" + value);
    }

    private static BackendResult benchmarkJava(Workload workload, CliOptions options) {
        printWorkerMeta("backendSelected", "java-direct");
        return measureWorkload(workload, options, true);
    }

    private static BackendResult benchmarkJni(Workload workload, CliOptions options) {
        WorkerTrace trace = WORKER_TRACE.get();
        trace.enter("jni_probe");
        NativeBackend backend = new NativeBackend(new JavaBackend());
        NativeContext context = backend.probe(true);
        if (!context.isAvailable()) {
            return BackendResult.failure(0, "Native backend unavailable: " + context.getMessage(), List.of(), "", "Native backend unavailable");
        }
        printWorkerMeta("backendSelected", "jni-direct");
        printWorkerMeta("nativeStatus", context.getStatus().toString());
        printWorkerMeta("nativeProvider", context.getProviderDescription());
        trace.complete("jni_probe");
        return measureWorkload(workload, options, false);
    }

    private static BackendResult measureWorkload(Workload workload, CliOptions options, boolean javaBackend) {
        long start = System.nanoTime();
        double[] samples = new double[options.measuredRuns()];
        WorkerTrace trace = WORKER_TRACE.get();
        for (int i = 0; i < options.warmupRuns(); i++) {
            trace.enter("warmup_" + (i + 1));
            runSingle(workload, javaBackend, options.threads());
            trace.complete("warmup_" + (i + 1));
        }
        for (int i = 0; i < options.measuredRuns(); i++) {
            trace.enter("measured_" + (i + 1));
            long runStart = System.nanoTime();
            runSingle(workload, javaBackend, options.threads());
            samples[i] = (System.nanoTime() - runStart) / 1e6;
            trace.complete("measured_" + (i + 1) + "=" + String.format(Locale.ROOT, "%.3fms", samples[i]));
        }
        return BackendResult.success(
            Arrays.stream(samples).min().orElse(Double.NaN),
            Arrays.stream(samples).average().orElse(Double.NaN),
            elapsedMs(start),
            "in-process worker",
            List.of()
        );
    }

    private static void runSingle(Workload workload, boolean javaBackend, int threads) {
        WorkerTrace trace = WORKER_TRACE.get();
        switch (workload.algorithmKey()) {
            case "gemm" -> runGemm(workload, javaBackend, threads, trace);
            case "lu" -> runLu(workload, javaBackend, trace);
            case "qr_factor" -> runQrFactor(workload, javaBackend, trace);
            case "qr_thin" -> runQrThin(workload, javaBackend, trace);
            case "qr_full" -> runQrFull(workload, javaBackend, trace);
            case "cholesky" -> runCholesky(workload, javaBackend, trace);
            case "hessenberg" -> runHessenberg(workload, trace);
            default -> throw new IllegalArgumentException("Unsupported algorithm: " + workload.algorithmKey());
        }
    }

    private static void runGemm(Workload workload, boolean javaBackend, int threads, WorkerTrace trace) {
        trace.enter("generate_inputs");
        double[] aData = randomData(workload.rows(), workload.inner(), workload.seed());
        double[] bData = randomData(workload.inner(), workload.cols(), workload.seed() + 1);
        double[] cData = new double[workload.rows() * workload.cols()];
        trace.complete("generate_inputs");
        if (javaBackend) {
            DispatchPolicy policy = DispatchPolicy.builder()
                .enableCuda(false)
                .enableParallel(threads > 1)
                .parallelism(threads)
                .enableBlas3(true)
                .enableSimd(true)
                .build();
            trace.enter("java_gemm");
            new JavaBackend().gemm(
                Matrix.wrap(aData, workload.rows(), workload.inner()),
                Matrix.wrap(bData, workload.inner(), workload.cols()),
                Matrix.wrap(cData, workload.rows(), workload.cols()),
                1.0, 0.0, policy
            );
            trace.complete("java_gemm");
        } else {
            trace.enter("jni_gemm");
            NativeBindings.nativeGemm(
                aData, workload.rows(), workload.inner(),
                bData, workload.inner(), workload.cols(),
                cData, workload.rows(), workload.cols(),
                1.0, 0.0,
                threads, NativeFlags.FORCE_BUILTIN
            );
            trace.complete("jni_gemm");
        }
    }

    private static void runLu(Workload workload, boolean javaBackend, WorkerTrace trace) {
        trace.enter("generate_inputs");
        double[] matrix = randomData(workload.rows(), workload.cols(), workload.seed());
        trace.complete("generate_inputs");
        printWorkerMeta("lu.blockThreshold", System.getProperty("net.faulj.decomposition.lu.blockThreshold", "384"));
        printWorkerMeta("lu.blockSize", System.getProperty("net.faulj.decomposition.lu.blockSize", "32"));
        printWorkerMeta("lu.maxAbs", Double.toString(maxAbs(matrix)));
        if (javaBackend) {
            trace.enter("java_lu_decompose");
            String previousBackend = System.getProperty("jlc.backend");
            System.setProperty("jlc.backend", "java");
            try {
                new LUDecomposition().decompose(Matrix.wrap(matrix, workload.rows(), workload.cols()));
                trace.complete("java_lu_decompose");
            } finally {
                if (previousBackend == null) {
                    System.clearProperty("jlc.backend");
                } else {
                    System.setProperty("jlc.backend", previousBackend);
                }
            }
        } else {
            trace.enter("jni_lu_factor");
            NativeBindings.nativeLuFactor(matrix, workload.rows(), new int[workload.rows()]);
            trace.complete("jni_lu_factor");
        }
    }

    private static void runQrFactor(Workload workload, boolean javaBackend, WorkerTrace trace) {
        trace.enter("generate_inputs");
        double[] matrix = randomData(workload.rows(), workload.cols(), workload.seed());
        trace.complete("generate_inputs");
        if (javaBackend) {
            trace.enter("java_qr_factor");
            HouseholderQR.factorize(Matrix.wrap(matrix, workload.rows(), workload.cols()));
            trace.complete("java_qr_factor");
        } else {
            trace.enter("jni_qr_factor");
            NativeBindings.nativeQrFactorizeOnly(matrix, workload.rows(), workload.cols());
            trace.complete("jni_qr_factor");
        }
    }

    private static void runQrThin(Workload workload, boolean javaBackend, WorkerTrace trace) {
        trace.enter("generate_inputs");
        double[] matrix = randomData(workload.rows(), workload.cols(), workload.seed());
        trace.complete("generate_inputs");
        if (javaBackend) {
            trace.enter("java_qr_thin");
            HouseholderQR.decomposeThin(Matrix.wrap(matrix, workload.rows(), workload.cols()));
            trace.complete("java_qr_thin");
        } else {
            int qCols = Math.min(workload.rows(), workload.cols());
            trace.enter("jni_qr_thin");
            NativeBindings.nativeQrDecompose(matrix, workload.rows(), workload.cols(), qCols,
                new double[workload.rows() * qCols], new double[qCols * workload.cols()]);
            trace.complete("jni_qr_thin");
        }
    }

    private static void runQrFull(Workload workload, boolean javaBackend, WorkerTrace trace) {
        trace.enter("generate_inputs");
        double[] matrix = randomData(workload.rows(), workload.cols(), workload.seed());
        trace.complete("generate_inputs");
        if (javaBackend) {
            trace.enter("java_qr_full");
            HouseholderQR.decompose(Matrix.wrap(matrix, workload.rows(), workload.cols()));
            trace.complete("java_qr_full");
        } else {
            trace.enter("jni_qr_full");
            NativeBindings.nativeQrDecompose(matrix, workload.rows(), workload.cols(), workload.rows(),
                new double[workload.rows() * workload.rows()], new double[workload.rows() * workload.cols()]);
            trace.complete("jni_qr_full");
        }
    }

    private static void runCholesky(Workload workload, boolean javaBackend, WorkerTrace trace) {
        trace.enter("generate_spd_inputs");
        double[] matrix = randomSpdData(workload.rows(), workload.seed());
        trace.complete("generate_spd_inputs");
        printWorkerMeta("cholesky.blockThreshold", System.getProperty("net.faulj.decomposition.cholesky.blockThreshold", "96"));
        printWorkerMeta("cholesky.blockSize", System.getProperty("net.faulj.decomposition.cholesky.blockSize", "32"));
        printWorkerMeta("cholesky.minDiag", Double.toString(minDiag(matrix, workload.rows())));
        printWorkerMeta("cholesky.maxDiag", Double.toString(maxDiag(matrix, workload.rows())));
        if (javaBackend) {
            trace.enter("java_cholesky");
            String previousBackend = System.getProperty("jlc.backend");
            System.setProperty("jlc.backend", "java");
            try {
                new CholeskyDecomposition().decompose(Matrix.wrap(matrix, workload.rows(), workload.cols()));
                trace.complete("java_cholesky");
            } finally {
                if (previousBackend == null) {
                    System.clearProperty("jlc.backend");
                } else {
                    System.setProperty("jlc.backend", previousBackend);
                }
            }
        } else {
            trace.enter("jni_cholesky");
            NativeBindings.nativeCholeskyDecompose(matrix, workload.rows());
            trace.complete("jni_cholesky");
        }
    }

    private static void runHessenberg(Workload workload, WorkerTrace trace) {
        trace.enter("generate_inputs");
        double[] matrix = randomData(workload.rows(), workload.cols(), workload.seed());
        trace.complete("generate_inputs");
        printWorkerMeta("hessenberg.blockSize", System.getProperty("net.faulj.decomposition.hessenberg.blockSize", "32"));
        trace.enter("java_hessenberg");
        HessenbergReduction.decompose(Matrix.wrap(matrix, workload.rows(), workload.cols()));
        trace.complete("java_hessenberg");
    }

    private static List<Workload> buildWorkloads(Set<String> includes) {
        List<Workload> workloads = new ArrayList<>();
        if (isIncluded(includes, "gemm")) {
            for (int size : GEMM_SQUARE_SIZES) workloads.add(Workload.gemm(size, size, size, "Square GEMM"));
            for (GemmShape shape : GEMM_RECT_SHAPES) workloads.add(Workload.gemm(shape.rows(), shape.inner(), shape.cols(), "Rectangular GEMM"));
        }
        if (isIncluded(includes, "lu")) {
            for (int size : LU_SQUARE_SIZES) workloads.add(Workload.square("LU", "lu", "lu", size, true, true, true, "Square only"));
        }
        if (isIncluded(includes, "qr_factor")) {
            for (int size : QR_FACTOR_SQUARE_SIZES) workloads.add(Workload.square("QR Factorize Only", "qr_factor", "qr_factor", size, true, true, true, "Square QR factorization only"));
            for (Shape shape : QR_RECT_SHAPES) workloads.add(Workload.rect("QR Factorize Only", "qr_factor", "qr_factor", shape.rows(), shape.cols(), true, true, true, rectangularNote(shape.rows(), shape.cols(), "QR factorization only")));
        }
        if (isIncluded(includes, "qr_thin")) {
            for (int size : QR_DECOMPOSE_SQUARE_SIZES) workloads.add(Workload.square("QR Thin", "qr_thin", "qr_thin", size, true, true, true, "Square thin QR"));
            for (Shape shape : QR_RECT_SHAPES) workloads.add(Workload.rect("QR Thin", "qr_thin", "qr_thin", shape.rows(), shape.cols(), true, true, true, rectangularNote(shape.rows(), shape.cols(), "thin QR")));
        }
        if (isIncluded(includes, "qr_full")) {
            for (int size : QR_DECOMPOSE_SQUARE_SIZES) workloads.add(Workload.square("QR Full", "qr_full", "qr_full", size, true, true, true, "Square full QR"));
            for (Shape shape : QR_RECT_SHAPES) workloads.add(Workload.rect("QR Full", "qr_full", "qr_full", shape.rows(), shape.cols(), true, true, true, rectangularNote(shape.rows(), shape.cols(), "full QR")));
        }
        if (isIncluded(includes, "cholesky")) {
            for (int size : CHOLESKY_SQUARE_SIZES) workloads.add(Workload.square("Cholesky", "cholesky", "cholesky", size, true, true, true, "Square SPD only"));
        }
        if (isIncluded(includes, "hessenberg")) {
            for (int size : HESSENBERG_SQUARE_SIZES) workloads.add(Workload.square("Hessenberg", "hessenberg", "hessenberg", size, true, false, false, "Square only; native path unavailable"));
        }
        return workloads;
    }

    private static Path resolveCppExecutable() {
        String configured = firstNonBlank(
            System.getProperty("jlc.native.algorithm.bench.path"),
            System.getProperty("faulj.native.algorithm.bench.path")
        );
        if (configured != null) {
            Path path = Paths.get(configured).toAbsolutePath().normalize();
            if (Files.exists(path)) return path;
        }
        String executableName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win")
            ? "jlc_native_algorithm_bench.exe"
            : "jlc_native_algorithm_bench";
        Path[] candidates = new Path[] {
            Paths.get("build", "native-backend", "lib", executableName),
            Paths.get("build", "native-backend", "lib", "Release", executableName)
        };
        for (Path candidate : candidates) {
            Path absolute = candidate.toAbsolutePath().normalize();
            if (Files.exists(absolute)) return absolute;
        }
        throw new IllegalStateException("Unable to locate native algorithm benchmark executable");
    }

    private static void printMarkdownTable(List<Row> rows) {
        System.out.println("| Algorithm | Shape | Java best ms | JNI-C++ best ms | C++ best ms | Fastest | Row wall s | Notes |");
        System.out.println("|---|---|---:|---:|---:|---|---:|---|");
        for (Row row : rows) {
            System.out.printf(Locale.ROOT, "| %s | %s | %s | %s | %s | %s | %.1f | %s |%n",
                row.workload().label(),
                row.workload().shapeLabel(),
                formatResult(row.javaResult()),
                formatResult(row.jniResult()),
                formatResult(row.cppResult()),
                fastest(row),
                row.wallMs() / 1000.0,
                row.workload().notes());
        }
    }

    private static void printDiagnostics(List<Row> rows) {
        System.out.println("Diagnostics:");
        for (Row row : rows) {
            printFailureDetails(row.workload(), "java", row.javaResult());
            printFailureDetails(row.workload(), "jni", row.jniResult());
            printFailureDetails(row.workload(), "cpp", row.cppResult());
        }
    }

    private static void printFailureDetails(Workload workload, String backend, BackendResult result) {
        if (result == null || result.status() == Status.OK || result.failureReport() == null || result.failureReport().isBlank()) {
            return;
        }
        System.out.println("---");
        System.out.println(result.failureReport());
    }

    private static String fastest(Row row) {
        double best = Double.POSITIVE_INFINITY;
        String label = "n/a";
        if (row.javaResult() != null && row.javaResult().status() == Status.OK && row.javaResult().bestMs() < best) {
            best = row.javaResult().bestMs();
            label = "Java";
        }
        if (row.jniResult() != null && row.jniResult().status() == Status.OK && row.jniResult().bestMs() < best) {
            best = row.jniResult().bestMs();
            label = "JNI-C++";
        }
        if (row.cppResult() != null && row.cppResult().status() == Status.OK && row.cppResult().bestMs() < best) {
            label = "C++";
        }
        return label;
    }

    private static String formatResult(BackendResult result) {
        if (result == null) return "N/A";
        return switch (result.status()) {
            case OK -> String.format(Locale.ROOT, "%.3f", result.bestMs());
            case TIMEOUT -> "TIMEOUT";
            case FAILURE -> "FAIL";
        };
    }

    private static long remainingBudgetMs(long rowStart, long budgetMs) {
        return Math.max(0L, budgetMs - elapsedMs(rowStart));
    }

    private static long elapsedMs(long startNanos) {
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
    }

    private static String tail(List<String> output) {
        if (output.isEmpty()) return "";
        int start = Math.max(0, output.size() - 20);
        return String.join("\n", output.subList(start, output.size()));
    }

    private static String stackTrace(Throwable throwable) {
        StringWriter writer = new StringWriter();
        PrintWriter printer = new PrintWriter(writer);
        throwable.printStackTrace(printer);
        printer.flush();
        return writer.toString();
    }

    private static String failureReport(Workload workload, String backend, List<String> command, long elapsedMs,
                                        String outputTail, String outcome, String stackTrace,
                                        String currentPhase, String completedPhases) {
        StringBuilder report = new StringBuilder();
        report.append("workload id: ").append(workload.id()).append('\n');
        report.append("algorithm/mode/shape/seed: ").append(workload.algorithmKey())
            .append(" / ").append(workload.label())
            .append(" / ").append(workload.shapeLabel())
            .append(" / ").append(workload.seed()).append('\n');
        report.append("backend selected: ").append(backend).append('\n');
        report.append("matrix dimensions: rows=").append(workload.rows())
            .append(" cols=").append(workload.cols())
            .append(" inner=").append(workload.inner()).append('\n');
        report.append((backend.equals("cpp") ? "native command args: " : "JVM args / native command args: "))
            .append(String.join(" ", command)).append('\n');
        report.append("elapsed time before failure: ").append(elapsedMs).append("ms").append('\n');
        if (currentPhase != null) report.append("current phase: ").append(currentPhase).append('\n');
        if (completedPhases != null) report.append("last completed algorithm phase: ").append(completedPhases).append('\n');
        report.append("outcome: ").append(outcome).append('\n');
        if (stackTrace != null && !stackTrace.isBlank()) {
            report.append("full exception stack trace:\n").append(stackTrace);
        }
        if (outputTail != null && !outputTail.isBlank()) {
            report.append("stdout/stderr tail:\n").append(outputTail).append('\n');
        }
        return report.toString();
    }

    private static int effectiveWarmupRuns(Workload workload, CliOptions options) {
        if ("lu".equals(workload.algorithmKey())) {
            return Math.max(options.warmupRuns(), 2);
        }
        return options.warmupRuns();
    }

    private static double[] randomData(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) data[i] = random.nextDouble() - 0.5;
        return data;
    }

    private static double[] randomSpdData(int size, long seed) {
        Random random = new Random(seed);
        double[] lower = new double[size * size];
        for (int row = 0; row < size; row++) {
            for (int col = 0; col <= row; col++) {
                lower[row * size + col] = random.nextDouble() - 0.5;
            }
            lower[row * size + row] += size;
        }
        double[] spd = new double[size * size];
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                double sum = 0.0;
                int upto = Math.min(row, col);
                for (int k = 0; k <= upto; k++) {
                    sum += lower[row * size + k] * lower[col * size + k];
                }
                spd[row * size + col] = sum;
            }
        }
        return spd;
    }

    private static double maxAbs(double[] values) {
        double max = 0.0;
        for (double value : values) max = Math.max(max, Math.abs(value));
        return max;
    }

    private static double minDiag(double[] values, int n) {
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < n; i++) min = Math.min(min, values[i * n + i]);
        return min;
    }

    private static double maxDiag(double[] values, int n) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) max = Math.max(max, values[i * n + i]);
        return max;
    }

    private static String rectangularNote(int rows, int cols, String label) {
        if (rows > cols) return "Tall-skinny " + label;
        if (rows < cols) return "Wide " + label;
        return "Square " + label;
    }

    private static boolean isIncluded(Set<String> includes, String algorithmKey) {
        return includes == null || includes.contains(algorithmKey);
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) return value.trim();
        }
        return null;
    }

    private record DoublePair(double best, double mean) {
    }

    private record Row(Workload workload, BackendResult javaResult, BackendResult jniResult,
                       BackendResult cppResult, long wallMs, String diagnostic) {
    }

    private record Workload(String label, String algorithmKey, String cppAlgorithmKey,
                            int rows, int cols, int inner, boolean supportsJava,
                            boolean supportsJni, boolean supportsCpp, String notes, long seed) {
        static Workload gemm(int rows, int inner, int cols, String notes) {
            return new Workload("GEMM", "gemm", "gemm", rows, cols, inner, true, true, true, notes, seedFor("gemm", rows, cols, inner));
        }

        static Workload square(String label, String algorithmKey, String cppAlgorithmKey, int size,
                               boolean java, boolean jni, boolean cpp, String notes) {
            return new Workload(label, algorithmKey, cppAlgorithmKey, size, size, size, java, jni, cpp, notes,
                seedFor(algorithmKey, size, size, size));
        }

        static Workload rect(String label, String algorithmKey, String cppAlgorithmKey, int rows, int cols,
                             boolean java, boolean jni, boolean cpp, String notes) {
            return new Workload(label, algorithmKey, cppAlgorithmKey, rows, cols, cols, java, jni, cpp, notes,
                seedFor(algorithmKey, rows, cols, cols));
        }

        static Workload fromWorkerOptions(CliOptions options) {
            return new Workload(options.algorithmLabel(), options.algorithm(), options.algorithm(),
                options.rows(), options.cols(), options.inner(), true, true, true, "worker", options.seed());
        }

        static Workload fromSingleOptions(CliOptions options) {
            String label = options.algorithmLabel();
            boolean cpp = !"hessenberg".equals(options.algorithm());
            boolean jni = !"hessenberg".equals(options.algorithm());
            return new Workload(label, options.algorithm(), options.algorithm(),
                options.rows(), options.cols(), options.inner() > 0 ? options.inner() : options.cols(),
                true, jni, cpp, "single workload", options.seed() != 0L ? options.seed() : seedFor(options.algorithm(), options.rows(), options.cols(), options.inner()));
        }

        String shapeLabel() {
            return "gemm".equals(algorithmKey) ? rows + "x" + inner + " * " + inner + "x" + cols : rows + "x" + cols;
        }

        String id() {
            return algorithmKey + "-" + rows + "x" + cols + "-k" + inner + "-seed" + seed;
        }

        private static long seedFor(String algorithmKey, int rows, int cols, int inner) {
            long hash = 1469598103934665603L;
            String text = algorithmKey + ":" + rows + ":" + cols + ":" + inner;
            for (int i = 0; i < text.length(); i++) {
                hash ^= text.charAt(i);
                hash *= 1099511628211L;
            }
            return hash == Long.MIN_VALUE ? 1L : Math.abs(hash);
        }
    }

    private record BackendResult(Status status, double bestMs, double meanMs, long wallMs, String detail,
                                 List<String> command, String outputTail, String failureReport) {
        static BackendResult success(double bestMs, double meanMs, long wallMs, String detail, List<String> command) {
            return new BackendResult(Status.OK, bestMs, meanMs, wallMs, detail, command, detail, null);
        }

        static BackendResult timeout(long wallMs, String detail, List<String> command, String outputTail, String failureReport) {
            return new BackendResult(Status.TIMEOUT, Double.NaN, Double.NaN, wallMs, detail, command, outputTail, failureReport);
        }

        static BackendResult failure(long wallMs, String detail, List<String> command, String outputTail, String failureReport) {
            return new BackendResult(Status.FAILURE, Double.NaN, Double.NaN, wallMs, detail, command, outputTail, failureReport);
        }
    }

    private enum Status {
        OK("ok"),
        TIMEOUT("timeout"),
        FAILURE("fail");

        private final String label;

        Status(String label) {
            this.label = label;
        }

        String label() {
            return label;
        }
    }

    private record Shape(int rows, int cols) {
    }

    private record GemmShape(int rows, int inner, int cols) {
    }

    private record CliOptions(boolean workerMode, String backend, String algorithm, int rows, int cols, int inner,
                              long seed, int warmupRuns, int measuredRuns, int threads, long timeoutSeconds,
                              Set<String> includes) {
        static CliOptions parse(String[] args) {
            boolean workerMode = false;
            String backend = null;
            String algorithm = null;
            int rows = 0;
            int cols = 0;
            int inner = 0;
            long seed = 0L;
            int warmupRuns = 1;
            int measuredRuns = 1;
            int threads = 1;
            long timeoutSeconds = DEFAULT_TIMEOUT_SECONDS;
            Set<String> includes = null;
            for (String arg : args) {
                if (arg == null) continue;
                if (arg.equals("--worker")) {
                    workerMode = true;
                } else if (arg.startsWith("--backend=")) {
                    backend = arg.substring("--backend=".length());
                } else if (arg.startsWith("--algorithm=")) {
                    algorithm = arg.substring("--algorithm=".length());
                } else if (arg.startsWith("--rows=")) {
                    rows = Integer.parseInt(arg.substring("--rows=".length()));
                } else if (arg.startsWith("--cols=")) {
                    cols = Integer.parseInt(arg.substring("--cols=".length()));
                } else if (arg.startsWith("--inner=")) {
                    inner = Integer.parseInt(arg.substring("--inner=".length()));
                } else if (arg.startsWith("--seed=")) {
                    seed = Long.parseLong(arg.substring("--seed=".length()));
                } else if (arg.startsWith("--warmup=")) {
                    warmupRuns = Integer.parseInt(arg.substring("--warmup=".length()));
                } else if (arg.startsWith("--runs=")) {
                    measuredRuns = Integer.parseInt(arg.substring("--runs=".length()));
                } else if (arg.startsWith("--threads=")) {
                    threads = Integer.parseInt(arg.substring("--threads=".length()));
                } else if (arg.startsWith("--timeoutSeconds=")) {
                    timeoutSeconds = Long.parseLong(arg.substring("--timeoutSeconds=".length()));
                } else if (arg.startsWith("--include=")) {
                    includes = new LinkedHashSet<>();
                    for (String part : arg.substring("--include=".length()).split(",")) {
                        String trimmed = part.trim().toLowerCase(Locale.ROOT);
                        if (!trimmed.isEmpty()) includes.add(trimmed);
                    }
                }
            }
            return new CliOptions(workerMode, backend, algorithm, rows, cols, inner, seed,
                warmupRuns, measuredRuns, threads, timeoutSeconds, includes);
        }

        String algorithmLabel() {
            return switch (algorithm) {
                case "gemm" -> "GEMM";
                case "lu" -> "LU";
                case "qr_factor" -> "QR Factorize Only";
                case "qr_thin" -> "QR Thin";
                case "qr_full" -> "QR Full";
                case "cholesky" -> "Cholesky";
                case "hessenberg" -> "Hessenberg";
                default -> algorithm;
            };
        }

        boolean hasSingleWorkload() {
            return !workerMode && algorithm != null && rows > 0 && cols > 0;
        }
    }

    private static final class WorkerTrace {
        private final List<String> completedPhases = new ArrayList<>();
        private String currentPhase = "init";
        private long phaseStartNs = System.nanoTime();

        void enter(String phase) {
            currentPhase = phase;
            phaseStartNs = System.nanoTime();
        }

        void complete(String label) {
            long ms = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - phaseStartNs);
            completedPhases.add(label + " (" + ms + "ms)");
            currentPhase = label;
            phaseStartNs = System.nanoTime();
        }

        String currentPhase() {
            return currentPhase;
        }

        List<String> completedPhases() {
            return completedPhases;
        }
    }

    private record WorkerMeta(String currentPhase, String completedPhases, String exceptionStackTrace) {
        static WorkerMeta parse(List<String> output) {
            String currentPhase = null;
            String completedPhases = null;
            String stackTrace = null;
            boolean inStack = false;
            StringBuilder stack = new StringBuilder();
            for (String line : output) {
                if (line.startsWith("worker_meta.currentPhase=")) {
                    currentPhase = line.substring("worker_meta.currentPhase=".length());
                } else if (line.startsWith("worker_meta.completedPhases=")) {
                    completedPhases = line.substring("worker_meta.completedPhases=".length());
                } else if (line.equals("worker_meta.stackTraceBegin")) {
                    inStack = true;
                } else if (line.equals("worker_meta.stackTraceEnd")) {
                    inStack = false;
                    stackTrace = stack.toString();
                } else if (inStack) {
                    stack.append(line).append('\n');
                }
            }
            return new WorkerMeta(currentPhase, completedPhases, stackTrace);
        }
    }
}
