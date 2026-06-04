package net.faulj.compute;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import jdk.incubator.vector.DoubleVector;
import net.faulj.matrix.Matrix;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public final class GemmBackendComparison {
    private static final DateTimeFormatter UTC_STAMP =
        DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'", Locale.US).withZone(ZoneOffset.UTC);

    private static final List<String> SUPPORTED_BACKENDS = List.of(
        "blas3-naive",
        "blas3-simd-1t",
        "blas3-parallel",
        "opt-1t",
        "opt-parallel",
        "cuda"
    );

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
        .enable(SerializationFeature.INDENT_OUTPUT);

    private GemmBackendComparison() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.fromSystemProperties();
        Instant startedAt = Instant.now();
        String runTimestamp = DateTimeFormatter.ISO_INSTANT.format(startedAt);
        String runStamp = UTC_STAMP.format(startedAt);

        Path reportDir = Path.of("build", "reports", "gemm");
        Files.createDirectories(reportDir);

        MachineMetadata metadata = MachineMetadata.collect(runTimestamp, runStamp, config);
        List<Row> rows = runBenchmark(config, metadata);
        assignBaselines(rows);

        Path stableCsv = reportDir.resolve("gemm-backend-comparison.csv");
        Path stampedCsv = reportDir.resolve("gemm-backend-comparison-" + runStamp + ".csv");
        Path json = reportDir.resolve("gemm-backend-comparison.json");

        writeCsv(rows, stableCsv);
        writeCsv(rows, stampedCsv);
        writeJson(metadata, rows, stableCsv, stampedCsv, json);

        printMetadata(metadata);
        printLegend();
        printTable(rows);
        System.out.println();
        System.out.println("Artifacts:");
        System.out.println("  " + stableCsv.toAbsolutePath());
        System.out.println("  " + stampedCsv.toAbsolutePath());
        System.out.println("  " + json.toAbsolutePath());

        boolean anyParityFailure = rows.stream().anyMatch(row -> "FAIL".equals(row.parityStatus));
        boolean anyCudaSkipped = rows.stream().anyMatch(row -> "cuda".equals(row.backend) && "SKIPPED".equals(row.parityStatus));
        if (anyParityFailure) {
            System.out.println();
            System.out.println("WARNING: at least one backend parity check failed. GFLOPs and speedup are omitted for failing rows.");
        }
        if (anyCudaSkipped) {
            System.out.println();
            System.out.println("NOTE: CUDA rows were skipped because CUDA was disabled, unavailable, or CudaGemm.gemm returned false.");
        }
    }

    private static List<Row> runBenchmark(Config config, MachineMetadata metadata) {
        List<Row> rows = new ArrayList<>();
        LinkedHashSet<String> requested = new LinkedHashSet<>(config.backends);

        for (int size : config.sizes) {
            Matrix a = GemmReference.seededMatrix(size, size, config.seed + size * 31L + 11L);
            Matrix b = GemmReference.seededMatrix(size, size, config.seed + size * 31L + 29L);
            OracleContext oracle = buildOracle(config, requested, size, a, b);

            if (oracle.reference == null) {
                for (String backend : config.backends) {
                    rows.add(Row.skipped(
                        metadata.timestampUtc,
                        size,
                        backend,
                        threadsForBackend(backend),
                        oracle.oracleType,
                        oracle.referenceBackend,
                        config.warmups,
                        config.iterations,
                        "Independent oracle size cap exceeded and opt-1t is unavailable for backend_reference parity."
                    ));
                }
                continue;
            }

            for (String backend : config.backends) {
                rows.add(runBackend(metadata.timestampUtc, config, size, backend, a, b, oracle));
            }
        }

        return rows;
    }

    private static OracleContext buildOracle(Config config, LinkedHashSet<String> requestedBackends, int size, Matrix a, Matrix b) {
        if (size <= config.independentOracleMaxSize) {
            double[] expected = GemmReference.gemm(
                a.getRawData(),
                b.getRawData(),
                new double[size * size],
                size,
                size,
                size,
                1.0,
                0.0
            );
            return new OracleContext(expected, "independent", "");
        }

        if (!requestedBackends.contains("opt-1t")) {
            return new OracleContext(null, "none", "");
        }

        BackendExecution reference = executeBackend("opt-1t", a, b, size);
        if (!reference.executed) {
            return new OracleContext(null, "backend_reference", "opt-1t");
        }
        return new OracleContext(reference.output, "backend_reference", "opt-1t");
    }

    private static Row runBackend(String timestamp, Config config, int size, String backend, Matrix a, Matrix b, OracleContext oracle) {
        int threads = threadsForBackend(backend);
        if ("blas3-naive".equals(backend) && size > config.naiveMaxSize) {
            return Row.skipped(
                timestamp,
                size,
                backend,
                threads,
                oracle.oracleType,
                oracle.referenceBackend,
                config.warmups,
                config.iterations,
                "Skipped because size exceeds gemm.naiveMaxSize=" + config.naiveMaxSize + "."
            );
        }
        if ("cuda".equals(backend) && config.cudaMode == CudaMode.OFF) {
            return Row.skipped(
                timestamp,
                size,
                backend,
                threads,
                oracle.oracleType,
                oracle.referenceBackend,
                config.warmups,
                config.iterations,
                "Skipped because gemm.cuda=off."
            );
        }

        BackendExecution execution;
        try {
            execution = executeBackend(backend, a, b, size);
        } catch (RuntimeException ex) {
            return Row.failed(
                timestamp,
                size,
                backend,
                threads,
                oracle.oracleType,
                oracle.referenceBackend,
                config.warmups,
                config.iterations,
                Double.NaN,
                Double.NaN,
                backendNotes(backend) + " Execution threw " + ex.getClass().getSimpleName() + ": " + safeMessage(ex)
            );
        }

        if (!execution.executed) {
            return Row.skipped(
                timestamp,
                size,
                backend,
                threads,
                oracle.oracleType,
                oracle.referenceBackend,
                config.warmups,
                config.iterations,
                backendNotes(backend) + " Backend did not execute on this machine."
            );
        }

        ErrorMetrics metrics = compare(execution.output, oracle.reference);
        double absTol = "cuda".equals(backend)
            ? GemmReference.cudaAbsTolerance(oracle.reference, size)
            : GemmReference.cpuAbsTolerance(oracle.reference, size);
        double relTol = "cuda".equals(backend) ? 1e-10 : 1e-12;
        boolean valid = metrics.maxAbsErr <= absTol && metrics.relResidual <= relTol;
        if (!valid) {
            return Row.failed(
                timestamp,
                size,
                backend,
                threads,
                oracle.oracleType,
                oracle.referenceBackend,
                config.warmups,
                config.iterations,
                metrics.relResidual,
                metrics.maxAbsErr,
                backendNotes(backend) + String.format(
                    Locale.US,
                    " Parity exceeded tolerances relTol=%g absTol=%g.",
                    relTol,
                    absTol
                )
            );
        }

        TimingStats timing = timeBackend(config, backend, a, b, size);
        return Row.valid(
            timestamp,
            size,
            backend,
            threads,
            oracle.oracleType,
            oracle.referenceBackend,
            metrics.relResidual,
            metrics.maxAbsErr,
            config.warmups,
            config.iterations,
            timing,
            backendNotes(backend)
        );
    }

    private static TimingStats timeBackend(Config config, String backend, Matrix a, Matrix b, int size) {
        long flops = 2L * size * size * size;
        int innerRepeats = (int) Math.max(1L, (20_000_000L + flops - 1L) / Math.max(1L, flops));
        double[] perCallNs = new double[config.iterations];
        double checksumAccumulator = 0.0;

        for (int warmup = 0; warmup < config.warmups; warmup++) {
            for (int repeat = 0; repeat < innerRepeats; repeat++) {
                BackendExecution execution = executeBackend(backend, a, b, size);
                if (!execution.executed) {
                    throw new IllegalStateException("Backend became unavailable during warmup: " + backend);
                }
                checksumAccumulator += checksum(execution.output);
            }
        }

        for (int iter = 0; iter < config.iterations; iter++) {
            long start = System.nanoTime();
            for (int repeat = 0; repeat < innerRepeats; repeat++) {
                BackendExecution execution = executeBackend(backend, a, b, size);
                if (!execution.executed) {
                    throw new IllegalStateException("Backend became unavailable during measurement: " + backend);
                }
                checksumAccumulator += checksum(execution.output);
            }
            long elapsed = System.nanoTime() - start;
            perCallNs[iter] = elapsed / (double) innerRepeats;
        }

        return TimingStats.from(perCallNs, flops, checksumAccumulator, innerRepeats);
    }

    private static BackendExecution executeBackend(String backend, Matrix a, Matrix b, int size) {
        Matrix c = new Matrix(size, size);
        switch (backend) {
            case "blas3-naive":
                BLAS3Kernels.gemm(a, b, c, 1.0, 0.0, naivePolicy());
                return BackendExecution.executed(c.getRawData());
            case "blas3-simd-1t":
                BLAS3Kernels.gemm(a, b, c, 1.0, 0.0, simdSingleThreadPolicy());
                return BackendExecution.executed(c.getRawData());
            case "blas3-parallel":
                BLAS3Kernels.gemm(a, b, c, 1.0, 0.0, blas3ParallelPolicy());
                return BackendExecution.executed(c.getRawData());
            case "opt-1t":
                OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, optSingleThreadPolicy());
                return BackendExecution.executed(c.getRawData());
            case "opt-parallel":
                OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, optParallelPolicy());
                return BackendExecution.executed(c.getRawData());
            case "cuda":
                return CudaGemm.gemm(a, b, c, 1.0, 0.0)
                    ? BackendExecution.executed(c.getRawData())
                    : BackendExecution.skipped();
            default:
                throw new IllegalArgumentException("Unsupported backend: " + backend);
        }
    }

    private static DispatchPolicy naivePolicy() {
        return DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(false)
            .parallelism(1)
            .enableBlas3(false)
            .enableSimd(false)
            .naiveThreshold(Integer.MAX_VALUE)
            .blockedThreshold(Integer.MAX_VALUE)
            .blas3Threshold(Integer.MAX_VALUE)
            .build();
    }

    private static DispatchPolicy simdSingleThreadPolicy() {
        return DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(false)
            .parallelism(1)
            .naiveThreshold(1)
            .blockedThreshold(1)
            .blas3Threshold(1)
            .enableBlas3(true)
            .enableSimd(true)
            .simdThreshold(Integer.MAX_VALUE)
            .build();
    }

    private static DispatchPolicy blas3ParallelPolicy() {
        return DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(true)
            .parallelism(Math.max(1, Runtime.getRuntime().availableProcessors()))
            .parallelThreshold(1)
            .naiveThreshold(1)
            .blockedThreshold(1)
            .blas3Threshold(1)
            .enableBlas3(true)
            .enableSimd(true)
            .simdThreshold(Integer.MAX_VALUE)
            .build();
    }

    private static DispatchPolicy optSingleThreadPolicy() {
        return DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(false)
            .parallelism(1)
            .build();
    }

    private static DispatchPolicy optParallelPolicy() {
        return DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(true)
            .parallelism(Math.max(1, Runtime.getRuntime().availableProcessors()))
            .parallelThreshold(1)
            .build();
    }

    private static void assignBaselines(List<Row> rows) {
        Map<Integer, List<Row>> bySize = new LinkedHashMap<>();
        for (Row row : rows) {
            bySize.computeIfAbsent(row.size, ignored -> new ArrayList<>()).add(row);
        }

        for (List<Row> sizeRows : bySize.values()) {
            Row baseline = sizeRows.stream()
                .filter(row -> row.valid && "blas3-naive".equals(row.backend))
                .findFirst()
                .orElseGet(() -> sizeRows.stream()
                    .filter(row -> row.valid && "opt-1t".equals(row.backend))
                    .findFirst()
                    .orElse(null));

            if (baseline == null || baseline.medianMs == null || baseline.medianMs <= 0.0) {
                continue;
            }

            for (Row row : sizeRows) {
                if (!row.valid || row.medianMs == null || row.medianMs <= 0.0) {
                    continue;
                }
                row.baselineLabel = baseline.backend;
                row.speedupVsBaseline = baseline.medianMs / row.medianMs;
            }
        }
    }

    private static ErrorMetrics compare(double[] actual, double[] expected) {
        double diffSquares = 0.0;
        double maxAbsErr = 0.0;
        for (int i = 0; i < expected.length; i++) {
            double diff = actual[i] - expected[i];
            diffSquares += diff * diff;
            maxAbsErr = Math.max(maxAbsErr, Math.abs(diff));
        }
        double refNorm = GemmReference.frobenius(expected);
        double relResidual = refNorm == 0.0 ? 0.0 : Math.sqrt(diffSquares) / refNorm;
        return new ErrorMetrics(relResidual, maxAbsErr);
    }

    private static double checksum(double[] values) {
        double sum = 0.0;
        for (int i = 0; i < values.length; i++) {
            sum = Math.fma(values[i], i + 1.0, sum);
        }
        return sum;
    }

    private static int threadsForBackend(String backend) {
        if ("opt-parallel".equals(backend) || "blas3-parallel".equals(backend)) {
            return Math.max(1, Runtime.getRuntime().availableProcessors());
        }
        return 1;
    }

    private static String backendNotes(String backend) {
        switch (backend) {
            case "blas3-naive":
                return "BLAS3Kernels.gemm forced to scalar/naive via DispatchPolicy. This is a production scalar baseline, not the oracle.";
            case "blas3-simd-1t":
                return "BLAS3Kernels.gemm single-thread BLAS3 path; uses dgemmSimd when Vector API is available, else scalar fallback.";
            case "blas3-parallel":
                return "BLAS3Kernels.gemm parallel path; current implementation is ForkJoin plus IntStream-style scalar blocked work, not SIMD parallel.";
            case "opt-1t":
                return "OptimizedBLAS3.gemm with parallelism=1 and CUDA disabled.";
            case "opt-parallel":
                return "OptimizedBLAS3.gemm with available processor count and CUDA disabled. This is the main production CPU optimized path.";
            case "cuda":
                return "CudaGemm.gemm direct CUDA path. Timing may include allocation, transfer, and launch overhead.";
            default:
                return "";
        }
    }

    private static void printMetadata(MachineMetadata metadata) {
        System.out.println("=== GEMM Backend Comparison ===");
        System.out.println("timestamp_utc=" + metadata.timestampUtc);
        System.out.println("os=" + metadata.osName + " " + metadata.osVersion + " arch=" + metadata.osArch);
        System.out.println("cpu_model=" + metadata.cpuModel);
        System.out.println("available_processors=" + metadata.availableProcessors);
        System.out.println("java=" + metadata.javaVersion + " jvm=" + metadata.jvmName + " " + metadata.jvmVersion + " vendor=" + metadata.jvmVendor);
        System.out.println("max_heap_bytes=" + metadata.maxHeapBytes);
        System.out.println("vector_api_available=" + metadata.vectorApiAvailable + " preferred_vector_lanes=" + metadata.preferredVectorLanes);
        System.out.println("gemm_block_sizes=" + metadata.gemmBlockSizes);
        System.out.println("cuda_status=" + metadata.cudaStatus);
        System.out.println("git_commit=" + metadata.gitCommit);
        System.out.println("seed=" + metadata.seed);
        System.out.println("sizes=" + metadata.sizes);
        System.out.println("warmups=" + metadata.warmups + " iterations=" + metadata.iterations);
    }

    private static void printLegend() {
        System.out.println();
        System.out.println("Backend legend:");
        for (String backend : SUPPORTED_BACKENDS) {
            System.out.println("  " + backend + ": " + backendNotes(backend));
        }
    }

    private static void printTable(List<Row> rows) {
        System.out.println();
        System.out.printf(
            Locale.US,
            "%-5s %-16s %-7s %-8s %-10s %-11s %-12s %-10s %-10s %-12s%n",
            "size",
            "backend",
            "valid",
            "status",
            "oracle",
            "median_ms",
            "gflops",
            "speedup",
            "threads",
            "checksum"
        );
        for (Row row : rows) {
            System.out.printf(
                Locale.US,
                "%-5d %-16s %-7s %-8s %-10s %-11s %-12s %-10s %-10d %-12s%n",
                row.size,
                row.backend,
                row.valid,
                row.parityStatus,
                row.oracleType,
                formatNumber(row.medianMs),
                formatNumber(row.gflopsMedian),
                formatNumber(row.speedupVsBaseline),
                row.threads,
                formatNumber(row.checksum)
            );
        }
    }

    private static void writeCsv(List<Row> rows, Path output) {
        List<String> lines = new ArrayList<>();
        lines.add("timestamp,size,m,k,n,backend,threads,valid,parity_status,oracle_type,reference_backend,rel_residual,max_abs_err,warmups,iters,median_ms,mean_ms,stddev_ms,min_ms,p10_ms,p25_ms,p75_ms,p90_ms,gflops_median,speedup_vs_baseline,baseline_label,checksum,notes");
        for (Row row : rows) {
            lines.add(String.join(",",
                csv(row.timestamp),
                Integer.toString(row.size),
                Integer.toString(row.m),
                Integer.toString(row.k),
                Integer.toString(row.n),
                csv(row.backend),
                Integer.toString(row.threads),
                Boolean.toString(row.valid),
                csv(row.parityStatus),
                csv(row.oracleType),
                csv(row.referenceBackend),
                csvNumber(row.relResidual),
                csvNumber(row.maxAbsErr),
                Integer.toString(row.warmups),
                Integer.toString(row.iters),
                csvNumber(row.medianMs),
                csvNumber(row.meanMs),
                csvNumber(row.stddevMs),
                csvNumber(row.minMs),
                csvNumber(row.p10Ms),
                csvNumber(row.p25Ms),
                csvNumber(row.p75Ms),
                csvNumber(row.p90Ms),
                csvNumber(row.gflopsMedian),
                csvNumber(row.speedupVsBaseline),
                csv(row.baselineLabel),
                csvNumber(row.checksum),
                csv(row.notes)
            ));
        }
        try {
            Files.write(output, lines, StandardCharsets.UTF_8);
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    private static void writeJson(MachineMetadata metadata, List<Row> rows, Path stableCsv, Path stampedCsv, Path json) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("metadata", metadata.toMap());

        Map<String, Object> artifacts = new LinkedHashMap<>();
        artifacts.put("stable_csv", stableCsv.toAbsolutePath().toString());
        artifacts.put("timestamped_csv", stampedCsv.toAbsolutePath().toString());
        artifacts.put("json", json.toAbsolutePath().toString());
        root.put("artifacts", artifacts);

        List<Map<String, Object>> jsonRows = new ArrayList<>();
        for (Row row : rows) {
            jsonRows.add(row.toMap());
        }
        root.put("rows", jsonRows);

        try {
            OBJECT_MAPPER.writeValue(json.toFile(), root);
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    private static String csv(String value) {
        String safe = value == null ? "" : value;
        return "\"" + safe.replace("\"", "\"\"") + "\"";
    }

    private static String csvNumber(Double value) {
        if (value == null || value.isNaN() || value.isInfinite()) {
            return "\"\"";
        }
        return csv(String.format(Locale.US, "%.9f", value));
    }

    private static String formatNumber(Double value) {
        if (value == null || value.isNaN() || value.isInfinite()) {
            return "-";
        }
        return String.format(Locale.US, "%.4f", value);
    }

    private static String safeMessage(Throwable throwable) {
        String message = throwable.getMessage();
        return message == null || message.isBlank() ? "(no message)" : message.replace('\n', ' ');
    }

    private enum CudaMode {
        AUTO,
        ON,
        OFF;

        static CudaMode parse(String value) {
            String normalized = value == null ? "auto" : value.trim().toLowerCase(Locale.ROOT);
            switch (normalized) {
                case "auto":
                    return AUTO;
                case "on":
                    return ON;
                case "off":
                    return OFF;
                default:
                    throw new IllegalArgumentException("Unsupported gemm.cuda value: " + value + " (expected auto|on|off)");
            }
        }
    }

    private static final class Config {
        final List<Integer> sizes;
        final int warmups;
        final int iterations;
        final List<String> backends;
        final long seed;
        final int naiveMaxSize;
        final int independentOracleMaxSize;
        final CudaMode cudaMode;

        private Config(List<Integer> sizes,
                       int warmups,
                       int iterations,
                       List<String> backends,
                       long seed,
                       int naiveMaxSize,
                       int independentOracleMaxSize,
                       CudaMode cudaMode) {
            this.sizes = sizes;
            this.warmups = warmups;
            this.iterations = iterations;
            this.backends = backends;
            this.seed = seed;
            this.naiveMaxSize = naiveMaxSize;
            this.independentOracleMaxSize = independentOracleMaxSize;
            this.cudaMode = cudaMode;
        }

        static Config fromSystemProperties() {
            boolean quick = Boolean.parseBoolean(System.getProperty("gemm.quick", "false"));
            List<Integer> sizes = parseSizes(System.getProperty("gemm.sizes", quick ? "128,256" : "128,256,512"));
            int warmups = Integer.getInteger("gemm.warmups", quick ? 2 : 8);
            int iterations = Integer.getInteger("gemm.iters", quick ? 5 : 25);
            List<String> backends = parseBackends(System.getProperty(
                "gemm.backends",
                "blas3-naive,blas3-simd-1t,blas3-parallel,opt-1t,opt-parallel,cuda"
            ));
            long seed = Long.getLong("gemm.seed", 20260604L);
            int naiveMaxSize = Integer.getInteger("gemm.naiveMaxSize", 512);
            int independentOracleMaxSize = Integer.getInteger("gemm.independentOracleMaxSize", 512);
            CudaMode cudaMode = CudaMode.parse(System.getProperty("gemm.cuda", "auto"));

            return new Config(sizes, warmups, iterations, backends, seed, naiveMaxSize, independentOracleMaxSize, cudaMode);
        }

        private static List<Integer> parseSizes(String raw) {
            List<Integer> sizes = new ArrayList<>();
            for (String token : raw.split(",")) {
                String trimmed = token.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }
                int size = Integer.parseInt(trimmed);
                if (size <= 0) {
                    throw new IllegalArgumentException("gemm.sizes must contain positive integers: " + raw);
                }
                sizes.add(size);
            }
            if (sizes.isEmpty()) {
                throw new IllegalArgumentException("gemm.sizes must not be empty");
            }
            return Collections.unmodifiableList(sizes);
        }

        private static List<String> parseBackends(String raw) {
            List<String> backends = new ArrayList<>();
            for (String token : raw.split(",")) {
                String backend = token.trim();
                if (backend.isEmpty()) {
                    continue;
                }
                if (!SUPPORTED_BACKENDS.contains(backend)) {
                    throw new IllegalArgumentException("Unsupported gemm backend: " + backend);
                }
                backends.add(backend);
            }
            if (backends.isEmpty()) {
                throw new IllegalArgumentException("gemm.backends must not be empty");
            }
            return Collections.unmodifiableList(backends);
        }
    }

    private static final class MachineMetadata {
        final String timestampUtc;
        final String timestampStamp;
        final String osName;
        final String osVersion;
        final String osArch;
        final int availableProcessors;
        final String javaVersion;
        final String jvmName;
        final String jvmVersion;
        final String jvmVendor;
        final long maxHeapBytes;
        final Boolean vectorApiAvailable;
        final Integer preferredVectorLanes;
        final String gemmBlockSizes;
        final String cudaStatus;
        final String cpuModel;
        final long seed;
        final List<Integer> sizes;
        final int warmups;
        final int iterations;
        final String gitCommit;
        final String cudaMode;

        private MachineMetadata(String timestampUtc,
                                String timestampStamp,
                                String osName,
                                String osVersion,
                                String osArch,
                                int availableProcessors,
                                String javaVersion,
                                String jvmName,
                                String jvmVersion,
                                String jvmVendor,
                                long maxHeapBytes,
                                Boolean vectorApiAvailable,
                                Integer preferredVectorLanes,
                                String gemmBlockSizes,
                                String cudaStatus,
                                String cpuModel,
                                long seed,
                                List<Integer> sizes,
                                int warmups,
                                int iterations,
                                String gitCommit,
                                String cudaMode) {
            this.timestampUtc = timestampUtc;
            this.timestampStamp = timestampStamp;
            this.osName = osName;
            this.osVersion = osVersion;
            this.osArch = osArch;
            this.availableProcessors = availableProcessors;
            this.javaVersion = javaVersion;
            this.jvmName = jvmName;
            this.jvmVersion = jvmVersion;
            this.jvmVendor = jvmVendor;
            this.maxHeapBytes = maxHeapBytes;
            this.vectorApiAvailable = vectorApiAvailable;
            this.preferredVectorLanes = preferredVectorLanes;
            this.gemmBlockSizes = gemmBlockSizes;
            this.cudaStatus = cudaStatus;
            this.cpuModel = cpuModel;
            this.seed = seed;
            this.sizes = sizes;
            this.warmups = warmups;
            this.iterations = iterations;
            this.gitCommit = gitCommit;
            this.cudaMode = cudaMode;
        }

        static MachineMetadata collect(String timestampUtc, String timestampStamp, Config config) {
            RuntimeMXBean runtime = ManagementFactory.getRuntimeMXBean();
            Boolean vectorApiAvailable = safeVectorApiAvailable();
            Integer preferredVectorLanes = safePreferredVectorLanes();
            String gemmBlockSizes = safeGemmBlockSizes();
            String cudaStatus = safeCudaStatus(config.cudaMode);

            return new MachineMetadata(
                timestampUtc,
                timestampStamp,
                System.getProperty("os.name", "unknown"),
                System.getProperty("os.version", "unknown"),
                System.getProperty("os.arch", "unknown"),
                Runtime.getRuntime().availableProcessors(),
                System.getProperty("java.version", "unknown"),
                runtime.getVmName(),
                runtime.getVmVersion(),
                System.getProperty("java.vendor", "unknown"),
                Runtime.getRuntime().maxMemory(),
                vectorApiAvailable,
                preferredVectorLanes,
                gemmBlockSizes,
                cudaStatus,
                detectCpuModel(),
                config.seed,
                config.sizes,
                config.warmups,
                config.iterations,
                bestEffortGitCommit(),
                config.cudaMode.name().toLowerCase(Locale.ROOT)
            );
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("timestamp_utc", timestampUtc);
            map.put("timestamp_stamp", timestampStamp);
            map.put("os_name", osName);
            map.put("os_version", osVersion);
            map.put("os_arch", osArch);
            map.put("available_processors", availableProcessors);
            map.put("java_version", javaVersion);
            map.put("jvm_name", jvmName);
            map.put("jvm_version", jvmVersion);
            map.put("jvm_vendor", jvmVendor);
            map.put("max_heap_bytes", maxHeapBytes);
            map.put("vector_api_available", vectorApiAvailable);
            map.put("preferred_vector_lanes", preferredVectorLanes);
            map.put("gemm_block_sizes", gemmBlockSizes);
            map.put("cuda_status", cudaStatus);
            map.put("cpu_model", cpuModel);
            map.put("seed", seed);
            map.put("sizes", sizes);
            map.put("warmups", warmups);
            map.put("iterations", iterations);
            map.put("git_commit", gitCommit);
            map.put("cuda_mode", cudaMode);
            return map;
        }

        private static Boolean safeVectorApiAvailable() {
            try {
                return VectorSupport.isVectorApiAvailable();
            } catch (Throwable ex) {
                return null;
            }
        }

        private static Integer safePreferredVectorLanes() {
            try {
                return DoubleVector.SPECIES_PREFERRED.length();
            } catch (Throwable ex) {
                return null;
            }
        }

        private static String safeGemmBlockSizes() {
            try {
                return String.valueOf(GemmDispatch.computeBlockSizes());
            } catch (Throwable ex) {
                return "unavailable: " + ex.getClass().getSimpleName();
            }
        }

        private static String safeCudaStatus(CudaMode mode) {
            try {
                CudaSupport.refresh();
                boolean available = mode != CudaMode.OFF && CudaSupport.isCudaAvailable();
                return "mode=" + mode.name().toLowerCase(Locale.ROOT) + ",available=" + available;
            } catch (Throwable ex) {
                return "mode=" + mode.name().toLowerCase(Locale.ROOT) + ",error=" + ex.getClass().getSimpleName();
            }
        }

        private static String detectCpuModel() {
            String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
            if (os.startsWith("windows")) {
                String env = System.getenv("PROCESSOR_IDENTIFIER");
                return env == null || env.isBlank() ? "unknown" : env;
            }
            if (os.startsWith("linux")) {
                try {
                    for (String line : Files.readAllLines(Path.of("/proc/cpuinfo"), StandardCharsets.UTF_8)) {
                        if (line.startsWith("model name")) {
                            int idx = line.indexOf(':');
                            return idx >= 0 ? line.substring(idx + 1).trim() : line.trim();
                        }
                    }
                } catch (IOException ignored) {
                }
                return "unknown";
            }
            if (os.startsWith("mac")) {
                return runCommand(List.of("sysctl", "-n", "machdep.cpu.brand_string")).orElse("unknown");
            }
            return "unknown";
        }

        private static String bestEffortGitCommit() {
            return runCommand(List.of("git", "rev-parse", "HEAD")).orElse("unavailable");
        }

        private static Optional<String> runCommand(List<String> command) {
            try {
                Process process = new ProcessBuilder(command)
                    .directory(Path.of(System.getProperty("user.dir")).toFile())
                    .redirectErrorStream(true)
                    .start();
                byte[] bytes = process.getInputStream().readAllBytes();
                int exit = process.waitFor();
                if (exit != 0) {
                    return Optional.empty();
                }
                String output = new String(bytes, StandardCharsets.UTF_8).trim();
                return output.isEmpty() ? Optional.empty() : Optional.of(output);
            } catch (IOException | InterruptedException ex) {
                if (ex instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                }
                return Optional.empty();
            }
        }
    }

    private static final class OracleContext {
        final double[] reference;
        final String oracleType;
        final String referenceBackend;

        private OracleContext(double[] reference, String oracleType, String referenceBackend) {
            this.reference = reference;
            this.oracleType = oracleType;
            this.referenceBackend = referenceBackend;
        }
    }

    private static final class BackendExecution {
        final boolean executed;
        final double[] output;

        private BackendExecution(boolean executed, double[] output) {
            this.executed = executed;
            this.output = output;
        }

        static BackendExecution executed(double[] output) {
            return new BackendExecution(true, output.clone());
        }

        static BackendExecution skipped() {
            return new BackendExecution(false, null);
        }
    }

    private static final class ErrorMetrics {
        final double relResidual;
        final double maxAbsErr;

        private ErrorMetrics(double relResidual, double maxAbsErr) {
            this.relResidual = relResidual;
            this.maxAbsErr = maxAbsErr;
        }
    }

    private static final class TimingStats {
        final double medianMs;
        final double meanMs;
        final double stddevMs;
        final double minMs;
        final double p10Ms;
        final double p25Ms;
        final double p75Ms;
        final double p90Ms;
        final double gflopsMedian;
        final double checksum;
        final int innerRepeats;

        private TimingStats(double medianMs,
                            double meanMs,
                            double stddevMs,
                            double minMs,
                            double p10Ms,
                            double p25Ms,
                            double p75Ms,
                            double p90Ms,
                            double gflopsMedian,
                            double checksum,
                            int innerRepeats) {
            this.medianMs = medianMs;
            this.meanMs = meanMs;
            this.stddevMs = stddevMs;
            this.minMs = minMs;
            this.p10Ms = p10Ms;
            this.p25Ms = p25Ms;
            this.p75Ms = p75Ms;
            this.p90Ms = p90Ms;
            this.gflopsMedian = gflopsMedian;
            this.checksum = checksum;
            this.innerRepeats = innerRepeats;
        }

        static TimingStats from(double[] perCallNs, long flops, double checksum, int innerRepeats) {
            double[] sorted = perCallNs.clone();
            Arrays.sort(sorted);

            double sum = 0.0;
            double min = Double.POSITIVE_INFINITY;
            for (double sample : perCallNs) {
                sum += sample;
                min = Math.min(min, sample);
            }
            double mean = sum / perCallNs.length;
            double variance = 0.0;
            if (perCallNs.length > 1) {
                for (double sample : perCallNs) {
                    double diff = sample - mean;
                    variance += diff * diff;
                }
                variance /= (perCallNs.length - 1);
            }
            double medianNs = percentile(sorted, 0.50);
            double medianMs = medianNs / 1_000_000.0;
            double gflopsMedian = medianNs <= 0.0 ? Double.NaN : flops / medianNs;

            return new TimingStats(
                medianMs,
                mean / 1_000_000.0,
                Math.sqrt(variance) / 1_000_000.0,
                min / 1_000_000.0,
                percentile(sorted, 0.10) / 1_000_000.0,
                percentile(sorted, 0.25) / 1_000_000.0,
                percentile(sorted, 0.75) / 1_000_000.0,
                percentile(sorted, 0.90) / 1_000_000.0,
                gflopsMedian,
                checksum,
                innerRepeats
            );
        }

        private static double percentile(double[] sorted, double q) {
            if (sorted.length == 0) {
                return Double.NaN;
            }
            if (sorted.length == 1) {
                return sorted[0];
            }
            double position = q * (sorted.length - 1);
            int lower = (int) Math.floor(position);
            int upper = (int) Math.ceil(position);
            if (lower == upper) {
                return sorted[lower];
            }
            double weight = position - lower;
            return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
        }
    }

    private static final class Row {
        final String timestamp;
        final int size;
        final int m;
        final int k;
        final int n;
        final String backend;
        final int threads;
        final boolean valid;
        final String parityStatus;
        final String oracleType;
        final String referenceBackend;
        final Double relResidual;
        final Double maxAbsErr;
        final int warmups;
        final int iters;
        final Double medianMs;
        final Double meanMs;
        final Double stddevMs;
        final Double minMs;
        final Double p10Ms;
        final Double p25Ms;
        final Double p75Ms;
        final Double p90Ms;
        final Double gflopsMedian;
        Double speedupVsBaseline;
        String baselineLabel;
        final Double checksum;
        final String notes;

        private Row(String timestamp,
                    int size,
                    String backend,
                    int threads,
                    boolean valid,
                    String parityStatus,
                    String oracleType,
                    String referenceBackend,
                    Double relResidual,
                    Double maxAbsErr,
                    int warmups,
                    int iters,
                    Double medianMs,
                    Double meanMs,
                    Double stddevMs,
                    Double minMs,
                    Double p10Ms,
                    Double p25Ms,
                    Double p75Ms,
                    Double p90Ms,
                    Double gflopsMedian,
                    Double checksum,
                    String notes) {
            this.timestamp = timestamp;
            this.size = size;
            this.m = size;
            this.k = size;
            this.n = size;
            this.backend = backend;
            this.threads = threads;
            this.valid = valid;
            this.parityStatus = parityStatus;
            this.oracleType = oracleType;
            this.referenceBackend = referenceBackend;
            this.relResidual = relResidual;
            this.maxAbsErr = maxAbsErr;
            this.warmups = warmups;
            this.iters = iters;
            this.medianMs = medianMs;
            this.meanMs = meanMs;
            this.stddevMs = stddevMs;
            this.minMs = minMs;
            this.p10Ms = p10Ms;
            this.p25Ms = p25Ms;
            this.p75Ms = p75Ms;
            this.p90Ms = p90Ms;
            this.gflopsMedian = gflopsMedian;
            this.checksum = checksum;
            this.notes = notes;
        }

        static Row valid(String timestamp,
                         int size,
                         String backend,
                         int threads,
                         String oracleType,
                         String referenceBackend,
                         double relResidual,
                         double maxAbsErr,
                         int warmups,
                         int iters,
                         TimingStats timing,
                         String notes) {
            String fullNotes = notes + " inner_repeats=" + timing.innerRepeats + ".";
            return new Row(
                timestamp,
                size,
                backend,
                threads,
                true,
                "PASS",
                oracleType,
                referenceBackend,
                relResidual,
                maxAbsErr,
                warmups,
                iters,
                timing.medianMs,
                timing.meanMs,
                timing.stddevMs,
                timing.minMs,
                timing.p10Ms,
                timing.p25Ms,
                timing.p75Ms,
                timing.p90Ms,
                timing.gflopsMedian,
                timing.checksum,
                fullNotes
            );
        }

        static Row failed(String timestamp,
                          int size,
                          String backend,
                          int threads,
                          String oracleType,
                          String referenceBackend,
                          int warmups,
                          int iters,
                          double relResidual,
                          double maxAbsErr,
                          String notes) {
            return new Row(
                timestamp,
                size,
                backend,
                threads,
                false,
                "FAIL",
                oracleType,
                referenceBackend,
                relResidual,
                maxAbsErr,
                warmups,
                iters,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                notes
            );
        }

        static Row skipped(String timestamp,
                           int size,
                           String backend,
                           int threads,
                           String oracleType,
                           String referenceBackend,
                           int warmups,
                           int iters,
                           String notes) {
            return new Row(
                timestamp,
                size,
                backend,
                threads,
                false,
                "SKIPPED",
                oracleType,
                referenceBackend,
                null,
                null,
                warmups,
                iters,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                notes
            );
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("timestamp", timestamp);
            map.put("size", size);
            map.put("m", m);
            map.put("k", k);
            map.put("n", n);
            map.put("backend", backend);
            map.put("threads", threads);
            map.put("valid", valid);
            map.put("parity_status", parityStatus);
            map.put("oracle_type", oracleType);
            map.put("reference_backend", referenceBackend);
            map.put("rel_residual", relResidual);
            map.put("max_abs_err", maxAbsErr);
            map.put("warmups", warmups);
            map.put("iters", iters);
            map.put("median_ms", medianMs);
            map.put("mean_ms", meanMs);
            map.put("stddev_ms", stddevMs);
            map.put("min_ms", minMs);
            map.put("p10_ms", p10Ms);
            map.put("p25_ms", p25Ms);
            map.put("p75_ms", p75Ms);
            map.put("p90_ms", p90Ms);
            map.put("gflops_median", gflopsMedian);
            map.put("speedup_vs_baseline", speedupVsBaseline);
            map.put("baseline_label", baselineLabel);
            map.put("checksum", checksum);
            map.put("notes", notes);
            return map;
        }
    }
}
