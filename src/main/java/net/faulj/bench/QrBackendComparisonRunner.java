package net.faulj.bench;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.BackendRegistry;
import net.faulj.nativeblas.BackendSnapshot;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;

/**
 * Compare Java and C++ QR backends across a size sweep and write calibrated dispatch buckets.
 */
public final class QrBackendComparisonRunner {
    private QrBackendComparisonRunner() {
    }

    public static void main(String[] args) {
        Shape[] shapes = new Shape[] {
            new Shape(32, 32), new Shape(48, 48), new Shape(64, 64), new Shape(96, 96),
            new Shape(128, 128), new Shape(160, 160), new Shape(192, 192), new Shape(256, 256),
            new Shape(384, 384), new Shape(512, 512), new Shape(768, 768), new Shape(1024, 1024),
            new Shape(1536, 1536), new Shape(2048, 2048)
        };
        int warmupRuns = 2;
        int measuredRuns = 3;
        Mode mode = Mode.FULL;
        Path calibrationOut = null;

        for (String arg : args) {
            if (arg == null) {
                continue;
            }
            if (arg.startsWith("--sizes=")) {
                shapes = squareShapes(parseSizes(arg.substring("--sizes=".length()), extractSquareSizes(shapes)));
            } else if (arg.startsWith("--shapes=")) {
                shapes = parseShapes(arg.substring("--shapes=".length()), shapes);
            } else if (arg.startsWith("--warmup=")) {
                warmupRuns = parsePositiveInt(arg.substring("--warmup=".length()), warmupRuns);
            } else if (arg.startsWith("--runs=")) {
                measuredRuns = parsePositiveInt(arg.substring("--runs=".length()), measuredRuns);
            } else if (arg.startsWith("--mode=")) {
                mode = Mode.parse(arg.substring("--mode=".length()));
            } else if (arg.startsWith("--calibrationOut=")) {
                calibrationOut = parseCalibrationOut(arg.substring("--calibrationOut=".length()).trim());
            }
        }

        System.out.println("QR_BACKEND_COMPARISON");
        System.out.println("mode=" + mode.cliName);
        System.out.println("warmupRuns=" + warmupRuns);
        System.out.println("measuredRuns=" + measuredRuns);

        Map<String, String> suggestedProperties = new java.util.LinkedHashMap<>();
        for (Shape shape : shapes) {
            Matrix a = randomMatrix(shape.rows, shape.cols, 1000L + shape.rows * 100_000L + shape.cols);
            Result javaResult = runBackend("java", a, mode, warmupRuns, measuredRuns);
            Result cppResult = runBackend("cpp", a, mode, warmupRuns, measuredRuns);
            String winner = javaResult.bestSeconds <= cppResult.bestSeconds ? "java" : "cpp";
            addCalibrationProperties(suggestedProperties, shape, mode, javaResult, cppResult, measuredRuns);

            System.out.printf(Locale.ROOT, "shape=%dx%d family=%s bandMetric=%d%n",
                shape.rows, shape.cols, shape.family().propertyKey(), shape.bandMetric());
            printResult("java", javaResult);
            printResult("cpp", cppResult);
            System.out.println("winner=" + winner);
            System.out.println("cpp_correctness=" + (cppResult.correctnessPass ? "PASS" : "FAIL"));
            if (Double.isFinite(cppResult.residual)) {
                System.out.printf(Locale.ROOT, "cpp_residual=%.6e%n", cppResult.residual);
            }
            if (Double.isFinite(cppResult.orthogonality)) {
                System.out.printf(Locale.ROOT, "cpp_orthogonality=%.6e%n", cppResult.orthogonality);
            }
        }

        if (calibrationOut != null) {
            writeCalibrationFile(calibrationOut, suggestedProperties, mode, warmupRuns, measuredRuns, shapes);
            System.out.println("calibration_written=" + calibrationOut.toAbsolutePath());
        }
    }

    private static Result runBackend(String backend, Matrix a, Mode mode, int warmupRuns, int measuredRuns) {
        System.setProperty("jlc.backend", "cpp".equals(backend) ? "native" : "java");
        System.setProperty("jlc.algorithm.qr.backend", "cpp".equals(backend) ? "cpp" : "java");

        for (int i = 0; i < warmupRuns; i++) {
            execute(a, mode);
        }

        double flops = (4.0 / 3.0) * a.getRowCount() * (double) a.getColumnCount() * a.getColumnCount();
        double best = Double.POSITIVE_INFINITY;
        double total = 0.0;
        for (int i = 0; i < measuredRuns; i++) {
            long start = System.nanoTime();
            execute(a, mode);
            double seconds = (System.nanoTime() - start) / 1e9;
            best = Math.min(best, seconds);
            total += seconds;
        }

        BackendSnapshot snapshot = BackendRegistry.snapshot();
        Correctness correctness = "cpp".equals(backend) ? validateCpp(a, mode) : Correctness.passed();
        return new Result(snapshot, best, total / measuredRuns, flops,
            correctness.pass, correctness.residual, correctness.orthogonality);
    }

    private static void execute(Matrix a, Mode mode) {
        if (mode == Mode.FACTORIZE) {
            HouseholderQR.factorize(a);
            return;
        }
        QRResult result = mode == Mode.THIN ? HouseholderQR.decomposeThin(a) : HouseholderQR.decompose(a);
        if (result.getR().getRowCount() == 0) {
            throw new IllegalStateException("Unexpected empty QR result");
        }
    }

    private static void printResult(String label, Result result) {
        System.out.println(label + "_requested=" + result.snapshot.requestedBackend().id());
        System.out.println(label + "_active=" + result.snapshot.activeBackend());
        System.out.printf(Locale.ROOT, "%s_best_ms=%.6f%n", label, result.bestSeconds * 1000.0);
        System.out.printf(Locale.ROOT, "%s_mean_ms=%.6f%n", label, result.meanSeconds * 1000.0);
        System.out.printf(Locale.ROOT, "%s_best_gflops=%.6f%n", label, result.flops / result.bestSeconds / 1e9);
        System.out.printf(Locale.ROOT, "%s_mean_gflops=%.6f%n", label, result.flops / result.meanSeconds / 1e9);
    }

    private static Correctness validateCpp(Matrix a, Mode mode) {
        try {
            Mode validationMode = mode == Mode.FACTORIZE ? Mode.FULL : mode;
            QRResult result = validationMode == Mode.THIN ? HouseholderQR.decomposeThin(a) : HouseholderQR.decompose(a);
            int k = Math.min(a.getRowCount(), a.getColumnCount());
            int expectedQCols = validationMode == Mode.THIN ? k : a.getRowCount();
            int expectedRRows = expectedQCols;
            double residual = reconstructionResidual(a, result.getQ(), result.getR());
            double orthogonality = orthogonalityError(result.getQ());
            boolean dimensionsPass = result.getQ().getRowCount() == a.getRowCount()
                && result.getQ().getColumnCount() == expectedQCols
                && result.getR().getRowCount() == expectedRRows
                && result.getR().getColumnCount() == a.getColumnCount();
            boolean pass = dimensionsPass && residual < 1e-8 && orthogonality < 1e-8;
            return new Correctness(pass, residual, orthogonality);
        } catch (RuntimeException | LinkageError ignored) {
            return Correctness.failed();
        }
    }

    private static double reconstructionResidual(Matrix a, Matrix q, Matrix r) {
        double[] aData = a.getRawData();
        double[] qData = q.getRawData();
        double[] rData = r.getRawData();
        int rows = a.getRowCount();
        int cols = a.getColumnCount();
        int qCols = q.getColumnCount();
        double diff = 0.0;
        double norm = 0.0;
        for (int row = 0; row < rows; row++) {
            int aBase = row * cols;
            int qBase = row * qCols;
            for (int col = 0; col < cols; col++) {
                double reconstructed = 0.0;
                for (int mid = 0; mid < qCols; mid++) {
                    reconstructed = Math.fma(qData[qBase + mid], rData[mid * cols + col], reconstructed);
                }
                double delta = aData[aBase + col] - reconstructed;
                diff = Math.fma(delta, delta, diff);
                norm = Math.fma(aData[aBase + col], aData[aBase + col], norm);
            }
        }
        return Math.sqrt(diff) / Math.max(1.0, Math.sqrt(norm));
    }

    private static double orthogonalityError(Matrix q) {
        double[] qData = q.getRawData();
        int rows = q.getRowCount();
        int cols = q.getColumnCount();
        double diff = 0.0;
        for (int left = 0; left < cols; left++) {
            for (int right = 0; right < cols; right++) {
                double dot = 0.0;
                for (int row = 0; row < rows; row++) {
                    dot = Math.fma(qData[row * cols + left], qData[row * cols + right], dot);
                }
                double expected = left == right ? 1.0 : 0.0;
                double delta = dot - expected;
                diff = Math.fma(delta, delta, diff);
            }
        }
        return Math.sqrt(diff);
    }

    private static void addCalibrationProperties(Map<String, String> out, Shape shape, Mode mode,
                                                 Result javaResult, Result cppResult, int measuredRuns) {
        String prefix = "bucket.qr." + mode.dispatchMode + "." + shape.family().propertyKey() + "."
            + shape.sizeBand() + ".1.";
        out.put(prefix + "java.samples", Integer.toString(measuredRuns));
        out.put(prefix + "java.meanNanos", Long.toString(Math.round(javaResult.meanSeconds * 1e9)));
        out.put(prefix + "java.minNanos", Long.toString(Math.round(javaResult.bestSeconds * 1e9)));
        out.put(prefix + "cpp.samples", Integer.toString(measuredRuns));
        out.put(prefix + "cpp.meanNanos", Long.toString(Math.round(cppResult.meanSeconds * 1e9)));
        out.put(prefix + "cpp.minNanos", Long.toString(Math.round(cppResult.bestSeconds * 1e9)));
        out.put(prefix + "cpp.correctness", cppResult.correctnessPass ? "PASS" : "FAIL");
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return Matrix.wrap(data, rows, cols);
    }

    private static int[] parseSizes(String value, int[] fallback) {
        try {
            String[] parts = value.split(",");
            List<Integer> sizes = new ArrayList<>();
            for (String part : parts) {
                int parsed = Integer.parseInt(part.trim());
                if (parsed > 0) {
                    sizes.add(parsed);
                }
            }
            if (sizes.isEmpty()) {
                return fallback;
            }
            int[] out = new int[sizes.size()];
            for (int i = 0; i < sizes.size(); i++) {
                out[i] = sizes.get(i);
            }
            return out;
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static Shape[] parseShapes(String value, Shape[] fallback) {
        try {
            String[] parts = value.split(",");
            List<Shape> parsed = new ArrayList<>();
            for (String part : parts) {
                String token = part.trim().toLowerCase(Locale.ROOT);
                if (token.isEmpty()) {
                    continue;
                }
                int separator = token.indexOf('x');
                if (separator <= 0 || separator >= token.length() - 1) {
                    continue;
                }
                int rows = Integer.parseInt(token.substring(0, separator).trim());
                int cols = Integer.parseInt(token.substring(separator + 1).trim());
                if (rows > 0 && cols > 0) {
                    parsed.add(new Shape(rows, cols));
                }
            }
            return parsed.isEmpty() ? fallback : parsed.toArray(Shape[]::new);
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static int[] extractSquareSizes(Shape[] shapes) {
        int[] out = new int[shapes.length];
        for (int i = 0; i < shapes.length; i++) {
            out[i] = shapes[i].problemSize();
        }
        return out;
    }

    private static Shape[] squareShapes(int[] sizes) {
        Shape[] shapes = new Shape[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            shapes[i] = new Shape(sizes[i], sizes[i]);
        }
        return shapes;
    }

    private static int parsePositiveInt(String value, int fallback) {
        try {
            int parsed = Integer.parseInt(value.trim());
            return parsed > 0 ? parsed : fallback;
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static Path parseCalibrationOut(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Path.of(value).toAbsolutePath().normalize();
    }

    private static void writeCalibrationFile(Path calibrationOut, Map<String, String> suggestedProperties, Mode mode,
                                             int warmupRuns, int measuredRuns, Shape[] shapes) {
        Properties properties = new Properties();
        if (Files.exists(calibrationOut)) {
            try (InputStream in = Files.newInputStream(calibrationOut)) {
                properties.load(in);
            } catch (IOException ignored) {
                properties.clear();
            }
        }
        properties.setProperty("version", "1");
        properties.setProperty("generatedAt", OffsetDateTime.now().toString());
        properties.setProperty("qr.mode." + mode.cliName + ".warmupRuns", Integer.toString(warmupRuns));
        properties.setProperty("qr.mode." + mode.cliName + ".measuredRuns", Integer.toString(measuredRuns));
        properties.setProperty("qr.mode." + mode.cliName + ".shapeCount", Integer.toString(shapes.length));
        properties.setProperty("os.name", System.getProperty("os.name", "unknown"));
        properties.setProperty("os.arch", System.getProperty("os.arch", "unknown"));
        properties.setProperty("java.version", System.getProperty("java.version", "unknown"));
        properties.setProperty("availableProcessors",
            Integer.toString(Runtime.getRuntime().availableProcessors()));
        for (Map.Entry<String, String> entry : suggestedProperties.entrySet()) {
            properties.setProperty(entry.getKey(), entry.getValue());
        }
        try {
            Path parent = calibrationOut.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            try (OutputStream out = Files.newOutputStream(calibrationOut)) {
                properties.store(out, "Algorithm dispatch calibration generated by QrBackendComparisonRunner");
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write QR calibration file: " + calibrationOut, e);
        }
    }

    private record Result(BackendSnapshot snapshot, double bestSeconds, double meanSeconds, double flops,
                          boolean correctnessPass, double residual, double orthogonality) {
    }

    private record Correctness(boolean pass, double residual, double orthogonality) {
        static Correctness passed() {
            return new Correctness(true, Double.NaN, Double.NaN);
        }

        static Correctness failed() {
            return new Correctness(false, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        }
    }

    private enum Mode {
        FACTORIZE("factorize", "factorize_only"),
        THIN("thin", "decompose_thin"),
        FULL("full", "decompose_full");

        private final String cliName;
        private final String dispatchMode;

        Mode(String cliName, String dispatchMode) {
            this.cliName = cliName;
            this.dispatchMode = dispatchMode;
        }

        static Mode parse(String value) {
            if (value == null) {
                return FULL;
            }
            return switch (value.trim().toLowerCase(Locale.ROOT)) {
                case "factorize", "factorize_only" -> FACTORIZE;
                case "thin", "decompose_thin" -> THIN;
                case "full", "decompose", "decompose_full" -> FULL;
                default -> FULL;
            };
        }
    }

    private record Shape(int rows, int cols) {
        int problemSize() {
            return Math.max(rows, cols);
        }

        int bandMetric() {
            return family() == QrShapeFamily.SQUARE ? Math.max(rows, cols) : Math.min(rows, cols);
        }

        QrShapeFamily family() {
            if ((long) rows >= 2L * (long) cols) {
                return QrShapeFamily.TALL;
            }
            if ((long) cols >= 2L * (long) rows) {
                return QrShapeFamily.WIDE;
            }
            return QrShapeFamily.SQUARE;
        }

        String sizeBand() {
            int dominant = Math.max(rows, cols);
            if (dominant <= 128) {
                return "small";
            }
            if (dominant <= 512) {
                return "medium";
            }
            return "large";
        }
    }

    private enum QrShapeFamily {
        SQUARE("square"),
        TALL("tall"),
        WIDE("wide");

        private final String propertyKey;

        QrShapeFamily(String propertyKey) {
            this.propertyKey = propertyKey;
        }

        String propertyKey() {
            return propertyKey;
        }
    }
}
