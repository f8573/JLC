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
import java.util.TreeMap;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;

/**
 * Compare Java and native QR backends across a size sweep and print a suggested auto-band config.
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
        String mode = "decompose";
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
                mode = arg.substring("--mode=".length()).trim().toLowerCase(Locale.ROOT);
            } else if (arg.startsWith("--calibrationOut=")) {
                calibrationOut = parseCalibrationOut(arg.substring("--calibrationOut=".length()).trim());
            }
        }

        boolean factorizeOnly = "factorize".equals(mode);
        System.out.println("QR_BACKEND_COMPARISON");
        System.out.println("mode=" + mode);
        System.out.println("warmupRuns=" + warmupRuns);
        System.out.println("measuredRuns=" + measuredRuns);

        List<Winner> winners = new ArrayList<>();
        for (Shape shape : shapes) {
            Matrix a = randomMatrix(shape.rows, shape.cols, 1000L + shape.rows * 100_000L + shape.cols);
            Result javaResult = runBackend("java", a, factorizeOnly, warmupRuns, measuredRuns);
            Result nativeResult = runBackend("native", a, factorizeOnly, warmupRuns, measuredRuns);
            String winner = javaResult.bestSeconds <= nativeResult.bestSeconds ? "java" : "native";
            winners.add(new Winner(shape, winner));

            System.out.printf(Locale.ROOT, "shape=%dx%d family=%s bandMetric=%d%n",
                shape.rows, shape.cols, shape.family().propertyKey(), shape.bandMetric());
            printResult("java", javaResult);
            printResult("native", nativeResult);
            System.out.println("winner=" + winner);
        }

        Map<String, String> suggestedProperties = new TreeMap<>();
        for (QrShapeFamily family : QrShapeFamily.values()) {
            String suggestedGrid = buildSuggestedGrid(winners, family);
            if (!suggestedGrid.isEmpty()) {
                System.out.println("suggested_" + family.propertyKey() + "_grid=" + suggestedGrid);
                if (factorizeOnly) {
                    String propertyKey = "jlc.native.qr." + family.propertyKey() + ".factorizeGrid";
                    System.out.println("suggested_property=" + propertyKey + "=" + suggestedGrid);
                    suggestedProperties.put(propertyKey, suggestedGrid);
                } else {
                    String propertyKey = "jlc.native.qr." + family.propertyKey() + ".decomposeGrid";
                    System.out.println("suggested_property=" + propertyKey + "=" + suggestedGrid);
                    suggestedProperties.put(propertyKey, suggestedGrid);
                }
            }
            String suggestedBands = buildSuggestedBands(winners, family);
            if (suggestedBands.isEmpty()) {
                continue;
            }
            System.out.println("suggested_" + family.propertyKey() + "_bands=" + suggestedBands);
            if (factorizeOnly) {
                String propertyKey = "jlc.native.qr." + family.propertyKey() + ".factorizeBands";
                System.out.println("suggested_property=" + propertyKey + "=" + suggestedBands);
                suggestedProperties.put(propertyKey, suggestedBands);
            } else {
                String propertyKey = "jlc.native.qr." + family.propertyKey() + ".decomposeBands";
                System.out.println("suggested_property=" + propertyKey + "=" + suggestedBands);
                suggestedProperties.put(propertyKey, suggestedBands);
            }
        }

        if (calibrationOut != null) {
            writeCalibrationFile(calibrationOut, suggestedProperties, mode, warmupRuns, measuredRuns, shapes);
            System.out.println("calibration_written=" + calibrationOut.toAbsolutePath());
        }
    }

    private static Result runBackend(String backend, Matrix a, boolean factorizeOnly, int warmupRuns, int measuredRuns) {
        System.setProperty("jlc.backend", "native".equals(backend) ? "native" : "java");
        System.setProperty("jlc.native.qr.provider", "native".equals(backend) ? "builtin" : "java");
        System.setProperty("jlc.native.qr.minSize", "1");

        for (int i = 0; i < warmupRuns; i++) {
            execute(a, factorizeOnly);
        }

        double flops = (4.0 / 3.0) * a.getRowCount() * (double) a.getColumnCount() * a.getColumnCount();
        double best = Double.POSITIVE_INFINITY;
        double total = 0.0;
        for (int i = 0; i < measuredRuns; i++) {
            long start = System.nanoTime();
            execute(a, factorizeOnly);
            double seconds = (System.nanoTime() - start) / 1e9;
            best = Math.min(best, seconds);
            total += seconds;
        }

        BackendSnapshot snapshot = BackendRegistry.snapshot();
        return new Result(snapshot, best, total / measuredRuns, flops);
    }

    private static void execute(Matrix a, boolean factorizeOnly) {
        if (factorizeOnly) {
            HouseholderQR.factorize(a);
            return;
        }
        QRResult result = HouseholderQR.decompose(a);
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

    private static String buildSuggestedBands(List<Winner> winners, QrShapeFamily family) {
        TreeMap<Integer, String> winnersByMetric = new TreeMap<>();
        for (Winner winner : winners.stream()
            .filter(winner -> winner.shape.family() == family)
            .sorted((left, right) -> Integer.compare(left.shape.bandMetric(), right.shape.bandMetric()))
            .toList()) {
            String existing = winnersByMetric.putIfAbsent(winner.shape.bandMetric(), winner.backend);
            if (existing != null && !existing.equals(winner.backend)) {
                return "";
            }
        }
        if (winnersByMetric.isEmpty()) {
            return "";
        }
        List<String> bands = new ArrayList<>();
        int bandStart = 1;
        Integer previousMetric = null;
        String current = null;
        for (Map.Entry<Integer, String> entry : winnersByMetric.entrySet()) {
            int metric = entry.getKey();
            String backend = entry.getValue();
            if (current == null) {
                current = backend;
            } else if (!backend.equals(current)) {
                bands.add(bandStart + "-" + previousMetric + ":" + current);
                bandStart = previousMetric + 1;
                current = backend;
            }
            previousMetric = metric;
        }
        bands.add(bandStart + "+:" + current);
        return String.join(",", bands);
    }

    private static String buildSuggestedGrid(List<Winner> winners, QrShapeFamily family) {
        Map<Integer, TreeMap<Integer, String>> winnersByShortThenLong = new TreeMap<>();
        for (Winner winner : winners) {
            if (winner.shape.family() != family) {
                continue;
            }
            winnersByShortThenLong
                .computeIfAbsent(winner.shape.bandMetric(), ignored -> new TreeMap<>())
                .put(winner.shape.problemSize(), winner.backend);
        }
        if (winnersByShortThenLong.isEmpty()) {
            return "";
        }

        List<String> rules = new ArrayList<>();
        for (Map.Entry<Integer, TreeMap<Integer, String>> shortEntry : winnersByShortThenLong.entrySet()) {
            int shortDim = shortEntry.getKey();
            TreeMap<Integer, String> longWinners = shortEntry.getValue();
            Integer previousLong = null;
            int longBandStart = 1;
            String current = null;
            for (Map.Entry<Integer, String> longEntry : longWinners.entrySet()) {
                int longDim = longEntry.getKey();
                String backend = longEntry.getValue();
                if (current == null) {
                    current = backend;
                } else if (!backend.equals(current)) {
                    rules.add(shortDim + "-" + shortDim + "x" + longBandStart + "-" + previousLong + ":" + current);
                    longBandStart = previousLong + 1;
                    current = backend;
                }
                previousLong = longDim;
            }
            if (current != null) {
                rules.add(shortDim + "-" + shortDim + "x" + longBandStart + "+:" + current);
            }
        }
        return String.join(",", rules);
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

    private static void writeCalibrationFile(Path calibrationOut, Map<String, String> suggestedProperties, String mode,
                                             int warmupRuns, int measuredRuns, Shape[] shapes) {
        Properties properties = new Properties();
        if (Files.exists(calibrationOut)) {
            try (InputStream in = Files.newInputStream(calibrationOut)) {
                properties.load(in);
            } catch (IOException ignored) {
                properties.clear();
            }
        }
        properties.setProperty("jlc.native.qr.calibration.generatedAt", OffsetDateTime.now().toString());
        properties.setProperty("jlc.native.qr.calibration.mode." + mode + ".warmupRuns", Integer.toString(warmupRuns));
        properties.setProperty("jlc.native.qr.calibration.mode." + mode + ".measuredRuns", Integer.toString(measuredRuns));
        properties.setProperty("jlc.native.qr.calibration.mode." + mode + ".shapeCount", Integer.toString(shapes.length));
        properties.setProperty("jlc.native.qr.calibration.os.name", System.getProperty("os.name", "unknown"));
        properties.setProperty("jlc.native.qr.calibration.os.arch", System.getProperty("os.arch", "unknown"));
        properties.setProperty("jlc.native.qr.calibration.java.version", System.getProperty("java.version", "unknown"));
        properties.setProperty("jlc.native.qr.calibration.availableProcessors",
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
                properties.store(out, "QR backend calibration generated by QrBackendComparisonRunner");
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write QR calibration file: " + calibrationOut, e);
        }
    }

    private record Result(BackendSnapshot snapshot, double bestSeconds, double meanSeconds, double flops) {
    }

    private record Winner(Shape shape, String backend) {
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
