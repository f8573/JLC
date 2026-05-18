package net.faulj.bench;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import net.faulj.nativeblas.BackendSnapshot;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Random;

final class BackendComparisonSupport {
    private BackendComparisonSupport() {
    }

    static void printResult(String label, Result result) {
        System.out.println(label + "_requested=" + result.snapshot().requestedBackend().id());
        System.out.println(label + "_active=" + result.snapshot().activeBackend());
        System.out.printf(Locale.ROOT, "%s_best_ms=%.6f%n", label, result.bestSeconds() * 1000.0);
        System.out.printf(Locale.ROOT, "%s_mean_ms=%.6f%n", label, result.meanSeconds() * 1000.0);
        if (result.flops() > 0.0) {
            System.out.printf(Locale.ROOT, "%s_best_gflops=%.6f%n", label, result.bestGflops());
            System.out.printf(Locale.ROOT, "%s_mean_gflops=%.6f%n", label, result.meanGflops());
        }
    }

    static void addCalibrationProperties(Map<String, String> out, String algorithm, String mode, Shape shape,
                                         Result javaResult, Result cppResult, int measuredRuns) {
        String prefix = "bucket." + normalizeToken(algorithm) + "." + normalizeToken(mode) + "."
            + shape.family().propertyKey() + "." + shape.sizeBand() + ".1.";
        out.put(prefix + "java.samples", Integer.toString(measuredRuns));
        out.put(prefix + "java.meanNanos", Long.toString(Math.round(javaResult.meanSeconds() * 1e9)));
        out.put(prefix + "java.minNanos", Long.toString(Math.round(javaResult.bestSeconds() * 1e9)));
        out.put(prefix + "cpp.samples", Integer.toString(measuredRuns));
        out.put(prefix + "cpp.meanNanos", Long.toString(Math.round(cppResult.meanSeconds() * 1e9)));
        out.put(prefix + "cpp.minNanos", Long.toString(Math.round(cppResult.bestSeconds() * 1e9)));
        out.put(prefix + "cpp.correctness", cppResult.correctnessPass() ? "PASS" : "FAIL");
    }

    static void writeCalibrationFile(Path calibrationOut, Map<String, String> suggestedProperties,
                                     String algorithm, String mode,
                                     int warmupRuns, int measuredRuns, Shape[] shapes,
                                     String comment) {
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
        properties.setProperty(normalizeToken(algorithm) + ".mode." + normalizeToken(mode) + ".warmupRuns",
            Integer.toString(warmupRuns));
        properties.setProperty(normalizeToken(algorithm) + ".mode." + normalizeToken(mode) + ".measuredRuns",
            Integer.toString(measuredRuns));
        properties.setProperty(normalizeToken(algorithm) + ".mode." + normalizeToken(mode) + ".shapeCount",
            Integer.toString(shapes.length));
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
                properties.store(out, comment);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write calibration file: " + calibrationOut, e);
        }
    }

    static int[] parseSizes(String value, int[] fallback) {
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

    static Shape[] parseShapes(String value, Shape[] fallback) {
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

    static Shape[] squareShapes(int[] sizes) {
        Shape[] shapes = new Shape[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            shapes[i] = new Shape(sizes[i], sizes[i]);
        }
        return shapes;
    }

    static int[] extractSquareSizes(Shape[] shapes) {
        int[] out = new int[shapes.length];
        for (int i = 0; i < shapes.length; i++) {
            out[i] = shapes[i].problemSize();
        }
        return out;
    }

    static int parsePositiveInt(String value, int fallback) {
        try {
            int parsed = Integer.parseInt(value.trim());
            return parsed > 0 ? parsed : fallback;
        } catch (Exception ignored) {
            return fallback;
        }
    }

    static Path parseCalibrationOut(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Path.of(value).toAbsolutePath().normalize();
    }

    static Matrix randomMatrix(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return Matrix.wrap(data, rows, cols);
    }

    static Matrix randomDiagonallyDominant(int n, long seed) {
        Random random = new Random(seed);
        double[] data = new double[n * n];
        for (int row = 0; row < n; row++) {
            double rowSum = 0.0;
            int rowOffset = row * n;
            for (int col = 0; col < n; col++) {
                if (row == col) {
                    continue;
                }
                double value = random.nextDouble() * 0.4 - 0.2;
                data[rowOffset + col] = value;
                rowSum += Math.abs(value);
            }
            data[rowOffset + row] = rowSum + 1.0 + random.nextDouble();
        }
        return Matrix.wrap(data, n, n);
    }

    static Matrix randomPositiveDefinite(int n, long seed) {
        Matrix base = randomMatrix(n, n, seed);
        Matrix spd = base.transpose().multiply(base);
        double[] data = spd.getRawData().clone();
        for (int i = 0; i < n; i++) {
            data[i * n + i] += n;
        }
        return Matrix.wrap(data, n, n);
    }

    static double relativeDifference(Matrix left, Matrix right) {
        return MatrixUtils.relativeError(left, right);
    }

    static String normalizeToken(String value) {
        return value.trim().toLowerCase(Locale.ROOT).replace('-', '_');
    }

    record Result(BackendSnapshot snapshot, double bestSeconds, double meanSeconds, double flops,
                  boolean correctnessPass, double residual, double orthogonality) {
        double bestGflops() {
            return flops > 0.0 ? flops / bestSeconds / 1e9 : Double.NaN;
        }

        double meanGflops() {
            return flops > 0.0 ? flops / meanSeconds / 1e9 : Double.NaN;
        }
    }

    record Correctness(boolean pass, double residual, double orthogonality) {
        static Correctness failed() {
            return new Correctness(false, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        }
    }

    record Shape(int rows, int cols) {
        int problemSize() {
            return Math.max(rows, cols);
        }

        ShapeFamily family() {
            if ((long) rows >= 2L * (long) cols) {
                return ShapeFamily.TALL;
            }
            if ((long) cols >= 2L * (long) rows) {
                return ShapeFamily.WIDE;
            }
            return ShapeFamily.SQUARE;
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

    enum ShapeFamily {
        SQUARE("square"),
        TALL("tall"),
        WIDE("wide");

        private final String propertyKey;

        ShapeFamily(String propertyKey) {
            this.propertyKey = propertyKey;
        }

        String propertyKey() {
            return propertyKey;
        }
    }
}
