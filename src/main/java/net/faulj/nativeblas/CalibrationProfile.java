package net.faulj.nativeblas;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import java.util.Properties;

final class CalibrationProfile {
    static final int CURRENT_VERSION = 1;
    static final CalibrationProfile EMPTY = new CalibrationProfile("", new Properties(), false);

    private final String path;
    private final Properties properties;
    private final boolean versionValid;

    private CalibrationProfile(String path, Properties properties, boolean versionValid) {
        this.path = path;
        this.properties = properties;
        this.versionValid = versionValid;
    }

    static CalibrationProfile load(String configuredPath) {
        if (configuredPath == null || configuredPath.isBlank()) {
            return EMPTY;
        }
        Properties loaded = new Properties();
        try (InputStream in = Files.newInputStream(Path.of(configuredPath))) {
            loaded.load(in);
        } catch (IOException | RuntimeException ignored) {
            loaded.clear();
        }
        boolean valid = parseInt(loaded.getProperty("version"), -1) == CURRENT_VERSION;
        return new CalibrationProfile(configuredPath, loaded, valid);
    }

    String path() {
        return path;
    }

    Optional<CalibrationBucket> bucket(AlgorithmDispatchRequest request) {
        if (!versionValid) {
            return Optional.empty();
        }
        String prefix = "bucket." + request.bucketKey() + ".";
        CalibrationStats javaStats = stats(prefix + "java.");
        CalibrationStats cppStats = stats(prefix + "cpp.");
        if (javaStats == null && cppStats == null) {
            return Optional.empty();
        }
        String correctness = properties.getProperty(prefix + "cpp.correctness", "UNKNOWN")
            .trim()
            .toUpperCase(Locale.ROOT);
        return Optional.of(new CalibrationBucket(javaStats, cppStats, "PASS".equals(correctness)));
    }

    private CalibrationStats stats(String prefix) {
        int samples = parseInt(properties.getProperty(prefix + "samples"), 0);
        double meanNanos = parseDouble(properties.getProperty(prefix + "meanNanos"), Double.NaN);
        double minNanos = parseDouble(properties.getProperty(prefix + "minNanos"), Double.NaN);
        double stddevNanos = parseDouble(properties.getProperty(prefix + "stddevNanos"), Double.NaN);
        if (samples <= 0 || !Double.isFinite(meanNanos) || meanNanos <= 0.0) {
            return null;
        }
        return new CalibrationStats(samples, meanNanos, minNanos, stddevNanos);
    }

    private static int parseInt(String value, int fallback) {
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static double parseDouble(String value, double fallback) {
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    record CalibrationStats(int samples, double meanNanos, double minNanos, double stddevNanos) {
    }

    record CalibrationBucket(CalibrationStats javaStats, CalibrationStats cppStats, boolean cppCorrectnessPass) {
    }
}
