package net.faulj.nativeblas;

import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

final class AlgorithmDispatch {
    private static final Object PROFILE_LOCK = new Object();
    private static final AtomicReference<LoadedProfile> PROFILE = new AtomicReference<>();

    private AlgorithmDispatch() {
    }

    static boolean shouldUseCpp(AlgorithmDispatchRequest request, NativeContext nativeContext) {
        AlgorithmBackend configured = configuredBackend(request.algorithm());
        if (configured == AlgorithmBackend.JAVA) {
            return false;
        }
        if (nativeContext == null || !nativeContext.isAvailable()) {
            return false;
        }
        if (configured == AlgorithmBackend.CPP) {
            return true;
        }

        Optional<CalibrationProfile.CalibrationBucket> bucket = calibrationProfile().bucket(request);
        if (bucket.isPresent()) {
            return calibratedCppWins(request, bucket.get());
        }
        return coldStartAllowsCpp(request);
    }

    static AlgorithmBackend configuredBackendForTests(String algorithm) {
        return configuredBackend(algorithm);
    }

    static ShapeFamily shapeFamilyForTests(int rows, int cols) {
        return ShapeFamily.fromDimensions(rows, cols);
    }

    static SizeBand sizeBandForTests(int rows, int cols) {
        return SizeBand.fromDimensions(rows, cols);
    }

    static void resetForTests() {
        PROFILE.set(null);
    }

    private static boolean calibratedCppWins(AlgorithmDispatchRequest request,
                                             CalibrationProfile.CalibrationBucket bucket) {
        CalibrationProfile.CalibrationStats javaStats = bucket.javaStats();
        CalibrationProfile.CalibrationStats cppStats = bucket.cppStats();
        if (!bucket.cppCorrectnessPass() || javaStats == null || cppStats == null) {
            return false;
        }
        if (cppStats.samples() < minimumSamples()) {
            return false;
        }
        double requiredSpeedup = sensitivityThreshold(request.algorithm());
        double speedup = javaStats.meanNanos() / cppStats.meanNanos();
        return speedup >= requiredSpeedup;
    }

    private static boolean coldStartAllowsCpp(AlgorithmDispatchRequest request) {
        if ("qr".equals(request.algorithm())) {
            return coldStartAllowsCppForQr(request);
        }
        if (isSensitivityCritical(request.algorithm())) {
            return false;
        }
        int dominant = Math.max(request.rows(), request.cols());
        int threshold = coldStartThreshold(request.algorithm());
        return threshold > 0 && dominant >= threshold;
    }

    private static boolean coldStartAllowsCppForQr(AlgorithmDispatchRequest request) {
        int dominant = Math.max(request.rows(), request.cols());
        // Conservative cold-start rule: native QR is promoted where repeated wins were
        // stable, while tall thin/full shapes stay on Java until calibration proves
        // otherwise on the target machine.
        if ("factorize_only".equals(request.mode())) {
            int threshold = parsePositiveInt("jlc.algorithm.qr.factorizeOnly.coldStartCppMinSize", 128);
            return dominant >= threshold;
        }
        if (!"decompose_thin".equals(request.mode()) && !"decompose_full".equals(request.mode())) {
            return false;
        }
        if (request.shapeFamily() == ShapeFamily.TALL) {
            return false;
        }
        int threshold = parsePositiveInt("jlc.algorithm.qr.decompose.coldStartCppMinSize", 128);
        return dominant >= threshold;
    }

    private static int minimumSamples() {
        return parsePositiveInt("jlc.algorithm.calibration.minSamples", 5);
    }

    private static double sensitivityThreshold(String algorithm) {
        if (isSensitivityCritical(algorithm)) {
            return parsePositiveDouble("jlc.algorithm.sensitive.speedupThreshold", 1.25);
        }
        String specific = System.getProperty("jlc.algorithm." + algorithm + ".speedupThreshold");
        if (specific != null && !specific.isBlank()) {
            return parsePositiveDoubleValue(specific, 1.10);
        }
        return parsePositiveDouble("jlc.algorithm.speedupThreshold", 1.10);
    }

    private static boolean isSensitivityCritical(String algorithm) {
        return "svd".equals(algorithm) || "schur".equals(algorithm) || "polar".equals(algorithm);
    }

    private static int coldStartThreshold(String algorithm) {
        String property = "jlc.algorithm." + algorithm + ".coldStartCppMinSize";
        int fallback = switch (algorithm) {
            case "gemm" -> 128;
            case "lu" -> 128;
            case "hessenberg" -> 128;
            case "bidiagonal" -> 128;
            case "qr" -> -1;
            default -> -1;
        };
        return parsePositiveInt(property, fallback);
    }

    private static AlgorithmBackend configuredBackend(String algorithm) {
        String normalized = normalizeAlgorithm(algorithm);
        AlgorithmBackend override = NativeAlgorithmScope.overrideFor(normalized);
        if (override != null) {
            return override;
        }
        String configured = firstNonBlank(
            System.getProperty("jlc.algorithm." + normalized + ".backend"),
            System.getProperty("faulj.algorithm." + normalized + ".backend"),
            System.getProperty("jlc.algorithm.backend"),
            System.getProperty("faulj.algorithm.backend")
        );
        return AlgorithmBackend.fromConfiguredValue(configured);
    }

    private static CalibrationProfile calibrationProfile() {
        String path = firstNonBlank(
            System.getProperty("jlc.algorithm.calibration.path"),
            System.getProperty("jlc.backend.calibration.path"),
            System.getProperty("faulj.algorithm.calibration.path"),
            System.getProperty("faulj.backend.calibration.path")
        );
        LoadedProfile current = PROFILE.get();
        if (current != null && current.matches(path)) {
            return current.profile();
        }
        synchronized (PROFILE_LOCK) {
            current = PROFILE.get();
            if (current != null && current.matches(path)) {
                return current.profile();
            }
            CalibrationProfile next = CalibrationProfile.load(path);
            PROFILE.set(new LoadedProfile(path == null ? "" : path, next));
            return next;
        }
    }

    private static String normalizeAlgorithm(String algorithm) {
        if (algorithm == null || algorithm.isBlank()) {
            return "";
        }
        return algorithm.trim().toLowerCase(Locale.ROOT).replace('-', '_');
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    private static int parsePositiveInt(String property, int fallback) {
        String value = System.getProperty(property);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            int parsed = Integer.parseInt(value.trim());
            return parsed > 0 ? parsed : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static double parsePositiveDouble(String property, double fallback) {
        String value = System.getProperty(property);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        return parsePositiveDoubleValue(value, fallback);
    }

    private static double parsePositiveDoubleValue(String value, double fallback) {
        try {
            double parsed = Double.parseDouble(value.trim());
            return parsed > 0.0 ? parsed : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private record LoadedProfile(String path, CalibrationProfile profile) {
        boolean matches(String configuredPath) {
            String normalized = configuredPath == null ? "" : configuredPath;
            return path.equals(normalized);
        }
    }
}
