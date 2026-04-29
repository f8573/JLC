package net.faulj.nativeblas;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

public final class NativeFactorizationSupport {
    private static final int DEFAULT_LU_MIN_SIZE = 128;
    private static final int DEFAULT_QR_MIN_SIZE = 1;
    private static final int DEFAULT_CHOLESKY_MIN_SIZE = 128;
    private static final String DEFAULT_QR_DECOMPOSE_BANDS = "1+:native";
    private static final String DEFAULT_QR_FACTORIZE_BANDS = "1+:native";
    private static final String DEFAULT_QR_TALL_FACTORIZE_GRID =
        "1-32x1+:native," +
        "33-64x1-1024:native," +
        "33-64x1025+:java," +
        "65+x1+:java";
    private static final Object QR_CALIBRATION_LOCK = new Object();
    private static volatile LoadedCalibration qrCalibration;

    private NativeFactorizationSupport() {
    }

    public static boolean tryLu(double[] packedLu, int n, int[] pivots) {
        return switch (resolveMode("lu", n, DEFAULT_LU_MIN_SIZE)) {
            case DISABLED -> false;
            case BUILTIN -> invokeBuiltinLu(packedLu, n, pivots);
            case VENDOR -> invokeVendorLu(packedLu, n, pivots);
        };
    }

    public static boolean tryQr(double[] aWork, int m, int n, boolean thin, double[] q, double[] r) {
        int qCols = thin ? Math.min(m, n) : m;
        return switch (resolveQrMode(m, n, false)) {
            case DISABLED -> false;
            case BUILTIN -> invokeBuiltinQr(aWork, m, n, qCols, q, r);
            case VENDOR -> invokeVendorQr(aWork, m, n, qCols, q, r);
        };
    }

    public static boolean tryQrFactorizeOnly(double[] aWork, int m, int n) {
        return switch (resolveQrMode(m, n, true)) {
            case DISABLED -> false;
            case BUILTIN -> invokeBuiltinQrFactorizeOnly(aWork, m, n);
            case VENDOR -> invokeVendorQrFactorizeOnly(aWork, m, n);
        };
    }

    public static boolean tryCholesky(double[] packedL, int n) {
        return switch (resolveMode("cholesky", n, DEFAULT_CHOLESKY_MIN_SIZE)) {
            case DISABLED -> false;
            case BUILTIN -> invokeBuiltinCholesky(packedL, n);
            case VENDOR -> invokeVendorCholesky(packedL, n);
        };
    }

    private static boolean invokeBuiltinLu(double[] packedLu, int n, int[] pivots) {
        try {
            NativeBindings.nativeLuFactor(packedLu, n, pivots);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static boolean invokeVendorLu(double[] packedLu, int n, int[] pivots) {
        try {
            NativeBindings.nativeLuFactorVendor(packedLu, n, pivots);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static boolean invokeBuiltinQr(double[] aWork, int m, int n, int qCols, double[] q, double[] r) {
        try {
            NativeBindings.nativeQrDecompose(aWork, m, n, qCols, q, r);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static boolean invokeBuiltinQrFactorizeOnly(double[] aWork, int m, int n) {
        try {
            NativeBindings.nativeQrFactorizeOnly(aWork, m, n);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static boolean invokeVendorQr(double[] aWork, int m, int n, int qCols, double[] q, double[] r) {
        try {
            NativeBindings.nativeQrDecomposeVendor(aWork, m, n, qCols, q, r);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static boolean invokeVendorQrFactorizeOnly(double[] aWork, int m, int n) {
        try {
            NativeBindings.nativeQrFactorizeOnlyVendor(aWork, m, n);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static boolean invokeBuiltinCholesky(double[] packedL, int n) {
        try {
            int info = NativeBindings.nativeCholeskyDecompose(packedL, n);
            if (info > 0) {
                throw new ArithmeticException("Matrix is not positive definite (non-positive pivot at " + (info - 1) + ")");
            }
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static boolean invokeVendorCholesky(double[] packedL, int n) {
        try {
            int info = NativeBindings.nativeCholeskyDecomposeVendor(packedL, n);
            if (info > 0) {
                throw new ArithmeticException("Matrix is not positive definite (non-positive pivot at " + (info - 1) + ")");
            }
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static Mode resolveMode(String algorithmKey, int problemSize, int defaultMinSize) {
        if (problemSize < minimumSize(algorithmKey, defaultMinSize)) {
            return Mode.DISABLED;
        }
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        if (!"native".equals(snapshot.activeBackend()) || !snapshot.nativeContext().isAvailable()) {
            return Mode.DISABLED;
        }

        String configured = firstNonBlank(
            System.getProperty("jlc.native." + algorithmKey + ".provider"),
            System.getProperty("faulj.native." + algorithmKey + ".provider"),
            System.getProperty("jlc.native.provider"),
            System.getProperty("faulj.native.provider")
        );
        if (configured == null || configured.isBlank()) {
            return Mode.BUILTIN;
        }
        return configuredMode(configured);
    }

    private static Mode resolveQrMode(int rows, int cols, boolean factorizeOnly) {
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        String configured = firstNonBlank(
            System.getProperty("jlc.native.qr.provider"),
            System.getProperty("faulj.native.qr.provider"),
            System.getProperty("jlc.native.provider"),
            System.getProperty("faulj.native.provider")
        );
        return resolveQrMode(rows, cols, factorizeOnly, snapshot, configured);
    }

    private static Mode autoQrMode(BackendSnapshot snapshot, int rows, int cols, QrShapeFamily family, int bandMetric,
                                   boolean factorizeOnly) {
        int shortDim = Math.min(rows, cols);
        int longDim = Math.max(rows, cols);
        String explicitGrid = qrGridPropertyValue(PropertySource.SYSTEM, family, factorizeOnly);
        String explicitBands = qrBandPropertyValue(PropertySource.SYSTEM, family, factorizeOnly);
        String calibrationGrid = qrGridPropertyValue(PropertySource.CALIBRATION, family, factorizeOnly);
        String calibrationBands = qrBandPropertyValue(PropertySource.CALIBRATION, family, factorizeOnly);

        String winner = selectBackendFromGrid(explicitGrid, shortDim, longDim);
        if (winner == null) {
            winner = selectBackendFromBands(explicitBands, bandMetric);
        }
        if (winner == null) {
            winner = selectBackendFromGrid(calibrationGrid, shortDim, longDim);
        }
        if (winner == null) {
            winner = selectBackendFromBands(calibrationBands, bandMetric);
        }
        if (winner == null) {
            winner = selectBackendFromGrid(defaultQrGrid(family, factorizeOnly), shortDim, longDim);
        }
        if (winner == null) {
            winner = selectBackendFromBands(factorizeOnly ? DEFAULT_QR_FACTORIZE_BANDS : DEFAULT_QR_DECOMPOSE_BANDS, bandMetric);
        }
        if (!"native".equals(winner) && !"builtin".equals(winner) && !"vendor".equals(winner)) {
            return Mode.DISABLED;
        }
        if (!"native".equals(snapshot.activeBackend()) || !snapshot.nativeContext().isAvailable()) {
            return Mode.DISABLED;
        }
        if ("vendor".equals(winner)) {
            return vendorAvailable() ? Mode.VENDOR : Mode.BUILTIN;
        }
        return Mode.BUILTIN;
    }

    private static String qrBandPropertyKey(QrShapeFamily family, boolean factorizeOnly) {
        return "jlc.native.qr." + family.propertyKey + "." + (factorizeOnly ? "factorizeBands" : "decomposeBands");
    }

    private static String qrBandPropertyKeyFaulj(QrShapeFamily family, boolean factorizeOnly) {
        return "faulj.native.qr." + family.propertyKey + "." + (factorizeOnly ? "factorizeBands" : "decomposeBands");
    }

    private static String qrGridPropertyKey(QrShapeFamily family, boolean factorizeOnly) {
        return "jlc.native.qr." + family.propertyKey + "." + (factorizeOnly ? "factorizeGrid" : "decomposeGrid");
    }

    private static String qrGridPropertyKeyFaulj(QrShapeFamily family, boolean factorizeOnly) {
        return "faulj.native.qr." + family.propertyKey + "." + (factorizeOnly ? "factorizeGrid" : "decomposeGrid");
    }

    private static Mode configuredMode(String configured) {
        return switch (configured.trim().toLowerCase()) {
            case "java", "disabled" -> Mode.DISABLED;
            case "vendor", "lapack", "mkl", "openblas" -> vendorAvailable() ? Mode.VENDOR : Mode.DISABLED;
            case "auto", "builtin", "native", "kernel" -> Mode.BUILTIN;
            default -> Mode.BUILTIN;
        };
    }

    private static String selectBackendFromBands(String spec, int problemSize) {
        if (spec == null || spec.isBlank()) {
            return null;
        }
        for (String rawBand : spec.split(",")) {
            String band = rawBand.trim();
            if (band.isEmpty()) {
                continue;
            }
            int colon = band.indexOf(':');
            if (colon <= 0 || colon >= band.length() - 1) {
                continue;
            }
            String range = band.substring(0, colon).trim();
            String backend = band.substring(colon + 1).trim().toLowerCase();
            if (matchesBand(range, problemSize)) {
                return backend;
            }
        }
        return null;
    }

    private static String selectBackendFromGrid(String spec, int shortDim, int longDim) {
        if (spec == null || spec.isBlank()) {
            return null;
        }
        for (String rawRule : spec.split(",")) {
            String rule = rawRule.trim();
            if (rule.isEmpty()) {
                continue;
            }
            int colon = rule.indexOf(':');
            if (colon <= 0 || colon >= rule.length() - 1) {
                continue;
            }
            String rectangle = rule.substring(0, colon).trim();
            String backend = rule.substring(colon + 1).trim().toLowerCase();
            int separator = rectangle.indexOf('x');
            if (separator <= 0 || separator >= rectangle.length() - 1) {
                continue;
            }
            String shortRange = rectangle.substring(0, separator).trim();
            String longRange = rectangle.substring(separator + 1).trim();
            if (matchesBand(shortRange, shortDim) && matchesBand(longRange, longDim)) {
                return backend;
            }
        }
        return null;
    }

    private static String qrBandPropertyValue(PropertySource source, QrShapeFamily family, boolean factorizeOnly) {
        return firstConfiguredValue(source,
            qrBandPropertyKey(family, factorizeOnly),
            qrBandPropertyKeyFaulj(family, factorizeOnly),
            "jlc.native.qr." + family.propertyKey + ".bands",
            "faulj.native.qr." + family.propertyKey + ".bands",
            factorizeOnly ? "jlc.native.qr.factorizeBands" : "jlc.native.qr.decomposeBands",
            factorizeOnly ? "faulj.native.qr.factorizeBands" : "faulj.native.qr.decomposeBands",
            "jlc.native.qr.bands",
            "faulj.native.qr.bands"
        );
    }

    private static String qrGridPropertyValue(PropertySource source, QrShapeFamily family, boolean factorizeOnly) {
        return firstConfiguredValue(source,
            qrGridPropertyKey(family, factorizeOnly),
            qrGridPropertyKeyFaulj(family, factorizeOnly),
            "jlc.native.qr." + family.propertyKey + ".grid",
            "faulj.native.qr." + family.propertyKey + ".grid",
            factorizeOnly ? "jlc.native.qr.factorizeGrid" : "jlc.native.qr.decomposeGrid",
            factorizeOnly ? "faulj.native.qr.factorizeGrid" : "faulj.native.qr.decomposeGrid",
            "jlc.native.qr.grid",
            "faulj.native.qr.grid"
        );
    }

    private static boolean matchesBand(String range, int problemSize) {
        if (range.endsWith("+")) {
            int lower = parseBandEndpoint(range.substring(0, range.length() - 1), 1);
            return problemSize >= lower;
        }
        int dash = range.indexOf('-');
        if (dash > 0 && dash < range.length() - 1) {
            int lower = parseBandEndpoint(range.substring(0, dash), 1);
            int upper = parseBandEndpoint(range.substring(dash + 1), Integer.MAX_VALUE);
            return problemSize >= lower && problemSize <= upper;
        }
        int exact = parseBandEndpoint(range, -1);
        return exact >= 0 && problemSize == exact;
    }

    private static int parseBandEndpoint(String value, int fallback) {
        try {
            return Math.max(1, Integer.parseInt(value.trim()));
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static int minimumSize(String algorithmKey, int defaultValue) {
        String configured = firstNonBlank(
            System.getProperty("jlc.native." + algorithmKey + ".minSize"),
            System.getProperty("faulj.native." + algorithmKey + ".minSize")
        );
        if (configured == null || configured.isBlank()) {
            return defaultValue;
        }
        try {
            return Math.max(1, Integer.parseInt(configured.trim()));
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static boolean vendorAvailable() {
        try {
            return NativeBindings.nativeVendorLapackAvailable();
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    private static String firstConfiguredValue(PropertySource source, String... keys) {
        for (String key : keys) {
            String value = configuredValue(source, key);
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    private static String configuredValue(PropertySource source, String key) {
        return switch (source) {
            case SYSTEM -> System.getProperty(key);
            case CALIBRATION -> qrCalibrationProperties().getProperty(key);
        };
    }

    private static Properties qrCalibrationProperties() {
        String configuredPath = firstNonBlank(
            System.getProperty("jlc.native.qr.calibration.path"),
            System.getProperty("faulj.native.qr.calibration.path")
        );
        if (configuredPath == null || configuredPath.isBlank()) {
            return EMPTY_CALIBRATION.properties;
        }

        LoadedCalibration current = qrCalibration;
        if (current != null && current.path.equals(configuredPath)) {
            return current.properties;
        }
        synchronized (QR_CALIBRATION_LOCK) {
            current = qrCalibration;
            if (current != null && current.path.equals(configuredPath)) {
                return current.properties;
            }
            qrCalibration = loadCalibration(configuredPath);
            return qrCalibration.properties;
        }
    }

    private static LoadedCalibration loadCalibration(String configuredPath) {
        Properties properties = new Properties();
        try (InputStream in = Files.newInputStream(Path.of(configuredPath))) {
            properties.load(in);
        } catch (IOException | RuntimeException ignored) {
            properties.clear();
        }
        return new LoadedCalibration(configuredPath, properties);
    }

    private enum Mode {
        DISABLED,
        BUILTIN,
        VENDOR
    }

    private enum PropertySource {
        SYSTEM,
        CALIBRATION
    }

    private static final LoadedCalibration EMPTY_CALIBRATION = new LoadedCalibration("", new Properties());

    static String qrModeForTests(int rows, int cols, boolean factorizeOnly) {
        return resolveQrMode(rows, cols, factorizeOnly).name();
    }

    static String qrShapeFamilyForTests(int rows, int cols) {
        return qrShapeFamily(rows, cols).name();
    }

    private static QrShapeFamily qrShapeFamily(int rows, int cols) {
        if ((long) rows >= 2L * (long) cols) {
            return QrShapeFamily.TALL;
        }
        if ((long) cols >= 2L * (long) rows) {
            return QrShapeFamily.WIDE;
        }
        return QrShapeFamily.SQUARE;
    }

    private static int qrBandMetric(int rows, int cols, QrShapeFamily family) {
        return family == QrShapeFamily.SQUARE ? Math.max(rows, cols) : Math.min(rows, cols);
    }

    static void resetCalibrationForTests() {
        synchronized (QR_CALIBRATION_LOCK) {
            qrCalibration = null;
        }
    }

    private static String defaultQrGrid(QrShapeFamily family, boolean factorizeOnly) {
        if (family == QrShapeFamily.TALL && factorizeOnly) {
            return DEFAULT_QR_TALL_FACTORIZE_GRID;
        }
        return null;
    }

    private static Mode resolveQrMode(int rows, int cols, boolean factorizeOnly, BackendSnapshot snapshot, String configured) {
        QrShapeFamily family = qrShapeFamily(rows, cols);
        int bandMetric = qrBandMetric(rows, cols, family);
        if (bandMetric < minimumSize("qr", DEFAULT_QR_MIN_SIZE)) {
            return Mode.DISABLED;
        }
        if (configured == null || configured.isBlank() || "auto".equalsIgnoreCase(configured.trim())) {
            return autoQrMode(snapshot, rows, cols, family, bandMetric, factorizeOnly);
        }
        if (!"native".equals(snapshot.activeBackend()) || !snapshot.nativeContext().isAvailable()) {
            return Mode.DISABLED;
        }
        return configuredMode(configured);
    }

    private enum QrShapeFamily {
        SQUARE("square"),
        TALL("tall"),
        WIDE("wide");

        private final String propertyKey;

        QrShapeFamily(String propertyKey) {
            this.propertyKey = propertyKey;
        }
    }

    private record LoadedCalibration(String path, Properties properties) {
    }
}
