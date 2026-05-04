package net.faulj.nativeblas;

enum ShapeFamily {
    SQUARE,
    TALL,
    WIDE;

    static ShapeFamily fromDimensions(int rows, int cols) {
        int safeRows = Math.max(1, rows);
        int safeCols = Math.max(1, cols);
        double squareMinRatio = parsePositiveDouble("jlc.algorithm.shape.square.minRatio", 0.5);
        double squareMaxRatio = parsePositiveDouble("jlc.algorithm.shape.square.maxRatio", 2.0);
        double ratio = safeRows / (double) safeCols;
        if (ratio >= squareMinRatio && ratio <= squareMaxRatio) {
            return SQUARE;
        }
        return safeRows > safeCols ? TALL : WIDE;
    }

    private static double parsePositiveDouble(String property, double fallback) {
        String value = System.getProperty(property);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            double parsed = Double.parseDouble(value.trim());
            return parsed > 0.0 ? parsed : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    String id() {
        return name().toLowerCase(java.util.Locale.ROOT);
    }
}
