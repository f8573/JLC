package net.faulj.nativeblas;

final class NativeValidationGuards {
    private NativeValidationGuards() {
    }

    static boolean allowSquare(String algorithm, int size, int validatedMaxSize) {
        int maxSize = positiveIntProperty("jlc.native." + algorithm + ".maxValidatedSize", validatedMaxSize);
        return maxSize <= 0 || size <= maxSize;
    }

    static boolean allowRectangular(String algorithm, int rows, int cols, int validatedMaxSize) {
        int maxSize = positiveIntProperty("jlc.native." + algorithm + ".maxValidatedSize", validatedMaxSize);
        return maxSize <= 0 || Math.max(rows, cols) <= maxSize;
    }

    private static int positiveIntProperty(String key, int fallback) {
        String value = System.getProperty(key);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }
}
