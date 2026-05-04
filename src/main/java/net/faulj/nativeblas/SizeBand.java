package net.faulj.nativeblas;

enum SizeBand {
    SMALL,
    MEDIUM,
    LARGE;

    static SizeBand fromDimensions(int rows, int cols) {
        int dominant = Math.max(Math.max(1, rows), Math.max(1, cols));
        int smallMax = parsePositiveInt("jlc.algorithm.size.small.max", 128);
        int mediumMax = parsePositiveInt("jlc.algorithm.size.medium.max", 512);
        if (dominant <= smallMax) {
            return SMALL;
        }
        if (dominant <= mediumMax) {
            return MEDIUM;
        }
        return LARGE;
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

    String id() {
        return name().toLowerCase(java.util.Locale.ROOT);
    }
}
