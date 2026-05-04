package net.faulj.nativeblas;

record AlgorithmDispatchRequest(
    String algorithm,
    String mode,
    int rows,
    int cols,
    int threadCount
) {
    AlgorithmDispatchRequest {
        if (algorithm == null || algorithm.isBlank()) {
            throw new IllegalArgumentException("algorithm must not be blank");
        }
        if (mode == null || mode.isBlank()) {
            throw new IllegalArgumentException("mode must not be blank");
        }
        rows = Math.max(1, rows);
        cols = Math.max(1, cols);
        threadCount = Math.max(1, threadCount);
        algorithm = normalizeToken(algorithm);
        mode = normalizeToken(mode);
    }

    ShapeFamily shapeFamily() {
        return ShapeFamily.fromDimensions(rows, cols);
    }

    SizeBand sizeBand() {
        return SizeBand.fromDimensions(rows, cols);
    }

    String bucketKey() {
        return algorithm + "." + mode + "." + shapeFamily().id() + "." + sizeBand().id() + "." + threadCount;
    }

    private static String normalizeToken(String value) {
        return value.trim().toLowerCase(java.util.Locale.ROOT).replace('-', '_');
    }
}
