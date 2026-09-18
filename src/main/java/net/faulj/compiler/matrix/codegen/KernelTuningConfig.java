package net.faulj.compiler.matrix.codegen;

/** Bounded and reproducible R5 tuning budget and decision policy. */
public record KernelTuningConfig(
    int warmupSamples,
    int measuredSamples,
    int maxCandidates,
    long maxBenchmarkMillis,
    double relativeMadThreshold,
    double promotionThreshold
) {
    public KernelTuningConfig {
        if (warmupSamples < 0) {
            throw new IllegalArgumentException("warmupSamples must not be negative");
        }
        if (measuredSamples < 1) {
            throw new IllegalArgumentException("measuredSamples must be positive");
        }
        if (maxCandidates < 1) {
            throw new IllegalArgumentException("maxCandidates must be positive");
        }
        if (maxBenchmarkMillis < 0L) {
            throw new IllegalArgumentException("maxBenchmarkMillis must not be negative");
        }
        if (!Double.isFinite(relativeMadThreshold) || relativeMadThreshold < 0.0) {
            throw new IllegalArgumentException("relativeMadThreshold must be finite and non-negative");
        }
        if (!Double.isFinite(promotionThreshold) || promotionThreshold < 0.0) {
            throw new IllegalArgumentException("promotionThreshold must be finite and non-negative");
        }
    }

    public static KernelTuningConfig defaults() {
        return new KernelTuningConfig(5, 9, KernelVariantGenerator.DEFAULT_MAX_CANDIDATES,
            0L, 0.10, 0.03);
    }

    public static KernelTuningConfig fromSystemProperties() {
        KernelTuningConfig defaults = defaults();
        return new KernelTuningConfig(
            integer("jlc.compiler.autotune.warmupSamples", defaults.warmupSamples()),
            Math.max(7, integer("jlc.compiler.autotune.samples", defaults.measuredSamples())),
            integer("jlc.compiler.autotune.maxCandidates", defaults.maxCandidates()),
            longValue("jlc.compiler.autotune.maxBenchmarkMillis", defaults.maxBenchmarkMillis()),
            decimal("jlc.compiler.autotune.relativeMadThreshold", defaults.relativeMadThreshold()),
            decimal("jlc.compiler.autotune.promotionThreshold", defaults.promotionThreshold()));
    }

    private static int integer(String key, int fallback) {
        try {
            int value = Integer.parseInt(System.getProperty(key, Integer.toString(fallback)).trim());
            return value >= 0 ? value : fallback;
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    private static long longValue(String key, long fallback) {
        try {
            long value = Long.parseLong(System.getProperty(key, Long.toString(fallback)).trim());
            return value >= 0L ? value : fallback;
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    private static double decimal(String key, double fallback) {
        try {
            double value = Double.parseDouble(System.getProperty(key, Double.toString(fallback)).trim());
            return Double.isFinite(value) && value >= 0.0 ? value : fallback;
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }
}
