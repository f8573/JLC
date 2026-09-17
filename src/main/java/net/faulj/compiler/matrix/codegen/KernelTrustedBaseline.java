package net.faulj.compiler.matrix.codegen;

/** A non-search baseline retained for comparison in calibration evidence. */
public record KernelTrustedBaseline(String name,
                                    String backend,
                                    boolean correctnessPassed,
                                    KernelBenchmarkStatistics statistics,
                                    String note) {
    public KernelTrustedBaseline {
        if (name == null || name.isBlank()) throw new IllegalArgumentException("Baseline name required");
        backend = backend == null ? "unknown" : backend;
        note = note == null ? "" : note;
    }

    public static KernelTrustedBaseline r2Java(KernelBenchmarkStatistics statistics) {
        return new KernelTrustedBaseline("R2_JAVA_FUSED", "java", true, statistics,
            "trusted Java fused path including its normal execution boundary");
    }

    public static KernelTrustedBaseline r4Scalar(KernelBenchmarkStatistics statistics) {
        return new KernelTrustedBaseline("R4_GENERATED_SCALAR_CPP", "scalar_cpp", true,
            statistics, "fixed generated scalar C++ baseline");
    }

    public static KernelTrustedBaseline r4Avx2(KernelBenchmarkStatistics statistics) {
        return new KernelTrustedBaseline("R4_BASELINE_AVX2", "avx2", true, statistics,
            "fixed R4 AVX2 u1 flat reference variant");
    }
}
