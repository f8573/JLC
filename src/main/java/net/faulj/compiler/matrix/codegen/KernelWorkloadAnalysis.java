package net.faulj.compiler.matrix.codegen;

/** Lightweight arithmetic-intensity explanation derived from verified R3 operations. */
public record KernelWorkloadAnalysis(
    int estimatedFpOpsPerElement,
    long estimatedBytesPerElement,
    double arithmeticIntensity,
    Classification classification
) {
    public enum Classification {
        BANDWIDTH_SENSITIVE,
        MIXED,
        COMPUTE_SENSITIVE
    }

    public static KernelWorkloadAnalysis from(PseudokernelPlan plan) {
        if (plan == null) throw new IllegalArgumentException("Plan is required");
        long bytes = plan.estimatedBytesPerElement();
        double intensity = bytes == 0L ? Double.POSITIVE_INFINITY
            : plan.arithmeticOpsPerElement() / (double) bytes;
        Classification classification = intensity < 0.125
            ? Classification.BANDWIDTH_SENSITIVE
            : intensity < 0.5 ? Classification.MIXED : Classification.COMPUTE_SENSITIVE;
        return new KernelWorkloadAnalysis(plan.arithmeticOpsPerElement(), bytes, intensity,
            classification);
    }

    public String diagnostic() {
        return "arithmetic-intensity: fpOpsPerElement=" + estimatedFpOpsPerElement
            + " bytesPerElement=" + estimatedBytesPerElement
            + " intensity=" + arithmeticIntensity + " class=" + classification;
    }
}
