package net.faulj.compiler.matrix.codegen;

/** Calibration-time timing seam for typed backend choices. */
@FunctionalInterface
public interface BackendBenchmarkRunner {
    KernelBenchmarkStatistics measure(BackendChoice choice,
                                      KernelTuningConfig config) throws Exception;
}
