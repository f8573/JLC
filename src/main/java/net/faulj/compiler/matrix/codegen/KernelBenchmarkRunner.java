package net.faulj.compiler.matrix.codegen;

import net.faulj.compiler.matrix.kernel.KernelBinding;

/** Calibration-time benchmark seam used by native harnesses and deterministic tests. */
@FunctionalInterface
public interface KernelBenchmarkRunner {
    KernelBenchmarkStatistics measure(CompiledKernelVariant variant,
                                      KernelBinding binding,
                                      KernelTuningConfig config) throws Exception;
}
