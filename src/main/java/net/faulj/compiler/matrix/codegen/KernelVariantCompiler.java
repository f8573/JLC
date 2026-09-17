package net.faulj.compiler.matrix.codegen;

/** Explicit calibration-time bridge from a candidate to a compiled invoker. */
@FunctionalInterface
public interface KernelVariantCompiler {
    CompiledKernelVariant compile(PseudokernelPlan plan,
                                  KernelVariantCandidate candidate) throws Exception;
}
