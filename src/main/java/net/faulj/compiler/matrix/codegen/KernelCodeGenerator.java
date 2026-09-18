package net.faulj.compiler.matrix.codegen;

/** Entry point for deterministic R4 source generation. */
public final class KernelCodeGenerator {
    private KernelCodeGenerator() {
    }

    public static PseudokernelPlan plan(net.faulj.compiler.matrix.kernel.KernelFunction function) {
        return PseudokernelPlanner.plan(function);
    }

    public static GeneratedKernelSource scalar(PseudokernelPlan plan) {
        return scalar(plan, CppEmissionOptions.standalone());
    }

    public static GeneratedKernelSource scalar(PseudokernelPlan plan, CppEmissionOptions options) {
        return ScalarCppEmitter.emit(plan, options);
    }

    public static GeneratedKernelSource avx2(PseudokernelPlan plan) {
        return avx2(plan, CppEmissionOptions.standalone());
    }

    public static GeneratedKernelSource avx2(PseudokernelPlan plan, CppEmissionOptions options) {
        return Avx2CppEmitter.emit(plan, options);
    }

    /** Emit the exact implementation described by an R5 variant signature. */
    public static GeneratedKernelSource variant(PseudokernelPlan plan,
                                                KernelVariantSignature variant) {
        return variant(plan, variant, CppEmissionOptions.standalone());
    }

    public static GeneratedKernelSource variant(PseudokernelPlan plan,
                                                KernelVariantSignature variant,
                                                CppEmissionOptions options) {
        return VariantCppEmitter.emit(plan, variant, options);
    }
}
