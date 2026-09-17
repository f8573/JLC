package net.faulj.compiler.matrix.codegen;

/** Compatibility entry point for the R4 strict scalar C++ implementation. */
public final class ScalarCppEmitter {
    private ScalarCppEmitter() {
    }

    public static GeneratedKernelSource emit(PseudokernelPlan plan) {
        return emit(plan, CppEmissionOptions.standalone());
    }

    public static GeneratedKernelSource emit(PseudokernelPlan plan,
                                             CppEmissionOptions options) {
        return VariantCppEmitter.emit(plan,
            KernelVariantSignature.legacyScalar(plan.signature()), options);
    }
}
