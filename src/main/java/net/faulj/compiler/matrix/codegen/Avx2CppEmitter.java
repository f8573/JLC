/**
 * Compatibility entry point for the fixed R4 AVX2 implementation.
 * R5's bounded emitter owns the actual variant expansion.
 */
package net.faulj.compiler.matrix.codegen;

public final class Avx2CppEmitter {
    private Avx2CppEmitter() {
    }

    public static GeneratedKernelSource emit(PseudokernelPlan plan) {
        return emit(plan, CppEmissionOptions.standalone());
    }

    public static GeneratedKernelSource emit(PseudokernelPlan plan,
                                             CppEmissionOptions options) {
        return VariantCppEmitter.emit(plan,
            KernelVariantSignature.baselineAvx2(plan.signature()), options);
    }
}
