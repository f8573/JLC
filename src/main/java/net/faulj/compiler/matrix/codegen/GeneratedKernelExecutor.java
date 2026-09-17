package net.faulj.compiler.matrix.codegen;

import net.faulj.compiler.matrix.kernel.KernelBinding;
import net.faulj.compiler.matrix.kernel.KernelBuffer;
import net.faulj.compiler.matrix.kernel.KernelLoweringResult;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.NativeGeneratedKernelSupport;

/** Registry-dispatch helper that preserves a trusted R2 fallback on misses. */
public final class GeneratedKernelExecutor {
    private GeneratedKernelExecutor() {
    }

    /**
     * Developer-selected generated path.  The mode maps to one exact
     * canonical variant; the registry is never asked to choose a different
     * implementation after this point.
     */
    public static boolean tryExecute(KernelLoweringResult lowering,
                                     KernelBinding binding,
                                     KernelBackendMode mode) {
        if (lowering == null || binding == null || mode == null || !mode.isGenerated()
            || !lowering.isEligible()) {
            return false;
        }
        PseudokernelPlan plan = PseudokernelPlanner.plan(lowering.program().function());
        KernelVariantSignature variant = mode.backend() == CodegenBackend.AVX2
            ? KernelVariantSignature.baselineAvx2(plan.signature())
            : KernelVariantSignature.legacyScalar(plan.signature());
        BackendChoice choice = mode.backend() == CodegenBackend.AVX2
            ? new GeneratedAvx2Backend(variant) : new ScalarNativeBackend(variant);
        return tryExecuteChoice(lowering, binding, choice, plan);
    }

    /**
     * Execute the exact generated variant carried by a typed choice.  This is
     * intentionally a validation-only gate followed by one exact registry
     * lookup; it does not benchmark, enumerate, or re-select variants.
     */
    public static boolean tryExecuteChoice(KernelLoweringResult lowering,
                                           KernelBinding binding,
                                           BackendChoice choice) {
        if (lowering == null) return false;
        return tryExecuteChoice(lowering, binding, choice,
            PseudokernelPlanner.plan(lowering.program().function()));
    }

    /**
     * Fast path for an already planned verified function.  Production callers
     * retain the immutable plan created with the fused region, so this avoids
     * re-running kernel verification/signature construction per invocation.
     */
    public static boolean tryExecuteChoice(KernelLoweringResult lowering,
                                           KernelBinding binding,
                                           BackendChoice choice,
                                           PseudokernelPlan plan) {
        if (lowering == null || binding == null || choice == null
            || choice instanceof R2JavaBackend || !lowering.isEligible()) {
            return false;
        }
        if (binding.function() != lowering.program().function()
            || !heapRealNoAliasBinding(binding)) {
            return false;
        }
        if (plan == null || plan.function() != lowering.program().function()) {
            return false;
        }
        KernelVariantSignature variant = choice.variantOptional().orElse(null);
        if (variant == null || !plan.signature().equals(variant.kernelSignature())) {
            return false;
        }
        if (choice instanceof GeneratedAvx2Backend
            && (!plan.avx2Eligible() || !RuntimeCpuFeatures.avx2Supported())) {
            return false;
        }
        if (choice instanceof GeneratedAvx2Backend
            && "avx2+fma".equals(variant.requiredCpuFeature())
            && !RuntimeCpuFeatures.fmaSupported()) {
            return false;
        }
        if (choice instanceof ScalarNativeBackend && !plan.scalarCppEligible()) {
            return false;
        }
        if (!NativeGeneratedKernelSupport.isAvailable()) {
            return false;
        }
        GeneratedKernelRegistry.Entry entry = GeneratedKernelRegistry.global().lookup(variant);
        if (entry == null || !entry.descriptor().variantSignature().equals(variant)) {
            return false;
        }
        // Invocation exceptions deliberately remain observable.  A false
        // return is the established JNI precondition miss contract.
        return entry.invoke(binding);
    }

    /** Profile-only dispatch path. It never compiles or benchmarks on a miss. */
    public static boolean tryExecuteTuned(KernelLoweringResult lowering,
                                          KernelBinding binding) {
        if (lowering == null || binding == null || !lowering.isEligible()) {
            return false;
        }
        if (binding.function() != lowering.program().function()
            || !heapRealNoAliasBinding(binding)) {
            return false;
        }
        PseudokernelPlan plan = PseudokernelPlanner.plan(lowering.program().function());
        BackendChoice choice = KernelDispatchSelector.global().selectChoice(plan.signature());
        if (choice instanceof R2JavaBackend || choice == null) {
            return false;
        }
        return tryExecuteChoice(lowering, binding, choice, plan);
    }

    private static boolean heapRealNoAliasBinding(KernelBinding binding) {
        KernelBuffer outputBuffer = binding.function().outputBuffers().get(0);
        Matrix output = binding.matrix(outputBuffer);
        if (!matches(output, outputBuffer)) {
            return false;
        }
        for (KernelBuffer inputBuffer : binding.function().inputBuffers()) {
            Matrix input = binding.matrix(inputBuffer);
            if (!matches(input, inputBuffer) || input == output) {
                return false;
            }
        }
        return true;
    }

    private static boolean matches(Matrix matrix, KernelBuffer buffer) {
        return matrix != null && matrix.getClass() == Matrix.class && !matrix.hasImagData()
            && matrix.getRowCount() == buffer.shape().rows()
            && matrix.getColumnCount() == buffer.shape().columns();
    }
}
