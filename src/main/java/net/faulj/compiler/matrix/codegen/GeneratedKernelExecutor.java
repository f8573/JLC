package net.faulj.compiler.matrix.codegen;

import net.faulj.compiler.matrix.kernel.KernelBinding;
import net.faulj.compiler.matrix.kernel.KernelBuffer;
import net.faulj.matrix.Matrix;
import net.faulj.compiler.matrix.kernel.KernelLoweringResult;

/** Registry-dispatch helper that preserves a trusted fallback on every miss. */
public final class GeneratedKernelExecutor {
    private GeneratedKernelExecutor() {
    }

    public static boolean tryExecute(KernelLoweringResult lowering,
                                     KernelBinding binding,
                                     KernelBackendMode mode) {
        if (lowering == null || binding == null || mode == null || !mode.isGenerated()
            || !lowering.isEligible()) {
            return false;
        }
        try {
            if (binding.function() != lowering.program().function()) {
                return false;
            }
            PseudokernelPlan plan = PseudokernelPlanner.plan(lowering.program().function());
            CodegenBackend backend = mode.backend();
            if (backend == CodegenBackend.AVX2
                && (!plan.avx2Eligible() || !RuntimeCpuFeatures.avx2Supported())) {
                return false;
            }
            if (backend == CodegenBackend.SCALAR_CPP && !plan.scalarCppEligible()) {
                return false;
            }
            if (!heapRealNoAliasBinding(binding)) {
                return false;
            }
            GeneratedKernelRegistry.Entry entry = GeneratedKernelRegistry.global()
                .lookup(plan.signature(), backend);
            return entry != null && entry.invoke(binding);
        } catch (RuntimeException | LinkageError failure) {
            return false;
        }
    }

    /** Profile-only dispatch path. It never compiles or benchmarks on a miss. */
    public static boolean tryExecuteTuned(KernelLoweringResult lowering,
                                          KernelBinding binding) {
        if (lowering == null || binding == null || !lowering.isEligible()) {
            return false;
        }
        try {
            if (binding.function() != lowering.program().function()
                || !heapRealNoAliasBinding(binding)) {
                return false;
            }
            PseudokernelPlan plan = PseudokernelPlanner.plan(lowering.program().function());
            KernelDispatchSelector.Selection selection = KernelDispatchSelector.global()
                .select(plan.signature());
            return selection.entry() != null && selection.entry().invoke(binding);
        } catch (RuntimeException | LinkageError failure) {
            return false;
        }
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
