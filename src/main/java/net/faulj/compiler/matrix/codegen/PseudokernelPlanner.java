package net.faulj.compiler.matrix.codegen;

import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelVerifier;

/** Creates the backend plan without mutating or enriching portable R3 IR. */
public final class PseudokernelPlanner {
    private PseudokernelPlanner() {
    }

    public static PseudokernelPlan plan(KernelFunction function) {
        if (function == null) {
            throw new IllegalArgumentException("Kernel function must not be null");
        }
        KernelVerifier.requireValid(function);
        return new PseudokernelPlan(
            function, KernelSignature.from(function), SimdEligibility.analyze(function));
    }
}
