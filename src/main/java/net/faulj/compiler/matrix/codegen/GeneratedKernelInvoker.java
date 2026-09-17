package net.faulj.compiler.matrix.codegen;

import net.faulj.compiler.matrix.kernel.KernelBinding;

/** Whole-region invocation contract for a compiled generated kernel. */
@FunctionalInterface
public interface GeneratedKernelInvoker {
    /**
     * Return true only when the complete output was produced. Implementations
     * must leave the output untouched when returning false.
     */
    boolean invoke(KernelBinding binding);
}
