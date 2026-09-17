package net.faulj.compiler.matrix.kernel;

/**
 * Scalar types understood by the portable Kernel IR.
 *
 * <p>R3 emits only {@link #FP64}. {@link #COMPLEX_FP64} is named so that a
 * future typed extension has an explicit place to land, but the current
 * lowerer and reference executor deliberately reject it.</p>
 */
public enum KernelValueType {
    FP64,
    COMPLEX_FP64
}
