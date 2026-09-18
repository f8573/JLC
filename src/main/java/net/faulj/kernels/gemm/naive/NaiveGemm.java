package net.faulj.kernels.gemm.naive;

import net.faulj.compute.BlockedMultiply;
import net.faulj.matrix.Matrix;

/**
 * Naive GEMM variants kept for validation and fallback behavior.
 */
public final class NaiveGemm {
    private NaiveGemm() {
    }

    public static Matrix multiply(Matrix a, Matrix b) {
        return BlockedMultiply.multiplyNaive(a, b);
    }
}
