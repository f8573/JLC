package net.faulj.kernels.gemm.blocked;

import net.faulj.compute.BlockedMultiply;
import net.faulj.matrix.Matrix;

/**
 * Blocked GEMM adapters that preserve existing blocked paths.
 */
public final class BlockedGemm {
    private BlockedGemm() {
    }

    public static Matrix multiply(Matrix a, Matrix b, int blockSize) {
        return BlockedMultiply.multiplyBlocked(a, b, blockSize);
    }

    public static Matrix multiplyParallel(Matrix a, Matrix b, int blockSize, int parallelism) {
        return BlockedMultiply.multiplyParallel(a, b, blockSize, parallelism);
    }
}
