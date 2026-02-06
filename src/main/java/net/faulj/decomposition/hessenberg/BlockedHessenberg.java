package net.faulj.decomposition.hessenberg;

import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.eigen.qr.BlockedHessenbergQR;
import net.faulj.matrix.Matrix;

/**
 * Blocked Hessenberg Reduction wrapper.
 * <p>
 * Delegates to {@link BlockedHessenbergQR} for the actual implementation.
 * This class provides a unified API in the hessenberg package.
 * </p>
 */
public class BlockedHessenberg {

    /**
     * Set the panel/block size used by the blocked algorithm.
     * @param bs block size (must be >= 1)
     */
    public static void setBlockSize(int bs) {
        BlockedHessenbergQR.setBlockSize(bs);
    }

    /**
     * Get the current block size.
     * @return current block size
     */
    public static int getBlockSize() {
        return BlockedHessenbergQR.getBlockSize();
    }

    /**
     * Reduces the matrix A to Hessenberg form.
     *
     * @param A The matrix to reduce.
     * @return The HessenbergResult (H, Q).
     */
    public static HessenbergResult decompose(Matrix A) {
        return BlockedHessenbergQR.decompose(A);
    }
}
