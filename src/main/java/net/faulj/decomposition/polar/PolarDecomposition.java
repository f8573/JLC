package net.faulj.decomposition.polar;

import net.faulj.decomposition.result.PolarResult;
import net.faulj.matrix.Matrix;

/**
 * Polar Decomposition wrapper.
 * <p>
 * Delegates to {@link net.faulj.polar.PolarDecomposition} for the actual implementation.
 * </p>
 */
public class PolarDecomposition {

    private final net.faulj.polar.PolarDecomposition delegate;

    /**
     * Create a polar decomposer.
     */
    public PolarDecomposition() {
        this.delegate = new net.faulj.polar.PolarDecomposition();
    }

    /**
     * Compute the polar decomposition of a matrix.
     *
     * @param A matrix to decompose
     * @return Polar result
     */
    public PolarResult decompose(Matrix A) {
        return delegate.decompose(A);
    }
}
