package net.faulj.decomposition.svd;

import net.faulj.decomposition.result.SVDResult;
import net.faulj.matrix.Matrix;

/**
 * SVD Decomposition wrapper.
 * <p>
 * Delegates to {@link net.faulj.svd.SVDecomposition} for the actual implementation.
 * </p>
 */
public class SVDecomposition {

    private final net.faulj.svd.SVDecomposition delegate;

    /**
     * Create an SVD decomposer using the default algorithm.
     */
    public SVDecomposition() {
        this.delegate = new net.faulj.svd.SVDecomposition();
    }

    /**
     * Compute the SVD of a matrix.
     *
     * @param A matrix to decompose
     * @return SVD result
     */
    public SVDResult decompose(Matrix A) {
        return delegate.decompose(A);
    }
}
