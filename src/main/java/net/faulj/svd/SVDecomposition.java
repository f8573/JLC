package net.faulj.svd;

import net.faulj.decomposition.result.SVDResult;
import net.faulj.matrix.Matrix;

/**
 * Computes the Singular Value Decomposition (SVD) of a matrix.
 * <p>
 * The default implementation uses a divide-and-conquer solver on the bidiagonal
 * reduction. You can select Golub-Kahan QR iteration via {@link SVDAlgorithm}.
 * </p>
 */
public class SVDecomposition {
    private final SVDAlgorithm algorithm;

    public SVDecomposition() {
        this(SVDAlgorithm.DIVIDE_AND_CONQUER);
    }

    public SVDecomposition(SVDAlgorithm algorithm) {
        if (algorithm == null) {
            throw new IllegalArgumentException("Algorithm must not be null");
        }
        this.algorithm = algorithm;
    }

    public SVDResult decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        return switch (algorithm) {
            case GOLUB_KAHAN_QR -> new GolubKahanSVD().decompose(A);
            case DIVIDE_AND_CONQUER -> new DivideAndConquerSVD().decompose(A);
        };
    }
}
