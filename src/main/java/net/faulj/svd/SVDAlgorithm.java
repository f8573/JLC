package net.faulj.svd;

/**
 * Supported SVD algorithm variants for {@link SVDecomposition}.
 */
public enum SVDAlgorithm {
    /**
     * Golub-Kahan bidiagonalization followed by implicit QR iteration on the bidiagonal matrix.
     */
    GOLUB_KAHAN_QR,
    /**
     * Divide-and-conquer solver on the tridiagonal eigenproblem from the bidiagonal form.
     */
    DIVIDE_AND_CONQUER
}
