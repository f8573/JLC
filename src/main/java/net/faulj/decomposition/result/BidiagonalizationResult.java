package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;

/**
 * Encapsulates the result of bidiagonal decomposition.
 * <p>
 * Represents A = U * B * V^T where U and V are orthogonal and B is bidiagonal.
 * </p>
 */
public class BidiagonalizationResult {
    private final Matrix U;
    private final Matrix B;
    private final Matrix V;
    private final Matrix A;
    /**
     * Create a bidiagonalization result container.
     *
     * @param A original matrix
     * @param U left orthogonal factor
     * @param B bidiagonal factor
     * @param V right orthogonal factor
     */
    public BidiagonalizationResult(Matrix A, Matrix U, Matrix B, Matrix V) {
        this.A = A;
        this.U = U;
        this.B = B;
        this.V = V;
    }

    /**
     * @return left orthogonal factor U
     */
    public Matrix getU() {
        return U;
    }

    /**
     * @return bidiagonal factor B
     */
    public Matrix getB() {
        return B;
    }

    /**
     * @return right orthogonal factor V
     */
    public Matrix getV() {
        return V;
    }

    /**
     * Reconstruct A from factors.
     *
     * @return reconstructed matrix
     */
    public Matrix reconstruct() {
        return U.multiply(B).multiply(V.transpose());
    }

    /**
     * Compute the Frobenius norm residual of the factorization.
     *
     * @return residual
     */
    public double residualNorm() {
        return MatrixUtils.relativeError(A, reconstruct());
    }

    /**
     * Verify orthogonality of a matrix against the identity.
     *
     * @param O matrix to verify
     * @return array with {orthogonalityError}
     */
    public double[] verifyOrthogonality(Matrix O) {
        return new double[]{MatrixUtils.orthogonalityError(O)};
    }
}
