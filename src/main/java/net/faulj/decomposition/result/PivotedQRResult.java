package net.faulj.decomposition.result;

import net.faulj.core.PermutationVector;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;

/**
 * Encapsulates the result of a column-pivoted QR decomposition: A * P = Q * R.
 * <p>
 * This result stores a column permutation P (as a permutation vector), along with
 * the orthogonal factor Q and upper trapezoidal factor R.
 * </p>
 */
public class PivotedQRResult {
    private final Matrix A;
    private final Matrix Q;
    private final Matrix R;
    private final PermutationVector P;

    /**
     * Create a column-pivoted QR result container.
     *
     * @param A original matrix
     * @param Q orthogonal factor
     * @param R upper trapezoidal factor
     * @param P column permutation vector
     */
    public PivotedQRResult(Matrix A, Matrix Q, Matrix R, PermutationVector P) {
        this.A = A;
        this.Q = Q;
        this.R = R;
        this.P = P;
    }

    /**
     * @return orthogonal factor Q
     */
    public Matrix getQ() {
        return Q;
    }

    /**
     * @return upper trapezoidal factor R
     */
    public Matrix getR() {
        return R;
    }

    /**
     * Column permutation used for A * P = Q * R.
     */
    /**
     * @return column permutation vector
     */
    public PermutationVector getP() {
        return P;
    }

    /**
     * Returns A with columns permuted by P (i.e., A * P).
     */
    /**
     * Apply the stored permutation to A (A * P).
     *
     * @return permuted matrix
     */
    public Matrix permutedA() {
        return applyColumnPermutation(A);
    }

    /**
     * Reconstructs Q * R (equals A * P).
     */
    /**
     * Reconstruct Q * R (equals A * P).
     *
     * @return reconstructed matrix
     */
    public Matrix reconstruct() {
        return Q.multiply(R);
    }

    /**
     * Compute the Frobenius norm residual of the factorization.
     *
     * @return residual
     */
    public double residualNorm() {
        return MatrixUtils.relativeError(permutedA(), reconstruct());
    }

    /**
     * Apply the stored column permutation to an arbitrary matrix.
     */
    /**
     * Apply the stored column permutation to an arbitrary matrix.
     *
     * @param M matrix to permute
     * @return permuted matrix
     */
    public Matrix applyColumnPermutation(Matrix M) {
        int m = M.getRowCount();
        int n = M.getColumnCount();
        if (P.size() != n) {
            throw new IllegalArgumentException("Permutation size mismatch");
        }
        Matrix result = new Matrix(m, n);
        for (int col = 0; col < n; col++) {
            int src = P.get(col);
            for (int row = 0; row < m; row++) {
                result.set(row, col, M.get(row, src));
            }
        }
        return result;
    }
}
