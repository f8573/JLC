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

    public PivotedQRResult(Matrix A, Matrix Q, Matrix R, PermutationVector P) {
        this.A = A;
        this.Q = Q;
        this.R = R;
        this.P = P;
    }

    public Matrix getQ() {
        return Q;
    }

    public Matrix getR() {
        return R;
    }

    /**
     * Column permutation used for A * P = Q * R.
     */
    public PermutationVector getP() {
        return P;
    }

    /**
     * Returns A with columns permuted by P (i.e., A * P).
     */
    public Matrix permutedA() {
        return applyColumnPermutation(A);
    }

    /**
     * Reconstructs Q * R (equals A * P).
     */
    public Matrix reconstruct() {
        return Q.multiply(R);
    }

    public double residualNorm() {
        return MatrixUtils.normResidual(permutedA(), reconstruct(), 1e-10);
    }

    public double residualElement() {
        return MatrixUtils.backwardErrorComponentwise(permutedA(), reconstruct(), 1e-10);
    }

    /**
     * Apply the stored column permutation to an arbitrary matrix.
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
