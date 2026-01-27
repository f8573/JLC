package net.faulj.decomposition.qr;

import net.faulj.core.PermutationVector;
import net.faulj.decomposition.result.PivotedQRResult;
import net.faulj.matrix.Matrix;

/**
 * Column-pivoted QR decomposition (rank-revealing):
 * A * P = Q * R, with P a column permutation.
 */
public class PivotedQR {
    private static final double EPS = 1e-12;

    /**
     * Compute a column-pivoted QR decomposition.
     *
     * @param A matrix to decompose
     * @return pivoted QR result containing Q, R, and permutation P
     */
    public static PivotedQRResult decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isReal()) {
            throw new UnsupportedOperationException("Pivoted QR requires a real-valued matrix");
        }
        int m = A.getRowCount();
        int n = A.getColumnCount();
        int kMax = Math.min(m, n);

        Matrix R = A.copy();
        Matrix Q = Matrix.Identity(m);
        PermutationVector P = new PermutationVector(n);

        for (int k = 0; k < kMax; k++) {
            int pivot = selectPivotColumn(R, k);
            if (pivot != k) {
                swapColumns(R, k, pivot);
                P.exchange(k, pivot);
            }

            int len = m - k;
            if (len <= 1) {
                continue;
            }

            double[] v = new double[len];
            for (int i = 0; i < len; i++) {
                v[i] = R.get(k + i, k);
            }

            double normX = 0.0;
            for (double vi : v) {
                normX += vi * vi;
            }
            normX = Math.sqrt(normX);
            if (normX <= EPS) {
                continue;
            }

            double beta = -Math.copySign(normX, v[0]);
            v[0] -= beta;
            double vv = 0.0;
            for (double vi : v) {
                vv += vi * vi;
            }
            if (vv <= EPS) {
                continue;
            }
            double tau = 2.0 / vv;

            applyHouseholderToR(R, v, tau, k);
            for (int i = 1; i < len; i++) {
                R.set(k + i, k, 0.0);
            }
            applyHouseholderToQ(Q, v, tau, k);
        }

        cleanupLower(R);
        return new PivotedQRResult(A, Q, R, P);
    }

    /**
     * Select the pivot column based on remaining column norms.
     *
     * @param R working matrix
     * @param startCol starting column index
     * @return pivot column index
     */
    private static int selectPivotColumn(Matrix R, int startCol) {
        int m = R.getRowCount();
        int n = R.getColumnCount();
        int pivot = startCol;
        double maxNorm = -1.0;
        for (int col = startCol; col < n; col++) {
            double norm = 0.0;
            for (int row = startCol; row < m; row++) {
                double v = R.get(row, col);
                norm += v * v;
            }
            if (norm > maxNorm) {
                maxNorm = norm;
                pivot = col;
            }
        }
        return pivot;
    }

    /**
     * Swap two columns in-place.
     *
     * @param M matrix to update
     * @param c1 first column index
     * @param c2 second column index
     */
    private static void swapColumns(Matrix M, int c1, int c2) {
        if (c1 == c2) {
            return;
        }
        int rows = M.getRowCount();
        for (int r = 0; r < rows; r++) {
            double tmp = M.get(r, c1);
            M.set(r, c1, M.get(r, c2));
            M.set(r, c2, tmp);
        }
    }

    /**
     * Apply a Householder reflector to R from the left.
     *
     * @param R matrix to update
     * @param v Householder vector
     * @param tau Householder scalar
     * @param k column/row index
     */
    private static void applyHouseholderToR(Matrix R, double[] v, double tau, int k) {
        int m = R.getRowCount();
        int n = R.getColumnCount();
        int len = v.length;
        for (int col = k; col < n; col++) {
            double dot = 0.0;
            for (int i = 0; i < len; i++) {
                dot += v[i] * R.get(k + i, col);
            }
            dot *= tau;
            for (int i = 0; i < len; i++) {
                double val = R.get(k + i, col) - v[i] * dot;
                R.set(k + i, col, val);
            }
        }
    }

    /**
     * Apply a Householder reflector to Q from the right (accumulation).
     *
     * @param Q matrix to update
     * @param v Householder vector
     * @param tau Householder scalar
     * @param k column/row index
     */
    private static void applyHouseholderToQ(Matrix Q, double[] v, double tau, int k) {
        int m = Q.getRowCount();
        int len = v.length;
        for (int row = 0; row < m; row++) {
            double dot = 0.0;
            for (int i = 0; i < len; i++) {
                dot += Q.get(row, k + i) * v[i];
            }
            dot *= tau;
            for (int i = 0; i < len; i++) {
                double val = Q.get(row, k + i) - dot * v[i];
                Q.set(row, k + i, val);
            }
        }
    }

    /**
     * Zero out small values below the diagonal in R.
     *
     * @param R matrix to clean
     */
    private static void cleanupLower(Matrix R) {
        int m = R.getRowCount();
        int n = R.getColumnCount();
        int limit = Math.min(m, n);
        for (int c = 0; c < limit; c++) {
            for (int r = c + 1; r < m; r++) {
                if (Math.abs(R.get(r, c)) < EPS) {
                    R.set(r, c, 0.0);
                }
            }
        }
    }
}
