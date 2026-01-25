package net.faulj.decomposition.qr;

import net.faulj.core.PermutationVector;
import net.faulj.decomposition.result.PivotedQRResult;
import net.faulj.matrix.Matrix;

/**
 * Strong rank-revealing QR (heuristic enhancement of column-pivoted QR).
 * <p>
 * Starts with a column-pivoted QR factorization and performs additional
 * column swaps using adjacent Givens rotations to amplify diagonal dominance
 * in the leading block, improving rank revelation.
 * </p>
 */
public class StrongRankRevealingQR {
    private static final double EPS = 1e-12;
    private static final double BOOST = 1.05;

    public static PivotedQRResult decompose(Matrix A) {
        PivotedQRResult base = PivotedQR.decompose(A);
        Matrix Q = base.getQ().copy();
        Matrix R = base.getR().copy();
        PermutationVector P = base.getP().copy();

        int m = R.getRowCount();
        int n = R.getColumnCount();
        int kMax = Math.min(m, n);

        for (int k = 0; k < kMax; k++) {
            int best = k;
            double bestVal = Math.abs(R.get(k, k));
            for (int j = k + 1; j < n; j++) {
                double val = Math.abs(R.get(k, j));
                if (val > bestVal * BOOST) {
                    bestVal = val;
                    best = j;
                }
            }
            if (best != k) {
                for (int j = best; j > k; j--) {
                    swapAdjacentColumnsWithGivens(Q, R, P, j - 1);
                }
            }
        }

        return new PivotedQRResult(A, Q, R, P);
    }

    private static void swapAdjacentColumnsWithGivens(Matrix Q, Matrix R, PermutationVector P, int k) {
        int m = R.getRowCount();
        int n = R.getColumnCount();
        if (k < 0 || k + 1 >= n) {
            return;
        }

        swapColumns(R, k, k + 1);
        P.exchange(k, k + 1);

        if (k + 1 >= m) {
            return;
        }

        double a = R.get(k, k);
        double b = R.get(k + 1, k);
        if (Math.abs(b) < EPS) {
            return;
        }
        double r = Math.hypot(a, b);
        double c = a / r;
        double s = -b / r;

        for (int col = k; col < n; col++) {
            double t1 = c * R.get(k, col) - s * R.get(k + 1, col);
            double t2 = s * R.get(k, col) + c * R.get(k + 1, col);
            R.set(k, col, t1);
            R.set(k + 1, col, t2);
        }

        for (int row = 0; row < m; row++) {
            double qk = Q.get(row, k);
            double qk1 = Q.get(row, k + 1);
            Q.set(row, k, c * qk + s * qk1);
            Q.set(row, k + 1, -s * qk + c * qk1);
        }
    }

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
}
