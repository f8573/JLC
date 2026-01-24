package net.faulj.givens;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.decomposition.result.QRResult;

/**
 * Handles the updating and downdating of QR decompositions using Givens rotations.
 * <p>
 * When a matrix <b>A</b> is modified by adding/removing rows or columns (rank-1 updates),
 * recomputing the QR decomposition from scratch (O(n³)) is inefficient. This class
 * provides O(n²) algorithms to update the existing Q and R factors.
 * </p>
 *
 * <h2>Supported Operations:</h2>
 * <ul>
 * <li><b>Row Insert:</b> Adding a new row to A (A -> [A; u^T]).</li>
 * <li><b>Row Delete:</b> Removing a row from A.</li>
 * <li><b>Rank-1 Update:</b> A -> A + uv^T.</li>
 * </ul>
 *
 * <h2>Methodology:</h2>
 * <p>
 * Updates are typically performed by appending the new data and then applying a sequence
 * of Givens rotations to "chase" the non-zero elements out of the lower triangular
 * portion, restoring the upper triangular structure of R.
 * </p>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li><b>Recursive Least Squares (RLS):</b> Online parameter estimation where data arrives sequentially.</li>
 * <li><b>Active Set Methods:</b> Adding/removing constraints in optimization problems.</li>
 * <li><b>Sliding Window Filtering:</b> Signal processing on a moving window of data.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.qr.GivensQR
 * @see GivensRotation
 */
public class QRUpdateDowndate {
    
    /**
     * Updates QR decomposition when a row is appended to the matrix.
     * Given Q*R = A, computes Q'*R' = [A; u^T].
     *
     * @param Q Orthogonal matrix from original QR
     * @param R Upper triangular matrix from original QR
     * @param u Row vector to append
     * @return Updated QRResult
     */
    public static QRResult appendRow(Matrix Q, Matrix R, Vector u) {
        if (Q == null || R == null || u == null) {
            throw new IllegalArgumentException("Inputs must not be null");
        }
        
        int m = Q.getRowCount();
        int n = R.getColumnCount();
        
        if (u.dimension() != n) {
            throw new IllegalArgumentException("Row dimension mismatch");
        }
        
        // Create augmented R: [R; u^T]
        Matrix R_aug = new Matrix(m + 1, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                R_aug.set(i, j, R.get(i, j));
            }
        }
        for (int j = 0; j < n; j++) {
            R_aug.set(m, j, u.get(j));
        }
        
        // Create augmented Q
        Matrix Q_aug = Matrix.Identity(m + 1);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                Q_aug.set(i, j, Q.get(i, j));
            }
        }
        
        // Apply Givens rotations to restore upper triangular form
        for (int j = 0; j < n; j++) {
            double a = R_aug.get(j, j);
            double b = R_aug.get(m, j);
            
            if (Math.abs(b) > 1e-14) {
                GivensRotation G = GivensRotation.compute(a, b);
                G.applyLeft(R_aug, j, m, j, n - 1);
                applyGivensTransposeRight(Q_aug, G, j, m);
            }
        }
        
        // Reconstruct A from Q' and R'
        Matrix A_new = Q_aug.multiply(R_aug);
        return new QRResult(A_new, Q_aug, R_aug);
    }
    
    /**
     * Performs rank-one update: A' = A + alpha * u * v^T.
     * Updates existing QR decomposition efficiently.
     *
     * @param Q Orthogonal matrix from A = QR
     * @param R Upper triangular matrix
     * @param u Column vector
     * @param v Column vector
     * @param alpha Scalar multiplier
     * @return Updated QRResult
     */
    public static QRResult rankOneUpdate(Matrix Q, Matrix R, Vector u, Vector v, double alpha) {
        if (Q == null || R == null || u == null || v == null) {
            throw new IllegalArgumentException("Inputs must not be null");
        }
        
        int m = Q.getRowCount();
        int n = R.getColumnCount();
        
        // Compute w = alpha * Q^T * u
        Matrix QT = Q.transpose();
        double[] w = new double[m];
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = 0; j < m; j++) {
                sum += QT.get(i, j) * u.get(j);
            }
            w[i] = alpha * sum;
        }
        
        // Update R: R' = R + w * v^T
        Matrix R_new = R.copy();
        for (int i = 0; i < Math.min(m, n); i++) {
            for (int j = 0; j < n; j++) {
                R_new.set(i, j, R_new.get(i, j) + w[i] * v.get(j));
            }
        }
        
        // Restore upper triangular form using Givens rotations
        Matrix Q_new = Q.copy();
        for (int j = 0; j < n; j++) {
            for (int i = Math.min(m, n) - 1; i > j; i--) {
                double a = R_new.get(i - 1, j);
                double b = R_new.get(i, j);
                
                if (Math.abs(b) > 1e-14) {
                    GivensRotation G = GivensRotation.compute(a, b);
                    G.applyLeft(R_new, i - 1, i, j, n - 1);
                    applyGivensTransposeRight(Q_new, G, i - 1, i);
                }
            }
        }
        
        Matrix A_new = Q_new.multiply(R_new);
        return new QRResult(A_new, Q_new, R_new);
    }
    
    private static void applyGivensTransposeRight(Matrix M, GivensRotation G, int i, int k) {
        int m = M.getRowCount();
        for (int row = 0; row < m; row++) {
            double valI = M.get(row, i);
            double valK = M.get(row, k);
            M.set(row, i, G.c * valI + G.s * valK);
            M.set(row, k, -G.s * valI + G.c * valK);
        }
    }
}