package net.faulj.givens;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.BidiagonalizationResult;

/**
 * Provides algorithms for reducing a matrix to bidiagonal form using Givens rotations.
 * <p>
 * Bidiagonalization is a crucial preliminary step in computing the Singular Value Decomposition (SVD).
 * While Householder reflections are typically used for the initial reduction due to efficiency
 * (Golub-Kahan), Givens rotations are often used during the iterative phase (e.g., QR iteration
 * on the bidiagonal matrix) or for specific sparse structures.
 * </p>
 *
 * <h2>Matrix Form:</h2>
 * <p>
 * A matrix B is upper bidiagonal if:
 * </p>
 * <pre>
 * ┌ x  x  0  0 ┐
 * │ 0  x  x  0 │
 * │ 0  0  x  x │
 * └ 0  0  0  x ┘
 * </pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li><b>SVD Computation:</b> Reducing a general dense matrix to bidiagonal form allows
 * singular values to be found via the implicit QR algorithm.</li>
 * <li><b>Least Squares:</b> LSQR and other iterative solvers often utilize bidiagonalization
 * (Lanczos bidiagonalization).</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.bidiagonal.Bidiagonalization
 * @see GivensRotation
 */
public class GivensBidiagonal {
    
    /**
     * Reduces matrix A to bidiagonal form using Givens rotations.
     *
     * @param A The matrix to bidiagonalize (m×n with m ≥ n)
     * @return BidiagonalizationResult containing U, B, and V
     * @throws IllegalArgumentException if A is null or has invalid dimensions
     */
    public BidiagonalizationResult bidiagonalize(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isReal()) {
            throw new UnsupportedOperationException("Givens bidiagonalization requires a real-valued matrix");
        }
        
        int m = A.getRowCount();
        int n = A.getColumnCount();
        
        if (m < n) {
            throw new IllegalArgumentException("Matrix must have m >= n for bidiagonalization");
        }
        
        Matrix B = A.copy();
        Matrix U = Matrix.Identity(m);
        Matrix V = Matrix.Identity(n);
        
        // Process each column
        for (int k = 0; k < n; k++) {
            // Zero out elements below diagonal in column k using left rotations
            for (int i = m - 1; i > k; i--) {
                double a = B.get(i - 1, k);
                double b = B.get(i, k);
                
                if (Math.abs(b) > 1e-14) {
                    GivensRotation G = GivensRotation.compute(a, b);
                    
                    // Apply to B: G^T * B
                    G.applyLeft(B, i - 1, i, k, n - 1);
                    
                    // Accumulate U: U = U * G (not G^T)
                    applyGivensRight(U, G, i - 1, i);
                }
            }
            
            // Zero out elements to the right of superdiagonal in row k
            if (k < n - 1) {
                for (int j = n - 1; j > k + 1; j--) {
                    double a = B.get(k, j - 1);
                    double b = B.get(k, j);
                    
                    if (Math.abs(b) > 1e-14) {
                        GivensRotation G = GivensRotation.compute(a, b);
                        
                        // Apply to B: B * G
                        G.applyRight(B, j - 1, j, k, m - 1);
                        
                        // Accumulate V: V = V * G
                        applyGivensRight(V, G, j - 1, j);
                    }
                }
            }
        }
        
        // Clean up small values
        cleanupSmallValues(B, 1e-12);
        
        return new BidiagonalizationResult(A, U, B, V);
    }
    
    /**
     * Apply a Givens rotation transpose on the right (column update).
     *
     * @param M matrix to update
     * @param G rotation parameters
     * @param i first column index
     * @param k second column index
     */
    private void applyGivensTransposeRight(Matrix M, GivensRotation G, int i, int k) {
        int m = M.getRowCount();
        for (int row = 0; row < m; row++) {
            double valI = M.get(row, i);
            double valK = M.get(row, k);
            M.set(row, i, G.c * valI + G.s * valK);
            M.set(row, k, -G.s * valI + G.c * valK);
        }
    }
    
    /**
     * Apply a Givens rotation on the right (column update).
     *
     * @param M matrix to update
     * @param G rotation parameters
     * @param i first column index
     * @param k second column index
     */
    private void applyGivensRight(Matrix M, GivensRotation G, int i, int k) {
        int m = M.getRowCount();
        for (int row = 0; row < m; row++) {
            double valI = M.get(row, i);
            double valK = M.get(row, k);
            M.set(row, i, G.c * valI - G.s * valK);
            M.set(row, k, G.s * valI + G.c * valK);
        }
    }
    
    /**
     * Zero out small values to improve numerical cleanliness.
     *
     * @param M matrix to clean
     * @param tol absolute tolerance
     */
    private void cleanupSmallValues(Matrix M, double tol) {
        int m = M.getRowCount();
        int n = M.getColumnCount();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(M.get(i, j)) < tol) {
                    M.set(i, j, 0.0);
                }
            }
        }
    }
}