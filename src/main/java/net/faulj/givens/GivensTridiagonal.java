package net.faulj.givens;

import net.faulj.matrix.Matrix;

/**
 * Provides algorithms for reducing a symmetric matrix to tridiagonal form using Givens rotations.
 * <p>
 * Tridiagonal reduction is the first step in the symmetric eigenvalue algorithm.
 * For a symmetric matrix A, there exists an orthogonal matrix Q such that T = Q<sup>T</sup>AQ is tridiagonal.
 * </p>
 *
 * <h2>Matrix Form:</h2>
 * <p>
 * A symmetric tridiagonal matrix T has the form:
 * </p>
 * <pre>
 * ┌ a  b  0  0 ┐
 * │ b  a  b  0 │
 * │ 0  b  a  b │
 * └ 0  0  b  a ┘
 * </pre>
 *
 * <h2>Computational Approach:</h2>
 * <p>
 * Givens rotations can selectively zero out elements below the first subdiagonal.
 * While Householder reflections are generally preferred for dense matrices (O(4n³/3) FLOPS),
 * Givens rotations are advantageous for:
 * </p>
 * <ul>
 * <li><b>Sparse Matrices:</b> Preserving sparsity patterns better than Householder reflections.</li>
 * <li><b>Parallel Computing:</b> Rotations on disjoint rows/columns can be applied simultaneously.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.symmetric.SymmetricEigenDecomposition
 * @see GivensRotation
 */
public class GivensTridiagonal {
    
    /**
     * Reduces a symmetric matrix to tridiagonal form using Givens rotations.
     * Returns {T, Q} where T = Q^T * A * Q and Q is orthogonal.
     *
     * @param A Symmetric matrix to tridiagonalize
     * @return Array of {T, Q} where T is tridiagonal and Q is orthogonal
     * @throws IllegalArgumentException if A is not symmetric
     */
    public static Matrix[] tridiagonalize(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        if (!A.isReal()) {
            throw new UnsupportedOperationException("Givens tridiagonalization requires a real-valued matrix");
        }
        
        // Verify symmetry
        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(A.get(i, j) - A.get(j, i)) > 1e-10) {
                    throw new IllegalArgumentException("Matrix must be symmetric");
                }
            }
        }
        
        Matrix T = A.copy();
        Matrix Q = Matrix.Identity(n);
        
        // Process each column
        for (int k = 0; k < n - 2; k++) {
            // Zero out elements below first subdiagonal in column k
            for (int i = n - 1; i > k + 1; i--) {
                double a = T.get(i - 1, k);
                double b = T.get(i, k);
                
                if (Math.abs(b) > 1e-14) {
                    GivensRotation G = GivensRotation.compute(a, b);
                    
                    // Apply from left: G^T * T
                    G.applyLeft(T, i - 1, i, k, n - 1);
                    
                    // Apply from right: T * G (maintains symmetry)
                    G.applyRight(T, i - 1, i, 0, n - 1);
                    
                    // Accumulate Q: Q = Q * G (not G^T)
                    applyGivensRight(Q, G, i - 1, i);
                }
            }
        }
        
        // Clean up small values
        cleanupSmallValues(T, 1e-12);
        
        return new Matrix[]{T, Q};
    }
    
    /**
     * Apply a Givens rotation on the right (column update).
     *
     * @param M matrix to update
     * @param G rotation parameters
     * @param i first column index
     * @param k second column index
     */
    private static void applyGivensRight(Matrix M, GivensRotation G, int i, int k) {
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
    private static void cleanupSmallValues(Matrix M, double tol) {
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