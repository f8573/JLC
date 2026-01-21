package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import net.faulj.vector.Vector;

/**
 * Encapsulates the result of Cholesky decomposition.
 * <p>
 * This class represents the factorization A = LL<sup>T</sup> where:
 * </p>
 * <ul>
 *   <li><b>L</b> - Lower triangular matrix with positive diagonal entries</li>
 *   <li><b>A</b> - Original symmetric positive definite matrix</li>
 * </ul>
 *
 * <h2>Matrix Properties:</h2>
 * <ul>
 *   <li><b>Lower triangular L:</b> L[i,j] = 0 for j &gt; i</li>
 *   <li><b>Positive diagonal:</b> L[i,i] &gt; 0 for all i</li>
 *   <li><b>Uniqueness:</b> L is unique for each positive definite A</li>
 *   <li><b>Upper factor:</b> U = L<sup>T</sup> is also available</li>
 * </ul>
 *
 * <h2>Positive Definite Requirement:</h2>
 * <p>
 * Cholesky decomposition exists if and only if A is symmetric positive definite:
 * </p>
 * <ul>
 *   <li>x<sup>T</sup>Ax &gt; 0 for all non-zero vectors x</li>
 *   <li>All eigenvalues are positive</li>
 *   <li>All leading principal minors are positive</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Symmetric positive definite matrix (e.g., covariance matrix)
 * Matrix A = new Matrix(new double[][] {
 *     {4,  2,  1},
 *     {2,  5,  3},
 *     {1,  3,  6}
 * });
 *
 * CholeskyDecomposition chol = new CholeskyDecomposition();
 * CholeskyResult result = chol.decompose(A);
 *
 * // Access factor
 * Matrix L = result.getL();          // Lower triangular
 * Matrix U = result.getU();          // Upper triangular (L^T)
 *
 * // Compute determinant efficiently
 * double det = result.getDeterminant();  // = [product of L[i,i]]^2
 *
 * // Verify factorization: A = L * L^T
 * Matrix reconstructed = result.reconstruct();
 * double error = result.getResidualNorm(A);
 *
 * // Solve Ax = b using forward/back substitution
 * Vector b = new Vector(new double[] {1, 2, 3});
 * Vector x = result.solve(b);
 * }</pre>
 *
 * <h2>Provided Operations:</h2>
 * <ul>
 *   <li><b>Factor access:</b> Get L and U = L<sup>T</sup></li>
 *   <li><b>Determinant:</b> Computed as [∏ L[i,i]]²</li>
 *   <li><b>System solving:</b> Forward and back substitution for Ax = b</li>
 *   <li><b>Inverse:</b> Compute A<sup>-1</sup> efficiently</li>
 *   <li><b>Reconstruction:</b> Verify A = LL<sup>T</sup></li>
 *   <li><b>Residual norm:</b> Measure factorization accuracy</li>
 * </ul>
 *
 * <h2>Solving Linear Systems:</h2>
 * <p>
 * To solve Ax = b using Cholesky:
 * </p>
 * <pre>
 * 1. Decompose: A = LL<sup>T</sup>
 * 2. Forward substitution: Solve Ly = b for y
 * 3. Back substitution: Solve L<sup>T</sup>x = y for x
 * </pre>
 * <p>
 * This is 2× faster than LU decomposition for symmetric positive definite systems.
 * </p>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Solving systems with SPD matrices</li>
 *   <li>Least squares via normal equations (A<sup>T</sup>A)</li>
 *   <li>Covariance matrix computations in statistics</li>
 *   <li>Generating multivariate normal random variables</li>
 *   <li>Kalman filtering and state estimation</li>
 *   <li>Optimization algorithms (Newton's method)</li>
 * </ul>
 *
 * <h2>Advantages over LU:</h2>
 * <ul>
 *   <li>Half the computational cost (n³/3 vs 2n³/3 flops)</li>
 *   <li>Half the storage (only L stored, not L and U)</li>
 *   <li>Better numerical stability for well-conditioned matrices</li>
 *   <li>No pivoting required</li>
 *   <li>Automatically detects non-positive-definiteness</li>
 * </ul>
 *
 * <h2>Immutability:</h2>
 * <p>
 * This class is immutable - all fields are final and matrices are not defensively copied.
 * Users should not modify returned matrices if factorization integrity is required.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.cholesky.CholeskyDecomposition
 * @see net.faulj.decomposition.lu.LUDecomposition
 * @see net.faulj.solve.LinearSolver
 */
public class CholeskyResult {
    private final Matrix L;
    private final Matrix A;

    public CholeskyResult(Matrix A, Matrix L) {
        this.A = A;
        this.L = L;
    }

    /**
     * Returns the lower triangular factor L.
     * @return lower triangular matrix with positive diagonal
     */
    public Matrix getL() {
        return L;
    }

    /**
     * Returns the upper triangular factor U = L^T.
     * @return upper triangular matrix
     */
    public Matrix getU() {
        return L.transpose();
    }

    /**
     * Reconstructs A = L * L^T for verification.
     * @return reconstructed matrix
     */
    public Matrix reconstruct() {
        return L.multiply(L.transpose());
    }

    /**
     * Computes the determinant from the Cholesky factors.
     * det(A) = [product of L[i,i]]^2
     * @return determinant of the original matrix
     */
    public double getDeterminant() {
        double prod = 1.0;
        for (int i = 0; i < L.getRowCount(); i++) {
            prod *= L.get(i, i);
        }
        return prod * prod;  // Square it
    }

    public double residualNorm() {
        return MatrixUtils.normResidual(A, reconstruct());
    }

    public double residualElement() {
        return MatrixUtils.backwardErrorComponentwise(A, reconstruct());
    }

    public double[] verifyOrthogonality(Matrix O) {
        Matrix I = Matrix.Identity(O.getRowCount());
        O = O.multiply(O.transpose());
        double n = MatrixUtils.normResidual(I, O);
        double e = MatrixUtils.backwardErrorComponentwise(I, O);
        return new double[]{n, e};
    }

    /**
     * Solves Ax = b using the Cholesky factors (A = L L^T).
     * Performs forward substitution Ly = b and back substitution L^T x = y.
     * @param b right-hand side vector
     * @return solution vector x
     */
    public Vector solve(Vector b) {
        if (b == null) throw new IllegalArgumentException("Right-hand side vector must not be null");
        int n = L.getRowCount();
        if (b.dimension() != n) throw new IllegalArgumentException("Dimension mismatch");

        double[] y = new double[n];
        // forward substitution: L * y = b
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += L.get(i, k) * y[k];
            }
            y[i] = (b.get(i) - sum) / L.get(i, i);
        }

        double[] x = new double[n];
        // back substitution: L^T * x = y
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int k = i + 1; k < n; k++) {
                sum += L.get(k, i) * x[k];
            }
            x[i] = (y[i] - sum) / L.get(i, i);
        }

        return new Vector(x);
    }

    /**
     * Computes the inverse of the original matrix using the Cholesky factors.
     * @return inverse matrix A^{-1}
     */
    public Matrix inverse() {
        int n = L.getRowCount();
        Vector[] cols = new Vector[n];
        for (int j = 0; j < n; j++) {
            double[] e = new double[n];
            e[j] = 1.0;
            Vector ej = new Vector(e);
            Vector x = solve(ej);
            cols[j] = x;
        }
        return new Matrix(cols);
    }
}
