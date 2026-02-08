package net.faulj.determinant;

import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;

/**
 * Computes the determinant of a matrix using LU Decomposition.
 * <p>
 * This class implements the O(n³) algorithm based on the decomposition A = P<sup>-1</sup>LU.
 * It is the preferred method for dense, square matrices of size n ≥ 4.
 * </p>
 *
 * <h2>Mathematical Theory:</h2>
 * <p>
 * Using the properties of determinants:
 * </p>
 * <ul>
 * <li>det(AB) = det(A)det(B)</li>
 * <li>det(L) = 1 (since L is unit lower triangular)</li>
 * <li>det(U) = Π u<sub>ii</sub> (product of diagonal entries)</li>
 * <li>det(P) = (-1)<sup>s</sup> (where s is the number of row exchanges)</li>
 * </ul>
 * <p>
 * The determinant is computed as:
 * </p>
 * <pre>
 * det(A) = (-1)<sup>s</sup> * det(L) * det(U)
 * = (-1)<sup>s</sup> * 1 * (u₁₁ * u₂₂ * ... * uₙₙ)
 * </pre>
 *
 * <h2>Numerical Stability:</h2>
 * <p>
 * This method uses partial pivoting (via the underlying LU Decomposition) to maintain
 * numerical stability. However, the determinant value can grow or shrink exponentially
 * with matrix size, potentially leading to overflow or underflow for large matrices.
 * In such cases, computing the log-determinant is often preferred.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = Matrix.random(10, 10);
 *
 * // Compute determinant via LU
 * double det = LUDeterminant.compute(A);
 *
 * if (Math.abs(det) > 1e-12) {
 * System.out.println("Matrix is invertible.");
 * System.out.println("Volume scaling factor: " + det);
 * } else {
 * System.out.println("Matrix is singular.");
 * }
 * }</pre>
 *
 * <h2>Algorithm Comparison:</h2>
 * <table border="1">
 * <tr><th>Method</th><th>Complexity</th><th>Stability</th><th>Use Case</th></tr>
 * <tr><td>LU Decomposition</td><td>O(n³)</td><td>High (w/ pivoting)</td><td>General purpose, n > 3</td></tr>
 * <tr><td>Laplace (Minors)</td><td>O(n!)</td><td>Exact (integers)</td><td>Theory, n ≤ 3</td></tr>
 * <tr><td>Gaussian Elimination</td><td>O(n³)</td><td>High</td><td>Alternative to LU</td></tr>
 * </table>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.lu.LUDecomposition
 * @see net.faulj.decomposition.result.LUResult
 */
public class LUDeterminant {

    /**
     * Computes the determinant of the given matrix using LU Decomposition.
     *
     * @param A the square matrix to process
     * @return the scalar determinant value
     * @throws IllegalArgumentException if the matrix is not square
     * @see LUDecomposition#decompose(Matrix)
     */
    public static double compute(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Determinant requires a square matrix");
        }
        if (!A.isReal()) {
            throw new IllegalArgumentException("LUDeterminant only supports real matrices. Use Determinant.computeComplex(Matrix).");
        }
        // LUResult already computes det(A) = det(P^-1) * det(L) * det(U)
        // where det(L)=1, det(P)=+/-1, det(U)=product of diagonal
        return new LUDecomposition().decompose(A).getDeterminant();
    }
}
