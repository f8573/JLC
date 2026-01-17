package net.faulj.determinant;

import net.faulj.matrix.Matrix;

/**
 * Strategy facade for computing matrix determinants.
 * <p>
 * The determinant is a scalar value that is a function of the entries of a square matrix.
 * It characterizes some properties of the matrix and the linear map represented by the matrix.
 * </p>
 *
 * <h2>Geometric Interpretation:</h2>
 * <p>
 * Geometrically, the absolute value of the determinant of a real matrix is the scale factor
 * for volume when the matrix acts as a linear transformation.
 * </p>
 * <ul>
 * <li><b>|det(A)| > 1:</b> The transformation expands volume.</li>
 * <li><b>|det(A)| < 1:</b> The transformation compresses volume.</li>
 * <li><b>det(A) = 0:</b> The transformation collapses volume to zero (matrix is singular).</li>
 * <li><b>det(A) < 0:</b> The transformation reverses orientation (e.g., includes a reflection).</li>
 * </ul>
 *
 * <h2>Strategy Selection:</h2>
 * <p>
 * This class automatically selects the most appropriate algorithm based on the matrix size:
 * </p>
 * <ul>
 * <li><b>n ≤ 3:</b> Uses direct formulas or Minors (low overhead).</li>
 * <li><b>n > 3:</b> Uses LU Decomposition (O(n³) efficiency).</li>
 * </ul>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li>det(I) = 1</li>
 * <li>det(A<sup>T</sup>) = det(A)</li>
 * <li>det(A<sup>-1</sup>) = 1 / det(A)</li>
 * <li>det(AB) = det(A)det(B)</li>
 * <li>det(cA) = c<sup>n</sup> det(A) (for n×n matrix)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = Matrix.random(5, 5);
 * double det = Determinant.compute(A);
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see LUDeterminant
 * @see MinorsDeterminant
 */
public class Determinant {

    /**
     * Computes the determinant of the given square matrix using the optimal strategy.
     *
     * @param A the matrix to analyze
     * @return the determinant of the matrix
     * @throws IllegalArgumentException if the matrix is not square
     */
    public static double compute(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Determinant is only defined for square matrices.");
        }

        int n = A.getRowCount();

        // For very small matrices, the overhead of object creation in LU decomposition
        // might outweigh the complexity of the recursive or direct formula.
        // However, standardizing on LU is robust.
        // Here we can dispatch based on size if desired.

        if (n <= 3) {
            return MinorsDeterminant.compute(A);
        } else {
            return LUDeterminant.compute(A);
        }
    }
}