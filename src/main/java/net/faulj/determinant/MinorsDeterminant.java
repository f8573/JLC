package net.faulj.determinant;

import net.faulj.matrix.Matrix;

/**
 * Computes the determinant using Laplace Expansion (Expansion by Minors).
 * <p>
 * This implementation uses the recursive definition of the determinant.
 * While mathematically elegant, this approach has factorial complexity O(n!)
 * and is computationally prohibitive for n > 10.
 * </p>
 *
 * <h2>Mathematical Definition:</h2>
 * <p>
 * For a matrix A of size n×n, the determinant is defined recursively:
 * </p>
 * <pre>
 * det(A) = Σ (-1)<sup>i+j</sup> * a<sub>ij</sub> * M<sub>ij</sub>
 * </pre>
 * <ul>
 * <li><b>i, j:</b> Row and column indices (expansion usually along first row, i=0)</li>
 * <li><b>a<sub>ij</sub>:</b> Element at position (i, j)</li>
 * <li><b>M<sub>ij</sub>:</b> Minor determinant (determinant of the (n-1)×(n-1) submatrix
 * obtained by removing row i and column j)</li>
 * </ul>
 *
 * <h2>Computational Path:</h2>
 * <ol>
 * <li>Base case: If n=1, return value. If n=2, return ad - bc.</li>
 * <li>Recursive step: Iterate through first row (or column).</li>
 * <li>Compute cofactor: (-1)<sup>row+col</sup> * a[row][col].</li>
 * <li>Extract submatrix (minor) and recurse.</li>
 * <li>Sum results.</li>
 * </ol>
 *
 * <h2>Complexity Analysis:</h2>
 * <ul>
 * <li><b>n=3:</b> 3! = 6 operations</li>
 * <li><b>n=5:</b> 5! = 120 operations</li>
 * <li><b>n=10:</b> 10! = 3,628,800 operations</li>
 * <li><b>n=20:</b> 20! ≈ 2.4 × 10¹⁸ (Infeasible)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][]{
 * {1, 2, 3},
 * {0, 4, 5},
 * {1, 0, 6}
 * });
 *
 * double det = MinorsDeterminant.compute(A); // Returns 1*24 - 2*(-5) + 3*(-4) ...
 * }</pre>
 *
 * <h2>Applicability:</h2>
 * <p>
 * This class is primarily intended for:
 * </p>
 * <ul>
 * <li>Educational purposes (demonstrating the definition)</li>
 * <li>Testing correctness of O(n³) algorithms on small inputs</li>
 * <li>Handling symbolic matrices (if extended)</li>
 * <li>Very small fixed-size matrices (2x2, 3x3) where overhead of LU is higher</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.matrix.Matrix#minor(int, int)
 */
public class MinorsDeterminant {

    /**
     * Computes the determinant recursively using expansion by minors.
     * <p>
     * <b>Warning:</b> This method is extremely slow for matrices larger than 10x10.
     * Consider using {@link LUDeterminant} for larger matrices.
     * </p>
     *
     * @param A the square matrix
     * @return the determinant value
     * @throws IllegalArgumentException if the matrix is not square
     */
    public static double compute(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Matrix must be square.");
        }
        if (!A.isReal()) {
            throw new IllegalArgumentException("MinorsDeterminant only supports real matrices. Use Determinant.computeComplex(Matrix).");
        }
        int n = A.getRowCount();

        // Base cases for recursion
        if (n == 1) {
            return A.get(0, 0);
        }
        if (n == 2) {
            return A.get(0, 0) * A.get(1, 1) - A.get(0, 1) * A.get(1, 0);
        }

        double det = 0.0;
        // Expand along the first row (row = 0)
        for (int col = 0; col < n; col++) {
            // Get the element a_0j
            double element = A.get(0, col);

            // Optimization: Skip if element is zero
            if (Math.abs(element) > 1e-15) {
                // Determine sign (-1)^(0+col)
                double sign = (col % 2 == 0) ? 1.0 : -1.0;

                // Get submatrix by removing row 0 and current column
                Matrix submatrix = A.minor(0, col);

                // Recursive call
                det += sign * element * compute(submatrix);
            }
        }
        return det;
    }
}
