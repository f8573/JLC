package net.faulj.inverse;

import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;
import net.faulj.solve.LUSolver;
import net.faulj.vector.Vector;

/**
 * Computes the matrix inverse using LU Decomposition.
 * <p>
 * This is the standard, numerically stable method for inverting dense matrices.
 * It works by decomposing the matrix into A = LU and then solving for the columns of the inverse.
 * </p>
 *
 * <h2>Algorithm:</h2>
 * <ol>
 * <li><b>Decomposition:</b> Factorize A into P·L·U (O(n<sup>3</sup>))</li>
 * <li><b>Forward/Back Substitution:</b> Solve A·x<sub>i</sub> = e<sub>i</sub> for each column i=1..n
 * <ul>
 * <li>e<sub>i</sub> is the i-th column of the identity matrix</li>
 * <li>x<sub>i</sub> becomes the i-th column of A<sup>-1</sup></li>
 * </ul>
 * </li>
 * </ol>
 *
 * <h2>Complexity:</h2>
 * <ul>
 * <li><b>Time:</b> O(n<sup>3</sup>)
 * <ul>
 * <li>Decomposition: ~2n<sup>3</sup>/3 flops</li>
 * <li>Solving n systems: n × (2n<sup>2</sup>) = 2n<sup>3</sup> flops</li>
 * </ul>
 * </li>
 * <li><b>Space:</b> O(n<sup>2</sup>) to store the result</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = MatrixFactory.random(5, 5);
 *
 * try {
 * Matrix A_inv = LUInverse.compute(A);
 * * // Validate: A * A_inv ≈ I
 * Matrix Identity = A.multiply(A_inv);
 * } catch (ArithmeticException e) {
 * // Handle singular matrix
 * }
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.lu.LUDecomposition
 * @see net.faulj.solve.LUSolver
 */
public class LUInverse {

    /**
     * Computes the inverse of a square matrix using LU decomposition.
     *
     * @param A The square matrix to invert.
     * @return The inverse matrix A<sup>-1</sup>.
     * @throws IllegalArgumentException if the matrix is not square.
     * @throws ArithmeticException if the matrix is singular (numerically rank deficient).
     */
    public static Matrix compute(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Matrix inversion requires a square matrix");
        }

        if (!A.isReal()) {
            return A.inverse();
        }

        int n = A.getRowCount();
        LUDecomposition decomp = new LUDecomposition();
        LUResult lu = decomp.decompose(A);

        if (lu.isSingular()) {
            throw new ArithmeticException("Matrix is singular (determinant is 0) and cannot be inverted");
        }

        LUSolver solver = new LUSolver();
        Vector[] invColumns = new Vector[n];

        // Solve Ax_i = e_i for each column i
        for (int i = 0; i < n; i++) {
            // Create standard basis vector e_i
            double[] eData = new double[n];
            eData[i] = 1.0;
            Vector e = new Vector(eData);

            // The solution x is the ith column of the inverse
            invColumns[i] = solver.solve(lu, e);
        }

        return new Matrix(invColumns);
    }
}