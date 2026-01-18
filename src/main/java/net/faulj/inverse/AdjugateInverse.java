package net.faulj.inverse;

import net.faulj.determinant.LUDeterminant;
import net.faulj.matrix.Matrix;

/**
 * Computes the inverse of a matrix using the Adjugate formula (Cramer's Rule).
 *
 * <h2>Formula:</h2>
 * <p>
 * The inverse is calculated using the relationship between the determinant and the adjugate matrix:
 * </p>
 * <pre>
 * A<sup>-1</sup> = (1 / det(A)) · adj(A)
 * </pre>
 *
 * <h2>Characteristics:</h2>
 * <table border="1">
 * <tr><th>Aspect</th><th>Description</th></tr>
 * <tr><td>Complexity</td><td>O(n<sup>5</sup>) - Extremely slow for large n</td></tr>
 * <tr><td>Stability</td><td>Potentially unstable for ill-conditioned matrices due to determinant overflow/underflow</td></tr>
 * <tr><td>Use Case</td><td>Theoretical demonstrations, small symbolic-like calculations, or 2x2/3x3 explicit inverses</td></tr>
 * </table>
 *
 * <h2>Alternatives:</h2>
 * <p>
 * For general purpose numerical inversion, use {@link LUInverse} or {@link GaussJordanInverse},
 * which operate in O(n<sup>3</sup>) time.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = ...;
 * try {
 * Matrix inv = AdjugateInverse.compute(A);
 * System.out.println("Inverse:\n" + inv);
 * } catch (ArithmeticException e) {
 * System.out.println("Matrix is singular!");
 * }
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see Adjugate
 * @see LUInverse
 */
public class AdjugateInverse {

    /**
     * Calculates the inverse of the matrix using the determinant and adjugate.
     *
     * @param A The square matrix to invert.
     * @return The inverse matrix A<sup>-1</sup>.
     * @throws ArithmeticException if the determinant is zero (matrix is singular) or near zero.
     * @throws IllegalArgumentException if the matrix is not square.
     */
    public static Matrix compute(Matrix A) {
        double det = LUDeterminant.compute(A);
        if (Math.abs(det) < 1e-12) {
            throw new ArithmeticException("Matrix is singular");
        }

        Matrix adj = Adjugate.compute(A);
        return adj.multiplyScalar(1.0 / det);
    }
}