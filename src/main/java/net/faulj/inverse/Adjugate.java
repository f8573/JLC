package net.faulj.inverse;

import net.faulj.determinant.LUDeterminant;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Computes the classical adjoint (adjugate) of a square matrix.
 * <p>
 * The adjugate of A, denoted adj(A), is the transpose of the cofactor matrix C.
 * </p>
 *
 * <h2>Mathematical Definition:</h2>
 * <p>
 * If A is an n×n matrix, its adjugate is defined as:
 * </p>
 * <pre>
 * adj(A)<sub>ij</sub> = C<sub>ji</sub> = (-1)<sup>i+j</sup> M<sub>ji</sub>
 * </pre>
 * <ul>
 * <li><b>C<sub>ji</sub></b> - The cofactor of the entry at row j, column i</li>
 * <li><b>M<sub>ji</sub></b> - The minor (determinant of the (n-1)×(n-1) submatrix) obtained by deleting row j and column i</li>
 * </ul>
 *
 * <h2>Relationship to Inverse:</h2>
 * <p>
 * The adjugate satisfies the fundamental property:
 * </p>
 * <pre>
 * A · adj(A) = adj(A) · A = det(A) · I
 * </pre>
 * <p>
 * Consequently, if det(A) ≠ 0:
 * </p>
 * <pre>
 * A<sup>-1</sup> = (1 / det(A)) · adj(A)
 * </pre>
 *
 * <h2>Computational Cost:</h2>
 * <p>
 * <b>WARNING:</b> This implementation is computationally expensive and intended primarily for
 * educational purposes or very small matrices.
 * </p>
 * <ul>
 * <li><b>Complexity:</b> O(n<sup>5</sup>) approx.</li>
 * <li><b>Reasoning:</b> It computes n<sup>2</sup> minors. Each minor requires a determinant calculation
 * of size (n-1), which takes O(n<sup>3</sup>) using LU decomposition.</li>
 * <li><b>Comparison:</b> Standard inversion methods like LU or Gauss-Jordan are O(n<sup>3</sup>).</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = ...; // Define 3x3 matrix
 * Matrix adj = Adjugate.compute(A);
 *
 * // Verify property: A * adj(A) = det(A) * I
 * double det = LUDeterminant.compute(A);
 * Matrix product = A.multiply(adj);
 * boolean propertyHolds = product.isDiagonal(det, 1e-9);
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.determinant.LUDeterminant
 * @see AdjugateInverse
 */
public class Adjugate {

    /**
     * Computes the adjugate matrix of the given square matrix.
     *
     * @param A The source square matrix.
     * @return The adjugate matrix adj(A).
     * @throws IllegalArgumentException if the matrix is not square.
     */
    public static Matrix compute(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Adjugate requires a square matrix");
        }

        int n = A.getRowCount();
        Vector[] cols = new Vector[n];

        for (int j = 0; j < n; j++) {
            double[] colData = new double[n];
            for (int i = 0; i < n; i++) {
                // Cofactor C_ij = (-1)^(i+j) * M_ij
                // Adjugate is the transpose, so entry (i,j) of Result is Cofactor(j,i)
                // adj(A)_ij = C_ji

                Matrix minor = A.minor(j, i); // Minor of element at row j, col i
                double det = LUDeterminant.compute(minor);

                double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
                colData[i] = sign * det;
            }
            cols[j] = new Vector(colData);
        }

        return new Matrix(cols);
    }
}