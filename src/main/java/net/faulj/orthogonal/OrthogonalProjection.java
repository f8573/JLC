package net.faulj.orthogonal;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Computes and manipulates orthogonal projection matrices.
 * <p>
 * An orthogonal projection matrix <b>P</b> maps any vector <b>v</b> to its orthogonal
 * projection <b>Pv</b> onto a specific subspace W.
 * </p>
 *
 * <h2>Formula:</h2>
 * <p>
 * If the columns of matrix <b>A</b> form a basis for subspace W, the projection matrix <b>P</b>
 * is given by:
 * </p>
 * <pre>
 * P = A(A<sup>T</sup>A)<sup>-1</sup>A<sup>T</sup>
 * </pre>
 * <p>
 * Note: If the columns of A are orthonormal (Q), the formula simplifies to <b>P = QQ<sup>T</sup></b>.
 * </p>
 *
 * <h2>Properties of Projection Matrices:</h2>
 * <ul>
 * <li><b>Idempotent:</b> P<sup>2</sup> = P (Projecting twice is the same as projecting once)</li>
 * <li><b>Symmetric:</b> P<sup>T</sup> = P (Self-adjoint operator)</li>
 * <li><b>Trace:</b> tr(P) = dim(W) (The rank of the projection)</li>
 * <li><b>Singular:</b> Not invertible (unless W is the entire space)</li>
 * </ul>
 *
 * <h2>Complementary Projection:</h2>
 * <p>
 * The matrix <b>I - P</b> projects onto the orthogonal complement W<sup>⊥</sup>.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create basis matrix A
 * Matrix A = ...;
 *
 * // Compute projection matrix for Col(A)
 * Matrix P = OrthogonalProjection.createMatrix(A);
 *
 * // Project vector b onto Col(A)
 * Vector b = ...;
 * Vector projection = P.multiply(b);
 *
 * // Verify idempotence
 * Matrix P2 = P.multiply(P); // Should equal P
 * }</pre>
 *
 * <h2>Computational Considerations:</h2>
 * <p>
 * Explicit calculation of P involving (A<sup>T</sup>A)<sup>-1</sup> can be numerically
 * unstable. Using an orthonormal basis (via QR decomposition) is preferred for
 * high precision:
 * </p>
 * <ol>
 * <li>Compute QR factorization: A = QR</li>
 * <li>P = Q Q<sup>T</sup></li>
 * </ol>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see BestApproximation
 * @see net.faulj.decomposition.qr.GramSchmidt
 */
public class OrthogonalProjection {
	public static Matrix createMatrix(Matrix A) {
		if (A == null) {
			throw new IllegalArgumentException("Matrix must not be null");
		}
		int rows = A.getRowCount();
		if (A.getColumnCount() == 0 || rows == 0) {
			return new Matrix(rows, rows);
		}
		Matrix At = A.transpose();
		Matrix AtAInv = At.multiply(A).inverse();
		return A.multiply(AtAInv).multiply(At);
	}
}
