package net.faulj.spaces;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Implements projection operators for vector subspaces.
 * <p>
 * This class handles the computation of projections of vectors onto subspaces.
 * A projection is a linear transformation \( P \) such that \( P^2 = P \) (idempotent).
 * </p>
 *
 * <h2>Types of Projections:</h2>
 * <ul>
 * <li><b>Orthogonal Projection:</b> Projects vector \( y \) onto subspace \( W \) such that
 * the error vector \( y - \hat{y} \) is orthogonal to \( W \).
 * <br>Matrix: \( P = A(A^TA)^{-1}A^T \) (where columns of A form a basis for W).
 * </li>
 * <li><b>Oblique Projection:</b> Projects onto a subspace along a specific non-orthogonal direction.</li>
 * </ul>
 *
 * <h2>Best Approximation Theorem:</h2>
 * <p>
 * The orthogonal projection \( \hat{y} \) of \( y \) onto \( W \) is the closest vector in
 * \( W \) to \( y \):
 * </p>
 * <pre>
 * ||y - \hat{y}|| < ||y - v||  for all v in W distinct from \hat{y}
 * </pre>
 *
 * <h2>Usage:</h2>
 * <p>
 * Projections are fundamental in:
 * </p>
 * <ul>
 * <li>Least squares solutions.</li>
 * <li>Gram-Schmidt orthogonalization.</li>
 * <li>Data compression and dimensionality reduction.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.orthogonal.OrthogonalProjection
 * @see OrthogonalComplement
 */
public class Projection {
	public Projection() {
		throw new RuntimeException("Class unfinished");
	}
}