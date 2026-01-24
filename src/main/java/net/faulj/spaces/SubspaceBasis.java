package net.faulj.spaces;

import net.faulj.matrix.Matrix;
import net.faulj.core.Tolerance;
import net.faulj.vector.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Utility class for computing bases of fundamental matrix subspaces.
 * <p>
 * This class provides static methods to determine the basis vectors for the three
 * fundamental subspaces associated with a matrix \( A \):
 * </p>
 * <ul>
 * <li><b>Row Space:</b> The span of the row vectors of \( A \).</li>
 * <li><b>Column Space:</b> The span of the column vectors of \( A \) (also known as the Image or Range).</li>
 * <li><b>Null Space:</b> The set of all vectors \( x \) such that \( Ax = 0 \) (also known as the Kernel).</li>
 * </ul>
 *
 * <h2>Fundamental Theorem of Linear Algebra:</h2>
 * <p>
 * The dimensions of these subspaces are related by the Rank-Nullity Theorem:
 * </p>
 * <ul>
 * <li>dim(Row Space) = dim(Column Space) = rank(A)</li>
 * <li>dim(Null Space) + rank(A) = n (where n is the number of columns)</li>
 * <li>Row Space \(\perp\) Null Space (in \(\mathbb{R}^n\))</li>
 * </ul>
 *
 * <h2>Algorithms:</h2>
 * <p>
 * Computations use numerically stable orthonormalization for row/column spaces and
 * RREF-based construction for null space:
 * </p>
 * <ol>
 * <li><b>Row Basis:</b> Orthonormalize the matrix rows (Modified Gram-Schmidt).</li>
 * <li><b>Column Basis:</b> Orthonormalize the matrix columns (Modified Gram-Schmidt).</li>
 * <li><b>Null Basis:</b> Build a basis from the RREF free variables, then orthonormalize.</li>
 * </ol>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = ...; // Define matrix
 *
 * // Compute bases
 * Set<Vector> rowBasis = SubspaceBasis.rowSpaceBasis(A);
 * Set<Vector> colBasis = SubspaceBasis.columnSpaceBasis(A);
 * Set<Vector> nullBasis = SubspaceBasis.nullSpaceBasis(A);
 *
 * System.out.println("Rank: " + rowBasis.size());
 * System.out.println("Nullity: " + nullBasis.size());
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.matrix.Matrix
 * @see net.faulj.vector.Vector
 */
public class SubspaceBasis {

	/**
	 * Computes a basis for the row space of the given matrix.
	 * <p>
	 * The row space is the subspace spanned by the rows of the matrix. This method
	 * orthonormalizes the matrix rows to produce a numerically stable basis.
	 * </p>
	 *
	 * <h3>Mathematical Properties:</h3>
	 * <ul>
	 * <li>The basis vectors are orthogonal to the Null Space.</li>
	 * <li>Elementary row operations do not change the row space.</li>
	 * <li>The size of the returned set equals the rank of the matrix.</li>
	 * </ul>
	 *
	 * @param m The matrix to analyze.
	 * @return A set of linearly independent {@link Vector} objects spanning the row space.
	 */
	public static Set<Vector> rowSpaceBasis(Matrix m) {
		Objects.requireNonNull(m, "Matrix must not be null");
		if (!m.isReal()) {
			throw new UnsupportedOperationException("Row space basis requires a real-valued matrix");
		}
		List<Vector> rows = new ArrayList<>();
		for (int r = 0; r < m.getRowCount(); r++) {
			rows.add(new Vector(m.getRow(r)));
		}
		return orthonormalize(rows, Tolerance.get());
	}

	/**
	 * Computes a basis for the column space (range) of the given matrix.
	 * <p>
	 * The column space is the subspace spanned by the columns of the matrix.
	 * This method orthonormalizes the matrix columns to produce a numerically stable basis.
	 * </p>
	 *
	 * <h3>Relation to Linear Systems:</h3>
	 * <p>
	 * The system \( Ax = b \) is solvable if and only if \( b \) lies in the column space.
	 * </p>
	 *
	 * @param m The matrix to analyze.
	 * @return A set of linearly independent {@link Vector} objects spanning the column space.
	 */
	public static Set<Vector> columnSpaceBasis(Matrix m) {
		Objects.requireNonNull(m, "Matrix must not be null");
		if (!m.isReal()) {
			throw new UnsupportedOperationException("Column space basis requires a real-valued matrix");
		}
		List<Vector> cols = new ArrayList<>();
		for (Vector col : m.getData()) {
			cols.add(new Vector(col.getData()));
		}
		return orthonormalize(cols, Tolerance.get());
	}

	/**
	 * Computes a basis for the null space (kernel) of the given matrix.
	 * <p>
	 * The null space consists of all vectors \( x \) such that \( Ax = 0 \).
	 * This method finds the general solution to the homogeneous system by identifying
	 * free variables in the RREF. Each free variable generates one basis vector.
	 * </p>
	 *
	 * <h3>Algorithm Details:</h3>
	 * <p>
	 * This implementation constructs the basis by separating pivot and free columns,
	 * effectively solving \( x_{pivot} = -F x_{free} \), where \( F \) is the
	 * matrix of coefficients for free variables in the equations for pivot variables.
	 * </p>
	 *
	 * @param m The matrix to analyze.
	 * @return A set of linearly independent {@link Vector} objects spanning the null space.
	 * Returns an empty set if the null space is trivial (contains only the zero vector).
	 */
	public static Set<Vector> nullSpaceBasis(Matrix m) {
		Objects.requireNonNull(m, "Matrix must not be null");
		if (!m.isReal()) {
			throw new UnsupportedOperationException("Null space basis requires a real-valued matrix");
		}
		if (m.getColumnCount() == 0) {
			return new LinkedHashSet<>();
		}

		Matrix rref = m.copy();
		rref.toReducedRowEchelonForm();
		int rows = rref.getRowCount();
		int cols = rref.getColumnCount();
		double tol = Tolerance.get();

		boolean[] pivotCols = new boolean[cols];
		int[] pivotRows = new int[cols];
		Arrays.fill(pivotRows, -1);

		for (int row = 0; row < rows; row++) {
			int lead = -1;
			for (int col = 0; col < cols; col++) {
				if (Math.abs(rref.get(row, col)) > tol) {
					lead = col;
					break;
				}
			}
			if (lead != -1) {
				pivotCols[lead] = true;
				pivotRows[lead] = row;
			}
		}

		List<Vector> basis = new ArrayList<>();
		for (int freeCol = 0; freeCol < cols; freeCol++) {
			if (pivotCols[freeCol]) {
				continue;
			}
			double[] vec = new double[cols];
			vec[freeCol] = 1.0;
			for (int pivotCol = 0; pivotCol < cols; pivotCol++) {
				int prow = pivotRows[pivotCol];
				if (prow == -1) {
					continue;
				}
				double val = rref.get(prow, freeCol);
				if (Math.abs(val) > tol) {
					vec[pivotCol] = -val;
				}
			}
			basis.add(new Vector(vec));
		}

		return orthonormalize(basis, tol);
	}

	private static Set<Vector> orthonormalize(List<Vector> vectors, double tol) {
		LinkedHashSet<Vector> result = new LinkedHashSet<>();
		List<double[]> basis = new ArrayList<>();
		for (Vector vector : vectors) {
			if (vector == null || vector.dimension() == 0) {
				continue;
			}
			double[] w = Arrays.copyOf(vector.getData(), vector.dimension());
			for (double[] q : basis) {
				double proj = dot(q, w);
				if (proj != 0.0) {
					for (int i = 0; i < w.length; i++) {
						w[i] -= proj * q[i];
					}
				}
			}
			double norm = norm2(w);
			if (norm > tol) {
				for (int i = 0; i < w.length; i++) {
					w[i] /= norm;
				}
				basis.add(w);
				result.add(new Vector(w));
			}
		}
		return result;
	}

	private static double dot(double[] a, double[] b) {
		double sum = 0.0;
		for (int i = 0; i < a.length; i++) {
			sum += a[i] * b[i];
		}
		return sum;
	}

	private static double norm2(double[] a) {
		double sum = 0.0;
		for (double v : a) {
			sum += v * v;
		}
		return Math.sqrt(sum);
	}
}
