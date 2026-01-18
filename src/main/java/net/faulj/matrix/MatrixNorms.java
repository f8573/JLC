package net.faulj.matrix;

/**
 * Utility class for computing various matrix norms.
 * <p>
 * A matrix norm is a function ||·||: K<sup>m×n</sup> → ℝ that assigns a strictly positive
 * length or size to all non-zero matrices. These norms are crucial for measuring errors,
 * analyzing stability, and determining convergence in numerical linear algebra.
 * </p>
 *
 * <h2>Supported Norms:</h2>
 * <ul>
 * <li><b>Frobenius Norm (||A||<sub>F</sub>):</b> Entry-wise Euclidean norm.</li>
 * <li><b>1-Norm (||A||<sub>1</sub>):</b> Maximum absolute column sum.</li>
 * <li><b>Infinity-Norm (||A||<sub>∞</sub>):</b> Maximum absolute row sum.</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <p>
 * All implemented norms satisfy the fundamental vector norm axioms:
 * </p>
 * <ul>
 * <li><b>Positivity:</b> ||A|| ≥ 0, and ||A|| = 0 iff A = 0</li>
 * <li><b>Homogeneity:</b> ||αA|| = |α| ||A||</li>
 * <li><b>Triangle Inequality:</b> ||A + B|| ≤ ||A|| + ||B||</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = MatrixFactory.random(3, 3);
 *
 * // Calculate norms
 * double fNorm = MatrixNorms.frobeniusNorm(A);
 * double infNorm = MatrixNorms.normInf(A);
 * double oneNorm = MatrixNorms.norm1(A);
 *
 * // Verify consistency (e.g., ||A||_2 <= ||A||_F)
 * System.out.println("Frobenius Norm: " + fNorm);
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Error estimation in linear system solving (||b - Ax||)</li>
 * <li>Condition number estimation (||A|| * ||A<sup>-1</sup>||)</li>
 * <li>Convergence criteria for iterative methods (e.g., Jacobi, Gauss-Seidel)</li>
 * <li>Perturbation theory analysis</li>
 * </ul>
 *
 * <h2>Immutability:</h2>
 * <p>
 * This is a utility class containing only static methods. It does not maintain state.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.matrix.Matrix
 * @see net.faulj.condition.ConditionNumber
 */
public class MatrixNorms {

	/**
	 * Computes the Frobenius norm of a matrix.
	 * <p>
	 * The Frobenius norm is defined as the square root of the sum of the absolute
	 * squares of its elements. It is equivalent to the Euclidean norm of the
	 * matrix elements flattened into a vector.
	 * </p>
	 *
	 * <h2>Formula:</h2>
	 * <pre>
	 * ||A||<sub>F</sub> = sqrt( Σ<sub>i</sub> Σ<sub>j</sub> |a<sub>ij</sub>|² )
	 * </pre>
	 *
	 * <h2>Properties:</h2>
	 * <ul>
	 * <li>Invariant under orthogonal transformations (rotations).</li>
	 * <li>Compatible with the vector 2-norm: ||Ax||<sub>2</sub> ≤ ||A||<sub>F</sub> ||x||<sub>2</sub>.</li>
	 * </ul>
	 *
	 * @param m The matrix to measure.
	 * @return The Frobenius norm (non-negative double).
	 */
	public static double frobeniusNorm(Matrix m) {
		double sum = 0.0;
		for (int i = 0; i < m.getRowCount(); i++) {
			for (int j = 0; j < m.getColumnCount(); j++) {
				double val = m.get(i, j);
				sum += val * val;
			}
		}
		return Math.sqrt(sum);
	}

	/**
	 * Computes the "1-norm" (Maximum Absolute Column Sum) of a matrix.
	 * <p>
	 * The 1-norm is the operator norm induced by the vector 1-norm. It represents
	 * the maximum factor by which the matrix can stretch a vector in the 1-norm.
	 * </p>
	 *
	 * <h2>Formula:</h2>
	 * <pre>
	 * ||A||<sub>1</sub> = max<sub>j</sub> ( Σ<sub>i</sub> |a<sub>ij</sub>| )
	 * </pre>
	 *
	 * <h2>Complexity:</h2>
	 * <p>
	 * O(m*n) where m is rows and n is columns.
	 * </p>
	 *
	 * @param m The matrix to measure.
	 * @return The maximum absolute column sum.
	 */
	public static double norm1(Matrix m) {
		double maxColSum = 0.0;
		for (int j = 0; j < m.getColumnCount(); j++) {
			double colSum = 0.0;
			for (int i = 0; i < m.getRowCount(); i++) {
				colSum += Math.abs(m.get(i, j));
			}
			if (colSum > maxColSum) {
				maxColSum = colSum;
			}
		}
		return maxColSum;
	}

	/**
	 * Computes the "Infinity-norm" (Maximum Absolute Row Sum) of a matrix.
	 * <p>
	 * The Infinity-norm is the operator norm induced by the vector infinity-norm.
	 * It is useful for bounding the spectral radius and error analysis.
	 * </p>
	 *
	 * <h2>Formula:</h2>
	 * <pre>
	 * ||A||<sub>∞</sub> = max<sub>i</sub> ( Σ<sub>j</sub> |a<sub>ij</sub>| )
	 * </pre>
	 *
	 * <h2>Relation to 1-Norm:</h2>
	 * <p>
	 * ||A||<sub>∞</sub> = ||A<sup>T</sup>||<sub>1</sub>
	 * </p>
	 *
	 * @param m The matrix to measure.
	 * @return The maximum absolute row sum.
	 */
	public static double normInf(Matrix m) {
		double maxRowSum = 0.0;
		for (int i = 0; i < m.getRowCount(); i++) {
			double rowSum = 0.0;
			for (int j = 0; j < m.getColumnCount(); j++) {
				rowSum += Math.abs(m.get(i, j));
			}
			if (rowSum > maxRowSum) {
				maxRowSum = rowSum;
			}
		}
		return maxRowSum;
	}
}