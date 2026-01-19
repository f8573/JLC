package net.faulj.orthogonal;

import net.faulj.matrix.Matrix;
import net.faulj.solve.LeastSquaresSolver;
import net.faulj.vector.Vector;

/**
 * Solves the Best Approximation Problem using orthogonal projections.
 * <p>
 * This class addresses the fundamental problem in linear algebra: Given a vector <b>b</b>
 * and a subspace <b>W</b>, find the vector <b>b̂</b> in W that is closest to <b>b</b>.
 * </p>
 *
 * <h2>The Best Approximation Theorem:</h2>
 * <p>
 * Let W be a subspace of ℝⁿ. For any vector <b>y</b> in ℝⁿ, the orthogonal projection
 * of <b>y</b> onto W is the unique vector <b>ŷ</b> in W such that:
 * </p>
 * <pre>
 * ||y - ŷ|| < ||y - v||
 * </pre>
 * <p>
 * for all <b>v</b> in W distinct from <b>ŷ</b>. The distance ||y - ŷ|| is the
 * minimum distance from <b>y</b> to W.
 * </p>
 *
 * <h2>Geometric Interpretation:</h2>
 * <ul>
 * <li>The error vector <b>z</b> = <b>y</b> - <b>ŷ</b> is orthogonal to the subspace W.</li>
 * <li><b>ŷ</b> is the "shadow" of <b>y</b> directly beneath it in W.</li>
 * <li>This forms a right triangle with vertices <b>0</b>, <b>ŷ</b>, and <b>y</b>, satisfying Pythagoras:
 * <br>||y||² = ||ŷ||² + ||y - ŷ||²</li>
 * </ul>
 *
 * <h2>Calculation via Normal Equations:</h2>
 * <p>
 * If A is a matrix whose columns form a basis for W, then <b>ŷ</b> = A<b>x̂</b>,
 * where <b>x̂</b> is the least-squares solution to A<b>x</b> = <b>y</b>.
 * This leads to the normal equations:
 * </p>
 * <pre>
 * A<sup>T</sup>A<b>x̂</b> = A<sup>T</sup><b>y</b>
 * </pre>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Basis for subspace W (columns of A)
 * Matrix A = MatrixFactory.fromColumns(v1, v2);
 * Vector y = new Vector(new double[]{1, 2, 3});
 *
 * // Find best approximation
 * Vector yHat = BestApproximation.findClosest(A, y);
 *
 * // Calculate error
 * Vector error = y.subtract(yHat);
 * double minDistance = error.norm2();
 *
 * System.out.println("Closest point: " + yHat);
 * System.out.println("Distance to subspace: " + minDistance);
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Least-squares solutions for inconsistent systems</li>
 * <li>Data fitting and regression analysis</li>
 * <li>Signal processing (noise reduction)</li>
 * <li>Function approximation in Hilbert spaces</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see OrthogonalProjection
 * @see net.faulj.solve.LeastSquaresSolver
 */
public class BestApproximation {
	public Vector findClosest(Vector y, Matrix A) {
		LeastSquaresSolver leastSquaresSolver = new LeastSquaresSolver();
		Vector xHat = leastSquaresSolver.solve(A,y);
		return A.multiply(xHat.toMatrix()).getData()[0];
	}
}