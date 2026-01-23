package net.faulj.svd;

import net.faulj.matrix.Matrix;

/**
 * Computes the Moore-Penrose Pseudoinverse (A<sup>+</sup>).
 * <p>
 * The pseudoinverse generalizes the matrix inverse to non-square and singular matrices.
 * It is defined via the SVD (A = UΣV<sup>T</sup>) as:
 * </p>
 * <pre>
 * A⁺ = V * Σ⁺ * Uᵀ
 * </pre>
 * <p>
 * Where Σ⁺ is formed by taking the reciprocal of non-zero singular values and transposing the diagonal matrix.
 * </p>
 *
 * <h2>Mathematical Properties:</h2>
 * <p>
 * A⁺ satisfies the four Moore-Penrose conditions:
 * </p>
 * <ol>
 * <li><b>AA⁺A = A</b> (A⁺ is a weak inverse)</li>
 * <li><b>A⁺AA⁺ = A⁺</b> (A is a weak inverse of A⁺)</li>
 * <li><b>(AA⁺)ᵀ = AA⁺</b> (AA⁺ is Hermitian/Symmetric)</li>
 * <li><b>(A⁺A)ᵀ = A⁺A</b> (A⁺A is Hermitian/Symmetric)</li>
 * </ol>
 *
 * <h2>Least Squares Solution:</h2>
 * <p>
 * For a linear system Ax = b, the vector x = A⁺b gives the solution with the minimum
 * Euclidean norm ||x||₂ among all solutions that minimize the residual ||Ax - b||₂.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}}); // 3x2 Matrix
 *
 * // Compute pseudoinverse
 * Pseudoinverse pinv = new Pseudoinverse();
 * Matrix A_plus = pinv.compute(A);
 *
 * // Solve Ax = b for least squares
 * Vector b = new Vector(new double[]{1, 1, 1});
 * Vector x = A_plus.operate(b);
 * }</pre>
 *
 * <h2>Tolerance Handling:</h2>
 * <ul>
 * <li>Singular values smaller than a threshold are treated as zero.</li>
 * <li>Default threshold: max(m,n) * σ₁ * machine_epsilon</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see SVDecomposition
 * @see net.faulj.solve.LeastSquaresSolver
 */
public class Pseudoinverse {
	private static final double EPS = 2.220446049250313e-16;

	public Pseudoinverse() {
	}

	/**
	 * Computes the Moore-Penrose pseudoinverse using the default tolerance.
	 *
	 * @param A input matrix
	 * @return pseudoinverse of A
	 */
	public Matrix compute(Matrix A) {
		if (A == null) {
			throw new IllegalArgumentException("Matrix must not be null");
		}
		if (!A.isReal()) {
			throw new UnsupportedOperationException("Pseudoinverse requires a real-valued matrix");
		}
		SVDecomposition svd = new SVDecomposition();
		net.faulj.decomposition.result.SVDResult result = svd.decompose(A);
		double tol = defaultTolerance(result.getSingularValues(), A.getRowCount(), A.getColumnCount());
		return computeFromSvd(result, tol);
	}

	/**
	 * Computes the Moore-Penrose pseudoinverse using a custom tolerance.
	 *
	 * @param A input matrix
	 * @param tolerance threshold below which singular values are treated as zero
	 * @return pseudoinverse of A
	 */
	public Matrix compute(Matrix A, double tolerance) {
		if (A == null) {
			throw new IllegalArgumentException("Matrix must not be null");
		}
		if (tolerance < 0) {
			throw new IllegalArgumentException("Tolerance must be non-negative");
		}
		if (!A.isReal()) {
			throw new UnsupportedOperationException("Pseudoinverse requires a real-valued matrix");
		}
		SVDecomposition svd = new SVDecomposition();
		net.faulj.decomposition.result.SVDResult result = svd.decompose(A);
		return computeFromSvd(result, tolerance);
	}

	private static Matrix computeFromSvd(net.faulj.decomposition.result.SVDResult result, double tolerance) {
		Matrix U = result.getU();
		Matrix V = result.getV();
		double[] singularValues = result.getSingularValues();

		int m = U.getRowCount();
		int n = V.getRowCount();
		int r = Math.min(Math.min(U.getColumnCount(), V.getColumnCount()), singularValues.length);

		if (r == 0) {
			return Matrix.zero(n, m);
		}

		Matrix Uthin = U.getColumnCount() == r ? U : U.crop(0, m - 1, 0, r - 1);
		Matrix Vthin = V.getColumnCount() == r ? V : V.crop(0, n - 1, 0, r - 1);

		Matrix sigmaPlus = new Matrix(r, r);
		for (int i = 0; i < r; i++) {
			double s = singularValues[i];
			if (Math.abs(s) > tolerance) {
				sigmaPlus.set(i, i, 1.0 / s);
			}
		}

		return Vthin.multiply(sigmaPlus).multiply(Uthin.transpose());
	}

	private static double defaultTolerance(double[] singularValues, int rows, int cols) {
		double maxSigma = 0.0;
		for (double s : singularValues) {
			maxSigma = Math.max(maxSigma, Math.abs(s));
		}
		if (maxSigma == 0.0) {
			return 0.0;
		}
		return Math.max(rows, cols) * maxSigma * EPS;
	}
}
