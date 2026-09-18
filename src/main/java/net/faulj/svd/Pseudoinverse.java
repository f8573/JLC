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

	/**
	 * Create a pseudoinverse helper.
	 */
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

	/**
	 * Compute the pseudoinverse from a precomputed SVD.
	 *
	 * @param result SVD result
	 * @param tolerance threshold below which singular values are treated as zero
	 * @return pseudoinverse matrix
	 */
	private static Matrix computeFromSvd(net.faulj.decomposition.result.SVDResult result, double tolerance) {
		Matrix U = result.getU();
		Matrix V = result.getV();
		double[] singularValues = result.getSingularValues();

		int m = U.getRowCount();
		int n = V.getRowCount();
		int uCols = U.getColumnCount();
		int vCols = V.getColumnCount();
		int svLen = singularValues.length;

		// Ensure we select the largest singular values first (robust if SVD
		// implementation returns unsorted singular values). Build an index
		// ordering by descending magnitude and then select those above the
		// tolerance.
		Integer[] order = new Integer[svLen];
		for (int i = 0; i < svLen; i++) order[i] = i;
		java.util.Arrays.sort(order, (a, b) -> Double.compare(Math.abs(singularValues[b]), Math.abs(singularValues[a])));
		java.util.ArrayList<Integer> keep = new java.util.ArrayList<>();
		for (int idx : order) {
			double s = singularValues[idx];
			if (Math.abs(s) > tolerance) keep.add(idx);
		}

		if (keep.isEmpty()) {
			return Matrix.zero(n, m);
		}

		int k = keep.size();
		Matrix U_k = new Matrix(m, k);
		Matrix V_k = new Matrix(n, k);
		Matrix sigmaPlus = new Matrix(k, k);

		for (int col = 0; col < k; col++) {
			int svIndex = keep.get(col);
			// Copy the corresponding column from U (m x uCols) and V (n x vCols)
			for (int row = 0; row < m; row++) {
				double val = (svIndex < uCols) ? U.get(row, svIndex) : 0.0;
				U_k.set(row, col, val);
			}
			for (int row = 0; row < n; row++) {
				double val = (svIndex < vCols) ? V.get(row, svIndex) : 0.0;
				V_k.set(row, col, val);
			}
			double s = singularValues[svIndex];
			sigmaPlus.set(col, col, (Math.abs(s) > 0.0) ? 1.0 / s : 0.0);
		}

		return V_k.multiply(sigmaPlus).multiply(U_k.transpose());
	}

	/**
	 * Compute the default tolerance for singular value cutoff.
	 *
	 * @param singularValues singular values
	 * @param rows row count
	 * @param cols column count
	 * @return tolerance value
	 */
	private static double defaultTolerance(double[] singularValues, int rows, int cols) {
		double maxSigma = 0.0;
		for (double s : singularValues) {
			maxSigma = Math.max(maxSigma, Math.abs(s));
		}
		if (maxSigma == 0.0) {
			return 0.0;
		}
		// Use a slightly more conservative cutoff to avoid inverting
		// tiny numerical singular values that arise from round-off
		// in rank-deficient matrices. Multiply EPS by 10 for safety.
		return Math.max(rows, cols) * maxSigma * (EPS * 10.0);
	}
}
