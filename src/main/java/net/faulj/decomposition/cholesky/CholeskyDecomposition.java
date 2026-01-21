package net.faulj.decomposition.cholesky;

import net.faulj.decomposition.result.CholeskyResult;
import net.faulj.matrix.Matrix;

/**
 * Cholesky decomposition implementation.
 */
public class CholeskyDecomposition {

	/**
	 * Decomposes a symmetric positive definite matrix A into L such that A = L * L^T.
	 * @param A symmetric positive definite matrix
	 * @return CholeskyResult containing lower triangular factor L
	 */
	public CholeskyResult decompose(Matrix A) {
		if (A == null) throw new IllegalArgumentException("Matrix A must not be null");
		if (!A.isSquare()) throw new IllegalArgumentException("Matrix must be square");

		int n = A.getRowCount();
		double[][] L = new double[n][n];

		for (int j = 0; j < n; j++) {
			// diagonal element
			double sum = 0.0;
			for (int k = 0; k < j; k++) {
				double v = L[j][k];
				sum += v * v;
			}
			double diag = A.get(j, j) - sum;
			if (diag <= 0.0) {
				throw new ArithmeticException("Matrix is not positive definite (non-positive pivot at " + j + ")");
			}
			L[j][j] = Math.sqrt(diag);

			// off-diagonal lower elements
			for (int i = j + 1; i < n; i++) {
				double s = 0.0;
				for (int k = 0; k < j; k++) {
					s += L[i][k] * L[j][k];
				}
				L[i][j] = (A.get(i, j) - s) / L[j][j];
			}
		}

		return new CholeskyResult(A, new Matrix(L));
	}
}
