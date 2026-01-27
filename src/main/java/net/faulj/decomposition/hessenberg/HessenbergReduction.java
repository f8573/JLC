package net.faulj.decomposition.hessenberg;

import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.matrix.Matrix;

/**
 * Reduces a square matrix to upper Hessenberg form using orthogonal Householder transformations.
 * <p>
 * The Hessenberg reduction factors a square matrix A into the form:
 * </p>
 * <pre>
 *   A = Q * H * Q<sup>T</sup>
 * </pre>
 * <p>
 * where:
 * </p>
 * <ul>
 *   <li><b>Q</b> is an n-by-n orthogonal matrix (Q<sup>T</sup>Q = I)</li>
 *   <li><b>H</b> is an upper Hessenberg matrix (zeros below first subdiagonal)</li>
 * </ul>
 *
 * <h2>Hessenberg Form:</h2>
 * <p>
 * An upper Hessenberg matrix has zeros below the first subdiagonal:
 * </p>
 * <pre>
 *   ┌ h₁₁ h₁₂ h₁₃ h₁₄ h₁₅ ┐
 *   │ h₂₁ h₂₂ h₂₃ h₂₄ h₂₅ │
 *   │  0  h₃₂ h₃₃ h₃₄ h₃₅ │
 *   │  0   0  h₄₃ h₄₄ h₄₅ │
 *   └  0   0   0  h₅₄ h₅₅ ┘
 * </pre>
 * <p>
 * For symmetric matrices, Hessenberg form reduces to tridiagonal form.
 * </p>
 *
 * <h2>Algorithm:</h2>
 * <p>
 * The algorithm applies Householder transformations from the left and right:
 * </p>
 * <ol>
 *   <li>For column k = 1 to n-2:</li>
 *   <li>Construct Householder reflector P<sub>k</sub> to zero elements below a[k+1,k]</li>
 *   <li>Apply similarity transformation: A = P<sub>k</sub> A P<sub>k</sub><sup>T</sup></li>
 *   <li>Accumulate Q = Q * P<sub>k</sub></li>
 * </ol>
 * <p>
 * The similarity transformation preserves eigenvalues: H has the same eigenvalues as A.
 * </p>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Hessenberg form only:</b> O(5n<sup>3</sup>/3) flops</li>
 *   <li><b>With accumulation of Q:</b> O(10n<sup>3</sup>/3) flops</li>
 *   <li><b>Space complexity:</b> O(n<sup>2</sup>) for H and Q</li>
 *   <li><b>Comparison:</b> Much cheaper than full QR iteration (O(n<sup>3</sup>) vs O(n<sup>4</sup>) for eigenvalues)</li>
 * </ul>
 *
 * <h2>Numerical Properties:</h2>
 * <ul>
 *   <li><b>Stability:</b> Backward stable (uses orthogonal transformations)</li>
 *   <li><b>Eigenvalue preservation:</b> H and A have identical eigenvalues</li>
 *   <li><b>Orthogonality:</b> Q satisfies Q<sup>T</sup>Q = I to machine precision</li>
 *   <li><b>Similarity:</b> A = QHQ<sup>T</sup> exact up to rounding errors</li>
 * </ul>
 *
 * <h2>Why Hessenberg Form?</h2>
 * <ul>
 *   <li><b>QR iteration efficiency:</b> One QR step on Hessenberg costs O(n<sup>2</sup>) vs O(n<sup>3</sup>) for general matrix</li>
 *   <li><b>Structure preservation:</b> QR iteration preserves Hessenberg form</li>
 *   <li><b>Eigenvalue computation:</b> Essential preprocessing for QR algorithm</li>
 *   <li><b>Nearly triangular:</b> Only one extra diagonal compared to triangular form</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>First stage of QR algorithm for eigenvalue computation</li>
 *   <li>Computing matrix exponentials and functions</li>
 *   <li>Solving differential equations (ODE systems)</li>
 *   <li>Control theory and system analysis</li>
 *   <li>Krylov subspace methods</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][] {
 *     {4, 2, 1, 3},
 *     {2, 5, 3, 1},
 *     {1, 3, 6, 2},
 *     {3, 1, 2, 4}
 * });
 *
 * Matrix[] result = HessenbergReduction.decompose(A);
 * Matrix H = result[0];  // Hessenberg form
 * Matrix Q = result[1];  // Orthogonal transformation
 *
 * // Verify: A = Q * H * Q^T
 * Matrix reconstructed = Q.multiply(H).multiply(Q.transpose());
 *
 * // H is upper Hessenberg (zeros below first subdiagonal)
 * for (int i = 0; i < n; i++) {
 *     for (int j = 0; j < i-1; j++) {
 *         assert Math.abs(H.get(i, j)) < 1e-10;
 *     }
 * }
 *
 * // Use H as input to QR iteration for eigenvalues
 * double[] eigenvalues = qrIteration(H);
 * }</pre>
 *
 * <h2>Symmetric Case - Tridiagonal Form:</h2>
 * <p>
 * When A is symmetric, Hessenberg form becomes tridiagonal:
 * </p>
 * <pre>
 *   ┌ d₁ e₁  0  0  0 ┐
 *   │ e₁ d₂ e₂  0  0 │
 *   │  0 e₂ d₃ e₃  0 │
 *   │  0  0 e₃ d₄ e₄ │
 *   └  0  0  0 e₄ d₅ ┘
 * </pre>
 * <p>
 * This offers additional computational advantages and specialized algorithms.
 * </p>
 *
 * <h2>Relationship to Schur Form:</h2>
 * <p>
 * Hessenberg reduction is the first step toward Schur decomposition:
 * </p>
 * <ol>
 *   <li>Reduce A to Hessenberg form H</li>
 *   <li>Apply QR iteration to H to obtain Schur form T</li>
 *   <li>Result: A = Q T Q<sup>T</sup> where T is quasi-triangular</li>
 * </ol>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Uses Householder reflections for numerical stability</li>
 *   <li>Accumulates orthogonal transformations efficiently</li>
 *   <li>Can compute H only (without Q) for faster execution</li>
 *   <li>Handles general nonsymmetric matrices</li>
 *   <li>Preserves sparsity patterns when possible</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.eigen.schur.RealSchurDecomposition
 * @see net.faulj.eigen.qr.ImplicitQRFrancis
 * @see net.faulj.decomposition.qr.HouseholderQR
 */
public class HessenbergReduction {
	private static final double EPS = 1e-12;

	/**
	 * Reduce a square matrix to Hessenberg form.
	 *
	 * @param A matrix to reduce
	 * @return result containing H and the orthogonal Q
	 */
	public static HessenbergResult decompose(Matrix A) {
		if (A == null) {
			throw new IllegalArgumentException("Matrix must not be null");
		}
		if (!A.isReal()) {
			throw new UnsupportedOperationException("Hessenberg reduction requires a real-valued matrix");
		}
		if (!A.isSquare()) {
			throw new ArithmeticException("Matrix must be square to compute Hessenberg form");
		}
		Matrix H = A.copy();
		int n = H.getRowCount();
		if (n <= 2) {
			return new HessenbergResult(A, H, Matrix.Identity(n));
		}

		Matrix Q = Matrix.Identity(n);
		double[] h = H.getRawData();
		double[] q = Q.getRawData();
		double[] v = new double[n];

		for (int k = 0; k < n - 2; k++) {
			int len = n - k - 1;
			int colIndex = k;
			int base = (k + 1) * n + colIndex;
			double x0 = h[base];
			double sigma = 0.0;
			for (int i = 1; i < len; i++) {
				double val = h[(k + 1 + i) * n + colIndex];
				sigma += val * val;
			}
			if (sigma <= EPS) {
				continue;
			}
			double mu = Math.sqrt(x0 * x0 + sigma);
			double beta = -Math.copySign(mu, x0);
			double v0 = x0 - beta;
			double v0sq = v0 * v0;
			if (v0sq <= EPS) {
				continue;
			}
			double tau = 2.0 * v0sq / (sigma + v0sq);

			v[0] = 1.0;
			for (int i = 1; i < len; i++) {
				v[i] = h[(k + 1 + i) * n + colIndex] / v0;
			}

			h[base] = beta;
			for (int i = 1; i < len; i++) {
				h[(k + 1 + i) * n + colIndex] = 0.0;
			}

			applyHouseholderLeft(h, n, k + 1, k + 1, v, len, tau);
			applyHouseholderRight(h, n, 0, k + 1, v, len, tau);
			applyHouseholderRight(q, n, 0, k + 1, v, len, tau);
		}

		return new HessenbergResult(A, H, Q);
	}

	/**
	 * Apply a Householder reflector from the left to a submatrix.
	 *
	 * @param data matrix data in row-major order
	 * @param size matrix dimension (square)
	 * @param startRow first row of the submatrix
	 * @param startCol first column of the submatrix
	 * @param v Householder vector (length len)
	 * @param len reflector length
	 * @param tau Householder scalar
	 */
	private static void applyHouseholderLeft(double[] data, int size, int startRow, int startCol,
											 double[] v, int len, double tau) {
		if (tau == 0.0 || len <= 1) {
			return;
		}
		for (int col = startCol; col < size; col++) {
			int idx = startRow * size + col;
			double dot = data[idx];
			int rowIdx = idx + size;
			for (int i = 1; i < len; i++) {
				dot += v[i] * data[rowIdx];
				rowIdx += size;
			}
			dot *= tau;
			data[idx] -= dot;
			rowIdx = idx + size;
			for (int i = 1; i < len; i++) {
				data[rowIdx] -= dot * v[i];
				rowIdx += size;
			}
		}
	}

	/**
	 * Apply a Householder reflector from the right to a submatrix.
	 *
	 * @param data matrix data in row-major order
	 * @param size matrix dimension (square)
	 * @param startRow first row of the submatrix
	 * @param startCol first column of the submatrix
	 * @param v Householder vector (length len)
	 * @param len reflector length
	 * @param tau Householder scalar
	 */
	private static void applyHouseholderRight(double[] data, int size, int startRow, int startCol,
											  double[] v, int len, double tau) {
		if (tau == 0.0 || len <= 1) {
			return;
		}
		for (int row = startRow; row < size; row++) {
			int idx = row * size + startCol;
			double dot = data[idx];
			for (int j = 1; j < len; j++) {
				dot += data[idx + j] * v[j];
			}
			dot *= tau;
			data[idx] -= dot;
			for (int j = 1; j < len; j++) {
				data[idx + j] -= dot * v[j];
			}
		}
	}
}
