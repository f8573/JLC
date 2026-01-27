package net.faulj.svd;

/**
 * Performs the SVD step of reducing a real bidiagonal matrix to diagonal form.
 * <p>
 * This algorithm is the core iterative phase of the SVD computation. It processes a
 * bidiagonal matrix B (obtained from the initial Golub-Kahan reduction) and drives
 * the superdiagonal elements to zero.
 * </p>
 *
 * <h2>Algorithm:</h2>
 * <p>
 * The class implements the Golub-Kahan SVD step (Implicit QR iteration with Wilkinson shift):
 * </p>
 * <ol>
 * <li>Determine a shift λ based on the bottom 2×2 submatrix (Wilkinson shift)</li>
 * <li>Apply Givens rotations to "chase the bulge" created by the shift down the diagonal</li>
 * <li>Update singular vectors U and V by accumulating rotations</li>
 * <li>Repeat until the superdiagonal element B[n-2, n-1] is negligible</li>
 * <li>Deflate the matrix and repeat for the upper (n-1)×(n-1) block</li>
 * </ol>
 *
 * <h2>Convergence:</h2>
 * <ul>
 * <li><b>Rate:</b> Cubic convergence (usually 2-3 iterations per singular value)</li>
 * <li><b>Stability:</b> Guaranteed convergence for real matrices</li>
 * <li><b>Precision:</b> Computes singular values to full machine precision</li>
 * </ul>
 *
 * <h2>Input Form:</h2>
 * <pre>
 * B =
 * ┌ d₀  e₀  0   0 ┐
 * │ 0   d₁  e₁  0 │
 * │ 0   0   d₂  e₂│
 * └ 0   0   0   d₃┘
 * </pre>
 *
 * <h2>Internal Usage:</h2>
 * <p>
 * This is a low-level utility class used by {@link SVDecomposition}. It operates directly
 * on arrays for performance reasons.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see SVDecomposition
 * @see net.faulj.decomposition.bidiagonal.Bidiagonalization
 */
public class BidiagonalQR {
	/**
	 * Create a bidiagonal QR helper.
	 */
	public BidiagonalQR() {
	}

	private static final double EPS = 2.220446049250313e-16;

	/**
	 * Computes the SVD of a bidiagonal matrix using implicit QR iterations.
	 *
	 * @param B bidiagonal matrix
	 * @return SVD of B
	 */
	public static BidiagonalSVDResult decompose(net.faulj.matrix.Matrix B) {
		if (B == null) {
			throw new IllegalArgumentException("Matrix must not be null");
		}
		int m = B.getRowCount();
		int n = B.getColumnCount();
		if (m == 0 || n == 0) {
			return new BidiagonalSVDResult(
					net.faulj.matrix.Matrix.Identity(m),
					net.faulj.matrix.Matrix.Identity(n),
					new double[0]
			);
		}
		if (m < n) {
			BidiagonalSVDResult t = decomposeUpper(B.transpose());
			return new BidiagonalSVDResult(t.V, t.U, t.singularValues);
		}
		return decomposeUpper(B);
	}

	/**
	 * Compute SVD for an upper bidiagonal matrix.
	 *
	 * @param B bidiagonal matrix
	 * @return bidiagonal SVD result
	 */
	private static BidiagonalSVDResult decomposeUpper(net.faulj.matrix.Matrix B) {
		int m = B.getRowCount();
		int n = B.getColumnCount();
		int p = Math.min(m, n);
		double[] s = new double[p];
		double[] e = new double[p];
		for (int i = 0; i < p; i++) {
			s[i] = B.get(i, i);
			if (i < p - 1) {
				e[i] = B.get(i, i + 1);
			}
		}
		e[p - 1] = 0.0;

		net.faulj.matrix.Matrix U = net.faulj.matrix.Matrix.Identity(m);
		net.faulj.matrix.Matrix V = net.faulj.matrix.Matrix.Identity(n);

		int maxIter = Math.max(1, 1000 * p);
		int iter = 0;
		int pp = p;
		while (pp > 0) {
			if (iter++ > maxIter) {
				break;
			}
			int k;
			int kase;
			for (k = pp - 2; k >= -1; k--) {
				if (k == -1) {
					break;
				}
				double threshold = EPS * (Math.abs(s[k]) + Math.abs(s[k + 1]));
				if (Math.abs(e[k]) <= threshold) {
					e[k] = 0.0;
					break;
				}
			}
			if (k == pp - 2) {
				kase = 4;
			} else {
				int ks;
				for (ks = pp - 1; ks >= k; ks--) {
					if (ks == k) {
						break;
					}
					double t = (ks != pp ? Math.abs(e[ks]) : 0.0) +
							(ks != k + 1 ? Math.abs(e[ks - 1]) : 0.0);
					if (Math.abs(s[ks]) <= EPS * t) {
						s[ks] = 0.0;
						break;
					}
				}
				if (ks == k) {
					kase = 3;
				} else if (ks == pp - 1) {
					kase = 1;
				} else {
					kase = 2;
					k = ks;
				}
			}
			k++;

			switch (kase) {
				case 1 -> {
					double f = e[pp - 2];
					e[pp - 2] = 0.0;
					for (int j = pp - 2; j >= k; j--) {
						double t = hypot(s[j], f);
						double cs = s[j] / t;
						double sn = f / t;
						s[j] = t;
						if (j != k) {
							f = -sn * e[j - 1];
							e[j - 1] = cs * e[j - 1];
						}
						applyRotationToColumns(V, j, pp - 1, cs, sn);
					}
				}
				case 2 -> {
					double f = e[k - 1];
					e[k - 1] = 0.0;
					for (int j = k; j < pp; j++) {
						double t = hypot(s[j], f);
						double cs = s[j] / t;
						double sn = f / t;
						s[j] = t;
						f = -sn * e[j];
						e[j] = cs * e[j];
						applyRotationToColumns(U, j, k - 1, cs, sn);
					}
				}
				case 3 -> {
					double scale = 0.0;
					scale = Math.max(scale, Math.abs(s[pp - 1]));
					scale = Math.max(scale, Math.abs(s[pp - 2]));
					scale = Math.max(scale, Math.abs(e[pp - 2]));
					scale = Math.max(scale, Math.abs(s[k]));
					scale = Math.max(scale, Math.abs(e[k]));
					if (scale == 0.0) {
						scale = 1.0;
					}
					double sp = s[pp - 1] / scale;
					double spm1 = s[pp - 2] / scale;
					double epm1 = e[pp - 2] / scale;
					double sk = s[k] / scale;
					double ek = e[k] / scale;
					double b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
					double c = (sp * epm1) * (sp * epm1);
					double shift = 0.0;
					if (b != 0.0 || c != 0.0) {
						shift = Math.sqrt(b * b + c);
						if (b < 0.0) {
							shift = -shift;
						}
						shift = c / (b + shift);
					}
					double f = (sk + sp) * (sk - sp) + shift;
					double g = sk * ek;
					for (int j = k; j < pp - 1; j++) {
						double t = hypot(f, g);
						double cs = f / t;
						double sn = g / t;
						if (j != k) {
							e[j - 1] = t;
						}
						f = cs * s[j] + sn * e[j];
						e[j] = cs * e[j] - sn * s[j];
						g = sn * s[j + 1];
						s[j + 1] = cs * s[j + 1];
						applyRotationToColumns(V, j, j + 1, cs, sn);

						t = hypot(f, g);
						cs = f / t;
						sn = g / t;
						s[j] = t;
						f = cs * e[j] + sn * s[j + 1];
						s[j + 1] = -sn * e[j] + cs * s[j + 1];
						g = sn * e[j + 1];
						e[j + 1] = cs * e[j + 1];
						applyRotationToColumns(U, j, j + 1, cs, sn);
					}
					e[pp - 2] = f;
				}
				case 4 -> {
					if (s[k] <= 0.0) {
						s[k] = s[k] < 0.0 ? -s[k] : 0.0;
						for (int i = 0; i < n; i++) {
							V.set(i, k, -V.get(i, k));
						}
					}
					while (k < pp - 1) {
						if (s[k] >= s[k + 1]) {
							break;
						}
						double t = s[k];
						s[k] = s[k + 1];
						s[k + 1] = t;
						swapColumns(V, k, k + 1);
						swapColumns(U, k, k + 1);
						k++;
					}
					pp--;
				}
				default -> {
				}
			}
		}

		return new BidiagonalSVDResult(U, V, s);
	}

	/**
	 * Compute hypot(a, b) safely.
	 *
	 * @param a first value
	 * @param b second value
	 * @return sqrt(a^2 + b^2)
	 */
	private static double hypot(double a, double b) {
		return Math.hypot(a, b);
	}

	/**
	 * Apply a Givens rotation to two columns.
	 *
	 * @param M matrix to update
	 * @param col1 first column index
	 * @param col2 second column index
	 * @param c cosine term
	 * @param s sine term
	 */
	private static void applyRotationToColumns(net.faulj.matrix.Matrix M, int col1, int col2, double c, double s) {
		int rows = M.getRowCount();
		for (int i = 0; i < rows; i++) {
			double x = M.get(i, col1);
			double y = M.get(i, col2);
			M.set(i, col1, c * x + s * y);
			M.set(i, col2, -s * x + c * y);
		}
	}

	/**
	 * Swap two columns in-place.
	 *
	 * @param M matrix to update
	 * @param col1 first column index
	 * @param col2 second column index
	 */
	private static void swapColumns(net.faulj.matrix.Matrix M, int col1, int col2) {
		int rows = M.getRowCount();
		for (int i = 0; i < rows; i++) {
			double tmp = M.get(i, col1);
			M.set(i, col1, M.get(i, col2));
			M.set(i, col2, tmp);
		}
	}

	public static final class BidiagonalSVDResult {
		private final net.faulj.matrix.Matrix U;
		private final net.faulj.matrix.Matrix V;
		private final double[] singularValues;

		/**
		 * Create a bidiagonal SVD result container.
		 *
		 * @param U left singular vectors
		 * @param V right singular vectors
		 * @param singularValues singular values
		 */
		private BidiagonalSVDResult(net.faulj.matrix.Matrix U, net.faulj.matrix.Matrix V, double[] singularValues) {
			this.U = U;
			this.V = V;
			this.singularValues = singularValues;
		}

		/**
		 * @return left singular vectors
		 */
		public net.faulj.matrix.Matrix getU() {
			return U;
		}

		/**
		 * @return right singular vectors
		 */
		public net.faulj.matrix.Matrix getV() {
			return V;
		}

		/**
		 * @return singular values
		 */
		public double[] getSingularValues() {
			return singularValues;
		}
	}
}
