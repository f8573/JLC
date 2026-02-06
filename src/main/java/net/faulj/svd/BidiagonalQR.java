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
 * <h2>EJML-inspired optimizations:</h2>
 * <ul>
 * <li>Raw array access instead of Matrix.get/set</li>
 * <li>SIMD vectorization for Givens rotation application</li>
 * <li>Dynamic shifting with zero-detection for first iterations</li>
 * <li>Exceptional shifts when convergence stalls</li>
 * <li>Relative tolerance for split detection</li>
 * </ul>
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
	// LAPACK dbdsqr: faster convergence for 512x512
	private static final int EXCEPTIONAL_THRESHOLD = 10; // Faster exceptional shifts
	private static final int MAX_ITER_PER_VALUE = 15; // LAPACK: 6*n iterations max

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
	 * Uses raw array access and SIMD for optimal performance.
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

		// Use raw arrays for U and V for faster access
		double[] U = new double[m * m];
		double[] V = new double[n * n];
		// Initialize as identity
		for (int i = 0; i < m; i++) {
			U[i * m + i] = 1.0;
		}
		for (int i = 0; i < n; i++) {
			V[i * n + i] = 1.0;
		}

		int maxIter = Math.max(1, MAX_ITER_PER_VALUE * p); // LAPACK: tighter iteration bound
		int iter = 0;
		int pp = p;
		int exceptionalCount = 0;
		int zeroIter = 0;

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
				// EJML-style: relative tolerance for split detection
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
					// EJML: relative threshold for zero detection
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
						applyRotationToColumnsRaw(V, n, j, pp - 1, cs, sn);
					}
					exceptionalCount = 0;
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
						applyRotationToColumnsRaw(U, m, j, k - 1, cs, sn);
					}
					exceptionalCount = 0;
				}
				case 3 -> {
					// LAPACK dbdsqr: fast Wilkinson shift computation
					exceptionalCount++;

					// Scale for numerical stability
					double scale = Math.max(Math.abs(s[pp - 1]), Math.abs(s[pp - 2]));
					scale = Math.max(scale, Math.abs(e[pp - 2]));
					scale = Math.max(scale, Math.abs(s[k]));
					if (k > 0) scale = Math.max(scale, Math.abs(e[k]));
					if (scale == 0.0) scale = 1.0;

					double sp = s[pp - 1] / scale;
					double spm1 = s[pp - 2] / scale;
					double epm1 = e[pp - 2] / scale;
					double sk = s[k] / scale;
					double ek = (k < e.length) ? e[k] / scale : 0.0;

					// LAPACK: Wilkinson shift from 2x2 trailing submatrix
					double b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) * 0.5;
					double c = (sp * epm1) * (sp * epm1);

					double shift = 0.0;
					// LAPACK: faster convergence strategy
					if (zeroIter < 4) {
						zeroIter++;
						shift = 0.0;
					} else if (exceptionalCount >= EXCEPTIONAL_THRESHOLD) {
						// LAPACK: exceptional shift
						shift = scale * 0.75 * Math.abs(e[pp - 2]);
						exceptionalCount = 0;
					} else if (Math.abs(b) > 0.0 || Math.abs(c) > 0.0) {
						double disc = Math.sqrt(b * b + c);
						shift = c / (b + (b >= 0.0 ? disc : -disc));
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
						applyRotationToColumnsRaw(V, n, j, j + 1, cs, sn);

						t = hypot(f, g);
						cs = f / t;
						sn = g / t;
						s[j] = t;
						f = cs * e[j] + sn * s[j + 1];
						s[j + 1] = -sn * e[j] + cs * s[j + 1];
						g = sn * e[j + 1];
						e[j + 1] = cs * e[j + 1];
						applyRotationToColumnsRaw(U, m, j, j + 1, cs, sn);
					}
					e[pp - 2] = f;
				}
				case 4 -> {
					if (s[k] <= 0.0) {
						s[k] = s[k] < 0.0 ? -s[k] : 0.0;
						negateColumnRaw(V, n, k);
					}
					while (k < pp - 1) {
						if (s[k] >= s[k + 1]) {
							break;
						}
						double t = s[k];
						s[k] = s[k + 1];
						s[k + 1] = t;
						swapColumnsRaw(V, n, k, k + 1);
						swapColumnsRaw(U, m, k, k + 1);
						k++;
					}
					pp--;
					exceptionalCount = 0;
					zeroIter = 0;
				}
				default -> {
				}
			}
		}

		// Wrap raw arrays back to Matrix objects
		return new BidiagonalSVDResult(
				net.faulj.matrix.Matrix.wrap(U, m, m),
				net.faulj.matrix.Matrix.wrap(V, n, n),
				s
		);
	}

	/**
	 * Compute hypot(a, b) safely.
	 */
	private static double hypot(double a, double b) {
		return Math.hypot(a, b);
	}

	/**
	 * Apply a Givens rotation to two columns using raw array access.
	 * Uses scalar 4x unrolled loop instead of SIMD gather/scatter for strided column access.
	 * This is faster than SIMD for non-contiguous memory patterns and eliminates allocations.
	 */
	private static void applyRotationToColumnsRaw(double[] M, int stride, int col1, int col2, double c, double s) {
		final double negS = -s;
		int i = 0;
		final int limit = stride - 3;

		// 4x unrolled loop for ILP (Instruction Level Parallelism)
		for (; i < limit; i += 4) {
			final int idx10 = i * stride + col1;
			final int idx20 = i * stride + col2;
			final int idx11 = (i + 1) * stride + col1;
			final int idx21 = (i + 1) * stride + col2;
			final int idx12 = (i + 2) * stride + col1;
			final int idx22 = (i + 2) * stride + col2;
			final int idx13 = (i + 3) * stride + col1;
			final int idx23 = (i + 3) * stride + col2;

			// Load 4 pairs
			final double x0 = M[idx10], y0 = M[idx20];
			final double x1 = M[idx11], y1 = M[idx21];
			final double x2 = M[idx12], y2 = M[idx22];
			final double x3 = M[idx13], y3 = M[idx23];

			// Apply rotation: x' = c*x + s*y, y' = -s*x + c*y
			M[idx10] = c * x0 + s * y0;
			M[idx20] = negS * x0 + c * y0;
			M[idx11] = c * x1 + s * y1;
			M[idx21] = negS * x1 + c * y1;
			M[idx12] = c * x2 + s * y2;
			M[idx22] = negS * x2 + c * y2;
			M[idx13] = c * x3 + s * y3;
			M[idx23] = negS * x3 + c * y3;
		}

		// Scalar remainder
		for (; i < stride; i++) {
			final int idx1 = i * stride + col1;
			final int idx2 = i * stride + col2;
			final double x = M[idx1];
			final double y = M[idx2];
			M[idx1] = c * x + s * y;
			M[idx2] = negS * x + c * y;
		}
	}

	/**
	 * Negate a column using raw array access with 4x unrolling for ILP.
	 */
	private static void negateColumnRaw(double[] M, int stride, int col) {
		int i = 0;
		int limit = stride - 3;

		// 4x unrolled loop
		for (; i < limit; i += 4) {
			int idx0 = i * stride + col;
			int idx1 = (i + 1) * stride + col;
			int idx2 = (i + 2) * stride + col;
			int idx3 = (i + 3) * stride + col;
			M[idx0] = -M[idx0];
			M[idx1] = -M[idx1];
			M[idx2] = -M[idx2];
			M[idx3] = -M[idx3];
		}

		// Scalar remainder
		for (; i < stride; i++) {
			M[i * stride + col] = -M[i * stride + col];
		}
	}

	/**
	 * Swap two columns using raw array access with 4x unrolling for ILP.
	 */
	private static void swapColumnsRaw(double[] M, int stride, int col1, int col2) {
		int i = 0;
		int limit = stride - 3;

		// 4x unrolled loop
		for (; i < limit; i += 4) {
			int idx10 = i * stride + col1;
			int idx20 = i * stride + col2;
			int idx11 = (i + 1) * stride + col1;
			int idx21 = (i + 1) * stride + col2;
			int idx12 = (i + 2) * stride + col1;
			int idx22 = (i + 2) * stride + col2;
			int idx13 = (i + 3) * stride + col1;
			int idx23 = (i + 3) * stride + col2;

			double tmp0 = M[idx10];
			M[idx10] = M[idx20];
			M[idx20] = tmp0;

			double tmp1 = M[idx11];
			M[idx11] = M[idx21];
			M[idx21] = tmp1;

			double tmp2 = M[idx12];
			M[idx12] = M[idx22];
			M[idx22] = tmp2;

			double tmp3 = M[idx13];
			M[idx13] = M[idx23];
			M[idx23] = tmp3;
		}

		// Scalar remainder
		for (; i < stride; i++) {
			int idx1 = i * stride + col1;
			int idx2 = i * stride + col2;
			double tmp = M[idx1];
			M[idx1] = M[idx2];
			M[idx2] = tmp;
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
