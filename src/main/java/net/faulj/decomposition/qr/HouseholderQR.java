package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import jdk.incubator.vector.*;

/**
 * Optimized Householder QR decomposition using vector API for BLAS3-like operations.
 * Maintains correctness of original algorithm while providing speedup through vectorization.
 */
public final class HouseholderQR {
	private static final double EPS = 1e-12;
	private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
	private static final int LANE_SIZE = SPECIES.length();

	private HouseholderQR() {}

	public static QRResult decompose(Matrix A) { return decompose(A, false); }
	public static QRResult decomposeThin(Matrix A) { return decompose(A, true); }

	private static QRResult decompose(Matrix A, boolean thin) {
		if (A == null) throw new IllegalArgumentException("Matrix must not be null");
		if (!A.isReal()) throw new UnsupportedOperationException("Householder QR requires real matrix");

		final int m = A.getRowCount();
		final int n = A.getColumnCount();
		final int kMax = Math.min(m, n);

		// Work on a copy
		Matrix R = A.copy();
		double[] a = R.getRawData();
		double[] tau = new double[kMax];

		// Factorization with vectorized column updates
		for (int k = 0; k < kMax; k++) {
			int len = m - k;
			if (len <= 1) { tau[k] = 0.0; continue; }

			int diag = k * n + k;
			double x0 = a[diag];

			// Compute sigma = ||x[1:]||^2 using vector API
			double sigma = computeSigmaVectorized(a, m, n, k);

			if (sigma <= EPS) {
				tau[k] = 0.0;
				continue;
			}

			double mu = Math.sqrt(x0 * x0 + sigma);
			double beta = -Math.copySign(mu, x0);
			double v0 = x0 - beta;
			double v0sq = v0 * v0;

			if (v0sq <= EPS) {
				tau[k] = 0.0;
				continue;
			}

			double tauK = 2.0 * v0sq / (sigma + v0sq);
			tau[k] = tauK;

			// Store R(k,k) = beta
			a[diag] = beta;

			// Store Householder vector below diagonal
			for (int r = k + 1; r < m; r++) {
				a[r * n + k] /= v0;
			}

			// Apply reflector to trailing matrix with vectorized updates
			applyReflectorVectorized(a, tauK, m, n, k);
		}

		// Build Q
		final int qCols = thin ? kMax : m;
		Matrix Q = new Matrix(m, qCols);
		double[] q = Q.getRawData();
		setIdentity(q, m, qCols);

		// Apply reflectors in reverse to build Q
		buildQVectorized(a, tau, q, m, n, qCols, kMax);

		// Zero below diagonal to make R explicit
		zeroBelowDiagonal(a, m, n);

		if (thin) {
			if (kMax == 0) return new QRResult(A, Q, R);
			Matrix Rthin = R.crop(0, kMax - 1, 0, n - 1);
			return new QRResult(A, Q, Rthin);
		}
		return new QRResult(A, Q, R);
	}

	/**
	 * Vectorized computation of sigma = ||x[1:]||^2 for column k
	 */
	private static double computeSigmaVectorized(double[] a, int m, int n, int k) {
		int startRow = k + 1;
		int colStart = k * n + k + 1;
		double sigma = 0.0;

		// Vectorized accumulation
		int upperBound = SPECIES.loopBound(m - startRow);
		int i = startRow;
		for (; i < upperBound + startRow; i += LANE_SIZE) {
			DoubleVector vec = DoubleVector.fromArray(SPECIES, a, i * n + k);
			DoubleVector squared = vec.mul(vec);
			sigma += squared.reduceLanes(VectorOperators.ADD);
		}

		// Process remaining elements
		for (; i < m; i++) {
			double v = a[i * n + k];
			sigma += v * v;
		}

		return sigma;
	}

	/**
	 * Vectorized application of Householder reflector to trailing matrix
	 */
	private static void applyReflectorVectorized(double[] a, double tauK, int m, int n, int k) {
		int startCol = k + 1;

		// Process columns in blocks for better cache utilization
		for (int colBlock = startCol; colBlock < n; colBlock += LANE_SIZE * 4) {
			int colEnd = Math.min(colBlock + LANE_SIZE * 4, n);

			for (int col = colBlock; col < colEnd; col++) {
				// Compute dot = v^T * AcolSegment
				double dot = a[k * n + col];

				int row = k + 1;
				int upperBound = SPECIES.loopBound(m - row);

				// Vectorized dot product accumulation
				DoubleVector dotVec = DoubleVector.zero(SPECIES);
				for (; row < upperBound + k + 1; row += LANE_SIZE) {
					DoubleVector vVec = DoubleVector.fromArray(SPECIES, a, row * n + k);
					DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, row * n + col);
					dotVec = vVec.fma(aVec, dotVec);
				}
				dot += dotVec.reduceLanes(VectorOperators.ADD);

				// Process remaining rows
				for (; row < m; row++) {
					dot += a[row * n + k] * a[row * n + col];
				}

				dot *= tauK;

				// Apply update: AcolSegment -= dot * v
				a[k * n + col] -= dot;

				row = k + 1;
				upperBound = SPECIES.loopBound(m - row);

				// Vectorized update of rows
				DoubleVector dotVecConst = DoubleVector.broadcast(SPECIES, dot);
				for (; row < upperBound + k + 1; row += LANE_SIZE) {
					int idx = row * n;
					DoubleVector vVec = DoubleVector.fromArray(SPECIES, a, idx + k);
					DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, idx + col);
					DoubleVector updated = aVec.sub(vVec.mul(dotVecConst));
					updated.intoArray(a, idx + col);
				}

				// Process remaining rows
				for (; row < m; row++) {
					a[row * n + col] -= dot * a[row * n + k];
				}
			}
		}
	}

	/**
	 * Vectorized construction of Q matrix
	 */
	private static void buildQVectorized(double[] a, double[] tau, double[] q,
										 int m, int n, int qCols, int kMax) {
		// Apply reflectors in reverse order
		for (int k = kMax - 1; k >= 0; k--) {
			double tauK = tau[k];
			if (tauK == 0.0) continue;

			int len = m - k;
			int vBase = (k + 1) * n + k;
			int qRowStart = k * qCols;

			// Process Q columns in blocks
			for (int colBlock = 0; colBlock < qCols; colBlock += LANE_SIZE * 4) {
				int colEnd = Math.min(colBlock + LANE_SIZE * 4, qCols);

				for (int col = colBlock; col < colEnd; col++) {
					int idx = qRowStart + col;

					// Compute dot = v^T * Q segment
					double dot = q[idx];

					int qIdx = idx + qCols;
					int vIdx = vBase;
					int i = 1;

					// Vectorized dot product
					int upperBound = SPECIES.loopBound(len - 1);
					DoubleVector dotVec = DoubleVector.zero(SPECIES);
					for (; i < upperBound + 1; i += LANE_SIZE) {
						DoubleVector vVec = DoubleVector.fromArray(SPECIES, a, vIdx);
						DoubleVector qVec = DoubleVector.fromArray(SPECIES, q, qIdx);
						dotVec = vVec.fma(qVec, dotVec);
						qIdx += qCols * LANE_SIZE;
						vIdx += n * LANE_SIZE;
					}
					dot += dotVec.reduceLanes(VectorOperators.ADD);

					// Process remaining elements
					for (; i < len; i++) {
						dot += a[vIdx] * q[qIdx];
						qIdx += qCols;
						vIdx += n;
					}

					dot *= tauK;

					// Apply update: Q segment -= dot * v
					q[idx] -= dot;

					qIdx = idx + qCols;
					vIdx = vBase;
					i = 1;

					// Vectorized update
					DoubleVector dotVecConst = DoubleVector.broadcast(SPECIES, dot);
					upperBound = SPECIES.loopBound(len - 1);
					for (; i < upperBound + 1; i += LANE_SIZE) {
						DoubleVector vVec = DoubleVector.fromArray(SPECIES, a, vIdx);
						DoubleVector qVec = DoubleVector.fromArray(SPECIES, q, qIdx);
						DoubleVector updated = qVec.sub(vVec.mul(dotVecConst));
						updated.intoArray(q, qIdx);
						qIdx += qCols * LANE_SIZE;
						vIdx += n * LANE_SIZE;
					}

					// Process remaining elements
					for (; i < len; i++) {
						q[qIdx] -= dot * a[vIdx];
						qIdx += qCols;
						vIdx += n;
					}
				}
			}
		}
	}

	private static void setIdentity(double[] q, int m, int n) {
		// Vectorized identity initialization
		int total = m * n;
		int i = 0;
		int upperBound = SPECIES.loopBound(total);

		DoubleVector zeroVec = DoubleVector.zero(SPECIES);
		for (; i < upperBound; i += LANE_SIZE) {
			zeroVec.intoArray(q, i);
		}

		// Process remaining elements
		for (; i < total; i++) {
			q[i] = 0.0;
		}

		// Set diagonal elements
		int d = Math.min(m, n);
		for (i = 0; i < d; i++) {
			q[i * n + i] = 1.0;
		}
	}

	private static void zeroBelowDiagonal(double[] a, int m, int n) {
		int limit = Math.min(m, n);

		// Vectorized zeroing of below-diagonal elements
		DoubleVector zeroVec = DoubleVector.zero(SPECIES);
		for (int c = 0; c < limit; c++) {
			int startRow = c + 1;
			int i = startRow;
			int upperBound = SPECIES.loopBound(m - startRow);

			// Vectorized store of zeros
			for (; i < upperBound + startRow; i += LANE_SIZE) {
				zeroVec.intoArray(a, i * n + c);
			}

			// Process remaining rows
			for (; i < m; i++) {
				a[i * n + c] = 0.0;
			}
		}
	}
}