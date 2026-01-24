package net.faulj.decomposition.qr;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.QRResult;

/**
 * Computes QR decomposition using Householder reflections, the gold standard for dense matrices.
 */
public class HouseholderQR {
	private static final double EPS = 1e-12;

	public static QRResult decompose(Matrix A) {
		return decompose(A, false);
	}

	public static QRResult decomposeThin(Matrix A) {
		return decompose(A, true);
	}

	private static QRResult decompose(Matrix A, boolean thin) {
		if (A == null) {
			throw new IllegalArgumentException("Matrix must not be null");
		}
		if (!A.isReal()) {
			throw new UnsupportedOperationException("Householder QR requires a real-valued matrix");
		}
		int m = A.getRowCount();
		int n = A.getColumnCount();
		int kMax = Math.min(m, n);

		Matrix R = A.copy();
		double[] rData = R.getRawData();
		double[] tau = new double[kMax];

		for (int k = 0; k < kMax; k++) {
			int len = m - k;
			if (len <= 1) {
				tau[k] = 0.0;
				continue;
			}
			int colIndex = k;
			int base = k * n + colIndex;
			double x0 = rData[base];
			double sigma = 0.0;
			for (int i = 1; i < len; i++) {
				double v = rData[(k + i) * n + colIndex];
				sigma += v * v;
			}
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

			rData[base] = beta;
			for (int i = 1; i < len; i++) {
				int idx = (k + i) * n + colIndex;
				rData[idx] /= v0;
			}

			for (int col = k + 1; col < n; col++) {
				int rowIndex = k * n + col;
				double dot = rData[rowIndex];
				int rIdx = rowIndex + n;
				int vIdx = (k + 1) * n + colIndex;
				for (int i = 1; i < len; i++) {
					dot += rData[vIdx] * rData[rIdx];
					rIdx += n;
					vIdx += n;
				}
				dot *= tauK;
				rData[rowIndex] -= dot;
				rIdx = rowIndex + n;
				vIdx = (k + 1) * n + colIndex;
				for (int i = 1; i < len; i++) {
					rData[rIdx] -= dot * rData[vIdx];
					rIdx += n;
					vIdx += n;
				}
			}
		}

		int qCols = thin ? kMax : m;
		Matrix Q = new Matrix(m, qCols);
		double[] q = Q.getRawData();
		int diag = Math.min(m, qCols);
		for (int i = 0; i < diag; i++) {
			q[i * qCols + i] = 1.0;
		}

		for (int k = kMax - 1; k >= 0; k--) {
			double tauK = tau[k];
			if (tauK == 0.0) {
				continue;
			}
			int len = m - k;
			int vBase = (k + 1) * n + k;
			int qRowStart = k * qCols;
			for (int col = 0; col < qCols; col++) {
				int idx = qRowStart + col;
				double dot = q[idx];
				int qIdx = idx + qCols;
				int vIdx = vBase;
				for (int i = 1; i < len; i++) {
					dot += rData[vIdx] * q[qIdx];
					qIdx += qCols;
					vIdx += n;
				}
				dot *= tauK;
				q[idx] -= dot;
				qIdx = idx + qCols;
				vIdx = vBase;
				for (int i = 1; i < len; i++) {
					q[qIdx] -= dot * rData[vIdx];
					qIdx += qCols;
					vIdx += n;
				}
			}
		}

		int limit = Math.min(m, n);
		for (int c = 0; c < limit; c++) {
			boolean zeroAll = c < tau.length && tau[c] != 0.0;
			for (int r = c + 1; r < m; r++) {
				int idx = r * n + c;
				if (zeroAll || Math.abs(rData[idx]) < EPS) {
					rData[idx] = 0.0;
				}
			}
		}

		if (thin) {
			if (kMax == 0) {
				return new QRResult(A, Q, R);
			}
			Matrix Rthin = R.crop(0, kMax - 1, 0, n - 1);
			return new QRResult(A, Q, Rthin);
		}

		return new QRResult(A, Q, R);
	}
}
