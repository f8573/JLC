package net.faulj.decomposition.qr;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;
import net.faulj.decomposition.result.QRResult;

/**
 * Computes QR decomposition using Householder reflections, the gold standard for dense matrices.
 */
public class HouseholderQR {
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
		Matrix Q = Matrix.Identity(m);

		for (int k = 0; k < kMax; k++) {
			int len = m - k;
			if (len <= 1) {
				continue;
			}
			double[] x = new double[len];
			for (int i = 0; i < len; i++) {
				x[i] = R.get(k + i, k);
			}

			double normX = 0.0;
			for (double xi : x) normX += xi * xi;
			normX = Math.sqrt(normX);

			boolean tailAllZero = true;
			for (int i = 1; i < len; i++) {
				if (Math.abs(x[i]) > 1e-12) {
					tailAllZero = false;
					break;
				}
			}
			if (tailAllZero && Math.abs(x[0] - normX) < 1e-12) {
				continue;
			}

			if (normX <= 1e-10) {
				continue;
			}

			Vector xVec = new net.faulj.vector.Vector(x);
			Vector hh = VectorUtils.householder(xVec);
			double tau = hh.get(hh.dimension() - 1);
			Vector v = hh.resize(hh.dimension() - 1);

			Matrix H = Matrix.Identity(m);
			for (int i = 0; i < len; i++) {
				for (int j = 0; j < len; j++) {
					double val = (i == j ? 1.0 : 0.0) - tau * v.get(i) * v.get(j);
					H.set(k + i, k + j, val);
				}
			}

			R = H.multiply(R);
			for (int i = 1; i < len; i++) {
				R.set(k + i, k, 0.0);
			}
			Q = Q.multiply(H);
		}

		for (int c = 0; c < Math.min(m, n); c++) {
			for (int r = c + 1; r < m; r++) {
				double val = R.get(r, c);
				if (Math.abs(val) < 1e-12) {
					R.set(r, c, 0.0);
				}
			}
		}

		if (thin) {
			if (kMax == 0) {
				return new QRResult(A, Q, R);
			}
			Matrix Qthin = Q.crop(0, m - 1, 0, kMax - 1);
			Matrix Rthin = R.crop(0, kMax - 1, 0, n - 1);
			return new QRResult(A, Qthin, Rthin);
		}

		return new QRResult(A, Q, R);
	}
}
