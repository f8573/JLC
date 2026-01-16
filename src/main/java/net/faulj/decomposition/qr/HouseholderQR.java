package net.faulj.decomposition.qr;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;
import net.faulj.decomposition.result.QRResult;

public class HouseholderQR {
	public static QRResult decompose(Matrix A) {
		if (!A.isSquare()) {
			throw new ArithmeticException("Matrix must be square to compute QR");
		}
		int n = A.getRowCount();
		Matrix R = A.copy();
		Matrix Q = Matrix.Identity(n);

		for (int k = 0; k < n - 1; k++) {
			int len = n - k;
			double[] x = new double[len];
			for (int i = 0; i < len; i++) {
				x[i] = R.get(k + i, k);
			}

			double normX = 0.0;
			for (double xi : x) normX += xi * xi;
			normX = Math.sqrt(normX);

			if (normX <= 1e-10) {
				continue;
			}

			double sign = x[0] >= 0 ? 1.0 : -1.0;
			double[] u = new double[len];
			u[0] = x[0] + sign * normX;
			for (int i = 1; i < len; i++) u[i] = x[i];

			double uNorm = 0.0;
			for (double ui : u) uNorm += ui * ui;
			uNorm = Math.sqrt(uNorm);
			if (uNorm <= 1e-10) continue;

			Vector hh = VectorUtils.householder(new net.faulj.vector.Vector(u));
			double tau = hh.get(hh.dimension() - 1);
			Vector v = hh.resize(hh.dimension() - 1);

			Matrix H = Matrix.Identity(n);
			for (int i = 0; i < len; i++) {
				for (int j = 0; j < len; j++) {
					double val = (i == j ? 1.0 : 0.0) - tau * v.get(i) * v.get(j);
					H.set(k + i, k + j, val);
				}
			}

			R = H.multiply(R);
			Q = Q.multiply(H);
		}

		return new QRResult(Q, R);
	}
}
