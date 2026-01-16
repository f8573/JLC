package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;

public class QRResult {
	private final Matrix Q;
	private final Matrix R;

	public QRResult(Matrix Q, Matrix R) {
		this.Q = Q;
		this.R = R;
	}

	public Matrix getQ() {
		return Q;
	}

	public Matrix getR() {
		return R;
	}
}
