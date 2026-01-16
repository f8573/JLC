package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;

/**
 * Result of QR decomposition: A = QR
 * Q is orthogonal, R is upper triangular.
 */
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

    public Matrix reconstruct() {
        return Q.multiply(R);
    }

    public double getResidualNorm(Matrix A) {
        Matrix QR = reconstruct();
        Matrix diff = A.subtract(QR);
        return diff.frobeniusNorm();
    }
}