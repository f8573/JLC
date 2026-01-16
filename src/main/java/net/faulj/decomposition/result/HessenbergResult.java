package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;

/**
 * Result of Hessenberg reduction: A = Q H Q^T
 * H is upper Hessenberg, Q is orthogonal (similarity transformation).
 */
public class HessenbergResult {
    private final Matrix H;
    private final Matrix Q;

    public HessenbergResult(Matrix H, Matrix Q) {
        this.H = H;
        this.Q = Q;
    }

    public Matrix getH() {
        return H;
    }

    public Matrix getQ() {
        return Q;
    }

    /**
     * Reconstructs A via A = Q H Q^T
     */
    public Matrix reconstruct() {
        return Q.multiply(H).multiply(Q.transpose());
    }

    public double getResidualNorm(Matrix A) {
        Matrix recon = reconstruct();
        return A.subtract(recon).frobeniusNorm();
    }
}