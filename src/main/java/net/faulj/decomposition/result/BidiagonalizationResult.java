package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;

/**
 * Encapsulates the result of bidiagonal decomposition.
 * <p>
 * Represents A = U * B * V^T where U and V are orthogonal and B is bidiagonal.
 * </p>
 */
public class BidiagonalizationResult {
    private final Matrix U;
    private final Matrix B;
    private final Matrix V;
    private final Matrix A;
    public BidiagonalizationResult(Matrix A, Matrix U, Matrix B, Matrix V) {
        this.A = A;
        this.U = U;
        this.B = B;
        this.V = V;
    }

    public Matrix getU() {
        return U;
    }

    public Matrix getB() {
        return B;
    }

    public Matrix getV() {
        return V;
    }

    public Matrix reconstruct() {
        return U.multiply(B).multiply(V.transpose());
    }

    public double residualNorm() {
        return MatrixUtils.normResidual(A, reconstruct(), 1e-10);
    }

    public double residualElement() {
        return MatrixUtils.backwardErrorComponentwise(A, reconstruct(), 1e-10);
    }

    public double[] verifyOrthogonality(Matrix O) {
        Matrix I = Matrix.Identity(O.getRowCount());
        O = O.multiply(O.transpose());
        double n = MatrixUtils.normResidual(I, O, 1e-10);
        double e = MatrixUtils.backwardErrorComponentwise(I, O, 1e-10);
        return new double[]{n, e};
    }
}
