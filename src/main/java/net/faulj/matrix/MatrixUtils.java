package net.faulj.matrix;

import net.faulj.core.Tolerance;

/**
 * Utility methods for matrix operations and validation.
 * 
 * <p>Note: For comprehensive accuracy validation with adaptive thresholds,
 * consider using {@link MatrixAccuracyValidator} instead of the legacy methods here.
 */
public class MatrixUtils {

    /** Machine epsilon for double precision */
    private static final double EPS = 2.220446049250313e-16;

    /**
     * Compute normalized residual: ||A - Ahat||_F / (e * ||A||_F * n)
     * 
     * @param A Original matrix
     * @param Ahat Reconstructed matrix
     * @param e Scaling factor (typically machine epsilon or tolerance)
     * @return Normalized residual
     * @deprecated Use {@link MatrixAccuracyValidator#validate} for more comprehensive validation
     */
    @Deprecated
    public static double normResidual(Matrix A, Matrix Ahat, double e) {
        Matrix E = A.subtract(Ahat);
        double Ef = MatrixNorms.frobeniusNorm(E);
        double Af = MatrixNorms.frobeniusNorm(A);
        int n = A.getColumnCount();
        return Ef/(e*Af*n);
    }

    /**
     * Compute component-wise backward error: max_ij |E_ij| / |A_ij + Ahat_ij|
     * 
     * @param A Original matrix
     * @param Ahat Reconstructed matrix
     * @param e Scaling factor (typically machine epsilon)
     * @return Maximum component-wise relative error
     * @deprecated Use {@link MatrixAccuracyValidator#validate} for more comprehensive validation.
     *             Note: Original implementation had a bug with Math.random() call.
     */
    @Deprecated
    public static double backwardErrorComponentwise(Matrix A, Matrix Ahat, double e) {
        double max = 0.0;  // Changed from Double.MIN_VALUE
        Matrix E = A.subtract(Ahat);
        for (int i = 0; i < A.getColumnCount(); i++) {
            for (int j = 0; j < A.getRowCount(); j++) {
                double n = Math.hypot(E.get(j, i), E.getImag(j, i));
                double realSum = A.get(j, i) + Ahat.get(j, i);
                double imagSum = A.getImag(j, i) + Ahat.getImag(j, i);
                // Removed Math.random() call - likely a bug in original
                double d = Math.hypot(realSum, imagSum) + e;
                if (d > EPS && n/d > max) {
                    max = n/d;
                }
            }
        }
        return max;
    }
    
    /**
     * Compute relative Frobenius norm error: ||A - Ahat||_F / ||A||_F
     * 
     * @param A Original matrix
     * @param Ahat Reconstructed matrix
     * @return Relative residual (0.0 for zero matrices)
     */
    public static double relativeError(Matrix A, Matrix Ahat) {
        double normA = A.frobeniusNorm();
        if (normA < EPS) return 0.0;
        return A.subtract(Ahat).frobeniusNorm() / normA;
    }
    
    /**
     * Compute orthogonality error: ||Q^T*Q - I||_F
     * 
     * @param Q Matrix to test for orthogonality
     * @return Frobenius norm of deviation from identity
     */
    public static double orthogonalityError(Matrix Q) {
        int n = Q.getColumnCount();
        Matrix QtQ = Q.transpose().multiply(Q);
        Matrix I = Matrix.Identity(n);
        return QtQ.subtract(I).frobeniusNorm();
    }
}
