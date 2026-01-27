package net.faulj.matrix;

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
