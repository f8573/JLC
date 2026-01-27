package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;

/**
 * Encapsulates the result of QR decomposition.
 * <p>
 * This class represents the factorization A = QR where:
 * </p>
 * <ul>
 *   <li><b>Q</b> - Orthogonal matrix (Q<sup>T</sup>Q = I) with m rows and m columns</li>
 *   <li><b>R</b> - Upper triangular matrix with m rows and n columns</li>
 *   <li><b>A</b> - Original m×n matrix being factored</li>
 * </ul>
 *
 * <h2>Matrix Properties:</h2>
 * <ul>
 *   <li><b>Orthogonality:</b> Q<sup>T</sup>Q = I (columns of Q are orthonormal)</li>
 *   <li><b>Upper triangular R:</b> R[i,j] = 0 for i &gt; j</li>
 *   <li><b>Norm preservation:</b> ||A||<sub>F</sub> = ||R||<sub>F</sub></li>
 *   <li><b>Rank revealing:</b> Diagonal elements of R indicate numerical rank</li>
 * </ul>
 *
 * <h2>Thin vs Full QR:</h2>
 * <p>
 * For m ≥ n (overdetermined systems), QR can be computed in two forms:
 * </p>
 * <ul>
 *   <li><b>Full QR:</b> Q is m×m, R is m×n</li>
 *   <li><b>Thin/Economy QR:</b> Q is m×n, R is n×n (more efficient)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * HouseholderQR qr = new HouseholderQR();
 * QRResult result = qr.decompose(A);
 *
 * // Access factors
 * Matrix Q = result.getQ();  // Orthogonal matrix
 * Matrix R = result.getR();  // Upper triangular
 *
 * // Verify orthogonality: Q^T * Q = I
 * Matrix QTQ = Q.transpose().multiply(Q);
 * Matrix identity = Matrix.Identity(Q.getColumnCount());
 * double orthError = QTQ.subtract(identity).frobeniusNorm();
 * System.out.println("Orthogonality error: " + orthError);  // Should be ~1e-15
 *
 * // Verify factorization: A = Q * R
 * Matrix reconstructed = result.reconstruct();
 * double factorError = result.getResidualNorm(A);
 * System.out.println("Factorization error: " + factorError);
 *
 * // Check for rank deficiency
 * int rank = countNonzeroDiagonal(R);
 * }</pre>
 *
 * <h2>Provided Operations:</h2>
 * <ul>
 *   <li><b>Factor access:</b> Get Q and R matrices</li>
 *   <li><b>Reconstruction:</b> Compute A = QR for verification</li>
 *   <li><b>Residual norm:</b> Measure factorization accuracy ||A - QR||<sub>F</sub></li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Solving least squares problems (minimize ||Ax - b||<sub>2</sub>)</li>
 *   <li>Computing orthonormal bases</li>
 *   <li>Rank determination and nullspace computation</li>
 *   <li>Matrix pseudoinverse calculation</li>
 *   <li>Eigenvalue algorithms (QR iteration)</li>
 *   <li>Gram-Schmidt orthogonalization</li>
 * </ul>
 *
 * <h2>Least Squares Solving:</h2>
 * <p>
 * To solve least squares min ||Ax - b||<sub>2</sub>:
 * </p>
 * <pre>{@code
 * // Method 1: Using Q and R directly
 * Vector y = Q.transpose().multiply(b);  // Q^T * b
 * Vector x = backSubstitution(R, y);     // Solve Rx = y
 *
 * // Method 2: Using normal equations (if implemented in result)
 * // R^T R x = R^T Q^T b  (but QR is more stable)
 * }</pre>
 *
 * <h2>Immutability:</h2>
 * <p>
 * This class is immutable - all fields are final and matrices are not defensively copied.
 * Users should not modify returned matrices if factorization integrity is required.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.qr.HouseholderQR
 * @see net.faulj.decomposition.qr.GivensQR
 * @see net.faulj.decomposition.qr.ModifiedGramSchmidt
 * @see net.faulj.solve.LeastSquaresSolver
 */
public class QRResult {
    private final Matrix A;
    private final Matrix Q;
    private final Matrix R;

    /**
     * Create a QR result container.
     *
     * @param A original matrix
     * @param Q orthogonal factor
     * @param R upper-triangular factor
     */
    public QRResult(Matrix A, Matrix Q, Matrix R) {
        this.A = A;
        this.Q = Q;
        this.R = R;
    }

    /**
     * @return orthogonal factor Q
     */
    public Matrix getQ() {
        return Q;
    }

    /**
     * @return upper-triangular factor R
     */
    public Matrix getR() {
        return R;
    }

    /**
     * Reconstruct A from QR factors.
     *
     * @return reconstructed matrix
     */
    public Matrix reconstruct() {
        return Q.multiply(R);
    }

    /**
     * Compute the Frobenius norm residual of the factorization.
     *
     * @return normalized residual
     */
    public double residualNorm() {
        return MatrixUtils.relativeError(A, reconstruct());
    }

    /**
     * Verify orthogonality of a matrix against the identity.
     *
     * @param O matrix to verify
     * @return array with {orthogonalityError}
     */
    public double[] verifyOrthogonality(Matrix O) {
        return new double[]{MatrixUtils.orthogonalityError(O)};
    }
}