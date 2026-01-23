package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;

/**
 * Encapsulates the result of Hessenberg reduction.
 * <p>
 * This class represents the similarity transformation A = QHQ<sup>T</sup> where:
 * </p>
 * <ul>
 *   <li><b>Q</b> - Orthogonal matrix (Q<sup>T</sup>Q = I) representing accumulated Householder reflections</li>
 *   <li><b>H</b> - Upper Hessenberg matrix (zeros below first subdiagonal)</li>
 *   <li><b>A</b> - Original square matrix being reduced</li>
 * </ul>
 *
 * <h2>Hessenberg Form:</h2>
 * <p>
 * An upper Hessenberg matrix has the structure:
 * </p>
 * <pre>
 *   ┌ * * * * * ┐
 *   │ * * * * * │
 *   │ 0 * * * * │
 *   │ 0 0 * * * │
 *   └ 0 0 0 * * ┘
 * </pre>
 * <p>
 * All elements below the first subdiagonal are zero: H[i,j] = 0 for i &gt; j+1.
 * </p>
 *
 * <h2>Similarity Transformation:</h2>
 * <p>
 * The reduction is a similarity transformation, which means:
 * </p>
 * <ul>
 *   <li><b>Eigenvalues preserved:</b> H and A have identical eigenvalues</li>
 *   <li><b>Characteristic polynomial:</b> det(λI - H) = det(λI - A)</li>
 *   <li><b>Trace preserved:</b> tr(H) = tr(A)</li>
 *   <li><b>Determinant preserved:</b> det(H) = det(A)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = Matrix.random(5, 5);
 * Matrix[] result = HessenbergReduction.decompose(A);
 * Matrix H = result[0];  // Hessenberg form
 * Matrix Q = result[1];  // Orthogonal transformation
 *
 * HessenbergResult hessResult = new HessenbergResult(H, Q);
 *
 * // Verify similarity: A = Q * H * Q^T
 * Matrix reconstructed = hessResult.reconstruct();
 * double error = hessResult.getResidualNorm(A);
 * System.out.println("Reconstruction error: " + error);  // Should be ~1e-14
 *
 * // Verify H is Hessenberg (zeros below first subdiagonal)
 * int n = H.getRowCount();
 * for (int i = 0; i < n; i++) {
 *     for (int j = 0; j < i - 1; j++) {
 *         assert Math.abs(H.get(i, j)) < 1e-12;
 *     }
 * }
 *
 * // Use H as input to QR iteration for eigenvalues
 * double[] eigenvalues = qrIteration(H);
 * }</pre>
 *
 * <h2>Provided Operations:</h2>
 * <ul>
 *   <li><b>Factor access:</b> Get H (Hessenberg) and Q (orthogonal) matrices</li>
 *   <li><b>Reconstruction:</b> Compute A = QHQ<sup>T</sup> for verification</li>
 *   <li><b>Residual norm:</b> Measure transformation accuracy ||A - QHQ<sup>T</sup>||<sub>F</sub></li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>First stage of QR algorithm for eigenvalue computation</li>
 *   <li>Prerequisite for Schur decomposition</li>
 *   <li>Computing matrix exponentials and functions</li>
 *   <li>Solving differential equations (e<sup>At</sup>)</li>
 *   <li>Control theory and system dynamics</li>
 *   <li>Krylov subspace methods</li>
 * </ul>
 *
 * <h2>Why Hessenberg Form Matters:</h2>
 * <ul>
 *   <li><b>QR iteration efficiency:</b> One QR step costs O(n<sup>2</sup>) vs O(n<sup>3</sup>) for dense matrix</li>
 *   <li><b>Structure preservation:</b> QR iteration maintains Hessenberg form</li>
 *   <li><b>Nearly triangular:</b> Already close to Schur form (eigenvalue revealing)</li>
 *   <li><b>Eigenvalue algorithms:</b> Essential preprocessing for all QR-based methods</li>
 * </ul>
 *
 * <h2>Symmetric Case - Tridiagonal:</h2>
 * <p>
 * When A is symmetric, H becomes tridiagonal (Hessenberg with additional symmetry):
 * </p>
 * <pre>
 *   ┌ d₁ e₁  0  0  0 ┐
 *   │ e₁ d₂ e₂  0  0 │
 *   │  0 e₂ d₃ e₃  0 │
 *   │  0  0 e₃ d₄ e₄ │
 *   └  0  0  0 e₄ d₅ ┘
 * </pre>
 * <p>
 * This offers additional computational advantages and specialized algorithms.
 * </p>
 *
 * <h2>Immutability:</h2>
 * <p>
 * This class is immutable - all fields are final and matrices are not defensively copied.
 * Users should not modify returned matrices if transformation integrity is required.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.hessenberg.HessenbergReduction
 * @see net.faulj.eigen.schur.RealSchurDecomposition
 * @see net.faulj.eigen.qr.ImplicitQRFrancis
 */
public class HessenbergResult {
    private final Matrix H;
    private final Matrix Q;
    private final Matrix A;
    public HessenbergResult(Matrix A, Matrix H, Matrix Q) {
        this.A = A;
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