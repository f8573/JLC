package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import net.faulj.scalar.Complex;

import java.util.Arrays;

/**
 * Encapsulates the result of Real Schur decomposition.
 * <p>
 * This class represents the similarity transformation A = UTU<sup>T</sup> where:
 * </p>
 * <ul>
 *   <li><b>U</b> - Orthogonal matrix (U<sup>T</sup>U = I) of Schur vectors</li>
 *   <li><b>T</b> - Quasi-upper triangular matrix in Real Schur form</li>
 *   <li><b>A</b> - Original square real matrix</li>
 * </ul>
 *
 * <h2>Real Schur Form:</h2>
 * <p>
 * A real matrix has Real Schur form T that is quasi-upper triangular:
 * </p>
 * <pre>
 *   ┌ λ₁  *   *   *   * ┐
 *   │  0  a  b   *   * │
 *   │  0  c  d   *   * │
 *   │  0  0  0  λ₄  * │
 *   └  0  0  0   0  λ₅ ┘
 * </pre>
 * <ul>
 *   <li><b>1×1 blocks:</b> Real eigenvalues on diagonal</li>
 *   <li><b>2×2 blocks:</b> Complex conjugate eigenvalue pairs</li>
 * </ul>
 *
 * <h2>Eigenvalue Extraction:</h2>
 * <p>
 * Eigenvalues are obtained directly from T:
 * </p>
 * <ul>
 *   <li><b>Real eigenvalues:</b> From 1×1 diagonal blocks T[i,i]</li>
 *   <li><b>Complex eigenvalues:</b> From 2×2 blocks via quadratic formula:
 *     <pre>
 *     λ = (a+d)/2 ± sqrt((a-d)² + 4bc) / 2
 *     </pre>
 *   </li>
 * </ul>
 *
 * <h2>Properties:</h2>
 * <ul>
 *   <li><b>Eigenvalue preservation:</b> T and A have identical eigenvalues</li>
 *   <li><b>Orthogonal U:</b> U<sup>T</sup>U = I to machine precision</li>
 *   <li><b>Similarity:</b> A = UTU<sup>T</sup> exact up to rounding errors</li>
 *   <li><b>Uniqueness:</b> Schur form is not unique (depends on ordering)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * RealSchurDecomposition schur = new RealSchurDecomposition();
 * SchurResult result = schur.decompose(A);
 *
 * // Access factors
 * Matrix T = result.getT();  // Quasi-triangular form
 * Matrix U = result.getU();  // Schur vectors
 *
 * // Get eigenvalues (already extracted)
 * double[] realParts = result.getRealEigenvalues();
 * double[] imagParts = result.getImagEigenvalues();
 *
 * // Display eigenvalues
 * for (int i = 0; i < realParts.length; i++) {
 *     if (Math.abs(imagParts[i]) < 1e-12) {
 *         System.out.println("λ" + i + " = " + realParts[i]);
 *     } else {
 *         System.out.println("λ" + i + " = " + realParts[i] + " ± " + imagParts[i] + "i");
 *     }
 * }
 *
 * // Verify similarity: A = U * T * U^T
 * Matrix reconstructed = U.multiply(T).multiply(U.transpose());
 * }</pre>
 *
 * <h2>Eigenvalue Representation:</h2>
 * <p>
 * This class stores eigenvalues in parallel arrays:
 * </p>
 * <ul>
 *   <li><b>realEigenvalues[i]:</b> Real part of i-th eigenvalue</li>
 *   <li><b>imagEigenvalues[i]:</b> Imaginary part of i-th eigenvalue</li>
 *   <li><b>Complex pairs:</b> Stored as conjugate pairs (λ and λ*)</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Computing all eigenvalues of a real matrix</li>
 *   <li>Matrix function evaluation (e<sup>A</sup>, log(A), A<sup>k</sup>)</li>
 *   <li>Solving matrix differential equations</li>
 *   <li>Stability analysis in control theory</li>
 *   <li>Computing eigenvectors (via back-transformation)</li>
 *   <li>Jordan normal form approximation</li>
 * </ul>
 *
 * <h2>Schur vs Eigendecomposition:</h2>
 * <table border="1">
 *   <tr><th>Aspect</th><th>Schur (UTU<sup>T</sup>)</th><th>Eigendecomposition (QΛQ<sup>-1</sup>)</th></tr>
 *   <tr><td>Existence</td><td>Always exists</td><td>Only for diagonalizable matrices</td></tr>
 *   <tr><td>U/Q orthogonal?</td><td>Yes</td><td>No (in general)</td></tr>
 *   <tr><td>Stability</td><td>Excellent</td><td>Poor for non-normal matrices</td></tr>
 *   <tr><td>Complex eigenvalues</td><td>2×2 blocks</td><td>Requires complex arithmetic</td></tr>
 * </table>
 *
 * <h2>Computational Path:</h2>
 * <ol>
 *   <li>Reduce A to Hessenberg form H (O(10n³/3))</li>
 *   <li>Apply QR iteration to H to get T (O(10n³), ~2-3 iterations per eigenvalue)</li>
 *   <li>Extract eigenvalues from diagonal blocks of T (O(n))</li>
 * </ol>
 *
 * <h2>Immutability:</h2>
 * <p>
 * This class is immutable - all fields are final. Eigenvalue arrays are not defensively
 * copied, so users should not modify them.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.eigen.schur.RealSchurDecomposition
 * @see net.faulj.eigen.schur.SchurEigenExtractor
 * @see net.faulj.eigen.qr.ImplicitQRFrancis
 * @see HessenbergResult
 */
public class SchurResult {
    private final Matrix A;
    private final Matrix T; // Quasi-upper triangular matrix
    private final Matrix U; // Unitary/Orthogonal matrix (Schur vectors)
    private final Complex[] eigenvalues;
    private final Matrix eigenvectors;

    /**
     * Create a Schur decomposition result.
     *
     * @param A original matrix
     * @param T quasi-upper triangular Schur form
     * @param U orthogonal Schur vectors
     * @param eigenvalues eigenvalues extracted from T
     */
    public SchurResult(Matrix A, Matrix T, Matrix U, Complex[] eigenvalues) {
        this(A, T, U, eigenvalues, null);
    }

    /**
     * Create a Schur decomposition result with optional eigenvectors.
     *
     * @param A original matrix
     * @param T quasi-upper triangular Schur form
     * @param U orthogonal Schur vectors
     * @param eigenvalues eigenvalues extracted from T
     * @param eigenvectors eigenvectors, or null if not computed
     */
    public SchurResult(Matrix A, Matrix T, Matrix U, Complex[] eigenvalues, Matrix eigenvectors) {
        this.A = A;
        this.T = T;
        this.U = U;
        this.eigenvalues = eigenvalues;
        this.eigenvectors = eigenvectors;
    }

    /**
     * @return Schur form T
     */
    public Matrix getT() { return T; }
    /**
     * @return Schur vectors U
     */
    public Matrix getU() { return U; }
    /**
     * @return eigenvalues as complex numbers
     */
    public Complex[] getEigenvalues() { return eigenvalues; }
    /**
     * @return eigenvectors, or null if not computed
     */
    public Matrix getEigenvectors() { return eigenvectors; }
    
    /**
     * Extracts the real parts of all eigenvalues.
     * @return Array of real parts of eigenvalues
     */
    public double[] getRealEigenvalues() {
        double[] real = new double[eigenvalues.length];
        for (int i = 0; i < eigenvalues.length; i++) {
            real[i] = eigenvalues[i].real;
        }
        return real;
    }
    
    /**
     * Extracts the imaginary parts of all eigenvalues.
     * @return Array of imaginary parts of eigenvalues
     */
    public double[] getImagEigenvalues() {
        double[] imag = new double[eigenvalues.length];
        for (int i = 0; i < eigenvalues.length; i++) {
            imag[i] = eigenvalues[i].imag;
        }
        return imag;
    }

    /**
     * @return string summary of the Schur result
     */
    @Override
    public String toString() {
        return "SchurResult{\n" +
                "  T=" + T.getRowCount() + "x" + T.getColumnCount() + "\n" +
                "  Eigenvectors=" + (eigenvectors == null ? "null" : eigenvectors.getRowCount() + "x" + eigenvectors.getColumnCount()) + "\n" +
                "  Eigenvalues=" + Arrays.toString(eigenvalues);
    }

    /**
     * Compute the Frobenius norm residual of the similarity transform.
     *
     * @return residual norm
     */
    public double residualNorm() {
        return MatrixUtils.relativeError(A, U.multiply(T).multiply(U.transpose()));
    }

    /**
     * Verify orthogonality of a candidate matrix.
     *
     * @param O matrix to test (typically U)
     * @return array containing orthogonality error
     */
    public double[] verifyOrthogonality(Matrix O) {
        return new double[]{MatrixUtils.orthogonalityError(O)};
    }
}
