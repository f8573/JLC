package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;

/**
 * Encapsulates the result of Polar decomposition.
 * <p>
 * This class represents the factorization A = UP where:
 * </p>
 * <ul>
 *   <li><b>U</b> - Unitary/orthogonal matrix (U<sup>T</sup>U = I)</li>
 *   <li><b>P</b> - Positive semi-definite Hermitian matrix (symmetric for real matrices)</li>
 *   <li><b>A</b> - Original m×n matrix</li>
 * </ul>
 *
 * <h2>Polar Decomposition Forms:</h2>
 * <p>
 * Two forms exist depending on factorization order:
 * </p>
 * <ul>
 *   <li><b>Right polar:</b> A = UP (U rotates, P stretches)</li>
 *   <li><b>Left polar:</b> A = P'U (P' stretches, U rotates)</li>
 * </ul>
 * <p>
 * This class represents the right polar form.
 * </p>
 *
 * <h2>Matrix Properties:</h2>
 * <ul>
 *   <li><b>Orthogonal U:</b> U<sup>T</sup>U = I (U is a rotation/reflection)</li>
 *   <li><b>Positive semi-definite P:</b> x<sup>T</sup>Px ≥ 0 for all x</li>
 *   <li><b>Symmetric P:</b> P = P<sup>T</sup> for real matrices</li>
 *   <li><b>Uniqueness:</b> Unique for invertible A; U unique if A is square</li>
 * </ul>
 *
 * <h2>Geometric Interpretation:</h2>
 * <p>
 * Polar decomposition separates the action of A into:
 * </p>
 * <ul>
 *   <li><b>P (stretching):</b> Scales along principal axes (eigenvalue directions)</li>
 *   <li><b>U (rotation):</b> Pure rotation/reflection with no scaling</li>
 * </ul>
 * <p>
 * This is analogous to polar form of complex numbers: z = re<sup>iθ</sup> = |z|·e<sup>iθ</sup>
 * </p>
 *
 * <h2>Computation via SVD:</h2>
 * <p>
 * Given A = UΣV<sup>T</sup> (SVD):
 * </p>
 * <pre>
 *   U<sub>polar</sub> = UV<sup>T</sup>
 *   P = VΣV<sup>T</sup>
 * </pre>
 * <p>
 * This shows P has eigenvalues equal to singular values of A.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][] {
 *     {3, 1},
 *     {1, 3}
 * });
 *
 * PolarDecomposition polar = new PolarDecomposition();
 * PolarResult result = polar.decompose(A);
 *
 * // Access factors
 * Matrix U = result.getU();  // Orthogonal/rotation component
 * Matrix P = result.getP();  // Positive semi-definite/stretching component
 *
 * // Verify factorization: A = U * P
 * Matrix reconstructed = result.reconstruct();
 * double error = result.getResidualNorm(A);
 *
 * // Verify U is orthogonal: U^T * U = I
 * Matrix UTU = U.transpose().multiply(U);
 * Matrix identity = Matrix.Identity(U.getRowCount());
 * double orthError = UTU.subtract(identity).frobeniusNorm();
 *
 * // Verify P is symmetric: P = P^T
 * Matrix PT = P.transpose();
 * double symError = P.subtract(PT).frobeniusNorm();
 *
 * // Verify P is positive semi-definite (all eigenvalues ≥ 0)
 * double[] eigenvalues = P.eigenvalues();
 * boolean isPosDef = Arrays.stream(eigenvalues).allMatch(λ -> λ >= -1e-10);
 * }</pre>
 *
 * <h2>Provided Operations:</h2>
 * <ul>
 *   <li><b>Factor access:</b> Get U (orthogonal) and P (positive semi-definite)</li>
 *   <li><b>Reconstruction:</b> Compute A = UP for verification</li>
 *   <li><b>Residual norm:</b> Measure factorization accuracy ||A - UP||<sub>F</sub></li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Mechanics and continuum mechanics (deformation gradient analysis)</li>
 *   <li>Computer graphics (decomposing transformations)</li>
 *   <li>Signal processing (optimal rotation estimation)</li>
 *   <li>Robotics (motion analysis and pose estimation)</li>
 *   <li>Crystallography (lattice deformation)</li>
 *   <li>Procrustes problem (orthogonal alignment)</li>
 *   <li>Matrix approximation by orthogonal matrices</li>
 * </ul>
 *
 * <h2>Procrustes Problem:</h2>
 * <p>
 * Find orthogonal Q minimizing ||A - QB||<sub>F</sub>. Solution: Q = U where A = UP
 * and P is computed from A. The polar factor U is the best orthogonal approximation.
 * </p>
 *
 * <h2>Special Cases:</h2>
 * <ul>
 *   <li><b>A orthogonal:</b> P = I, U = A</li>
 *   <li><b>A positive definite:</b> U = I, P = A</li>
 *   <li><b>A singular:</b> U not unique, but decomposition exists</li>
 *   <li><b>A = 0:</b> U arbitrary orthogonal, P = 0</li>
 * </ul>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Via SVD:</b> O(mn·min(m,n)) flops (dominant cost is SVD)</li>
 *   <li><b>Space:</b> O(m² + n²) for storing U and P</li>
 *   <li><b>Alternative methods:</b> Newton iteration, matrix sign function</li>
 * </ul>
 *
 * <h2>Relationship to SVD:</h2>
 * <p>
 * Polar decomposition can be viewed as a regrouping of SVD factors:
 * </p>
 * <pre>
 *   SVD:   A = (UΣ)(V<sup>T</sup>)
 *   Polar: A = (UV<sup>T</sup>)(VΣV<sup>T</sup>) = U<sub>polar</sub> · P
 * </pre>
 *
 * <h2>Uniqueness:</h2>
 * <ul>
 *   <li><b>P is always unique</b> (positive semi-definite square root of A<sup>T</sup>A)</li>
 *   <li><b>U is unique if A is invertible</b></li>
 *   <li><b>If A is singular:</b> Infinitely many U satisfy A = UP</li>
 * </ul>
 *
 * <h2>Numerical Stability:</h2>
 * <p>
 * When computed via SVD, polar decomposition inherits SVD's excellent stability:
 * </p>
 * <ul>
 *   <li>Backward stable computation</li>
 *   <li>Orthogonality of U maintained to machine precision</li>
 *   <li>Symmetry and positive semi-definiteness of P preserved</li>
 * </ul>
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
 * @see net.faulj.polar.PolarDecomposition
 * @see net.faulj.svd.SVDecomposition
 * @see SVDResult
 */
public class PolarResult {
    private final Matrix U;  // Orthogonal matrix
    private final Matrix P;  // Positive semi-definite matrix

    public PolarResult(Matrix U, Matrix P) {
        this.U = U;
        this.P = P;
    }

    /**
     * Returns the orthogonal factor U.
     * @return orthogonal matrix (rotation/reflection component)
     */
    public Matrix getU() {
        return U;
    }

    /**
     * Returns the positive semi-definite factor P.
     * @return symmetric positive semi-definite matrix (stretching component)
     */
    public Matrix getP() {
        return P;
    }

    /**
     * Reconstructs A = U * P for verification.
     * @return reconstructed matrix
     */
    public Matrix reconstruct() {
        return U.multiply(P);
    }

    /**
     * Computes reconstruction error ||A - UP||_F
     * @param A original matrix
     * @return Frobenius norm of reconstruction error
     */
    public double getResidualNorm(Matrix A) {
        Matrix reconstructed = reconstruct();
        return A.subtract(reconstructed).frobeniusNorm();
    }
}
