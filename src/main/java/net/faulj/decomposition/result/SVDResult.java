package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;

/**
 * Encapsulates the result of Singular Value Decomposition (SVD).
 * <p>
 * This class represents the factorization A = UΣV<sup>T</sup> where:
 * </p>
 * <ul>
 *   <li><b>U</b> - Left singular vectors (m×m orthogonal matrix)</li>
 *   <li><b>Σ</b> - Diagonal matrix of singular values (m×n, non-negative, descending order)</li>
 *   <li><b>V</b> - Right singular vectors (n×n orthogonal matrix)</li>
 *   <li><b>A</b> - Original m×n matrix</li>
 * </ul>
 *
 * <h2>Singular Values:</h2>
 * <p>
 * Singular values σ<sub>i</sub> are non-negative and ordered:
 * </p>
 * <pre>
 *   σ₁ ≥ σ₂ ≥ ... ≥ σ<sub>min(m,n)</sub> ≥ 0
 * </pre>
 * <p>
 * They represent the "stretching" factors of the linear transformation A.
 * </p>
 *
 * <h2>Thin vs Full SVD:</h2>
 * <ul>
 *   <li><b>Full SVD:</b> U is m×m, V is n×n, Σ is m×n</li>
 *   <li><b>Thin/Economy SVD:</b> U is m×r, V is n×r, Σ is r×r (where r = min(m,n))</li>
 *   <li><b>Compact SVD:</b> Only non-zero singular values and corresponding vectors</li>
 * </ul>
 *
 * <h2>Matrix Properties:</h2>
 * <ul>
 *   <li><b>Orthogonality:</b> U<sup>T</sup>U = I and V<sup>T</sup>V = I</li>
 *   <li><b>Singular values:</b> σᵢ = sqrt(λᵢ) where λᵢ are eigenvalues of A<sup>T</sup>A</li>
 *   <li><b>Rank:</b> Number of non-zero singular values</li>
 *   <li><b>Norms:</b> ||A||₂ = σ₁ (largest singular value)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][] {
 *     {3, 2, 2},
 *     {2, 3, -2}
 * });
 *
 * SVDecomposition svd = new SVDecomposition();
 * SVDResult result = svd.decompose(A);
 *
 * // Access factors
 * Matrix U = result.getU();               // Left singular vectors
 * double[] sigma = result.getSingularValues();  // Singular values array
 * Matrix Sigma = result.getSigma();       // Diagonal matrix form
 * Matrix V = result.getV();               // Right singular vectors
 *
 * // Get matrix properties
 * int rank = result.getRank();
 * double conditionNumber = result.getConditionNumber();  // σ₁/σᵣ
 * double norm2 = result.getNorm2();       // ||A||₂ = σ₁
 *
 * // Verify factorization: A = U * Σ * V^T
 * Matrix reconstructed = result.reconstruct();
 * double error = result.getResidualNorm(A);
 *
 * // Compute pseudoinverse: A⁺ = V * Σ⁺ * U^T
 * Matrix pseudoInverse = result.pseudoInverse();
 * }</pre>
 *
 * <h2>Provided Operations:</h2>
 * <ul>
 *   <li><b>Factor access:</b> Get U, Σ, and V matrices</li>
 *   <li><b>Singular values:</b> Access as array or diagonal matrix</li>
 *   <li><b>Rank determination:</b> Count non-zero singular values</li>
 *   <li><b>Condition number:</b> Ratio σ<sub>max</sub>/σ<sub>min</sub></li>
 *   <li><b>Matrix norms:</b> 2-norm and Frobenius norm</li>
 *   <li><b>Pseudoinverse:</b> Moore-Penrose pseudoinverse A<sup>+</sup></li>
 *   <li><b>Low-rank approximation:</b> Best rank-k approximation</li>
 *   <li><b>Reconstruction:</b> Verify A = UΣV<sup>T</sup></li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Solving least squares problems (including rank-deficient)</li>
 *   <li>Computing pseudoinverse for singular/rectangular matrices</li>
 *   <li>Rank and nullspace determination</li>
 *   <li>Data compression and dimensionality reduction (PCA)</li>
 *   <li>Image processing and denoising</li>
 *   <li>Recommender systems and collaborative filtering</li>
 *   <li>Signal processing (Karhunen-Loève transform)</li>
 *   <li>Matrix approximation and compression</li>
 * </ul>
 *
 * <h2>Rank and Nullspace:</h2>
 * <pre>{@code
 * // Determine numerical rank with tolerance
 * int rank = 0;
 * double tol = 1e-10;
 * for (double s : sigma) {
 *     if (s > tol) rank++;
 * }
 *
 * // Nullspace basis: last (n - rank) columns of V
 * Matrix nullspace = V.getColumns(rank, n - 1);
 * }</pre>
 *
 * <h2>Low-Rank Approximation:</h2>
 * <p>
 * The best rank-k approximation to A (in Frobenius norm) is:
 * </p>
 * <pre>
 *   A<sub>k</sub> = Σ<sub>i=1..k</sub> σᵢ u<sub>i</sub> v<sub>i</sub><sup>T</sup>
 * </pre>
 * <pre>{@code
 * // Best rank-k approximation
 * int k = 2;
 * Matrix U_k = U.getColumns(0, k - 1);
 * Matrix V_k = V.getColumns(0, k - 1);
 * Matrix Sigma_k = Sigma.crop(0, k - 1, 0, k - 1);
 * Matrix A_k = U_k.multiply(Sigma_k).multiply(V_k.transpose());
 * }</pre>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Full SVD:</b> O(min(mn², m²n)) flops</li>
 *   <li><b>Thin SVD:</b> O(mn·min(m,n)) flops (more efficient)</li>
 *   <li><b>Algorithms:</b> Golub-Kahan bidiagonalization + QR iteration</li>
 *   <li><b>Space:</b> O(m·min(m,n) + n·min(m,n)) for thin SVD</li>
 * </ul>
 *
 * <h2>Numerical Stability:</h2>
 * <p>
 * SVD is the most numerically stable matrix decomposition:
 * </p>
 * <ul>
 *   <li>Backward stable algorithm</li>
 *   <li>Reliable rank determination even for ill-conditioned matrices</li>
 *   <li>Best method for computing pseudoinverse</li>
 *   <li>Orthogonal U and V to machine precision</li>
 * </ul>
 *
 * <h2>Relationship to Other Decompositions:</h2>
 * <ul>
 *   <li><b>Eigendecomposition:</b> If A = A<sup>T</sup>, then U = V and Σ = |Λ|</li>
 *   <li><b>Polar decomposition:</b> A = (UV<sup>T</sup>)(VΣV<sup>T</sup>) = UP</li>
 *   <li><b>QR decomposition:</b> A = QR ≈ (U)(ΣV<sup>T</sup>)</li>
 * </ul>
 *
 * <h2>Immutability:</h2>
 * <p>
 * This class is immutable - all fields are final. Singular value arrays are not
 * defensively copied, so users should not modify them.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.svd.SVDecomposition
 * @see net.faulj.svd.ThinSVD
 * @see net.faulj.svd.Pseudoinverse
 * @see net.faulj.svd.RankEstimation
 */
public class SVDResult {
    private final Matrix A;
    private final Matrix U;
    private final double[] singularValues;
    private final Matrix V;

    public SVDResult(Matrix A, Matrix U, double[] singularValues, Matrix V) {
        this.A = A;
        this.U = U;
        this.singularValues = singularValues;
        this.V = V;
    }

    /**
     * Returns the left singular vectors matrix U.
     * @return m×m (or m×r for thin SVD) orthogonal matrix
     */
    public Matrix getU() {
        return U;
    }

    /**
     * Returns the singular values as an array (descending order).
     * @return array of non-negative singular values
     */
    public double[] getSingularValues() {
        return singularValues;
    }

    /**
     * Returns the singular values as a diagonal matrix Σ.
     * @return diagonal matrix with singular values
     */
    public Matrix getSigma() {
        int m = U.getColumnCount();
        int n = V.getColumnCount();
        Matrix sigma = new Matrix(m, n);
        for (int i = 0; i < Math.min(singularValues.length, Math.min(m, n)); i++) {
            sigma.set(i, i, singularValues[i]);
        }
        return sigma;
    }

    /**
     * Returns the right singular vectors matrix V.
     * @return n×n (or n×r for thin SVD) orthogonal matrix
     */
    public Matrix getV() {
        return V;
    }

    /**
     * Reconstructs A = U * Σ * V^T for verification.
     * @return reconstructed matrix
     */
    public Matrix reconstruct() {
        Matrix sigma = getSigma();
        return U.multiply(sigma).multiply(V.transpose());
    }

    /**
     * Computes reconstruction error ||A - UΣV^T||_F
     * @return Frobenius norm of reconstruction error
     */
    public double residualNorm() {
        return MatrixUtils.normResidual(A, reconstruct(), 1e-10);
    }

    public double residualElement() {
        return MatrixUtils.backwardErrorComponentwise(A, reconstruct(), 1e-10);
    }

    public double[] verifyOrthogonality(Matrix O) {
        Matrix I = Matrix.Identity(O.getRowCount());
        double n = MatrixUtils.normResidual(I, O, 1e-10);
        double e = MatrixUtils.backwardErrorComponentwise(I, O, 1e-10);
        return new double[]{n, e};
    }

    /**
     * Determines the numerical rank (number of singular values above tolerance).
     * @param tolerance threshold for considering singular value as zero
     * @return numerical rank
     */
    public int getRank(double tolerance) {
        int rank = 0;
        for (double s : singularValues) {
            if (s > tolerance) rank++;
        }
        return rank;
    }

    /**
     * Returns the 2-norm of the matrix (largest singular value).
     * @return ||A||₂ = σ₁
     */
    public double getNorm2() {
        return singularValues.length > 0 ? singularValues[0] : 0.0;
    }

    /**
     * Returns the condition number in the 2-norm.
     * @return cond(A) = σ<sub>max</sub> / σ<sub>min</sub>
     */
    public double getConditionNumber() {
        if (singularValues.length == 0) return Double.POSITIVE_INFINITY;
        double minSigma = singularValues[singularValues.length - 1];
        if (minSigma < 1e-15) return Double.POSITIVE_INFINITY;
        return singularValues[0] / minSigma;
    }
}
