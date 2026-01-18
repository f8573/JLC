package net.faulj.svd;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.SVDResult;

/**
 * Computes the Singular Value Decomposition (SVD) of a matrix.
 * <p>
 * This class performs the factorization A = UΣV<sup>T</sup> where:
 * </p>
 * <ul>
 * <li><b>A</b> - Original m×n rectangular real matrix</li>
 * <li><b>U</b> - m×m Orthogonal matrix (U<sup>T</sup>U = I) containing left singular vectors</li>
 * <li><b>Σ</b> - m×n Diagonal matrix containing singular values (σ)</li>
 * <li><b>V</b> - n×n Orthogonal matrix (V<sup>T</sup>V = I) containing right singular vectors</li>
 * </ul>
 *
 * <h2>Mathematical Definition:</h2>
 * <p>
 * For any real matrix A, there exist orthogonal matrices U and V such that:
 * </p>
 * <pre>
 * A = U * diag(σ₁, σ₂, ..., σᵣ) * Vᵀ
 * </pre>
 * <p>
 * Where singular values are non-negative and ordered: σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = MatrixFactory.createRandom(5, 3);
 *
 * // Perform SVD
 * SVDecomposition svd = new SVDecomposition();
 * SVDResult result = svd.decompose(A);
 *
 * // Access components
 * Matrix U = result.getU();       // Left singular vectors
 * double[] S = result.getSingularValues(); // Singular values vector
 * Matrix V = result.getV();       // Right singular vectors
 *
 * // Verify: A ≈ U * Σ * V^T
 * Matrix Sigma = MatrixFactory.diagonal(S, 5, 3);
 * Matrix reconstructed = U.multiply(Sigma).multiply(V.transpose());
 * }</pre>
 *
 * <h2>Computational Path:</h2>
 * <ol>
 * <li><b>Bidiagonalization:</b> Reduce A to bidiagonal form B using Householder reflections (Golub-Kahan).
 * <br>A = P * B * Qᵀ</li>
 * <li><b>Diagonalization:</b> Apply implicit QR iteration with Wilkinson shift to B to zero out the superdiagonal.
 * <br>B → Σ</li>
 * <li><b>Accumulation:</b> Accumulate Householder and Givens rotations to form U and V.</li>
 * </ol>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li><b>Dimensionality Reduction:</b> Principal Component Analysis (PCA)</li>
 * <li><b>Data Compression:</b> Low-rank matrix approximation (Eckart-Young-Mirsky theorem)</li>
 * <li><b>Pseudoinverse:</b> Computing the Moore-Penrose inverse</li>
 * <li><b>Noise Reduction:</b> Truncated SVD to remove small singular values</li>
 * <li><b>Condition Number:</b> Ratio of largest to smallest singular value</li>
 * </ul>
 *
 * <h2>Performance Notes:</h2>
 * <ul>
 * <li>Complexity is approximately O(mn² + n³) for m ≥ n</li>
 * <li>This implementation computes the "Full SVD" (U is m×m)</li>
 * <li>For m &gt;&gt; n, consider using {@link ThinSVD} for efficiency</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.result.SVDResult
 * @see ThinSVD
 * @see BidiagonalQR
 */
public class SVDecomposition {
}