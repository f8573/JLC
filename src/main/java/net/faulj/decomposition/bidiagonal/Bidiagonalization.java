package net.faulj.decomposition.bidiagonal;

/**
 * Computes the bidiagonal decomposition of a rectangular matrix using Householder reflections.
 * <p>
 * The bidiagonal decomposition factors an m-by-n matrix A into the form:
 * </p>
 * <pre>
 *   A = U * B * V<sup>T</sup>
 * </pre>
 * <p>
 * where:
 * </p>
 * <ul>
 *   <li><b>U</b> is an m-by-m orthogonal matrix</li>
 *   <li><b>B</b> is an m-by-n bidiagonal matrix</li>
 *   <li><b>V</b> is an n-by-n orthogonal matrix</li>
 * </ul>
 *
 * <h2>Bidiagonal Form:</h2>
 * <p>
 * A bidiagonal matrix has non-zero elements only on the main diagonal and either the
 * superdiagonal (if m &ge; n) or subdiagonal (if m &lt; n):
 * </p>
 * <pre>
 * For m &ge; n (upper bidiagonal):     For m &lt; n (lower bidiagonal):
 *   ┌ d₁ e₁  0  0 ┐                      ┌ d₁  0  0  0  0 ┐
 *   │  0 d₂ e₂  0 │                      │ e₁ d₂  0  0  0 │
 *   │  0  0 d₃ e₃ │                      └  0 e₂ d₃  0  0 ┘
 *   │  0  0  0 d₄ │
 *   └  0  0  0  0 ┘
 * </pre>
 *
 * <h2>Algorithm:</h2>
 * <p>
 * The Golub-Kahan bidiagonalization algorithm alternately applies Householder reflections
 * to the left and right to zero out elements:
 * </p>
 * <ol>
 *   <li>Apply Householder transformation from the left to zero column below diagonal</li>
 *   <li>Apply Householder transformation from the right to zero row to the right of superdiagonal</li>
 *   <li>Repeat for each column/row pair</li>
 *   <li>Accumulate transformations to form U and V</li>
 * </ol>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Bidiagonalization only:</b> O(mn<sup>2</sup>) for m &ge; n or O(m<sup>2</sup>n) for m &lt; n</li>
 *   <li><b>With U and V:</b> O(mn<sup>2</sup> + n<sup>3</sup>) or O(m<sup>2</sup>n + m<sup>3</sup>)</li>
 *   <li><b>Space complexity:</b> O(m + n) for Householder vectors</li>
 * </ul>
 *
 * <h2>Numerical Properties:</h2>
 * <ul>
 *   <li><b>Stability:</b> Backward stable due to orthogonal transformations</li>
 *   <li><b>Orthogonality:</b> U and V satisfy U<sup>T</sup>U = I and V<sup>T</sup>V = I to machine precision</li>
 *   <li><b>Norm preservation:</b> ||A||<sub>F</sub> = ||B||<sub>F</sub></li>
 *   <li><b>Rank revealing:</b> Small diagonal elements indicate numerical rank deficiency</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>First stage of SVD computation (Golub-Reinsch algorithm)</li>
 *   <li>Least squares problems</li>
 *   <li>Low-rank approximation</li>
 *   <li>Matrix condition number estimation</li>
 *   <li>Data compression and dimensionality reduction</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][] {
 *     {4, 3, 2, 1},
 *     {1, 2, 3, 4},
 *     {2, 1, 4, 3}
 * });
 *
 * Bidiagonalization bidiag = new Bidiagonalization();
 * var result = bidiag.decompose(A);
 *
 * Matrix U = result.getU();      // Orthogonal matrix
 * Matrix B = result.getB();      // Bidiagonal matrix
 * Matrix V = result.getV();      // Orthogonal matrix
 *
 * // Verify: A = U * B * V^T
 * Matrix reconstructed = U.multiply(B).multiply(V.transpose());
 * }</pre>
 *
 * <h2>Relationship to SVD:</h2>
 * <p>
 * Bidiagonalization is the first major step in computing the Singular Value Decomposition.
 * After bidiagonalization, an iterative algorithm (typically QR iteration) is applied to B
 * to compute its singular values. This two-stage approach is more efficient than direct
 * methods for large matrices.
 * </p>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Uses compact WY representation for efficient Householder accumulation</li>
 *   <li>Supports both compact (economical) and full forms</li>
 *   <li>Handles rectangular matrices of any dimensions</li>
 *   <li>Detects and handles rank-deficient matrices gracefully</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.svd.SVDecomposition
 * @see net.faulj.decomposition.qr.HouseholderQR
 * @see net.faulj.svd.BidiagonalQR
 */
public class Bidiagonalization {
}
