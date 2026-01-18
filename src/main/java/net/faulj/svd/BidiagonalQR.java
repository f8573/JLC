package net.faulj.svd;

/**
 * Performs the SVD step of reducing a real bidiagonal matrix to diagonal form.
 * <p>
 * This algorithm is the core iterative phase of the SVD computation. It processes a
 * bidiagonal matrix B (obtained from the initial Golub-Kahan reduction) and drives
 * the superdiagonal elements to zero.
 * </p>
 *
 * <h2>Algorithm:</h2>
 * <p>
 * The class implements the Golub-Kahan SVD step (Implicit QR iteration with Wilkinson shift):
 * </p>
 * <ol>
 * <li>Determine a shift λ based on the bottom 2×2 submatrix (Wilkinson shift)</li>
 * <li>Apply Givens rotations to "chase the bulge" created by the shift down the diagonal</li>
 * <li>Update singular vectors U and V by accumulating rotations</li>
 * <li>Repeat until the superdiagonal element B[n-2, n-1] is negligible</li>
 * <li>Deflate the matrix and repeat for the upper (n-1)×(n-1) block</li>
 * </ol>
 *
 * <h2>Convergence:</h2>
 * <ul>
 * <li><b>Rate:</b> Cubic convergence (usually 2-3 iterations per singular value)</li>
 * <li><b>Stability:</b> Guaranteed convergence for real matrices</li>
 * <li><b>Precision:</b> Computes singular values to full machine precision</li>
 * </ul>
 *
 * <h2>Input Form:</h2>
 * <pre>
 * B = ┌ d₀  e₀  0   0 ┐
 * │ 0   d₁  e₁  0 │
 * │ 0   0   d₂  e₂│
 * └ 0   0   0   d₃┘
 * </pre>
 *
 * <h2>Internal Usage:</h2>
 * <p>
 * This is a low-level utility class used by {@link SVDecomposition}. It operates directly
 * on arrays for performance reasons.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see SVDecomposition
 * @see net.faulj.decomposition.bidiagonal.Bidiagonalization
 */
public class BidiagonalQR {
}