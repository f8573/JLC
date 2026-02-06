package net.faulj.eigen.schur;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.qr.ImplicitQRFrancis;
import net.faulj.matrix.Matrix;

/**
 * Computes the Real Schur Decomposition of a square real matrix.
 * <p>
 * This class factorizes a general square matrix A into the form:
 * </p>
 * <pre>
 * A = U T U<sup>T</sup>
 * </pre>
 * <p>
 * where:
 * </p>
 * <ul>
 * <li><b>U</b> is an orthogonal matrix (U<sup>T</sup>U = I) containing the Schur vectors.</li>
 * <li><b>T</b> is a quasi-upper triangular matrix (Real Schur form).</li>
 * </ul>
 *
 * <h2>Real Schur Form:</h2>
 * <p>
 * The matrix T has a block upper-triangular structure where the diagonal blocks are either:
 * </p>
 * <ul>
 * <li><b>1×1 blocks:</b> Corresponding to real eigenvalues.</li>
 * <li><b>2×2 blocks:</b> Corresponding to complex conjugate eigenvalue pairs.</li>
 * </ul>
 *
 * <h2>Algorithm:</h2>
 * <p>
 * The decomposition is performed in two main stages:
 * </p>
 * <ol>
 * <li><b>Hessenberg Reduction:</b> The matrix A is reduced to upper Hessenberg form H
 * using orthogonal similarity transformations (Householder reflections).
 * <br><i>Cost:</i> ~10n³/3 flops.</li>
 * <li><b>Implicit QR Iteration:</b> The Francis double-shift QR algorithm is applied to H
 * to iteratively reduce subdiagonal elements to zero (or negligible size), converging to T.
 * <br><i>Cost:</i> ~10n³ to 20n³ flops depending on convergence.</li>
 * </ol>
 *
 * <h2>Numerical Properties:</h2>
 * <ul>
 * <li><b>Stability:</b> The algorithm is backward stable, meaning the computed decomposition corresponds
 * exacty to a matrix A + E where ||E|| is small (close to machine epsilon).</li>
 * <li><b>Robustness:</b> Uses implicit shifts and deflation to handle multiple and clustered eigenvalues.</li>
 * <li><b>Orthogonality:</b> The matrix U is formed by accumulating orthogonal transformations,
 * guaranteeing orthogonality to machine precision.</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = ...; // Any square matrix
 *
 * // Compute decomposition
 * SchurResult result = RealSchurDecomposition.decompose(A);
 *
 * Matrix T = result.getT();
 * Matrix U = result.getU();
 *
 * // Verify A ≈ U * T * U^T
 * Matrix reconstructed = U.multiply(T).multiply(U.transpose());
 *
 * // Inspect eigenvalues
 * double[] realEigs = result.getRealEigenvalues();
 * double[] imagEigs = result.getImagEigenvalues();
 * }</pre>
 *
 * <h2>Relation to Eigendecomposition:</h2>
 * <p>
 * While the Eigendecomposition A = QΛQ⁻¹ may involve a non-orthogonal Q (which can be ill-conditioned),
 * the Schur decomposition always uses an orthogonal U. This makes it the preferred method for numerical
 * tasks requiring invariant subspaces, such as solving Riccati equations or computing matrix functions.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see SchurResult
 * @see net.faulj.decomposition.hessenberg.HessenbergReduction
 * @see net.faulj.eigen.qr.ImplicitQRFrancis
 */
public class RealSchurDecomposition {

    /**
     * Computes the Real Schur Decomposition of a square matrix A.
     * <p>
     * This method is the primary entry point for the decomposition. It orchestrates the
     * Hessenberg reduction and the subsequent QR iteration.
     * </p>
     *
     * @param A The square real matrix to decompose. Must not be null.
     * @return The {@link SchurResult} containing matrices T, U, and the extracted eigenvalues.
     * @throws IllegalArgumentException If the matrix A is not square.
     * @throws ArithmeticException If the QR algorithm fails to converge (rare).
     */
    public static SchurResult decompose(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Schur decomposition requires a square matrix.");
        }

        // Use ImplicitQRFrancis with EJML-style implicit double-shift QR
        // This is O(n³) vs O(n⁴) for explicit QR iteration
        return ImplicitQRFrancis.decompose(A);
    }

    /**
     * Computes just the Schur form T without returning the full result.
     * Useful for benchmarking to reduce allocation overhead.
     *
     * @param A The square real matrix to decompose.
     * @return The quasi-upper triangular Schur form T.
     */
    public static Matrix schurT(Matrix A) {
        SchurResult result = decompose(A);
        return result.getT();
    }
}
