package net.faulj.polar;

import net.faulj.decomposition.result.PolarResult;
import net.faulj.matrix.Matrix;
import net.faulj.svd.ThinSVD;
import net.faulj.decomposition.result.SVDResult;
import net.faulj.nativeblas.AlgorithmBackend;
import net.faulj.nativeblas.NativeAlgorithmScope;

/**
 * Computes the Polar Decomposition of a matrix using Singular Value Decomposition (SVD).
 * <p>
 * The polar decomposition factorizes a matrix <b>A</b> into the product of an
 * orthogonal matrix <b>U</b> and a positive semi-definite matrix <b>P</b>:
 * </p>
 * <pre>
 * A = U * P
 * </pre>
 * <ul>
 * <li><b>U (Unitary/Orthogonal):</b> Represents a rotation or reflection (isometry).</li>
 * <li><b>P (Positive Semi-Definite):</b> Represents a scaling or stretching deformation along orthogonal axes.</li>
 * </ul>
 *
 * <h2>Algorithm:</h2>
 * <p>
 * This implementation computes the decomposition via the Singular Value Decomposition (SVD) of A.
 * Given the SVD of A:
 * </p>
 * <pre>
 * A = W * Σ * V<sup>T</sup>
 * </pre>
 * <p>
 * The polar factors are constructed as:
 * </p>
 * <ul>
 * <li><b>U</b> = W * V<sup>T</sup></li>
 * <li><b>P</b> = V * Σ * V<sup>T</sup></li>
 * </ul>
 * <p>
 * This method is known as the "direct" or "SVD-based" method. It is numerically stable
 * and provides the closest unitary approximation to A (in the Frobenius norm sense).
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create a matrix
 * Matrix A = new Matrix(new double[][] {
 * {1, 2},
 * {3, 4}
 * });
 *
 * // Compute decomposition
 * PolarDecomposition polar = new PolarDecomposition();
 * PolarResult result = polar.decompose(A);
 *
 * // Extract factors
 * Matrix U = result.getU(); // Rotation
 * Matrix P = result.getP(); // Stretch
 *
 * // Verify A = UP
 * Matrix reconstructed = U.multiply(P);
 * }</pre>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li><b>Existence:</b> Exists for any square or rectangular matrix.</li>
 * <li><b>Uniqueness:</b>
 * <ul>
 * <li><b>P</b> is always unique.</li>
 * <li><b>U</b> is unique if A is invertible (full rank).</li>
 * </ul>
 * </li>
 * <li><b>Optimality:</b> The factor U is the solution to the orthogonal Procrustes problem:
 * <br><i>min || A - Q ||<sub>F</sub></i> subject to <i>Q<sup>T</sup>Q = I</i>.</li>
 * </ul>
 *
 * <h2>Computational Complexity:</h2>
 * <p>
 * The cost is dominated by the SVD computation:
 * </p>
 * <ul>
 * <li><b>Time:</b> O(min(m,n) * mn) flops. For a square n×n matrix, approximately O(20n³).</li>
 * <li><b>Space:</b> O(mn) to store the result matrices.</li>
 * </ul>
 *
 * <h2>Numerical Stability:</h2>
 * <p>
 * The SVD-based approach is backward stable. The computed factors U and P satisfy
 * <i>A = UP + E</i> with ||E|| ≈ ε||A||. The orthogonality of U is guaranteed to
 * machine precision.
 * </p>
 *
 * <h2>Alternative Methods:</h2>
 * <p>
 * While this class uses SVD, other methods exist (e.g., Newton iteration for the matrix sign function).
 * The SVD method is generally preferred for its robustness, especially for singular or
 * near-singular matrices.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.result.PolarResult
 * @see net.faulj.svd.SVDecomposition
 */
public class PolarDecomposition {

    /**
     * Decomposes the given matrix A into the product of an orthogonal matrix U
     * and a positive semi-definite matrix P.
     *
     * @param A The matrix to decompose. Must not be null.
     * @return The resulting {@link PolarResult} containing matrices U and P.
     * @throws IllegalArgumentException if the matrix dimensions are invalid or memory allocation fails.
     */
    public static PolarResult decompose(Matrix A) {
        AlgorithmBackend svdBackend = nativeSvdBackend(A);
        SVDResult svd = NativeAlgorithmScope.withOverride("svd", svdBackend, () -> new ThinSVD().decompose(A));
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        int rank = Math.min(rows, cols);

        Matrix W = thinColumns(svd.getU(), rows, rank);
        Matrix V = thinColumns(svd.getV(), cols, rank);
        Matrix Sigma = diagonal(svd.getSingularValues(), rank);
        Matrix P = V.multiply(Sigma).multiply(V.transpose());
        Matrix U = W.multiply(V.transpose());
        return new PolarResult(A, U, P);
    }

    private static Matrix thinColumns(Matrix factor, int expectedRows, int rank) {
        if (rank == 0) {
            return new Matrix(expectedRows, 0);
        }
        return factor.getColumnCount() == rank
            ? factor
            : factor.crop(0, expectedRows - 1, 0, rank - 1);
    }

    private static Matrix diagonal(double[] singularValues, int rank) {
        Matrix sigma = new Matrix(rank, rank);
        int diag = Math.min(rank, singularValues.length);
        for (int i = 0; i < diag; i++) {
            sigma.set(i, i, singularValues[i]);
        }
        return sigma;
    }

    private static AlgorithmBackend nativeSvdBackend(Matrix A) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        int threads = defaultThreadCount();
        boolean useCpp = net.faulj.nativeblas.BackendRegistry.shouldUseCppForAlgorithm("polar", "decompose", rows, cols, threads);
        return useCpp ? AlgorithmBackend.CPP : AlgorithmBackend.JAVA;
    }

    private static int defaultThreadCount() {
        net.faulj.compute.DispatchPolicy policy = net.faulj.compute.DispatchPolicy.defaultPolicy();
        return policy.isParallelEnabled() ? policy.getParallelism() : 1;
    }
}
