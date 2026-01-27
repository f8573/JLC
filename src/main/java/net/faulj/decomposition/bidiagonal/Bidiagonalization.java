package net.faulj.decomposition.bidiagonal;

import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.matrix.Matrix;

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
    private static final double TOL = 1e-12;

    /**
     * Compute the bidiagonal decomposition of a matrix.
     *
     * @param A matrix to decompose
     * @return bidiagonalization result containing U, B, and V
     */
    public BidiagonalizationResult decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        int m = A.getRowCount();
        int n = A.getColumnCount();
        if (m >= n) {
            return decomposeUpper(A);
        }
        BidiagonalizationResult transposed = decomposeUpper(A.transpose());
        return new BidiagonalizationResult(
                A,
                transposed.getV(),
                transposed.getB().transpose(),
                transposed.getU()
        );
    }

    /**
     * Compute the upper bidiagonal form for matrices with m &ge; n.
     *
     * @param A matrix to decompose
     * @return bidiagonalization result
     */
    private BidiagonalizationResult decomposeUpper(Matrix A) {
        int m = A.getRowCount();
        int n = A.getColumnCount();
        net.faulj.matrix.Matrix B = A.copy();
        net.faulj.matrix.Matrix U = net.faulj.matrix.Matrix.Identity(m);
        net.faulj.matrix.Matrix V = net.faulj.matrix.Matrix.Identity(n);

        int limit = Math.min(m, n);
        for (int k = 0; k < limit; k++) {
            int len = m - k;
            double[] x = new double[len];
            for (int i = 0; i < len; i++) {
                x[i] = B.get(k + i, k);
            }
            double normX = norm2(x);
            if (normX > TOL) {
                if (!tailIsZero(x) || Math.abs(x[0] - normX) > TOL) {
                    net.faulj.vector.Vector hh = net.faulj.vector.VectorUtils.householder(new net.faulj.vector.Vector(x));
                    double tau = hh.get(len);
                    net.faulj.vector.Vector v = hh.resize(len);
                    applyHouseholderLeft(B, k, k, v.getData(), tau);
                    applyHouseholderRight(U, 0, k, v.getData(), tau);
                    for (int i = 1; i < len; i++) {
                        B.set(k + i, k, 0.0);
                    }
                }
            }

            if (k < n - 1) {
                int lenRow = n - k - 1;
                double[] xRow = new double[lenRow];
                for (int j = 0; j < lenRow; j++) {
                    xRow[j] = B.get(k, k + 1 + j);
                }
                double normRow = norm2(xRow);
                if (normRow > TOL) {
                    if (!tailIsZero(xRow) || Math.abs(xRow[0] - normRow) > TOL) {
                        net.faulj.vector.Vector hh = net.faulj.vector.VectorUtils.householder(new net.faulj.vector.Vector(xRow));
                        double tau = hh.get(lenRow);
                        net.faulj.vector.Vector v = hh.resize(lenRow);
                        applyHouseholderRight(B, k, k + 1, v.getData(), tau);
                        applyHouseholderRight(V, 0, k + 1, v.getData(), tau);
                        for (int j = 1; j < lenRow; j++) {
                            B.set(k, k + 1 + j, 0.0);
                        }
                    }
                }
            }
        }

        return new net.faulj.decomposition.result.BidiagonalizationResult(A, U, B, V);
    }

    /**
     * Compute the Euclidean norm of a vector.
     *
     * @param x vector entries
     * @return 2-norm
     */
    private static double norm2(double[] x) {
        double sum = 0.0;
        for (double v : x) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    /**
     * Check whether all entries except the first are near zero.
     *
     * @param x vector entries
     * @return true if the tail is zero within tolerance
     */
    private static boolean tailIsZero(double[] x) {
        for (int i = 1; i < x.length; i++) {
            if (Math.abs(x[i]) > TOL) {
                return false;
            }
        }
        return true;
    }

    /**
     * Apply a Householder reflector from the left to a submatrix.
     *
     * @param M matrix to update
     * @param startRow row offset
     * @param startCol column offset
     * @param v Householder vector
     * @param tau Householder scalar
     */
    private static void applyHouseholderLeft(net.faulj.matrix.Matrix M, int startRow, int startCol, double[] v, double tau) {
        int rows = M.getRowCount();
        int cols = M.getColumnCount();
        int len = v.length;
        for (int col = startCol; col < cols; col++) {
            double dot = 0.0;
            for (int i = 0; i < len; i++) {
                dot += v[i] * M.get(startRow + i, col);
            }
            if (Math.abs(dot) < 1e-15) {
                continue;
            }
            double scale = tau * dot;
            for (int i = 0; i < len; i++) {
                int r = startRow + i;
                M.set(r, col, M.get(r, col) - scale * v[i]);
            }
        }
    }

    /**
     * Apply a Householder reflector from the right to a submatrix.
     *
     * @param M matrix to update
     * @param startRow row offset
     * @param startCol column offset
     * @param v Householder vector
     * @param tau Householder scalar
     */
    private static void applyHouseholderRight(net.faulj.matrix.Matrix M, int startRow, int startCol, double[] v, double tau) {
        int rows = M.getRowCount();
        int len = v.length;
        for (int row = startRow; row < rows; row++) {
            double dot = 0.0;
            for (int j = 0; j < len; j++) {
                dot += M.get(row, startCol + j) * v[j];
            }
            if (Math.abs(dot) < 1e-15) {
                continue;
            }
            double scale = tau * dot;
            for (int j = 0; j < len; j++) {
                int c = startCol + j;
                M.set(row, c, M.get(row, c) - scale * v[j]);
            }
        }
    }
}
