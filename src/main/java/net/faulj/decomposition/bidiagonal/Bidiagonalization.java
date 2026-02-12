package net.faulj.decomposition.bidiagonal;

import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

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
        // First pass: normalize and orthogonalize any meaningful input columns
        for (int i = 0; i < cols; i++) {
            net.faulj.vector.Vector col = thin.getData()[i].copy();
            double norm = col.norm2();
            net.faulj.vector.Vector candidate = (norm > TOL) ? col : null;
            if (candidate != null) {
                candidate = orthogonalize(candidate, accepted);
            }
            double candNorm = candidate == null ? 0.0 : candidate.norm2();
            if (candNorm <= TOL) {
                // Try simple identity fallback to find a usable basis vector
                for (int b = 0; b < rows; b++) {
                    double[] data = new double[rows];
                    data[b] = 1.0;
                    net.faulj.vector.Vector fallback = new net.faulj.vector.Vector(data);
                    fallback = orthogonalize(fallback, accepted);
                    double fallbackNorm = fallback.norm2();
                    if (fallbackNorm > TOL) {
                        candidate = fallback;
                        candNorm = fallbackNorm;
                        break;
                    }
                }
            }
            if (candidate != null && candNorm > TOL) {
                candidate = candidate.multiplyScalar(1.0 / candNorm);
                basis[i] = candidate;
                accepted.add(candidate);
            } else {
                // leave a zero placeholder; will be filled later
                basis[i] = net.faulj.vector.VectorUtils.zero(rows);
            }
        }
        int count = accepted.size();
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
 * <h2>EJML-inspired optimizations:</h2>
 * <ul>
 *   <li>Raw array access instead of Matrix.get/set</li>
 *   <li>SIMD vectorization for Householder applications</li>
 *   <li>Max-element normalization for numerical stability</li>
 *   <li>Efficient skip of zero-norm columns/rows</li>
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
    private static final double SAFE_MIN = Double.MIN_NORMAL;
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int LANE_SIZE = SPECIES.length();
    // LAPACK: use dgebrd-style blocked algorithm for larger matrices
    private static final int BLOCK_THRESHOLD = 128;

    // ThreadLocal workspace to avoid allocations in hot paths
    private static final ThreadLocal<HouseholderWorkspace> HH_WS =
            ThreadLocal.withInitial(HouseholderWorkspace::new);

    /** Enable verbose bidiagonalization runtime diagnostics (temporary) */
    private static final boolean DEBUG = false;

    private static final class HouseholderWorkspace {
        double[] dotBuffer = new double[4096];

        double[] ensureDotBuffer(int size) {
            if (dotBuffer.length < size) {
                dotBuffer = new double[size];
            }
            return dotBuffer;
        }
    }

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
     * Uses raw array access and SIMD for optimal performance.
     *
     * @param A matrix to decompose
     * @return bidiagonalization result
     */
    private BidiagonalizationResult decomposeUpper(Matrix A) {
        int m = A.getRowCount();
        int n = A.getColumnCount();

        // Use raw arrays for faster access
        double[] b = A.copy().getRawData();
        double[] u = new double[m * m];
        double[] v = new double[n * n];

        // Initialize U and V as identity
        for (int i = 0; i < m; i++) {
            u[i * m + i] = 1.0;
        }
        for (int i = 0; i < n; i++) {
            v[i * n + i] = 1.0;
        }

        int limit = Math.min(m, n);

        // Workspace for Householder vectors
        double[] vCol = new double[m];
        double[] vRow = new double[n];

        // Store reflectors to assemble U and V after bidiagonalization
        double[][] leftVecs = new double[limit][];
        double[] leftTaus = new double[limit];
        int[] leftLens = new int[limit];

        double[][] rightVecs = new double[limit][];
        double[] rightTaus = new double[limit];
        int[] rightLens = new int[limit];

        for (int k = 0; k < limit; k++) {
            // ===== Column Householder =====
            int len = m - k;

            // LAPACK dlarfg-style: faster scaled norm computation
            double x0 = b[k * n + k];
            double scale = Math.abs(x0);
            double ssq = 1.0;

            for (int i = 1; i < len; i++) {
                double absxi = Math.abs(b[(k + i) * n + k]);
                if (absxi > scale) {
                    double temp = scale / absxi;
                    ssq = 1.0 + ssq * temp * temp;
                    scale = absxi;
                } else if (absxi > 0.0) {
                    double temp = absxi / scale;
                    ssq += temp * temp;
                }
                vCol[i] = b[(k + i) * n + k];
            }
            vCol[0] = x0;

            double xnorm = scale * Math.sqrt(ssq);
            if (xnorm > SAFE_MIN) {
                double beta = (x0 >= 0.0) ? -xnorm : xnorm;
                double tau = (beta - x0) / beta;
                double invV0 = 1.0 / (x0 - beta);

                if (Math.abs(tau) > SAFE_MIN) {

                    // Build normalized Householder vector
                    vCol[0] = 1.0;
                    for (int i = 1; i < len; i++) {
                        vCol[i] *= invV0;
                    }

                    // Apply Householder from left to B
                    applyHouseholderLeftRaw(b, n, k, k, vCol, len, tau);
                    // Store left reflector for later assembly of U
                    leftVecs[k] = java.util.Arrays.copyOf(vCol, len);
                    leftTaus[k] = tau;
                    leftLens[k] = len;

                    // Set the column to bidiagonal form
                    b[k * n + k] = beta;
                    for (int i2 = 1; i2 < len; i2++) {
                        b[(k + i2) * n + k] = 0.0;
                    }
                }
            }

            // ===== Row Householder =====
            if (k < n - 1) {
                int lenRow = n - k - 1;

                // LAPACK dlarfg-style for row
                double x0Row = b[k * n + k + 1];
                double scaleRow = Math.abs(x0Row);
                double ssqRow = 1.0;

                for (int j = 1; j < lenRow; j++) {
                    double absxj = Math.abs(b[k * n + k + 1 + j]);
                    if (absxj > scaleRow) {
                        double temp = scaleRow / absxj;
                        ssqRow = 1.0 + ssqRow * temp * temp;
                        scaleRow = absxj;
                    } else if (absxj > 0.0) {
                        double temp = absxj / scaleRow;
                        ssqRow += temp * temp;
                    }
                    vRow[j] = b[k * n + k + 1 + j];
                }
                vRow[0] = x0Row;

                double xnormRow = scaleRow * Math.sqrt(ssqRow);
                if (xnormRow > SAFE_MIN) {
                    double betaRow = (x0Row >= 0.0) ? -xnormRow : xnormRow;
                    double tauRow = (betaRow - x0Row) / betaRow;
                    double invV0Row = 1.0 / (x0Row - betaRow);

                    if (Math.abs(tauRow) > SAFE_MIN) {

                        // Build normalized Householder vector
                        vRow[0] = 1.0;
                        for (int j = 1; j < lenRow; j++) {
                            vRow[j] *= invV0Row;
                        }

                        // Apply Householder from right to B
                        applyHouseholderRightRaw(b, n, k, k + 1, vRow, lenRow, tauRow);
                        // Store right reflector for later assembly of V
                        rightVecs[k] = java.util.Arrays.copyOf(vRow, lenRow);
                        rightTaus[k] = tauRow;
                        rightLens[k] = lenRow;

                        // Set the row to bidiagonal form
                        b[k * n + k + 1] = betaRow;
                        for (int j2 = 1; j2 < lenRow; j2++) {
                            b[k * n + k + 1 + j2] = 0.0;
                        }
                    }
                }
            }
        }

        // Assemble U by applying stored left reflectors to identity in reverse order
        for (int i = limit - 1; i >= 0; i--) {
            if (leftVecs[i] != null && Math.abs(leftTaus[i]) > 0.0) {
                applyHouseholderLeftRaw(u, m, i, 0, leftVecs[i], leftLens[i], leftTaus[i]);
            }
        }

        // Assemble V by applying stored right reflectors to identity in forward order
        for (int i = 0; i < limit; i++) {
            if (rightVecs[i] != null && Math.abs(rightTaus[i]) > 0.0) {
                applyHouseholderRightRaw(v, n, 0, i + 1, rightVecs[i], rightLens[i], rightTaus[i]);
            }
        }

        Matrix Umat = Matrix.wrap(u, m, m);
        Matrix Bmat = Matrix.wrap(b, m, n);
        Matrix Vmat = Matrix.wrap(v, n, n);

        return new BidiagonalizationResult(A, Umat, Bmat, Vmat);
    }

    /**
     * Apply Householder reflector from the left using raw arrays.
     * LAPACK dlarfx-style: 8x unrolling for better ILP on column operations.
     */
    private static void applyHouseholderLeftRaw(double[] M, int cols, int startRow, int startCol,
                                                 double[] v, int len, double tau) {
        if (DEBUG) {
            int rows = M.length / cols;
            try {
                System.out.println("applyHouseholderLeftRaw called: startRow=" + startRow + " startCol=" + startCol + " len=" + len + " tau=" + tau);
                System.out.println("v (first " + Math.min(len, 8) + ")=" + java.util.Arrays.toString(java.util.Arrays.copyOf(v, Math.min(len, 8))));
                System.out.println("Before left apply M: " + MatrixUtils.matrixSummary(Matrix.wrap(M, rows, cols), 3, 3));
            } catch (Exception ignored) {
            }
        }
        // LAPACK: process 8 columns at a time for maximum ILP
        int col = startCol;
        int limit = cols - 7;

        for (; col < limit; col += 8) {
            // Compute 8 dot products in parallel
            double dot0 = 0.0, dot1 = 0.0, dot2 = 0.0, dot3 = 0.0;
            double dot4 = 0.0, dot5 = 0.0, dot6 = 0.0, dot7 = 0.0;

            for (int i = 0; i < len; i++) {
                int rowBase = (startRow + i) * cols;
                double vi = v[i];
                dot0 += vi * M[rowBase + col];
                dot1 += vi * M[rowBase + col + 1];
                dot2 += vi * M[rowBase + col + 2];
                dot3 += vi * M[rowBase + col + 3];
                dot4 += vi * M[rowBase + col + 4];
                dot5 += vi * M[rowBase + col + 5];
                dot6 += vi * M[rowBase + col + 6];
                dot7 += vi * M[rowBase + col + 7];
            }

            if (DEBUG) {
                System.out.println(String.format("left dots at col=%d: [%.6e, %.6e, %.6e, %.6e, %.6e, %.6e, %.6e, %.6e]", col, dot0, dot1, dot2, dot3, dot4, dot5, dot6, dot7));
            }

            double scale0 = tau * dot0, scale1 = tau * dot1;
            double scale2 = tau * dot2, scale3 = tau * dot3;
            double scale4 = tau * dot4, scale5 = tau * dot5;
            double scale6 = tau * dot6, scale7 = tau * dot7;

            // Update 8 columns
            for (int i = 0; i < len; i++) {
                int rowBase = (startRow + i) * cols;
                double vi = v[i];
                M[rowBase + col] -= scale0 * vi;
                M[rowBase + col + 1] -= scale1 * vi;
                M[rowBase + col + 2] -= scale2 * vi;
                M[rowBase + col + 3] -= scale3 * vi;
                M[rowBase + col + 4] -= scale4 * vi;
                M[rowBase + col + 5] -= scale5 * vi;
                M[rowBase + col + 6] -= scale6 * vi;
                M[rowBase + col + 7] -= scale7 * vi;
            }
            if (DEBUG) {
                int rows = M.length / cols;
                System.out.println("After left block update M: " + MatrixUtils.matrixSummary(Matrix.wrap(M, rows, cols), 3, 3));
            }
        }

        // Scalar remainder
        for (; col < cols; col++) {
            double dot = 0.0;
            for (int i = 0; i < len; i++) {
                dot += v[i] * M[(startRow + i) * cols + col];
            }
            if (DEBUG) {
                System.out.println(String.format("left dot at col=%d: %.6e", col, dot));
            }
            double scale = tau * dot;
            for (int i = 0; i < len; i++) {
                M[(startRow + i) * cols + col] -= scale * v[i];
            }
            if (DEBUG) {
                int rows = M.length / cols;
                System.out.println("After left scalar update M: " + MatrixUtils.matrixSummary(Matrix.wrap(M, rows, cols), 3, 3));
            }
        }
    }

    /**
     * Apply Householder reflector from the right using raw arrays.
     * Optimized with AVX2 SIMD for contiguous row access.
     */
    private static void applyHouseholderRightRaw(double[] M, int cols, int startRow, int startCol,
                                                  double[] v, int len, double tau) {
        int rows = M.length / cols;
        if (DEBUG) {
            try {
                System.out.println("applyHouseholderRightRaw called: startRow=" + startRow + " startCol=" + startCol + " len=" + len + " tau=" + tau);
                System.out.println("vRow (first " + Math.min(len, 8) + ")=" + java.util.Arrays.toString(java.util.Arrays.copyOf(v, Math.min(len, 8))));
                System.out.println("Before right apply M: " + MatrixUtils.matrixSummary(Matrix.wrap(M, rows, cols), 3, 3));
            } catch (Exception ignored) {
            }
        }
        int vecLen = SPECIES.length();
        int upperBound = SPECIES.loopBound(len);

        for (int row = startRow; row < rows; row++) {
            int rowBase = row * cols;
            int idx = rowBase + startCol;

            // Vectorized dot product: row * v
            double dot = 0.0;
            int j = 0;

            // SIMD loop for dot product
            for (; j < upperBound; j += vecLen) {
                DoubleVector mVec = DoubleVector.fromArray(SPECIES, M, idx + j);
                DoubleVector vVec = DoubleVector.fromArray(SPECIES, v, j);
                dot += mVec.mul(vVec).reduceLanes(VectorOperators.ADD);
            }

            // Scalar remainder
            for (; j < len; j++) {
                dot += M[idx + j] * v[j];
            }

            if (DEBUG) {
                System.out.println(String.format("right dot at row=%d: %.6e", row, dot));
            }

            // Do not skip tiny dots here; perform the update even if dot is near zero
            // to keep behavior symmetric with the left-apply helper and avoid
            // accumulation mismatches for very small couplings.

            double scale = tau * dot;
            DoubleVector scaleVec = DoubleVector.broadcast(SPECIES, scale);

            // Vectorized update: row -= scale * v
            j = 0;
            for (; j < upperBound; j += vecLen) {
                DoubleVector mVec = DoubleVector.fromArray(SPECIES, M, idx + j);
                DoubleVector vVec = DoubleVector.fromArray(SPECIES, v, j);
                mVec.sub(vVec.mul(scaleVec)).intoArray(M, idx + j);
            }

            // Scalar remainder
            for (; j < len; j++) {
                M[idx + j] -= scale * v[j];
            }
            if (DEBUG) {
                System.out.println("After right update (row " + row + ") M: " + MatrixUtils.matrixSummary(Matrix.wrap(M, rows, cols), 3, 3));
            }
        }
        if (DEBUG) {
            System.out.println("applyHouseholderRightRaw complete M: " + MatrixUtils.matrixSummary(Matrix.wrap(M, rows, cols), 3, 3));
        }
    }
}
