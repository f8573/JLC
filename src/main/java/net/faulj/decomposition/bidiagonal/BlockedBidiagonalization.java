package net.faulj.decomposition.bidiagonal;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import net.faulj.compute.BLAS3Kernels;
import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.matrix.Matrix;

/**
 * Blocked Bidiagonalization using Golub-Kahan algorithm with BLAS-3 trailing updates.
 *
 * <h2>Algorithm:</h2>
 * <p>
 * Reduces a matrix A to bidiagonal form B using orthogonal transformations:
 * A = U * B * V^T
 * </p>
 *
 * <h2>Performance:</h2>
 * <p>
 * Uses blocked Householder reflectors and BLAS-3 matrix-matrix operations
 * for trailing updates, achieving significantly better cache utilization
 * than the unblocked algorithm.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 */
public class BlockedBidiagonalization {

    private static final double EPS = 1e-12;
    private static final double TOL = 1e-15;
    private static final int DEFAULT_BLOCK_SIZE = 64;
    private static volatile int BLOCK_SIZE = DEFAULT_BLOCK_SIZE;

    // Vector API SIMD support for 4x-8x speedup on dot products
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    // Workspace buffers to reduce allocation overhead
    private static final ThreadLocal<WorkspaceBuffers> WORKSPACE =
        ThreadLocal.withInitial(WorkspaceBuffers::new);

    private static class WorkspaceBuffers {
        double[] V = new double[2048 * 256];      // Pre-allocated V matrix buffer
        double[] T = new double[256 * 256];       // Pre-allocated T matrix buffer
        double[] W = new double[2048 * 2048];     // Pre-allocated W matrix buffer
        double[] W2 = new double[2048 * 2048];    // Pre-allocated W2 matrix buffer
    }

    /**
     * Set the panel/block size at runtime.
     */
    public static void setBlockSize(int bs) {
        if (bs < 1) {
            throw new IllegalArgumentException("blockSize must be >= 1");
        }
        BLOCK_SIZE = bs;
    }

    public static int getBlockSize() {
        return BLOCK_SIZE;
    }

    /**
     * Compute blocked bidiagonalization.
     */
    public static BidiagonalizationResult decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }

        int m = A.getRowCount();
        int n = A.getColumnCount();

        if (m >= n) {
            return decomposeUpper(A);
        }

        // For m < n, transpose, decompose, then transpose back
        BidiagonalizationResult transposed = decomposeUpper(A.transpose());
        return new BidiagonalizationResult(
            A,
            transposed.getV(),
            transposed.getB().transpose(),
            transposed.getU()
        );
    }

    /**
     * Decompose when m >= n (upper bidiagonal result).
     */
    private static BidiagonalizationResult decomposeUpper(Matrix A) {
        int m = A.getRowCount();
        int n = A.getColumnCount();

        Matrix B = A.copy();
        Matrix U = Matrix.Identity(m);
        Matrix V = Matrix.Identity(n);

        double[] b = B.getRawData();
        double[] u = U.getRawData();
        double[] v = V.getRawData();

        // Adaptive block size selection for better performance
        int blockSize = selectBlockSize(m, n);

        for (int k = 0; k < n; k += blockSize) {
            int panelWidth = Math.min(blockSize, n - k);

            // Factor panel (alternating left/right Householder)
            double[][] VL = new double[panelWidth][];  // Left Householder vectors
            double[][] VR = new double[panelWidth][];  // Right Householder vectors
            double[] tauL = new double[panelWidth];
            double[] tauR = new double[panelWidth];

            factorPanel(b, m, n, k, panelWidth, VL, VR, tauL, tauR);

            // Apply accumulated transformations to trailing matrix using BLAS-3
            if (k + panelWidth < n) {
                applyPanelLeft(b, m, n, k, panelWidth, VL, tauL);
                applyPanelRight(b, m, n, k, panelWidth, VR, tauR);
            }

            // Accumulate transformations into U and V
            accumulateU(u, m, k, panelWidth, VL, tauL);
            accumulateV(v, n, k, panelWidth, VR, tauR);
        }

        // Zero out fill-in
        zeroBidiagonal(b, m, n);

        return new BidiagonalizationResult(A, U, B, V);
    }

    /**
     * Select optimal block size based on matrix dimensions.
     * Larger blocks for larger matrices improve BLAS-3 efficiency.
     */
    private static int selectBlockSize(int m, int n) {
        int minDim = Math.min(m, n);
        if (minDim >= 2048) {
            return 256;  // Very large matrices
        } else if (minDim >= 1024) {
            return 128;  // Large matrices
        } else if (minDim >= 512) {
            return 96;   // Medium matrices
        } else {
            return 64;   // Small to medium matrices
        }
    }

    /**
     * Factor panel with alternating left/right Householder reflections.
     * Optimized with SIMD and inline operations.
     */
    private static void factorPanel(double[] b, int m, int n, int k, int panelWidth,
                                    double[][] VL, double[][] VR,
                                    double[] tauL, double[] tauR) {
        for (int j = 0; j < panelWidth && (k + j) < n; j++) {
            int col = k + j;

            // Left Householder: zero out column below diagonal
            if (col < m) {
                int len = m - col;

                // Scalar norm for STRIDED column access (can't use SIMD efficiently)
                double sum = 0.0;
                for (int i = 0; i < len; i++) {
                    double val = b[(col + i) * n + col];
                    sum += val * val;
                }
                double norm = Math.sqrt(sum);

                if (norm > TOL) {
                    double beta = -Math.copySign(norm, b[col * n + col]);
                    double v0 = b[col * n + col] - beta;

                    if (Math.abs(v0) > EPS) {
                        double tau = 2.0 * v0 * v0 / (norm * norm);
                        tauL[j] = tau;

                        // Build normalized Householder vector
                        double[] v = new double[len];
                        v[0] = 1.0;
                        double invV0 = 1.0 / v0;
                        for (int i = 1; i < len; i++) {
                            v[i] = b[(col + i) * n + col] * invV0;
                        }
                        VL[j] = v;

                        // Apply to remaining columns in panel (SIMD-optimized)
                        applyHouseholderLeftOptimized(b, n, col, col, v, tau, m, n);

                        // Store result
                        b[col * n + col] = beta;
                        for (int i = 1; i < len; i++) {
                            b[(col + i) * n + col] = 0.0;
                        }
                    } else {
                        tauL[j] = 0.0;
                        VL[j] = null;
                    }
                } else {
                    tauL[j] = 0.0;
                    VL[j] = null;
                }
            }

            // Right Householder: zero out row to the right of superdiagonal
            if (col < n - 1 && col < m) {
                int lenRow = n - col - 1;
                int rowStart = col * n + col + 1;

                // Inline SIMD norm computation
                double normRow = norm2SIMD(b, rowStart, lenRow);

                if (normRow > TOL) {
                    double beta = -Math.copySign(normRow, b[rowStart]);
                    double v0 = b[rowStart] - beta;

                    if (Math.abs(v0) > EPS) {
                        double tau = 2.0 * v0 * v0 / (normRow * normRow);
                        tauR[j] = tau;

                        // Build normalized Householder vector
                        double[] v = new double[lenRow];
                        v[0] = 1.0;
                        double invV0 = 1.0 / v0;
                        for (int i = 1; i < lenRow; i++) {
                            v[i] = b[rowStart + i] * invV0;
                        }
                        VR[j] = v;

                        // Apply to remaining rows in panel (SIMD-optimized)
                        applyHouseholderRightOptimized(b, n, col, col + 1, v, tau, m, n);

                        // Store result
                        b[col * n + (col + 1)] = beta;
                        for (int jj = 1; jj < lenRow; jj++) {
                            b[col * n + (col + 1 + jj)] = 0.0;
                        }
                    } else {
                        tauR[j] = 0.0;
                        VR[j] = null;
                    }
                } else {
                    tauR[j] = 0.0;
                    VR[j] = null;
                }
            } else {
                tauR[j] = 0.0;
                VR[j] = null;
            }
        }
    }

    /**
     * Apply left Householder reflections using direct array access.
     */
    private static void applyHouseholderLeft(double[] b, int ldb, int startRow, int startCol,
                                            double[] v, double tau, int m, int n) {
        int len = v.length;
        for (int col = startCol; col < n; col++) {
            double dot = 0.0;
            for (int i = 0; i < len; i++) {
                dot += v[i] * b[(startRow + i) * ldb + col];
            }
            dot *= tau;
            for (int i = 0; i < len; i++) {
                b[(startRow + i) * ldb + col] -= dot * v[i];
            }
        }
    }

    /**
     * Apply right Householder reflections using direct array access.
     */
    private static void applyHouseholderRight(double[] b, int ldb, int startRow, int startCol,
                                             double[] v, double tau, int m, int n) {
        int len = v.length;
        for (int row = startRow; row < m; row++) {
            int rowOffset = row * ldb;
            double dot = 0.0;
            for (int j = 0; j < len; j++) {
                dot += v[j] * b[rowOffset + startCol + j];
            }
            dot *= tau;
            for (int j = 0; j < len; j++) {
                b[rowOffset + startCol + j] -= dot * v[j];
            }
        }
    }

    /**
     * Apply panel of left Householder reflections to trailing matrix using BLAS-3.
     * Computes: B[k:m, k+panelWidth:n] -= V * T * V^T * B[k:m, k+panelWidth:n]
     */
    private static void applyPanelLeft(double[] b, int m, int n, int k, int panelWidth,
                                      double[][] VL, double[] tauL) {
        // Count non-null reflectors
        int actualCount = 0;
        for (int j = 0; j < panelWidth; j++) {
            if (VL[j] != null && tauL[j] > 0) {
                actualCount++;
            }
        }
        if (actualCount == 0) return;

        // Build compact V matrix (m_trail × actualCount) stored column-major
        int rowStart = k;
        int m_trail = m - rowStart;
        int colStart = k + panelWidth;
        int nCols = n - colStart;

        if (m_trail <= 0 || nCols <= 0) return;

        // For small trailing blocks, use sequential application (less overhead)
        if (actualCount < 4 || nCols < 32) {
            for (int j = 0; j < panelWidth; j++) {
                if (VL[j] != null && tauL[j] > 0) {
                    int col = k + j;
                    applyHouseholderLeftOptimized(b, n, col, k + panelWidth, VL[j], tauL[j], m, n);
                }
            }
            return;
        }

        // Use pre-allocated workspace buffers
        WorkspaceBuffers ws = WORKSPACE.get();
        int vSize = m_trail * actualCount;
        int tSize = actualCount * actualCount;
        int wSize = actualCount * nCols;

        // Ensure buffers are large enough
        if (vSize > ws.V.length) {
            ws.V = new double[vSize * 2];
        }
        if (tSize > ws.T.length) {
            ws.T = new double[tSize * 2];
        }
        if (wSize > ws.W.length) {
            ws.W = new double[wSize * 2];
            ws.W2 = new double[wSize * 2];
        }

        double[] V = ws.V;
        double[] T = ws.T;
        double[] W = ws.W;
        double[] W2 = ws.W2;

        int vIdx = 0;
        for (int j = 0; j < panelWidth; j++) {
            if (VL[j] == null || tauL[j] <= 0) continue;

            int col = k + j;
            int vStart = col - rowStart;
            int vLen = VL[j].length;

            // Copy Householder vector into V matrix (optimized loop)
            int baseIdx = vIdx;
            for (int i = 0; i < vStart; i++) {
                V[i * actualCount + baseIdx] = 0.0;
            }
            for (int i = 0; i < vLen; i++) {
                V[(vStart + i) * actualCount + baseIdx] = VL[j][i];
            }
            for (int i = vStart + vLen; i < m_trail; i++) {
                V[i * actualCount + baseIdx] = 0.0;
            }

            vIdx++;
        }

        // Build upper triangular T matrix
        buildT(V, m_trail, actualCount, tauL, panelWidth, T);

        // Apply blocked update: B -= V * T * V^T * B
        // Step 1: W = V^T * B[rowStart:m, colStart:n]
        BLAS3Kernels.gemmStridedTransA(
            V, 0, actualCount,
            b, rowStart * n + colStart, n,
            W, 0, nCols,
            m_trail, actualCount, nCols,
            1.0, 0.0, actualCount);

        // Step 2: W2 = T * W
        BLAS3Kernels.gemmStrided(
            T, 0, actualCount,
            W, 0, nCols,
            W2, 0, nCols,
            actualCount, actualCount, nCols,
            1.0, 0.0, actualCount);

        // Step 3: B -= V * W2
        BLAS3Kernels.gemmStrided(
            V, 0, actualCount,
            W2, 0, nCols,
            b, rowStart * n + colStart, n,
            m_trail, actualCount, nCols,
            -1.0, 1.0, actualCount);
    }

    /**
     * SIMD-optimized Householder for LEFT updates (column-wise).
     * Uses BLAS-2 rank-1 update: B -= tau * v * (v^T * B)
     * Scalar version for strided column access in row-major matrix.
     */
    private static void applyHouseholderLeftOptimized(double[] b, int ldb, int startRow,
                                                     int startCol, double[] v, double tau,
                                                     int m, int n) {
        int len = v.length;
        int endCol = Math.min(n, startCol + 256);
        int nCols = endCol - startCol;

        // Step 1: Compute all dot products (v^T * B) - scalar for column access
        double[] vTB = new double[nCols];
        for (int c = 0; c < nCols; c++) {
            int col = startCol + c;
            double dot = 0.0;
            for (int i = 0; i < len; i++) {
                dot += v[i] * b[(startRow + i) * ldb + col];
            }
            vTB[c] = tau * dot;
        }

        // Step 2: Rank-1 update B -= v * vTB^T - scalar for column access
        for (int c = 0; c < nCols; c++) {
            int col = startCol + c;
            double factor = vTB[c];
            for (int i = 0; i < len; i++) {
                b[(startRow + i) * ldb + col] -= factor * v[i];
            }
        }
    }

    /**
     * SIMD-optimized Householder for RIGHT updates (row-wise).
     * Uses BLAS-2 rank-1 update: B -= (B * v) * tau * v^T
     * SIMD version - contiguous row access in row-major matrix!
     */
    private static void applyHouseholderRightOptimized(double[] b, int ldb, int startRow,
                                                      int startCol, double[] v, double tau,
                                                      int m, int n) {
        int len = v.length;
        int endRow = Math.min(m, startRow + 256);
        int nRows = endRow - startRow;

        // Step 1: Compute all dot products (B * v) using SIMD (contiguous row access!)
        double[] Bv = new double[nRows];
        for (int r = 0; r < nRows; r++) {
            int row = startRow + r;
            int rowOffset = row * ldb + startCol;

            // SIMD dot product on contiguous row data
            double dot = 0.0;
            int vectorLoopBound = SPECIES.loopBound(len);

            int j = 0;
            for (; j < vectorLoopBound; j += SPECIES.length()) {
                DoubleVector vVec = DoubleVector.fromArray(SPECIES, v, j);
                DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, rowOffset + j);
                dot += vVec.mul(bVec).reduceLanes(VectorOperators.ADD);
            }

            // Scalar tail
            for (; j < len; j++) {
                dot += v[j] * b[rowOffset + j];
            }

            Bv[r] = tau * dot;
        }

        // Step 2: Rank-1 update B -= Bv * v^T using SIMD (contiguous row access!)
        for (int r = 0; r < nRows; r++) {
            int row = startRow + r;
            int rowOffset = row * ldb + startCol;
            double factor = Bv[r];

            // SIMD axpy on contiguous row data
            int vectorLoopBound = SPECIES.loopBound(len);
            int j = 0;

            for (; j < vectorLoopBound; j += SPECIES.length()) {
                DoubleVector vVec = DoubleVector.fromArray(SPECIES, v, j);
                DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, rowOffset + j);
                DoubleVector result = bVec.sub(vVec.mul(factor));
                result.intoArray(b, rowOffset + j);
            }

            // Scalar tail
            for (; j < len; j++) {
                b[rowOffset + j] -= factor * v[j];
            }
        }
    }

    /**
     * Apply panel of right Householder reflections to trailing matrix using BLAS-3.
     * Computes: B[k:m, k+panelWidth:n] -= B[k:m, k+panelWidth:n] * V * T * V^T
     */
    private static void applyPanelRight(double[] b, int m, int n, int k, int panelWidth,
                                       double[][] VR, double[] tauR) {
        // For right updates, we work on B[k:m, k+panelWidth:n]
        int rowStart = k;
        int nRows = m - rowStart;
        int colStart = k + panelWidth;
        int n_trail = n - colStart;

        if (nRows <= 0 || n_trail <= 0) return;

        // Count reflectors that actually affect the trailing matrix
        int actualCount = 0;
        for (int j = 0; j < panelWidth; j++) {
            int col = k + j;
            int vrStart = col + 1;
            if (VR[j] != null && tauR[j] > 0 && vrStart <= colStart && col < m) {
                actualCount++;
            }
        }
        if (actualCount == 0) return;

        // Use pre-allocated workspace buffers
        WorkspaceBuffers ws = WORKSPACE.get();
        int vSize = n_trail * actualCount;
        int tSize = actualCount * actualCount;
        int wSize = nRows * actualCount;

        // Ensure buffers are large enough
        if (vSize > ws.V.length) {
            ws.V = new double[vSize * 2];
        }
        if (tSize > ws.T.length) {
            ws.T = new double[tSize * 2];
        }
        if (wSize > ws.W.length) {
            ws.W = new double[wSize * 2];
            ws.W2 = new double[wSize * 2];
        }

        double[] V = ws.V;
        double[] T = ws.T;
        double[] W = ws.W;
        double[] W2 = ws.W2;

        int vIdx = 0;
        for (int j = 0; j < panelWidth; j++) {
            int col = k + j;
            int vrStart = col + 1;
            if (VR[j] == null || tauR[j] <= 0 || vrStart > colStart || col >= m) continue;

            // VR[j] starts at column vrStart = k+j+1
            int vStart = Math.max(0, vrStart - colStart);
            int vLen = VR[j].length;

            // Copy Householder vector into V matrix (optimized)
            int overlap = Math.min(vLen, n_trail - vStart);
            int srcStart = Math.max(0, colStart - vrStart);

            for (int i = 0; i < vStart; i++) {
                V[i * actualCount + vIdx] = 0.0;
            }

            for (int i = 0; i < overlap && srcStart + i < vLen; i++) {
                V[(vStart + i) * actualCount + vIdx] = VR[j][srcStart + i];
            }

            for (int i = vStart + overlap; i < n_trail; i++) {
                V[i * actualCount + vIdx] = 0.0;
            }

            vIdx++;
        }

        if (vIdx == 0) return;

        // Build upper triangular T matrix
        buildTRight(V, n_trail, vIdx, tauR, panelWidth, T);

        // Apply blocked update: B -= B * V * T * V^T
        // Step 1: W = B[rowStart:m, colStart:n] * V
        BLAS3Kernels.gemmStrided(
            b, rowStart * n + colStart, n,
            V, 0, vIdx,
            W, 0, vIdx,
            nRows, n_trail, vIdx,
            1.0, 0.0, vIdx);

        // Step 2: W2 = W * T
        BLAS3Kernels.gemmStrided(
            W, 0, vIdx,
            T, 0, vIdx,
            W2, 0, vIdx,
            nRows, vIdx, vIdx,
            1.0, 0.0, vIdx);

        // Step 3: B -= W2 * V^T
        BLAS3Kernels.gemmStridedColMajorB(
            W2, 0, vIdx,
            V, 0, vIdx,
            b, rowStart * n + colStart, n,
            nRows, vIdx, n_trail,
            -1.0, 1.0, vIdx);
    }

    /**
     * Accumulate left transformations into U.
     */
    private static void accumulateU(double[] u, int m, int k, int panelWidth,
                                    double[][] VL, double[] tauL) {
        for (int j = panelWidth - 1; j >= 0; j--) {
            if (VL[j] != null && tauL[j] > 0) {
                int col = k + j;
                applyHouseholderRight(u, m, 0, col, VL[j], tauL[j], m, m);
            }
        }
    }

    /**
     * Accumulate right transformations into V.
     */
    private static void accumulateV(double[] v, int n, int k, int panelWidth,
                                    double[][] VR, double[] tauR) {
        for (int j = panelWidth - 1; j >= 0; j--) {
            if (VR[j] != null && tauR[j] > 0) {
                int col = k + j + 1;
                if (col < n) {
                    applyHouseholderRight(v, n, 0, col, VR[j], tauR[j], n, n);
                }
            }
        }
    }

    /**
     * Zero out everything except diagonal and superdiagonal.
     */
    private static void zeroBidiagonal(double[] b, int m, int n) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j < i || j > i + 1) {
                    b[i * n + j] = 0.0;
                }
            }
        }
    }

    /**
     * Build upper triangular T matrix for compact WY representation (left reflections).
     * T is upper triangular with T[i,i] = tau[i].
     * For j < i: T[j,i] = -tau[i] * V[:,j]^T * V[:,i] * sum(T[j,k] * work[k], k=j..i-1)
     */
    private static void buildT(double[] V, int m, int k, double[] tau, int tauLen, double[] T) {
        for (int i = 0; i < k * k; i++) {
            T[i] = 0.0;
        }

        double[] work = new double[k];

        for (int i = 0; i < k; i++) {
            // Find corresponding non-zero tau
            int tauSrc = -1;
            int count = 0;
            for (int t = 0; t < tauLen; t++) {
                if (tau[t] > 0) {
                    if (count == i) {
                        tauSrc = t;
                        break;
                    }
                    count++;
                }
            }

            if (tauSrc < 0 || tau[tauSrc] == 0.0) {
                T[i * k + i] = 0.0;
                continue;
            }

            double tauI = tau[tauSrc];

            // Compute work[j] = -tau[i] * V[:,j]^T * V[:,i]
            for (int j = 0; j < i; j++) {
                double dot = 0.0;
                for (int r = 0; r < m; r++) {
                    dot += V[r * k + j] * V[r * k + i];
                }
                work[j] = -tauI * dot;
            }

            // Compute T[j,i] = sum(T[j,l] * work[l], l=j..i-1)
            for (int j = 0; j < i; j++) {
                double sum = 0.0;
                for (int l = j; l < i; l++) {
                    sum += T[j * k + l] * work[l];
                }
                T[j * k + i] = sum;
            }

            T[i * k + i] = tauI;
        }
    }

    /**
     * Build upper triangular T matrix for compact WY representation (right reflections).
     */
    private static void buildTRight(double[] V, int n, int k, double[] tau, int tauLen, double[] T) {
        // Same algorithm as buildT, just different interpretation
        buildT(V, n, k, tau, tauLen, T);
    }

    /**
     * SIMD-optimized dot product using Vector API.
     * Processes 4-8 elements at once for 3-5x speedup.
     */
    private static double dotSIMD(double[] v, int vOffset, double[] b, int bOffset, int len, int stride) {
        int vectorLoopBound = SPECIES.loopBound(len);
        double sum = 0.0;

        int i = 0;
        // Vector loop: process SPECIES.length() elements at once
        for (; i < vectorLoopBound; i += SPECIES.length()) {
            DoubleVector vVec = DoubleVector.fromArray(SPECIES, v, vOffset + i);

            // Load from b with stride
            double[] bTemp = new double[SPECIES.length()];
            for (int j = 0; j < SPECIES.length(); j++) {
                bTemp[j] = b[bOffset + i * stride + j * stride];
            }
            DoubleVector bVec = DoubleVector.fromArray(SPECIES, bTemp, 0);

            sum += vVec.mul(bVec).reduceLanes(VectorOperators.ADD);
        }

        // Scalar tail loop
        for (; i < len; i++) {
            sum += v[vOffset + i] * b[bOffset + i * stride];
        }

        return sum;
    }

    /**
     * SIMD-optimized axpy: y = y - alpha * x
     * Processes 4-8 elements at once for 3-5x speedup.
     */
    private static void axpySIMD(double alpha, double[] x, int xOffset, double[] y, int yOffset, int len, int stride) {
        int vectorLoopBound = SPECIES.loopBound(len);

        int i = 0;
        // Vector loop
        for (; i < vectorLoopBound; i += SPECIES.length()) {
            DoubleVector xVec = DoubleVector.fromArray(SPECIES, x, xOffset + i);

            // Load from y with stride
            double[] yTemp = new double[SPECIES.length()];
            for (int j = 0; j < SPECIES.length(); j++) {
                yTemp[j] = y[yOffset + i * stride + j * stride];
            }
            DoubleVector yVec = DoubleVector.fromArray(SPECIES, yTemp, 0);

            // Compute y = y - alpha * x
            DoubleVector result = yVec.sub(xVec.mul(alpha));

            // Store back with stride
            double[] resultArray = result.toArray();
            for (int j = 0; j < SPECIES.length(); j++) {
                y[yOffset + i * stride + j * stride] = resultArray[j];
            }
        }

        // Scalar tail loop
        for (; i < len; i++) {
            y[yOffset + i * stride] -= alpha * x[xOffset + i];
        }
    }

    /**
     * SIMD-optimized norm computation: inline and vectorized.
     */
    private static double norm2SIMD(double[] x, int offset, int len) {
        int vectorLoopBound = SPECIES.loopBound(len);
        double sum = 0.0;

        int i = 0;
        // Vector loop
        for (; i < vectorLoopBound; i += SPECIES.length()) {
            DoubleVector vVec = DoubleVector.fromArray(SPECIES, x, offset + i);
            sum += vVec.mul(vVec).reduceLanes(VectorOperators.ADD);
        }

        // Scalar tail loop
        for (; i < len; i++) {
            double val = x[offset + i];
            sum += val * val;
        }

        return Math.sqrt(sum);
    }

    /**
     * Compute Euclidean norm using direct array access.
     */
    private static double norm2(double[] x) {
        double sum = 0.0;
        for (double v : x) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }
}
