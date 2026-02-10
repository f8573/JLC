package net.faulj.eigen.qr;

import net.faulj.kernels.gemm.Gemm;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.matrix.Matrix;

/**
 * Implements Blocked Hessenberg Reduction.
 * <p>
 * Reduces a general square matrix A to Upper Hessenberg form H using orthogonal
 * similarity transformations: H = Q<sup>T</sup>AQ.
 * </p>
 *
 * <h2>Algorithm:</h2>
 * <p>
 * This implementation aggregates Householder reflectors into a compact WY block
 * reflector and applies trailing updates using matrix-matrix operations (BLAS-3 style).
 * </p>
 * <ul>
 * <li><b>Panel Factorization:</b> Decomposes a block of columns using Householder reflectors.</li>
 * <li><b>Matrix Update:</b> Applies the accumulated block transformations to the trailing submatrix.</li>
 * </ul>
 *
 * <h2>Performance:</h2>
 * <p>
 * For large matrices, this reduces memory traffic compared to a
 * naive element-by-element implementation by using blocked GEMM kernels.
 * </p>
 *
 * <h2>Structure:</h2>
 * <pre>
 * Original A         ->       Hessenberg H
 * [ x x x x ]              [ x x x x ]
 * [ x x x x ]   Q^T A Q    [ x x x x ]
 * [ x x x x ]  =========>  [ 0 x x x ]
 * [ x x x x ]              [ 0 0 x x ]
 * </pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @see HessenbergResult
 */
public class BlockedHessenbergQR {

    private static final double EPS = 1e-12;

    // LAPACK dlahrd: optimal block size for 512x512 is 32-48
    // Larger blocks = fewer updates but more memory traffic
    private static final int DEFAULT_BLOCK_SIZE = 32;
    private static volatile int BLOCK_SIZE = initBlockSize();

    private static int initBlockSize() {
        String[] props = new String[]{
            "net.faulj.eigen.qr.BlockedHessenbergQR.blockSize",
            "net.faulj.eigen.qr.blockSize"
        };
        for (String p : props) {
            String val = System.getProperty(p);
            if (val != null && !val.isEmpty()) {
                try {
                    int parsed = Integer.parseInt(val);
                    if (parsed >= 1) {
                        return parsed;
                    } else {
                        System.err.println("Invalid blockSize (must be >=1): " + val + " for property " + p);
                    }
                } catch (NumberFormatException ignored) {
                    System.err.println("Invalid integer for property " + p + ": " + val);
                }
            }
        }
        return DEFAULT_BLOCK_SIZE;
    }

    /**
     * Set the panel/block size used by the blocked algorithm at runtime.
     * Must be >= 1. This is primarily intended for benchmarking and testing.
     */
    public static void setBlockSize(int bs) {
        if (bs < 1) {
            throw new IllegalArgumentException("blockSize must be >= 1");
        }
        BLOCK_SIZE = bs;
    }

    /**
     * Return the current block size.
     */
    public static int getBlockSize() {
        return BLOCK_SIZE;
    }

    /**
     * Reduces the matrix A to Hessenberg form.
     *
     * @param A The matrix to reduce.
     * @return The HessenbergResult (H, Q).
     */
    public static HessenbergResult decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isReal()) {
            throw new UnsupportedOperationException("Hessenberg reduction requires a real-valued matrix");
        }
        if (!A.isSquare()) {
            throw new ArithmeticException("Matrix must be square to compute Hessenberg form");
        }
        Matrix H = A.copy();
        int n = H.getRowCount();
        if (n <= 2) {
            return new HessenbergResult(A, H, Matrix.Identity(n));
        }
        Matrix Q = Matrix.Identity(n);

        double[] h = H.getRawData();
        double[] q = Q.getRawData();

        double[] work = new double[n * n];
        for (int k = 0; k < n - 2; k += BLOCK_SIZE) {
            int panelWidth = Math.min(BLOCK_SIZE, (n - 2) - k);
            int m = n - (k + 1);
            if (panelWidth <= 0 || m <= 0) {
                break;
            }

            double[] V = new double[m * panelWidth];
            double[] T = new double[panelWidth * panelWidth];
            double[] tau = new double[panelWidth];

            System.arraycopy(h, 0, work, 0, h.length);
            factorPanelUnblocked(work, n, k, panelWidth, V, tau);
            buildT(V, m, panelWidth, tau, T);

            applyBlockLeft(h, n, k, k, V, T, m, panelWidth);
            applyBlockRight(h, n, k, k + 1, V, T, m, panelWidth);
            applyBlockRight(q, n, k, k + 1, V, T, m, panelWidth);
        }

        // Clear any small numerical fill-in below the first subdiagonal.
        for (int col = 0; col < n - 2; col++) {
            for (int row = col + 2; row < n; row++) {
                h[row * n + col] = 0.0;
            }
        }

        return new HessenbergResult(A, H, Q);
    }

    private static void factorPanelUnblocked(double[] work, int n, int k, int panelWidth,
                                             double[] V, double[] tau) {
        int m = n - (k + 1);
        if (m <= 0 || panelWidth <= 0) {
            return;
        }

        double[] vWork = new double[m];

        for (int j = 0; j < panelWidth; j++) {
            int col = k + j;

            int rowStart = col + 1;
            int len = n - rowStart;
            if (len <= 0) {
                tau[j] = 0.0;
                continue;
            }

            int base = rowStart * n + col;
            double x0 = work[base];
            double sigma = 0.0;
            for (int i = 1; i < len; i++) {
                double val = work[(rowStart + i) * n + col];
                sigma += val * val;
            }
            if (sigma <= EPS) {
                tau[j] = 0.0;
                continue;
            }

            double mu = Math.sqrt(x0 * x0 + sigma);
            double beta = -Math.copySign(mu, x0);
            double v0 = x0 - beta;
            double v0sq = v0 * v0;
            if (v0sq <= EPS) {
                tau[j] = 0.0;
                continue;
            }

            double tauJ = 2.0 * v0sq / (sigma + v0sq);
            tau[j] = tauJ;

            int zeroPrefix = rowStart - (k + 1);
            for (int i = 0; i < zeroPrefix; i++) {
                vWork[i] = 0.0;
            }
            int localOffset = zeroPrefix;
            vWork[localOffset] = 1.0;
            for (int i = 1; i < len; i++) {
                vWork[localOffset + i] = work[(rowStart + i) * n + col] / v0;
            }
            for (int i = localOffset + len; i < m; i++) {
                vWork[i] = 0.0;
            }

            for (int r = 0; r < m; r++) {
                V[r * panelWidth + j] = vWork[r];
            }

            for (int col2 = col + 1; col2 < n; col2++) {
                double dot = work[rowStart * n + col2];
                int idx = (rowStart + 1) * n + col2;
                for (int i = 1; i < len; i++) {
                    dot += vWork[localOffset + i] * work[idx];
                    idx += n;
                }
                dot *= tauJ;
                work[rowStart * n + col2] -= dot;
                idx = (rowStart + 1) * n + col2;
                for (int i = 1; i < len; i++) {
                    work[idx] -= dot * vWork[localOffset + i];
                    idx += n;
                }
            }

            for (int row = 0; row < n; row++) {
                int idx = row * n + col + 1;
                double dot = work[idx];
                for (int i = 1; i < len; i++) {
                    dot += work[idx + i] * vWork[localOffset + i];
                }
                dot *= tauJ;
                work[idx] -= dot;
                for (int i = 1; i < len; i++) {
                    work[idx + i] -= dot * vWork[localOffset + i];
                }
            }

            work[base] = beta;
            for (int i = 1; i < len; i++) {
                work[(rowStart + i) * n + col] = 0.0;
            }
        }
    }

    private static void buildT(double[] V, int m, int k,
                               double[] tau, double[] T) {
        for (int i = 0; i < k * k; i++) {
            T[i] = 0.0;
        }

        double[] work = new double[k];
        for (int i = 0; i < k; i++) {
            double tauI = tau[i];
            if (tauI == 0.0) {
                T[i * k + i] = 0.0;
                continue;
            }
            for (int j = 0; j < i; j++) {
                double dot = 0.0;
                for (int r = 0; r < m; r++) {
                    dot += V[r * k + j] * V[r * k + i];
                }
                work[j] = -tauI * dot;
            }
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

    private static void applyBlockLeft(double[] a, int n, int k, int colStart,
                                       double[] V, double[] T,
                                       int m, int panelWidth) {
        if (colStart >= n) {
            return;
        }
        int nCols = n - colStart;

        double[] W = new double[panelWidth * nCols];
        Gemm.gemmStridedTransA(
            V, 0, panelWidth,
            a, (k + 1) * n + colStart, n,
            W, 0, nCols,
            m, panelWidth, nCols,
            1.0, 0.0, panelWidth);

        double[] W2 = new double[panelWidth * nCols];
        Gemm.gemmStridedTransA(
            T, 0, panelWidth,
            W, 0, nCols,
            W2, 0, nCols,
            panelWidth, panelWidth, nCols,
            1.0, 0.0, panelWidth);

        Gemm.gemmStrided(
            V, 0, panelWidth,
            W2, 0, nCols,
            a, (k + 1) * n + colStart, n,
            m, panelWidth, nCols,
            -1.0, 1.0, panelWidth);
    }

    private static void applyBlockRight(double[] a, int n, int k, int colStart,
                                        double[] V, double[] T,
                                        int m, int panelWidth) {
        int fullStart = k + 1;
        if (fullStart >= n || colStart >= n) {
            return;
        }
        int nCols = n - colStart;
        int colOffset = colStart - fullStart;
        if (nCols <= 0 || colOffset < 0) {
            return;
        }

        double[] W = new double[n * panelWidth];
        Gemm.gemmStrided(
            a, fullStart, n,
            V, 0, panelWidth,
            W, 0, panelWidth,
            n, m, panelWidth,
            1.0, 0.0, panelWidth);

        double[] W2 = new double[n * panelWidth];
        Gemm.gemmStrided(
            W, 0, panelWidth,
            T, 0, panelWidth,
            W2, 0, panelWidth,
            n, panelWidth, panelWidth,
            1.0, 0.0, panelWidth);

        Gemm.gemmStridedColMajorB(
            W2, 0, panelWidth,
            V, colOffset * panelWidth, panelWidth,
            a, colStart, n,
            n, panelWidth, nCols,
            -1.0, 1.0, panelWidth);
    }
}
