package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import net.faulj.compute.BLAS3Kernels;
import net.faulj.compute.DispatchPolicy;
import jdk.incubator.vector.*;

/**
 * Householder QR decomposition using column-major internal representation
 * for optimal vectorization and cache performance.
 * 
 * Internally works with A^T (column-major) to make column operations contiguous.
 * This enables true BLAS3-like performance through SIMD vectorization.
 */
public final class HouseholderQR {
    private static final double EPS = 1e-12;
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int LANE_SIZE = SPECIES.length();
    private static final int BLOCK_SIZE = 64; // Optimal for L1 cache
    private static final int TRANSPOSE_BLOCK = 32;
    private static final int TRAILING_BLOCK = 128;

    private HouseholderQR() {}

    public static QRResult decompose(Matrix A) { return decompose(A, false); }
    public static QRResult decomposeThin(Matrix A) { return decompose(A, true); }

    private static QRResult decompose(Matrix A, boolean thin) {
        if (A == null) throw new IllegalArgumentException("Matrix must not be null");
        if (!A.isReal()) throw new UnsupportedOperationException("Householder QR requires real matrix");

        final int m = A.getRowCount();
        final int n = A.getColumnCount();
        final int kMax = Math.min(m, n);

        // Convert to column-major (transposed) for contiguous column access
        double[] AT = toColumnMajor(A.getRawData(), m, n);
        double[] tau = new double[kMax];
        Workspace ws = new Workspace(m, n);

        // Perform blocked Householder QR factorization
        factorizeBlocked(AT, tau, m, n, kMax, ws);

        // Build Q in column-major, then transpose back to row-major
        final int qRows = thin ? kMax : m;
        double[] QT = buildQ(AT, tau, m, n, qRows, kMax);

        // Extract R from column-major representation
        Matrix R = extractR(AT, m, n, kMax, thin);

        Matrix Q = fromColumnMajor(QT, m, qRows);

        return new QRResult(A, Q, R);
    }

    /**
     * Convert row-major matrix to column-major (transposed)
     */
    private static double[] toColumnMajor(double[] a, int rows, int cols) {
        double[] at = new double[rows * cols];
        transposeBlocked(a, at, rows, cols);
        return at;
    }

    /**
     * Convert column-major matrix back to row-major
     */
    private static Matrix fromColumnMajor(double[] at, int rows, int cols) {
        double[] a = new double[rows * cols];
        transposeBlocked(at, a, cols, rows);
        return Matrix.wrap(a, rows, cols);
    }

    private static void transposeBlocked(double[] src, double[] dst, int rows, int cols) {
        int block = TRANSPOSE_BLOCK;
        for (int i = 0; i < rows; i += block) {
            int iMax = Math.min(i + block, rows);
            for (int j = 0; j < cols; j += block) {
                int jMax = Math.min(j + block, cols);
                for (int r = i; r < iMax; r++) {
                    int srcBase = r * cols;
                    for (int c = j; c < jMax; c++) {
                        dst[c * rows + r] = src[srcBase + c];
                    }
                }
            }
        }
    }

    /**
     * Blocked Householder QR factorization (LAPACK style)
     * Works on column-major matrix AT (n x m) where original A is m x n
     */
    private static void factorizeBlocked(double[] AT, double[] tau, int m, int n, int kMax, Workspace ws) {
        // Process matrix in blocks for cache efficiency
        for (int kStart = 0; kStart < kMax; kStart += BLOCK_SIZE) {
            int kEnd = Math.min(kStart + BLOCK_SIZE, kMax);
            
            // 1. Compute panel factorization (current block)
            factorizePanel(AT, tau, m, n, kStart, kEnd);
            
            // 2. Update trailing matrix if there are more columns
            if (kEnd < n) {
                updateTrailingMatrix(AT, tau, m, n, kStart, kEnd, ws);
            }
        }
    }

    /**
     * Factorize a panel of columns [kStart, kEnd)
     */
    private static void factorizePanel(double[] AT, double[] tau, int m, int n, 
                                       int kStart, int kEnd) {
        for (int k = kStart; k < kEnd; k++) {
            // Compute Householder vector for column k
            double beta = computeHouseholder(AT, tau, m, n, k, k);
            
            // Apply to remaining columns in panel
            for (int j = k + 1; j < kEnd; j++) {
                applyHouseholderToColumn(AT, tau[k], m, n, k, j);
            }
        }
    }

    /**
     * Update trailing matrix after panel factorization
     */
    private static void updateTrailingMatrix(double[] AT, double[] tau, int m, int n,
                                             int kStart, int kEnd, Workspace ws) {
        int trailingCols = n - kEnd;
        int panelSizeActual = kEnd - kStart;
        int panelSize = ws.panelSize;

        if (trailingCols <= 0 || panelSizeActual <= 0) {
            return;
        }

        int rows = m;

        // Build packed V and V^T for compact WY representation
        packV(AT, m, kStart, kEnd, panelSize, ws.vPack, ws.vtPack);

        // Build T for compact WY representation
        buildT(ws.vPack, tau, panelSizeActual, panelSize, rows, kStart, ws.t);

        DispatchPolicy policy = DispatchPolicy.defaultPolicy();

        int blockCols = ws.blockCols;
        for (int colStart = 0; colStart < trailingCols; colStart += blockCols) {
            int blockColsUsed = Math.min(blockCols, trailingCols - colStart);

            packTrailingBlock(AT, m, kEnd, kStart, colStart, blockColsUsed, blockCols, ws.cBlock);

            Matrix Vt = Matrix.wrap(ws.vtPack, panelSize, rows);
            Matrix Cb = Matrix.wrap(ws.cBlock, rows, blockCols);
            Matrix Wm = Matrix.wrap(ws.w, panelSize, blockCols);

            // W = V^T * C
            BLAS3Kernels.gemm(Vt, Cb, Wm, 1.0, 0.0, policy);

            // W = T^T * W
            applyTTranspose(ws.t, ws.w, panelSize, blockCols, ws.temp);

            // C = C - V * W
            Matrix V = Matrix.wrap(ws.vPack, rows, panelSize);
            BLAS3Kernels.gemm(V, Wm, Cb, -1.0, 1.0, policy);

            unpackTrailingBlock(AT, m, kEnd, kStart, colStart, blockColsUsed, blockCols, ws.cBlock);
        }
    }

    /**
     * Compute W = V^T * AT_trailing where V contains Householder vectors
     */

    /**
     * Compute Householder vector for column k starting at row startRow
     */
    private static double computeHouseholder(double[] AT, double[] tau,
                                             int m, int n, int k, int startRow) {
        int colBase = k * m;
        int len = m - startRow;
        
        if (len <= 1) {
            tau[k] = 0.0;
            return AT[colBase + startRow];
        }
        
        // Extract column segment (contiguous in column-major)
        double x0 = AT[colBase + startRow];
        
        // Compute sigma = ||x[1:]||^2 using vector API
        double sigma = 0.0;
        int i = startRow + 1;
        int upperBound = SPECIES.loopBound(m);
        
        DoubleVector sumVec = DoubleVector.zero(SPECIES);
        for (; i <= upperBound - LANE_SIZE; i += LANE_SIZE) {
            DoubleVector vec = DoubleVector.fromArray(SPECIES, AT, colBase + i);
            sumVec = sumVec.add(vec.mul(vec));
        }
        sigma = sumVec.reduceLanes(VectorOperators.ADD);
        
        // Process remainder
        for (; i < m; i++) {
            double v = AT[colBase + i];
            sigma += v * v;
        }
        
        // Compute Householder vector using LAPACK-style stable formula
        double xnorm = Math.sqrt(sigma);
        if (xnorm == 0.0) {
            if (x0 >= 0.0) {
                tau[k] = 0.0;
                return x0;
            } else {
                tau[k] = 2.0;
                AT[colBase + startRow] = -x0;
                return -x0;
            }
        }

        double beta = -Math.copySign(Math.hypot(x0, xnorm), x0);
        double tauK = (beta - x0) / beta;
        double v0Inv = 1.0 / (x0 - beta);

        for (i = startRow + 1; i < m; i++) {
            AT[colBase + i] *= v0Inv;
        }

        tau[k] = tauK;
        AT[colBase + startRow] = beta;
        return beta;
    }

    /**
     * Build the upper triangular T for compact WY representation
     */
    private static void buildT(double[] vPack, double[] tau, int panelSizeActual, int panelSize,
                               int rows, int kStart, double[] T) {
        java.util.Arrays.fill(T, 0.0);
        for (int i = 0; i < panelSizeActual; i++) {
            double tauK = tau[kStart + i];
            T[i * panelSize + i] = tauK;
            int rowStart = kStart + i;

            for (int r = 0; r < i; r++) {
                double dot = 0.0;
                for (int row = rowStart; row < rows; row++) {
                    dot += vPack[row * panelSize + r] * vPack[row * panelSize + i];
                }
                T[r * panelSize + i] = -tauK * dot;
            }

            for (int r = 0; r < i; r++) {
                double sum = 0.0;
                for (int c = r; c < i; c++) {
                    sum += T[r * panelSize + c] * T[c * panelSize + i];
                }
                T[r * panelSize + i] = sum;
            }
        }
    }

    /**
     * Apply W = T^T * W for upper triangular T
     */
    private static void applyTTranspose(double[] T, double[] W,
                                        int panelSize, int trailingCols, double[] temp) {
        for (int i = 0; i < panelSize; i++) {
            for (int j = 0; j < trailingCols; j++) {
                double sum = 0.0;
                for (int k = 0; k <= i; k++) {
                    sum = Math.fma(T[k * panelSize + i], W[k * trailingCols + j], sum);
                }
                temp[i * trailingCols + j] = sum;
            }
        }

        System.arraycopy(temp, 0, W, 0, panelSize * trailingCols);
    }

    /**
     * Apply Householder reflector from column k to column j
     */
    private static void applyHouseholderToColumn(double[] AT, double tauK,
                                                 int m, int n, int k, int j) {
        int colKBase = k * m;
        int colJBase = j * m;
        
        // Compute dot = v^T * col_j
        double dot = AT[colJBase + k]; // First element (v0 = 1)
        
        // Vectorized dot product
        int i = k + 1;
        int upperBound = SPECIES.loopBound(m);
        
        DoubleVector dotVec = DoubleVector.zero(SPECIES);
        for (; i <= upperBound - LANE_SIZE; i += LANE_SIZE) {
            DoubleVector vVec = DoubleVector.fromArray(SPECIES, AT, colKBase + i);
            DoubleVector aVec = DoubleVector.fromArray(SPECIES, AT, colJBase + i);
            dotVec = dotVec.add(vVec.mul(aVec));
        }
        dot += dotVec.reduceLanes(VectorOperators.ADD);
        
        // Process remainder
        for (; i < m; i++) {
            dot += AT[colKBase + i] * AT[colJBase + i];
        }
        
        dot *= tauK;
        
        // Apply update: col_j -= dot * v
        AT[colJBase + k] -= dot;
        
        // Vectorized update
        i = k + 1;
        for (; i <= upperBound - LANE_SIZE; i += LANE_SIZE) {
            DoubleVector vVec = DoubleVector.fromArray(SPECIES, AT, colKBase + i);
            DoubleVector aVec = DoubleVector.fromArray(SPECIES, AT, colJBase + i);
            aVec.sub(vVec.mul(dot)).intoArray(AT, colJBase + i);
        }
        
        // Process remainder
        for (; i < m; i++) {
            AT[colJBase + i] -= dot * AT[colKBase + i];
        }
    }

    /**
     * Build Q matrix from Householder vectors
     */
    private static double[] buildQ(double[] AT, double[] tau,
                                   int m, int n, int qRows, int kMax) {
        // Initialize Q^T as identity (column-major)
        double[] QT = new double[m * qRows];

        int diagLimit = Math.min(m, qRows);
        for (int i = 0; i < diagLimit; i++) {
            QT[i * m + i] = 1.0;
        }

        // Apply Householder reflectors in reverse order to build Q^T
        for (int k = kMax - 1; k >= 0; k--) {
            double tauK = tau[k];
            if (tauK == 0.0) continue;

            int colKBase = k * m;

            for (int j = k; j < qRows; j++) {
                int colQBase = j * m;
                double dot = QT[colQBase + k];

                int i = k + 1;
                int upperBound = SPECIES.loopBound(m);

                DoubleVector dotVec = DoubleVector.zero(SPECIES);
                for (; i <= upperBound - LANE_SIZE; i += LANE_SIZE) {
                    DoubleVector vVec = DoubleVector.fromArray(SPECIES, AT, colKBase + i);
                    DoubleVector qVec = DoubleVector.fromArray(SPECIES, QT, colQBase + i);
                    dotVec = dotVec.add(vVec.mul(qVec));
                }
                dot += dotVec.reduceLanes(VectorOperators.ADD);

                for (; i < m; i++) {
                    dot += AT[colKBase + i] * QT[colQBase + i];
                }

                dot *= tauK;

                QT[colQBase + k] -= dot;

                i = k + 1;
                for (; i <= upperBound - LANE_SIZE; i += LANE_SIZE) {
                    DoubleVector vVec = DoubleVector.fromArray(SPECIES, AT, colKBase + i);
                    DoubleVector qVec = DoubleVector.fromArray(SPECIES, QT, colQBase + i);
                    qVec.sub(vVec.mul(dot)).intoArray(QT, colQBase + i);
                }

                for (; i < m; i++) {
                    QT[colQBase + i] -= dot * AT[colKBase + i];
                }
            }
        }

        return QT;
    }

    private static void packV(double[] AT, int m, int kStart, int kEnd, int panelSize,
                              double[] vPack, double[] vtPack) {
        int panelSizeActual = kEnd - kStart;

        for (int i = 0; i < panelSize; i++) {
            int k = kStart + i;
            for (int row = 0; row < m; row++) {
                double val;
                if (i >= panelSizeActual) {
                    val = 0.0;
                } else if (row < kStart) {
                    val = 0.0;
                } else {
                    int r = row - kStart;
                    if (r < i) {
                        val = 0.0;
                    } else if (r == i) {
                        val = 1.0;
                    } else {
                        val = AT[(k * m) + row];
                    }
                }
                vPack[row * panelSize + i] = val;
                vtPack[i * m + row] = val;
            }
        }
    }

    private static void packTrailingBlock(double[] AT, int m, int kEnd, int kStart,
                                          int colStart, int blockColsUsed, int blockCols, double[] cBlock) {
        int trailingStart = kEnd + colStart;
        for (int row = 0; row < m; row++) {
            for (int j = 0; j < blockCols; j++) {
                if (j < blockColsUsed) {
                    int col = trailingStart + j;
                    cBlock[row * blockCols + j] = AT[col * m + row];
                } else {
                    cBlock[row * blockCols + j] = 0.0;
                }
            }
        }
    }

    private static void unpackTrailingBlock(double[] AT, int m, int kEnd, int kStart,
                                            int colStart, int blockColsUsed, int blockCols, double[] cBlock) {
        int trailingStart = kEnd + colStart;
        for (int row = 0; row < m; row++) {
            for (int j = 0; j < blockColsUsed; j++) {
                int col = trailingStart + j;
                AT[col * m + row] = cBlock[row * blockCols + j];
            }
        }
    }

    private static void packQBlock(double[] Q, int m, int qRows, int kStart,
                                   int colStart, int blockColsUsed, int blockCols, double[] cBlock) {
        for (int row = 0; row < m; row++) {
            int rowBase = row * qRows;
            for (int j = 0; j < blockCols; j++) {
                if (j < blockColsUsed) {
                    cBlock[row * blockCols + j] = Q[rowBase + colStart + j];
                } else {
                    cBlock[row * blockCols + j] = 0.0;
                }
            }
        }
    }

    private static void unpackQBlock(double[] Q, int m, int qRows, int kStart,
                                     int colStart, int blockColsUsed, int blockCols, double[] cBlock) {
        for (int row = 0; row < m; row++) {
            int rowBase = row * qRows;
            for (int j = 0; j < blockColsUsed; j++) {
                Q[rowBase + colStart + j] = cBlock[row * blockCols + j];
            }
        }
    }

    private static final class Workspace {
        final double[] vPack;
        final double[] vtPack;
        final double[] cBlock;
        final double[] w;
        final double[] t;
        final double[] temp;
        final int blockCols;
        final int panelSize;

        Workspace(int m, int n) {
            int maxPanel = BLOCK_SIZE;
            int maxRows = m;
            int maxCols = Math.min(TRAILING_BLOCK, Math.max(m, n));
            this.blockCols = maxCols;
            this.panelSize = maxPanel;
            this.vPack = new double[maxRows * maxPanel];
            this.vtPack = new double[maxPanel * maxRows];
            this.cBlock = new double[maxRows * maxCols];
            this.w = new double[maxPanel * maxCols];
            this.t = new double[maxPanel * maxPanel];
            this.temp = new double[maxPanel * maxCols];
        }
    }

    /**
     * Extract R matrix from column-major factorization result
     */
    private static Matrix extractR(double[] AT, int m, int n, int kMax, boolean thin) {
        int rRows = thin ? kMax : m;
        int rCols = n;
        double[] R = new double[rRows * rCols];
        
        // Extract upper triangular part
        for (int c = 0; c < rCols; c++) {
            int srcBase = c * m;
            int rowLimit = Math.min(c + 1, rRows);

            for (int r = 0; r < rowLimit; r++) {
                R[r * rCols + c] = AT[srcBase + r];
            }

            for (int r = rowLimit; r < rRows; r++) {
                R[r * rCols + c] = 0.0;
            }
        }
        
        return Matrix.wrap(R, rRows, rCols);
    }
}