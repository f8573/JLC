package net.faulj.decomposition.hessenberg;

import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.NativeLapackHessenbergSupport;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Reduces a square matrix to upper Hessenberg form using orthogonal Householder transformations.
 */
public class HessenbergReduction {
    private static final double SAFE_MIN = Double.MIN_NORMAL;
    private static final int DEFAULT_BLOCK_SIZE = 32;
    private static final int PARALLEL_THRESHOLD = 200;

    private enum Mode {
        H_ONLY,
        H_AND_Q
    }

    private static final class ReductionWorkspace {
        final double[] reflector;
        double[] V;
        double[] T;
        double[] W;
        double[] WT;
        double[] tauBlock;
        double[] work;

        ReductionWorkspace(int n) {
            this.reflector = new double[n];
        }

        void ensureBlockCapacity(int blockLen, int panelSize, int n) {
            int vSize = blockLen * panelSize;
            int squareSize = panelSize * panelSize;
            int wideSize = n * panelSize;

            if (V == null || V.length < vSize) {
                V = new double[vSize];
            }
            if (T == null || T.length < squareSize) {
                T = new double[squareSize];
            }
            if (W == null || W.length < wideSize) {
                W = new double[wideSize];
            }
            if (WT == null || WT.length < wideSize) {
                WT = new double[wideSize];
            }
            if (tauBlock == null || tauBlock.length < panelSize) {
                tauBlock = new double[panelSize];
            }
            if (work == null || work.length < panelSize) {
                work = new double[panelSize];
            }
        }
    }

    private static final class ReductionState {
        final Matrix hessenberg;
        final double[] tau;

        ReductionState(Matrix hessenberg, double[] tau) {
            this.hessenberg = hessenberg;
            this.tau = tau;
        }
    }

    public static HessenbergResult decompose(Matrix A) {
        validateInput(A);
        int n = A.getRowCount();
        if (n <= 2) {
            Matrix h = A.copy();
            return new HessenbergResult(A, h, Matrix.Identity(n));
        }

        Matrix nativeH = A.copy();
        double[] nativeQ = new double[n * n];
        if (NativeLapackHessenbergSupport.tryDecompose(nativeH.getRawData(), n, nativeQ)) {
            zeroBelowSubdiagonal(nativeH.getRawData(), n);
            return new HessenbergResult(A, nativeH, Matrix.wrap(nativeQ, n, n));
        }

        ReductionState state = reduce(A, Mode.H_AND_Q);
        Matrix H = state.hessenberg;
        double[] h = H.getRawData();
        ReductionWorkspace workspace = new ReductionWorkspace(n);
        double[] q = accumulateQBlocked(h, state.tau, n, workspace);
        zeroBelowSubdiagonal(h, n);
        return new HessenbergResult(A, H, Matrix.wrap(q, n, n));
    }

    /**
     * Reduce a matrix to Hessenberg form without forming Q.
     * Useful for benchmarking to reduce allocation pressure.
     */
    public static Matrix reduceToHessenberg(Matrix A) {
        validateInput(A);
        if (A.getRowCount() <= 2) {
            return A.copy();
        }

        Matrix nativeH = A.copy();
        if (NativeLapackHessenbergSupport.tryReduce(nativeH.getRawData(), nativeH.getRowCount())) {
            zeroBelowSubdiagonal(nativeH.getRawData(), nativeH.getRowCount());
            return nativeH;
        }

        ReductionState state = reduce(A, Mode.H_ONLY);
        zeroBelowSubdiagonal(state.hessenberg.getRawData(), state.hessenberg.getRowCount());
        return state.hessenberg;
    }

    private static ReductionState reduce(Matrix A, Mode mode) {
        Matrix H = A.copy();
        int n = H.getRowCount();
        double[] h = H.getRawData();
        double[] tau = new double[n - 2];
        ReductionWorkspace workspace = new ReductionWorkspace(n);
        reduceUnblocked(h, n, tau, workspace.reflector);
        return new ReductionState(H, tau);
    }

    private static void reduceUnblocked(double[] h, int n, double[] tau, double[] reflector) {
        for (int k = 0; k < n - 2; k++) {
            factorColumn(h, n, k, tau, reflector);
            double tauK = tau[k];
            if (tauK == 0.0) {
                continue;
            }
            applyFromLeft(h, n, k, tauK, reflector);
            applyFromRight(h, n, k, tauK, reflector);
        }
    }

    private static void validateInput(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isReal()) {
            throw new UnsupportedOperationException("Hessenberg reduction requires a real-valued matrix");
        }
        if (!A.isSquare()) {
            throw new ArithmeticException("Matrix must be square to compute Hessenberg form");
        }
    }

    private static void factorColumn(double[] h, int n, int k, double[] tau, double[] v) {
        int start = k + 1;
        int len = n - start;
        int base = start * n + k;

        double x0 = h[base];
        double scale = Math.abs(x0);
        double ssq = 1.0;
        for (int i = 1; i < len; i++) {
            double absxi = Math.abs(h[(start + i) * n + k]);
            if (absxi > scale) {
                double temp = scale / absxi;
                ssq = 1.0 + ssq * temp * temp;
                scale = absxi;
            } else if (absxi > 0.0) {
                double temp = absxi / scale;
                ssq += temp * temp;
            }
        }

        double xnorm = scale == 0.0 ? 0.0 : scale * Math.sqrt(ssq);
        if (!(xnorm >= SAFE_MIN)) {
            tau[k] = 0.0;
            return;
        }

        double beta = x0 >= 0.0 ? -xnorm : xnorm;
        double tauK = (beta - x0) / beta;
        double invV0 = 1.0 / (x0 - beta);

        tau[k] = tauK;
        h[base] = beta;
        v[0] = 1.0;
        for (int i = 1; i < len; i++) {
            double vi = h[(start + i) * n + k] * invV0;
            v[i] = vi;
            h[(start + i) * n + k] = vi;
        }
    }

    private static void applyFromLeft(double[] h, int n, int k, double tau, double[] v) {
        int start = k + 1;
        int len = n - start;
        int numCols = n - start;

        if (numCols >= PARALLEL_THRESHOLD && n >= 400) {
            final int fStart = start;
            final int fLen = len;
            final int fLimit = len - 3;
            IntStream.range(0, numCols).parallel().forEach(colOff -> {
                int col = fStart + colOff;
                double dot = h[fStart * n + col];
                int i = 1;
                for (; i < fLimit; i += 4) {
                    dot = Math.fma(v[i], h[(fStart + i) * n + col], dot);
                    dot = Math.fma(v[i + 1], h[(fStart + i + 1) * n + col], dot);
                    dot = Math.fma(v[i + 2], h[(fStart + i + 2) * n + col], dot);
                    dot = Math.fma(v[i + 3], h[(fStart + i + 3) * n + col], dot);
                }
                for (; i < fLen; i++) {
                    dot = Math.fma(v[i], h[(fStart + i) * n + col], dot);
                }
                dot *= tau;
                h[fStart * n + col] -= dot;
                i = 1;
                for (; i < fLimit; i += 4) {
                    h[(fStart + i) * n + col] -= dot * v[i];
                    h[(fStart + i + 1) * n + col] -= dot * v[i + 1];
                    h[(fStart + i + 2) * n + col] -= dot * v[i + 2];
                    h[(fStart + i + 3) * n + col] -= dot * v[i + 3];
                }
                for (; i < fLen; i++) {
                    h[(fStart + i) * n + col] -= dot * v[i];
                }
            });
            return;
        }

        for (int col = start; col < n; col++) {
            double dot = h[start * n + col];
            int i = 1;
            int limit = len - 3;
            for (; i < limit; i += 4) {
                dot = Math.fma(v[i], h[(start + i) * n + col], dot);
                dot = Math.fma(v[i + 1], h[(start + i + 1) * n + col], dot);
                dot = Math.fma(v[i + 2], h[(start + i + 2) * n + col], dot);
                dot = Math.fma(v[i + 3], h[(start + i + 3) * n + col], dot);
            }
            for (; i < len; i++) {
                dot = Math.fma(v[i], h[(start + i) * n + col], dot);
            }
            dot *= tau;
            h[start * n + col] -= dot;
            i = 1;
            for (; i < limit; i += 4) {
                h[(start + i) * n + col] -= dot * v[i];
                h[(start + i + 1) * n + col] -= dot * v[i + 1];
                h[(start + i + 2) * n + col] -= dot * v[i + 2];
                h[(start + i + 3) * n + col] -= dot * v[i + 3];
            }
            for (; i < len; i++) {
                h[(start + i) * n + col] -= dot * v[i];
            }
        }
    }

    private static void applyFromRight(double[] h, int n, int k, double tau, double[] v) {
        int start = k + 1;
        int len = n - start;

        if (n >= PARALLEL_THRESHOLD && n >= 400) {
            final int fStart = start;
            final int fLen = len;
            final int fLimit = len - 3;
            IntStream.range(0, n).parallel().forEach(row -> {
                int idx = row * n + fStart;
                double dot = h[idx];
                int j = 1;
                for (; j < fLimit; j += 4) {
                    dot = Math.fma(h[idx + j], v[j], dot);
                    dot = Math.fma(h[idx + j + 1], v[j + 1], dot);
                    dot = Math.fma(h[idx + j + 2], v[j + 2], dot);
                    dot = Math.fma(h[idx + j + 3], v[j + 3], dot);
                }
                for (; j < fLen; j++) {
                    dot = Math.fma(h[idx + j], v[j], dot);
                }
                dot *= tau;
                h[idx] -= dot;
                j = 1;
                for (; j < fLimit; j += 4) {
                    h[idx + j] -= dot * v[j];
                    h[idx + j + 1] -= dot * v[j + 1];
                    h[idx + j + 2] -= dot * v[j + 2];
                    h[idx + j + 3] -= dot * v[j + 3];
                }
                for (; j < fLen; j++) {
                    h[idx + j] -= dot * v[j];
                }
            });
            return;
        }

        for (int row = 0; row < n; row++) {
            int idx = row * n + start;
            double dot = h[idx];
            int j = 1;
            int limit = len - 3;
            for (; j < limit; j += 4) {
                dot = Math.fma(h[idx + j], v[j], dot);
                dot = Math.fma(h[idx + j + 1], v[j + 1], dot);
                dot = Math.fma(h[idx + j + 2], v[j + 2], dot);
                dot = Math.fma(h[idx + j + 3], v[j + 3], dot);
            }
            for (; j < len; j++) {
                dot = Math.fma(h[idx + j], v[j], dot);
            }
            dot *= tau;
            h[idx] -= dot;
            j = 1;
            for (; j < limit; j += 4) {
                h[idx + j] -= dot * v[j];
                h[idx + j + 1] -= dot * v[j + 1];
                h[idx + j + 2] -= dot * v[j + 2];
                h[idx + j + 3] -= dot * v[j + 3];
            }
            for (; j < len; j++) {
                h[idx + j] -= dot * v[j];
            }
        }
    }

    private static double[] accumulateQBlocked(double[] h, double[] tau, int n, ReductionWorkspace workspace) {
        double[] q = new double[n * n];
        for (int i = 0; i < n; i++) {
            q[i * n + i] = 1.0;
        }

        int blockSize = blockSize();
        for (int kBlock = 0; kBlock < n - 2; kBlock += blockSize) {
            int blockCount = Math.min(blockSize, n - 2 - kBlock);
            int blockStart = kBlock + 1;
            int blockLen = n - blockStart;

            int activeCount = 0;
            for (int j = 0; j < blockCount; j++) {
                if (tau[kBlock + j] != 0.0) {
                    activeCount++;
                }
            }
            if (activeCount == 0) {
                continue;
            }

            workspace.ensureBlockCapacity(blockLen, activeCount, n);
            Arrays.fill(workspace.V, 0, blockLen * activeCount, 0.0);
            Arrays.fill(workspace.T, 0, activeCount * activeCount, 0.0);

            int column = 0;
            for (int j = 0; j < blockCount; j++) {
                int k = kBlock + j;
                double tauK = tau[k];
                if (tauK == 0.0) {
                    continue;
                }

                int rowStart = k + 1;
                int len = n - rowStart;
                int localOffset = rowStart - blockStart;
                workspace.tauBlock[column] = tauK;
                workspace.V[localOffset * activeCount + column] = 1.0;
                for (int i = 1; i < len; i++) {
                    workspace.V[(localOffset + i) * activeCount + column] = h[(rowStart + i) * n + k];
                }
                column++;
            }

            buildBlockReflectorFactor(workspace.V, blockLen, activeCount, workspace.tauBlock, workspace.T, workspace.work);
            applyBlockRight(q, n, blockStart, blockLen, activeCount, workspace);
        }

        return q;
    }

    private static void buildBlockReflectorFactor(double[] V, int blockLen, int panelSize,
                                                  double[] tau, double[] T, double[] work) {
        Arrays.fill(T, 0, panelSize * panelSize, 0.0);
        for (int i = 0; i < panelSize; i++) {
            double tauI = tau[i];
            if (tauI == 0.0) {
                T[i * panelSize + i] = 0.0;
                continue;
            }

            for (int j = 0; j < i; j++) {
                double dot = 0.0;
                for (int row = 0; row < blockLen; row++) {
                    dot = Math.fma(V[row * panelSize + j], V[row * panelSize + i], dot);
                }
                work[j] = -tauI * dot;
            }

            for (int j = 0; j < i; j++) {
                double sum = 0.0;
                for (int l = j; l < i; l++) {
                    sum = Math.fma(T[j * panelSize + l], work[l], sum);
                }
                T[j * panelSize + i] = sum;
            }

            T[i * panelSize + i] = tauI;
        }
    }

    private static void applyBlockRight(double[] a, int n, int blockStart, int blockLen, int panelSize,
                                        ReductionWorkspace workspace) {
        double[] V = workspace.V;
        double[] T = workspace.T;
        double[] W = workspace.W;
        double[] WT = workspace.WT;
        int wideSize = n * panelSize;

        Arrays.fill(W, 0, wideSize, 0.0);
        Arrays.fill(WT, 0, wideSize, 0.0);

        Gemm.gemmStrided(
            a, blockStart, n,
            V, 0, panelSize,
            W, 0, panelSize,
            n, blockLen, panelSize,
            1.0, 0.0, panelSize
        );

        Gemm.gemmStrided(
            W, 0, panelSize,
            T, 0, panelSize,
            WT, 0, panelSize,
            n, panelSize, panelSize,
            1.0, 0.0, panelSize
        );

        Gemm.gemmStridedColMajorB(
            WT, 0, panelSize,
            V, 0, panelSize,
            a, blockStart, n,
            n, panelSize, blockLen,
            -1.0, 1.0, panelSize
        );
    }

    private static void zeroBelowSubdiagonal(double[] h, int n) {
        for (int col = 0; col < n - 2; col++) {
            for (int row = col + 2; row < n; row++) {
                h[row * n + col] = 0.0;
            }
        }
    }

    private static int blockSize() {
        return firstPositiveIntegerProperty(
            "net.faulj.decomposition.hessenberg.blockSize",
            "net.faulj.eigen.qr.BlockedHessenbergQR.blockSize",
            "net.faulj.eigen.qr.blockSize"
        );
    }

    private static int firstPositiveIntegerProperty(String... keys) {
        for (String key : keys) {
            String value = System.getProperty(key);
            if (value == null || value.isBlank()) {
                continue;
            }
            try {
                int parsed = Integer.parseInt(value.trim());
                if (parsed >= 1) {
                    return parsed;
                }
            } catch (NumberFormatException ignored) {
                // Fall through to the next property key.
            }
        }
        return DEFAULT_BLOCK_SIZE;
    }
}
