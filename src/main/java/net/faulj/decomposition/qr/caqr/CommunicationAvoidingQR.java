package net.faulj.decomposition.qr.caqr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.util.PerfTimers;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Communication-Avoiding QR (CAQR) using a two-level TSQR reduction.
 */
public final class CommunicationAvoidingQR {

    private CommunicationAvoidingQR() {}

    public static CAQRPanelResult factorPanel(double[] A, int aOffset, int m, int b, int lda, QRConfig cfg) {
        cfg = cfg == null ? QRConfig.defaultConfig() : cfg;
        if (A == null) {
            throw new IllegalArgumentException("Panel must not be null");
        }
        if (m < 0 || b < 0 || lda < Math.max(1, m) || aOffset < 0) {
            throw new IllegalArgumentException("Invalid CAQR panel dimensions");
        }
        if (b > m) {
            throw new IllegalArgumentException("CAQR panel width must not exceed row count");
        }
        int required = aOffset + (b == 0 ? 0 : (b - 1) * lda + m);
        if (required > A.length) {
            throw new IllegalArgumentException("Panel storage is smaller than requested dimensions");
        }

        int workers = Math.max(1, cfg.p == 0 ? Runtime.getRuntime().availableProcessors() : cfg.p);
        WorkspaceManager ws = new WorkspaceManager(m, b, workers, cfg.alignmentBytes);
        double[] rTop = new double[b * b];
        if (m == 0 || b == 0) {
            return new CAQRPanelResult(rTop, ws.yCombinedOffset, ws.tCombinedOffset, ws);
        }

        double[] panel = copyColumnMajorPanel(A, aOffset, m, b, lda);
        double[] tau = new double[b];
        factorizePanel(panel, tau, m, b);

        for (int row = 0; row < b; row++) {
            for (int col = row; col < b; col++) {
                rTop[row * b + col] = panel[col * m + row];
            }
        }

        double[] y = buildY(panel, m, b);
        double[] t = buildT(y, tau, m, b);
        DoubleBuffer db = ws.doubleBuffer();
        synchronized (db) {
            DoubleBuffer view = db.duplicate();
            view.position(ws.yCombinedOffset);
            view.put(y);
            view.position(ws.tCombinedOffset);
            view.put(t);
        }
        return new CAQRPanelResult(rTop, ws.yCombinedOffset, ws.tCombinedOffset, ws);
    }

    public static void applyWYUpdate(double[] A, int aOffset, int m, int n, int lda, WorkspaceManager workspace) {
        if (m <= 0 || n <= 0) return;

        long tApply = PerfTimers.start();

        DoubleBuffer db = workspace.doubleBuffer();
        int b = workspace.panelWidth;
        if (b <= 0) return;

        double[] Y = new double[m * b];
        double[] T = new double[b * b];
        synchronized (db) {
            DoubleBuffer view = db.duplicate();
            view.position(workspace.yCombinedOffset);
            view.get(Y, 0, Y.length);
            view.position(workspace.tCombinedOffset);
            view.get(T, 0, T.length);
        }

        double[] Z = new double[b * n];
        long tZ = PerfTimers.start();
        Gemm.gemmStrided(true,
                       Y, 0, b,
                       A, aOffset, lda,
                       Z, 0, n,
                       m, b, n,
                       1.0, 0.0);
        PerfTimers.record("CAQR.applyWY.Z", tZ);

        double[] W = new double[b * n];
        long tW = PerfTimers.start();
        Gemm.gemmStrided(true,
                       T, 0, b,
                       Z, 0, n,
                       W, 0, n,
                       b, b, n,
                       1.0, 0.0);
        PerfTimers.record("CAQR.applyWY.W", tW);

        long tUpd = PerfTimers.start();
        Gemm.gemmStrided(Y, 0, b,
                       W, 0, n,
                       A, aOffset, lda,
                       m, b, n,
                       -1.0, 1.0);
        PerfTimers.record("CAQR.applyWY.UPDATE", tUpd);

        PerfTimers.record("CAQR.applyWY.total", tApply);
    }

    /**
     * Factor a tall-or-square matrix with a two-level TSQR reduction.
     */
    public static QRResult factor(Matrix A, boolean thin, QRConfig cfg) {
        cfg = cfg == null ? QRConfig.defaultConfig() : cfg;
        if (A == null) throw new IllegalArgumentException("Matrix must not be null");
        if (!A.isReal()) throw new UnsupportedOperationException("CAQR requires real matrix");

        final int m = A.getRowCount();
        final int n = A.getColumnCount();

        if (!thin || m < n || n == 0) {
            return net.faulj.decomposition.qr.HouseholderQR.decomposeHouseholder(A, thin);
        }

        int p = cfg.p > 0 ? cfg.p : Math.min(Math.max(1, cfg.numThreads), m);
        p = Math.min(p, m);
        int maxLeaves = Math.max(1, m / n);
        p = Math.min(p, maxLeaves);

        int base = m / p;
        int rem = m % p;
        List<Matrix> qBlocks = new ArrayList<>(p);
        List<Matrix> rBlocks = new ArrayList<>(p);
        int rowStart = 0;
        for (int i = 0; i < p; i++) {
            int rows = base + (i < rem ? 1 : 0);
            Matrix Ai = A.crop(rowStart, rowStart + rows - 1, 0, n - 1);
            QRResult res = net.faulj.decomposition.qr.HouseholderQR.decomposeHouseholder(Ai, true);
            qBlocks.add(res.getQ());
            rBlocks.add(res.getR().crop(0, n - 1, 0, n - 1));
            rowStart += rows;
        }

        Matrix stackedR = new Matrix(p * n, n);
        for (int i = 0; i < p; i++) {
            Matrix rBlock = rBlocks.get(i);
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < n; c++) {
                    stackedR.set(i * n + r, c, rBlock.get(r, c));
                }
            }
        }

        QRResult root = net.faulj.decomposition.qr.HouseholderQR.decomposeHouseholder(stackedR, true);
        Matrix stackedQ = root.getQ();
        Matrix finalR = root.getR();

        Matrix finalQ = new Matrix(m, n);
        rowStart = 0;
        for (int i = 0; i < p; i++) {
            Matrix rootBlock = stackedQ.crop(i * n, (i + 1) * n - 1, 0, n - 1);
            Matrix qBlock = qBlocks.get(i).multiply(rootBlock);
            for (int r = 0; r < qBlock.getRowCount(); r++) {
                for (int c = 0; c < n; c++) {
                    finalQ.set(rowStart + r, c, qBlock.get(r, c));
                }
            }
            rowStart += qBlock.getRowCount();
        }

        return new QRResult(A, finalQ, finalR);
    }

    private static double[] copyColumnMajorPanel(double[] source, int offset, int rows, int cols, int lda) {
        double[] panel = new double[rows * cols];
        for (int col = 0; col < cols; col++) {
            System.arraycopy(source, offset + col * lda, panel, col * rows, rows);
        }
        return panel;
    }

    private static void factorizePanel(double[] panel, double[] tau, int rows, int cols) {
        for (int k = 0; k < cols; k++) {
            computeHouseholder(panel, tau, rows, k);
            for (int col = k + 1; col < cols; col++) {
                applyHouseholder(panel, tau[k], rows, k, col);
            }
        }
    }

    private static void computeHouseholder(double[] panel, double[] tau, int rows, int col) {
        int base = col * rows;
        int len = rows - col;
        if (len <= 1) {
            tau[col] = 0.0;
            return;
        }

        double max = 0.0;
        for (int row = col; row < rows; row++) {
            max = Math.max(max, Math.abs(panel[base + row]));
        }
        if (max < Double.MIN_NORMAL) {
            tau[col] = 0.0;
            return;
        }

        double invMax = 1.0 / max;
        double x0 = panel[base + col] * invMax;
        double sigma = 0.0;
        for (int row = col + 1; row < rows; row++) {
            double value = panel[base + row] * invMax;
            sigma = Math.fma(value, value, sigma);
        }

        double xnorm = Math.sqrt(sigma);
        if (xnorm == 0.0) {
            if (x0 >= 0.0) {
                tau[col] = 0.0;
            } else {
                tau[col] = 2.0;
                panel[base + col] = -panel[base + col];
            }
            return;
        }

        double normBeta = -Math.copySign(Math.hypot(x0, xnorm), x0);
        tau[col] = (normBeta - x0) / normBeta;
        double invV0 = 1.0 / ((x0 - normBeta) * max);
        for (int row = col + 1; row < rows; row++) {
            panel[base + row] *= invV0;
        }
        panel[base + col] = normBeta * max;
    }

    private static void applyHouseholder(double[] panel, double tau, int rows, int reflectorCol, int targetCol) {
        if (tau == 0.0) {
            return;
        }
        int reflectorBase = reflectorCol * rows;
        int targetBase = targetCol * rows;
        double dot = panel[targetBase + reflectorCol];
        for (int row = reflectorCol + 1; row < rows; row++) {
            dot = Math.fma(panel[reflectorBase + row], panel[targetBase + row], dot);
        }
        dot *= tau;
        panel[targetBase + reflectorCol] -= dot;
        for (int row = reflectorCol + 1; row < rows; row++) {
            panel[targetBase + row] -= dot * panel[reflectorBase + row];
        }
    }

    private static double[] buildY(double[] panel, int rows, int cols) {
        double[] y = new double[rows * cols];
        for (int col = 0; col < cols; col++) {
            for (int row = col; row < rows; row++) {
                y[row * cols + col] = row == col ? 1.0 : panel[col * rows + row];
            }
        }
        return y;
    }

    private static double[] buildT(double[] y, double[] tau, int rows, int cols) {
        double[] t = new double[cols * cols];
        for (int i = 0; i < cols; i++) {
            t[i * cols + i] = tau[i];
            for (int r = 0; r < i; r++) {
                double dot = 0.0;
                for (int row = i; row < rows; row++) {
                    dot = Math.fma(y[row * cols + r], y[row * cols + i], dot);
                }
                t[r * cols + i] = -tau[i] * dot;
            }
            for (int r = 0; r < i; r++) {
                double sum = 0.0;
                for (int c = r; c < i; c++) {
                    sum += t[r * cols + c] * t[c * cols + i];
                }
                t[r * cols + i] = sum;
            }
        }
        return t;
    }
}
