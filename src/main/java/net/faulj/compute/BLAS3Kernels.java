package net.faulj.compute;

import net.faulj.matrix.Matrix;

/**
 * Provides optimized Level-3 BLAS kernel operations.
 */
public class BLAS3Kernels {
    private static final double ZERO_EPS = 1e-15;

    private BLAS3Kernels() {
    }

    public static Matrix gemm(Matrix a, Matrix b) {
        return gemm(a, b, DispatchPolicy.defaultPolicy());
    }

    public static Matrix gemm(Matrix a, Matrix b, DispatchPolicy policy) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Matrices must not be null");
        }
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int k2 = b.getRowCount();
        int n = b.getColumnCount();
        if (k != k2) {
            throw new IllegalArgumentException("Inner dimensions must agree for multiplication: " + k + " != " + k2);
        }
        Matrix c = net.faulj.matrix.Matrix.zero(m, n);
        dgemmNaive(a, b, c, 1.0, 0.0);
        return c;
    }

    public static void gemm(Matrix a, Matrix b, Matrix c, double alpha, double beta, DispatchPolicy policy) {
        dgemmNaive(a, b, c, alpha, beta);
    }

    static void dgemm(Matrix a, Matrix b, Matrix c, double alpha, double beta, int blockSize) {
        dgemmNaive(a, b, c, alpha, beta);
    }

    static void dgemmParallel(Matrix a, Matrix b, Matrix c, double alpha, double beta, int blockSize, int parallelism) {
        dgemmNaive(a, b, c, alpha, beta);
    }

    private static void dgemmNaive(Matrix a, Matrix b, Matrix c, double alpha, double beta) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();

        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] cd = c.getRawData();
        double[] ai = a.getRawImagData();
        double[] bi = b.getRawImagData();
        double[] ci = null;
        boolean complex = ai != null || bi != null || c.getRawImagData() != null;
        if (complex) {
            ci = c.ensureImagData();
        }

        if (beta == 0.0) {
            java.util.Arrays.fill(cd, 0.0);
        } else if (beta != 1.0) {
            for (int i = 0; i < cd.length; i++) {
                cd[i] *= beta;
            }
        }

        for (int i = 0; i < m; i++) {
            int aRowOffset = i * k;
            int cRowOffset = i * n;
            for (int kk = 0; kk < k; kk++) {
                double aVal = ad[aRowOffset + kk];
                double aImg = ai == null ? 0.0 : ai[aRowOffset + kk];
                if (Math.abs(aVal) <= ZERO_EPS && Math.abs(aImg) <= ZERO_EPS) {
                    continue;
                }
                if (alpha != 1.0) {
                    aVal *= alpha;
                    aImg *= alpha;
                }
                int bRowOffset = kk * n;
                int cIndex = cRowOffset;
                int bIndex = bRowOffset;
                if (!complex) {
                    for (int j = 0; j < n; j++) {
                        cd[cIndex++] += aVal * bd[bIndex++];
                    }
                } else {
                    for (int j = 0; j < n; j++) {
                        double bVal = bd[bIndex];
                        double bImg = bi == null ? 0.0 : bi[bIndex];
                        cd[cIndex] += aVal * bVal - aImg * bImg;
                        ci[cIndex] += aVal * bImg + aImg * bVal;
                        cIndex++;
                        bIndex++;
                    }
                }
            }
        }
    }
}
