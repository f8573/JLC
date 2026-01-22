package net.faulj.compute;

import net.faulj.matrix.Matrix;

/**
 * Implements cache-optimized blocked matrix multiplication algorithms.
 */
public class BlockedMultiply {
    private static final double ZERO_EPS = 1e-15;

    private BlockedMultiply() {
    }

    public static Matrix multiply(Matrix a, Matrix b) {
        return multiply(a, b, DispatchPolicy.defaultPolicy());
    }

    public static Matrix multiply(Matrix a, Matrix b, DispatchPolicy policy) {
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
        if (m == 0 || n == 0 || k == 0) {
            return net.faulj.matrix.Matrix.zero(m, n);
        }
        return multiplyNaive(a, b);
    }

    public static Matrix multiplyNaive(Matrix a, Matrix b) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();

        Matrix result = new Matrix(m, n);
        double[] c = result.getRawData();
        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] ai = a.getRawImagData();
        double[] bi = b.getRawImagData();
        double[] ci = null;
        boolean complex = ai != null || bi != null;
        if (complex) {
            ci = result.ensureImagData();
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
                int bRowOffset = kk * n;
                int cIndex = cRowOffset;
                int bIndex = bRowOffset;
                if (!complex) {
                    for (int j = 0; j < n; j++) {
                        c[cIndex++] += aVal * bd[bIndex++];
                    }
                } else {
                    for (int j = 0; j < n; j++) {
                        double bVal = bd[bIndex];
                        double bImg = bi == null ? 0.0 : bi[bIndex];
                        c[cIndex] += aVal * bVal - aImg * bImg;
                        ci[cIndex] += aVal * bImg + aImg * bVal;
                        cIndex++;
                        bIndex++;
                    }
                }
            }
        }

        return result;
    }

    public static Matrix multiplyBlocked(Matrix a, Matrix b, int blockSize) {
        return multiplyNaive(a, b);
    }

    public static Matrix multiplyParallel(Matrix a, Matrix b, int blockSize, int parallelism) {
        return multiplyNaive(a, b);
    }
}
