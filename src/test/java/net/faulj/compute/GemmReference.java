package net.faulj.compute;

import net.faulj.matrix.Matrix;

import java.util.Arrays;
import java.util.SplittableRandom;

import static org.junit.Assert.assertTrue;

final class GemmReference {
    private GemmReference() {
    }

    static double[] seededArray(int length, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        double[] data = new double[length];
        for (int i = 0; i < length; i++) {
            double value = random.nextDouble(-1.0, 1.0);
            if (i % 11 == 0) {
                value = 0.0;
            }
            data[i] = value;
        }
        return data;
    }

    static Matrix seededMatrix(int rows, int cols, long seed) {
        return Matrix.wrap(seededArray(rows * cols, seed), rows, cols);
    }

    static double[] gemm(double[] a, double[] b, double[] c, int m, int k, int n, double alpha, double beta) {
        double[] out = Arrays.copyOf(c, c.length);
        for (int i = 0; i < m; i++) {
            int cRow = i * n;
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[cRow + j] = alpha * sum + beta * out[cRow + j];
            }
        }
        return out;
    }

    static double[] gemmStrided(double[] a, int aOff, int lda,
                                double[] b, int bOff, int ldb,
                                double[] c, int cOff, int ldc,
                                int m, int k, int n, double alpha, double beta) {
        double[] out = extractRowMajor(c, cOff, ldc, m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    sum += a[aOff + i * lda + p] * b[bOff + p * ldb + j];
                }
                out[i * n + j] = alpha * sum + beta * out[i * n + j];
            }
        }
        return out;
    }

    static double[] gemmStridedTransA(double[] a, int aOff, int lda,
                                      double[] b, int bOff, int ldb,
                                      double[] c, int cOff, int ldc,
                                      int m, int k, int n, double alpha, double beta) {
        double[] out = extractRowMajor(c, cOff, ldc, k, n);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < m; p++) {
                    sum += a[aOff + p * lda + i] * b[bOff + p * ldb + j];
                }
                out[i * n + j] = alpha * sum + beta * out[i * n + j];
            }
        }
        return out;
    }

    static double[] gemmStridedColMajorA(double[] a, int aOff, int lda,
                                         double[] b, int bOff, int ldb,
                                         double[] c, int cOff, int ldc,
                                         int m, int k, int n, double alpha, double beta) {
        double[] out = extractRowMajor(c, cOff, ldc, m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    sum += a[aOff + p * lda + i] * b[bOff + p * ldb + j];
                }
                out[i * n + j] = alpha * sum + beta * out[i * n + j];
            }
        }
        return out;
    }

    static double[] gemmStridedColMajorB(double[] a, int aOff, int lda,
                                         double[] b, int bOff, int ldb,
                                         double[] c, int cOff, int ldc,
                                         int m, int k, int n, double alpha, double beta) {
        double[] out = extractRowMajor(c, cOff, ldc, m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    sum += a[aOff + i * lda + p] * b[bOff + j * ldb + p];
                }
                out[i * n + j] = alpha * sum + beta * out[i * n + j];
            }
        }
        return out;
    }

    static double[] extractRowMajor(double[] backing, int offset, int ld, int rows, int cols) {
        double[] out = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(backing, offset + i * ld, out, i * cols, cols);
        }
        return out;
    }

    static double[] toColumnMajor(double[] rowMajor, int rows, int cols, int ld) {
        double[] out = new double[Math.max(0, cols * ld)];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[j * ld + i] = rowMajor[i * cols + j];
            }
        }
        return out;
    }

    static double maxAbs(double[] values) {
        double max = 0.0;
        for (double value : values) {
            max = Math.max(max, Math.abs(value));
        }
        return max;
    }

    static double frobenius(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value * value;
        }
        return Math.sqrt(sum);
    }

    static void assertParity(String path,
                             int m, int k, int n,
                             double alpha, double beta, long seed,
                             double[] expected, double[] actual,
                             double relTol, double absTol) {
        double maxAbsErr = 0.0;
        double diffSum = 0.0;
        for (int i = 0; i < expected.length; i++) {
            double diff = actual[i] - expected[i];
            maxAbsErr = Math.max(maxAbsErr, Math.abs(diff));
            diffSum += diff * diff;
        }
        double refNorm = frobenius(expected);
        double relErr = refNorm == 0.0 ? 0.0 : Math.sqrt(diffSum) / refNorm;
        String message = String.format(
            "%s parity failed for shape m=%d k=%d n=%d alpha=%s beta=%s seed=%d maxAbsErr=%s relErr=%s absTol=%s relTol=%s",
            path, m, k, n, alpha, beta, seed, maxAbsErr, relErr, absTol, relTol
        );

        if (refNorm == 0.0) {
            assertTrue(message, maxAbsErr <= absTol);
        } else {
            assertTrue(message, maxAbsErr <= absTol);
            assertTrue(message, relErr <= relTol);
        }
    }

    static double cpuAbsTolerance(double[] expected, int k) {
        return Math.max(1e-12, 1e-9 * Math.max(1.0, maxAbs(expected)) * Math.max(1.0, k / 16.0));
    }

    static double cudaAbsTolerance(double[] expected, int k) {
        return Math.max(1e-12, 1e-8 * Math.max(1.0, maxAbs(expected)) * Math.max(1.0, k / 16.0));
    }
}
