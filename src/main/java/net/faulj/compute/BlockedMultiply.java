package net.faulj.compute;

import net.faulj.matrix.Matrix;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

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
        if (n == 1) {
            return multiplyVector(a, b);
        }
        DispatchPolicy active = policy == null ? DispatchPolicy.defaultPolicy() : policy;
        DispatchPolicy.Algorithm algorithm = active.selectForMultiply(m, n, k);
        boolean parallel = active.shouldParallelize(m, n, k);
        int blockSize = active.blockSize(m, n, k);
        int parallelism = active.getParallelism();
        if (algorithm == DispatchPolicy.Algorithm.CUDA) {
            Matrix cuda = CudaGemm.multiply(a, b);
            if (cuda != null) {
                return cuda;
            }
            algorithm = active.selectCpuAlgorithm(m, n, k);
        }
        switch (algorithm) {
            case BLAS3:
                return BLAS3Kernels.gemm(a, b, active);
            case PARALLEL:
                return multiplyParallel(a, b, blockSize, parallelism);
            case BLOCKED:
            case STRASSEN:
                if (parallel) {
                    return multiplyParallel(a, b, blockSize, parallelism);
                }
                return multiplyBlocked(a, b, blockSize);
            case NAIVE:
            case SPECIALIZED:
            default:
                if (parallel) {
                    return multiplyNaiveParallel(a, b, parallelism);
                }
                return multiplyNaive(a, b);
        }
    }

    private static Matrix multiplyVector(Matrix a, Matrix b) {
        int m = a.getRowCount();
        int k = a.getColumnCount();

        Matrix result = new Matrix(m, 1);
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

        if (!complex) {
            for (int i = 0; i < m; i++) {
                int aRowOffset = i * k;
                double sum = 0.0;
                for (int kk = 0; kk < k; kk++) {
                    sum = Math.fma(ad[aRowOffset + kk], bd[kk], sum);
                }
                c[i] = sum;
            }
        } else {
            for (int i = 0; i < m; i++) {
                int aRowOffset = i * k;
                double sumR = 0.0;
                double sumI = 0.0;
                for (int kk = 0; kk < k; kk++) {
                    double aVal = ad[aRowOffset + kk];
                    double aImg = ai == null ? 0.0 : ai[aRowOffset + kk];
                    double bVal = bd[kk];
                    double bImg = bi == null ? 0.0 : bi[kk];
                    sumR = Math.fma(aVal, bVal, sumR);
                    sumR = Math.fma(-aImg, bImg, sumR);
                    sumI = Math.fma(aVal, bImg, sumI);
                    sumI = Math.fma(aImg, bVal, sumI);
                }
                c[i] = sumR;
                ci[i] = sumI;
            }
        }

        return result;
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
                        c[cIndex] = Math.fma(aVal, bd[bIndex], c[cIndex]);
                        cIndex++;
                        bIndex++;
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
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();
        int bs = Math.max(1, blockSize);

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

        for (int ii = 0; ii < m; ii += bs) {
            int iMax = Math.min(ii + bs, m);
            for (int kk = 0; kk < k; kk += bs) {
                int kMax = Math.min(kk + bs, k);
                for (int jj = 0; jj < n; jj += bs) {
                    int jMax = Math.min(jj + bs, n);
                    for (int i = ii; i < iMax; i++) {
                        int aRowOffset = i * k;
                        int cRowOffset = i * n;
                        for (int kIndex = kk; kIndex < kMax; kIndex++) {
                            double aVal = ad[aRowOffset + kIndex];
                            double aImg = ai == null ? 0.0 : ai[aRowOffset + kIndex];
                            if (Math.abs(aVal) <= ZERO_EPS && Math.abs(aImg) <= ZERO_EPS) {
                                continue;
                            }
                            int bRowOffset = kIndex * n;
                            int cIndex = cRowOffset + jj;
                            int bIndex = bRowOffset + jj;
                            if (!complex) {
                                for (int j = jj; j < jMax; j++) {
                                    c[cIndex] = Math.fma(aVal, bd[bIndex], c[cIndex]);
                                    cIndex++;
                                    bIndex++;
                                }
                            } else {
                                for (int j = jj; j < jMax; j++) {
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
                }
            }
        }

        return result;
    }

    public static Matrix multiplyParallel(Matrix a, Matrix b, int blockSize, int parallelism) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();
        int bs = Math.max(1, blockSize);

        int blockRows = (m + bs - 1) / bs;
        int threads = Math.max(1, parallelism);
        if (threads <= 1 || blockRows <= 1) {
            return multiplyBlocked(a, b, bs);
        }

        Matrix result = new Matrix(m, n);
        double[] c = result.getRawData();
        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] ai = a.getRawImagData();
        double[] bi = b.getRawImagData();
        double[] ci;
        boolean complex = ai != null || bi != null;
        if (complex) {
            ci = result.ensureImagData();
        } else {
            ci = null;
        }

        ForkJoinPool pool = new ForkJoinPool(threads);
        try {
            pool.submit(() ->
                IntStream.range(0, blockRows).parallel().forEach(blockRow -> {
                    int ii = blockRow * bs;
                    int iMax = Math.min(ii + bs, m);
                    for (int kk = 0; kk < k; kk += bs) {
                        int kMax = Math.min(kk + bs, k);
                        for (int jj = 0; jj < n; jj += bs) {
                            int jMax = Math.min(jj + bs, n);
                            for (int i = ii; i < iMax; i++) {
                                int aRowOffset = i * k;
                                int cRowOffset = i * n;
                                for (int kIndex = kk; kIndex < kMax; kIndex++) {
                                    double aVal = ad[aRowOffset + kIndex];
                                    double aImg = ai == null ? 0.0 : ai[aRowOffset + kIndex];
                                    if (Math.abs(aVal) <= ZERO_EPS && Math.abs(aImg) <= ZERO_EPS) {
                                        continue;
                                    }
                                    int bRowOffset = kIndex * n;
                                    int cIndex = cRowOffset + jj;
                                    int bIndex = bRowOffset + jj;
                                    if (!complex) {
                                        for (int j = jj; j < jMax; j++) {
                                            c[cIndex] = Math.fma(aVal, bd[bIndex], c[cIndex]);
                                            cIndex++;
                                            bIndex++;
                                        }
                                    } else {
                                        for (int j = jj; j < jMax; j++) {
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
                        }
                    }
                })
            ).get();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            return multiplyBlocked(a, b, bs);
        } catch (ExecutionException ex) {
            return multiplyBlocked(a, b, bs);
        } finally {
            pool.shutdown();
        }

        return result;
    }

    public static Matrix multiplyNaiveParallel(Matrix a, Matrix b, int parallelism) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();

        int threads = Math.max(1, parallelism);
        if (threads <= 1 || m <= 1) {
            return multiplyNaive(a, b);
        }

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

        ForkJoinPool pool = new ForkJoinPool(threads);
        try {
            double[] finalCi = ci;
            boolean finalComplex = complex;
            pool.submit(() ->
                IntStream.range(0, m).parallel().forEach(i -> {
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
                        if (!finalComplex) {
                            for (int j = 0; j < n; j++) {
                                c[cIndex] = Math.fma(aVal, bd[bIndex], c[cIndex]);
                                cIndex++;
                                bIndex++;
                            }
                        } else {
                            for (int j = 0; j < n; j++) {
                                double bVal = bd[bIndex];
                                double bImg = bi == null ? 0.0 : bi[bIndex];
                                c[cIndex] += aVal * bVal - aImg * bImg;
                                finalCi[cIndex] += aVal * bImg + aImg * bVal;
                                cIndex++;
                                bIndex++;
                            }
                        }
                    }
                })
            ).get();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            return multiplyNaive(a, b);
        } catch (ExecutionException ex) {
            return multiplyNaive(a, b);
        } finally {
            pool.shutdown();
        }

        return result;
    }
}
