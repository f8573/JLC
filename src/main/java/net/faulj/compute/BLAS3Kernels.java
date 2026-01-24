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
        gemm(a, b, c, 1.0, 0.0, policy);
        return c;
    }

    public static void gemm(Matrix a, Matrix b, Matrix c, double alpha, double beta, DispatchPolicy policy) {
        if (a == null || b == null || c == null) {
            throw new IllegalArgumentException("Matrices must not be null");
        }
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int k2 = b.getRowCount();
        int n = b.getColumnCount();
        if (k != k2) {
            throw new IllegalArgumentException("Inner dimensions must agree for multiplication: " + k + " != " + k2);
        }
        if (c.getRowCount() != m || c.getColumnCount() != n) {
            throw new IllegalArgumentException("Output matrix dimensions must be " + m + "x" + n);
        }

        DispatchPolicy active = policy == null ? DispatchPolicy.defaultPolicy() : policy;
        DispatchPolicy.Algorithm algorithm = active.selectForMultiply(m, n, k);
        boolean parallel = active.shouldParallelize(m, n, k);
        int blockSize = active.blockSize(m, n, k);
        int parallelism = active.getParallelism();

        if (algorithm == DispatchPolicy.Algorithm.CUDA) {
            if (CudaGemm.gemm(a, b, c, alpha, beta)) {
                return;
            }
            algorithm = active.selectCpuAlgorithm(m, n, k);
        }
        switch (algorithm) {
            case PARALLEL:
                dgemmParallel(a, b, c, alpha, beta, blockSize, parallelism);
                return;
            case BLOCKED:
            case BLAS3:
            case STRASSEN:
                if (parallel) {
                    dgemmParallel(a, b, c, alpha, beta, blockSize, parallelism);
                } else {
                    dgemm(a, b, c, alpha, beta, blockSize);
                }
                return;
            case NAIVE:
            case SPECIALIZED:
            default:
                if (parallel) {
                    dgemmParallel(a, b, c, alpha, beta, 1, parallelism);
                } else {
                    dgemmNaive(a, b, c, alpha, beta);
                }
        }
    }

    static void dgemm(Matrix a, Matrix b, Matrix c, double alpha, double beta, int blockSize) {
        dgemmNaive(a, b, c, alpha, beta);
    }

    static void dgemmParallel(Matrix a, Matrix b, Matrix c, double alpha, double beta, int blockSize, int parallelism) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();

        int threads = Math.max(1, parallelism);
        if (threads <= 1 || m == 0 || n == 0 || k == 0) {
            dgemm(a, b, c, alpha, beta, blockSize);
            return;
        }

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

        double[] cdBase = null;
        double[] ciBase = null;
        if (beta != 0.0) {
            cdBase = java.util.Arrays.copyOf(cd, cd.length);
            if (complex) {
                ciBase = java.util.Arrays.copyOf(ci, ci.length);
            }
        }

        if (beta == 0.0) {
            java.util.Arrays.fill(cd, 0.0);
            if (complex) {
                java.util.Arrays.fill(ci, 0.0);
            }
        } else if (beta != 1.0) {
            for (int i = 0; i < cd.length; i++) {
                cd[i] *= beta;
            }
            if (complex) {
                for (int i = 0; i < ci.length; i++) {
                    ci[i] *= beta;
                }
            }
        }

        int bs = Math.max(1, blockSize);
        double[] ciLocal = ci;
        java.util.concurrent.ForkJoinPool pool = new java.util.concurrent.ForkJoinPool(threads);
        try {
            if (bs <= 1) {
                pool.submit(() ->
                    java.util.stream.IntStream.range(0, m).parallel().forEach(row -> {
                        int aRowOffset = row * k;
                        int cRowOffset = row * n;
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
                                    ciLocal[cIndex] += aVal * bImg + aImg * bVal;
                                    cIndex++;
                                    bIndex++;
                                }
                            }
                        }
                    })
                ).get();
            } else {
                int blockRows = (m + bs - 1) / bs;
                pool.submit(() ->
                    java.util.stream.IntStream.range(0, blockRows).parallel().forEach(blockRow -> {
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
                                        if (alpha != 1.0) {
                                            aVal *= alpha;
                                            aImg *= alpha;
                                        }
                                        int bRowOffset = kIndex * n;
                                        int cIndex = cRowOffset + jj;
                                        int bIndex = bRowOffset + jj;
                                        if (!complex) {
                                            for (int j = jj; j < jMax; j++) {
                                                cd[cIndex++] += aVal * bd[bIndex++];
                                            }
                                        } else {
                                            for (int j = jj; j < jMax; j++) {
                                                double bVal = bd[bIndex];
                                                double bImg = bi == null ? 0.0 : bi[bIndex];
                                                cd[cIndex] += aVal * bVal - aImg * bImg;
                                                ciLocal[cIndex] += aVal * bImg + aImg * bVal;
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
            }
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            restoreOriginal(cdBase, cd, ciBase, ci);
            dgemm(a, b, c, alpha, beta, blockSize);
        } catch (java.util.concurrent.ExecutionException ex) {
            restoreOriginal(cdBase, cd, ciBase, ci);
            dgemm(a, b, c, alpha, beta, blockSize);
        } finally {
            pool.shutdown();
        }
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

    private static void restoreOriginal(double[] cdBase, double[] cd, double[] ciBase, double[] ci) {
        if (cdBase == null) {
            return;
        }
        System.arraycopy(cdBase, 0, cd, 0, cd.length);
        if (ciBase != null && ci != null) {
            System.arraycopy(ciBase, 0, ci, 0, ci.length);
        }
    }
}
