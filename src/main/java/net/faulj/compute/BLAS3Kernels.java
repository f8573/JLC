package net.faulj.compute;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Provides optimized Level-3 BLAS kernel operations.
 */
public class BLAS3Kernels {
    private static final double ZERO_EPS = 1e-15;
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int PREFETCH_DISTANCE = 2;
    private static final int DEFAULT_K_UNROLL = 4;
    private static final ThreadLocal<double[]> PACK_A_BUFFER = new ThreadLocal<>();
    private static final ThreadLocal<double[]> PACK_B_BUFFER = new ThreadLocal<>();

    private BLAS3Kernels() {
    }

    /**
     * Multiply two matrices using the default dispatch policy.
     *
     * @param a left matrix
     * @param b right matrix
     * @return product matrix
     */
    public static Matrix gemm(Matrix a, Matrix b) {
        return gemm(a, b, DispatchPolicy.defaultPolicy());
    }

    /**
     * Multiply two matrices using a specific dispatch policy.
     *
     * @param a left matrix
     * @param b right matrix
     * @param policy dispatch policy
     * @return product matrix
     */
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
        Matrix c = (a instanceof OffHeapMatrix || b instanceof OffHeapMatrix)
            ? new OffHeapMatrix(m, n)
            : net.faulj.matrix.Matrix.zero(m, n);
        gemm(a, b, c, 1.0, 0.0, policy);
        return c;
    }

    /**
     * Compute $C = \alpha AB + \beta C$ using a dispatch policy.
     *
     * @param a left matrix
     * @param b right matrix
     * @param c output matrix
     * @param alpha scaling for AB
     * @param beta scaling for C
     * @param policy dispatch policy
     */
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
            case SIMD:
                if (parallel) {
                    dgemmParallel(a, b, c, alpha, beta, blockSize, parallelism);
                } else {
                    dgemmSimd(a, b, c, alpha, beta, blockSize);
                }
                return;
            case BLOCKED:
            case STRASSEN:
                if (parallel) {
                    dgemmParallel(a, b, c, alpha, beta, blockSize, parallelism);
                } else {
                    dgemm(a, b, c, alpha, beta, blockSize);
                }
                return;
            case BLAS3:
                if (parallel) {
                    dgemmParallel(a, b, c, alpha, beta, blockSize, parallelism);
                } else if (active.isSimdEnabled() && active.isSimdAvailable()) {
                    dgemmSimd(a, b, c, alpha, beta, blockSize);
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

    static void dgemmSimd(Matrix a, Matrix b, Matrix c, double alpha, double beta, int blockSize) {
        if (a instanceof OffHeapMatrix && b instanceof OffHeapMatrix && c instanceof OffHeapMatrix) {
            dgemmSimdOffHeap((OffHeapMatrix) a, (OffHeapMatrix) b, (OffHeapMatrix) c, alpha, beta, blockSize);
            return;
        }
        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] cd = c.getRawData();
        double[] ai = a.getRawImagData();
        double[] bi = b.getRawImagData();
        if (ai != null || bi != null || c.getRawImagData() != null) {
            dgemmNaive(a, b, c, alpha, beta);
            return;
        }

        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();
        int bs = Math.max(1, blockSize);

        dgemmSimdPacked(ad, bd, cd, m, k, n, alpha, beta, bs);
    }

    static void dgemmSimdOffHeap(OffHeapMatrix a, OffHeapMatrix b, OffHeapMatrix c,
                                double alpha, double beta, int blockSize) {
        if (a.getRawImagData() != null || b.getRawImagData() != null || c.getRawImagData() != null) {
            dgemmNaive(a, b, c, alpha, beta);
            return;
        }

        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] cd = c.getRawData();
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();
        int bs = Math.max(1, blockSize);

        dgemmSimdPacked(ad, bd, cd, m, k, n, alpha, beta, bs);
        c.syncToOffHeap();
    }

    static void dgemmSimdPacked(double[] ad, double[] bd, double[] cd,
                               int m, int k, int n, double alpha, double beta, int blockSize) {
        if (beta == 0.0) {
            java.util.Arrays.fill(cd, 0.0);
        } else if (beta != 1.0) {
            for (int i = 0; i < cd.length; i++) {
                cd[i] *= beta;
            }
        }

        int vecLen = SPECIES.length();
        int microRows = microRowsForVector(vecLen);
        int kUnroll = kUnrollForVector(vecLen);

        for (int jj = 0; jj < n; jj += blockSize) {
            int jMax = Math.min(jj + blockSize, n);
            int blockWidth = jMax - jj;
            int packedN = roundUp(blockWidth, vecLen);
            for (int kk = 0; kk < k; kk += blockSize) {
                int kMax = Math.min(kk + blockSize, k);
                int kBlock = kMax - kk;
                double[] bPack = ensureCapacity(PACK_B_BUFFER, kBlock * packedN);
                packB(bd, n, kk, kBlock, jj, blockWidth, packedN, bPack);
                for (int ii = 0; ii < m; ii += blockSize) {
                    int iMax = Math.min(ii + blockSize, m);
                    for (int i = ii; i < iMax; i += microRows) {
                        int rows = Math.min(microRows, iMax - i);
                        double[] aPack = ensureCapacity(PACK_A_BUFFER, rows * kBlock);
                        packA(ad, k, i, rows, kk, kBlock, alpha, aPack);
                        int cOffset = i * n + jj;
                        microKernel(rows, kBlock, packedN, blockWidth, aPack, bPack,
                            cd, cOffset, n, vecLen, kUnroll);
                    }
                }
            }
        }
    }

    private static void microKernel(int rows, int kBlock, int packedN, int blockWidth,
                                    double[] aPack, double[] bPack, double[] c,
                                    int cOffset, int ldc, int vecLen, int kUnroll) {
        boolean aligned = (blockWidth % vecLen) == 0;
        int jLimit = aligned ? blockWidth : blockWidth - (blockWidth % vecLen);

        int row1Offset = kBlock;
        int row2Offset = row1Offset + kBlock;
        int row3Offset = row2Offset + kBlock;

        for (int j = 0; j < jLimit; j += vecLen) {
            int cBase = cOffset + j;
            DoubleVector c0 = DoubleVector.fromArray(SPECIES, c, cBase);
            DoubleVector c1 = rows > 1 ? DoubleVector.fromArray(SPECIES, c, cBase + ldc) : DoubleVector.zero(SPECIES);
            DoubleVector c2 = rows > 2 ? DoubleVector.fromArray(SPECIES, c, cBase + 2 * ldc) : DoubleVector.zero(SPECIES);
            DoubleVector c3 = rows > 3 ? DoubleVector.fromArray(SPECIES, c, cBase + 3 * ldc) : DoubleVector.zero(SPECIES);

            int p = 0;
            int kLimit = kBlock - (kBlock % kUnroll);
            for (; p < kLimit; p += kUnroll) {
                int bBase = p * packedN + j;
                DoubleVector b0 = DoubleVector.fromArray(SPECIES, bPack, bBase);
                DoubleVector b1 = DoubleVector.fromArray(SPECIES, bPack, bBase + packedN);
                DoubleVector b2 = DoubleVector.fromArray(SPECIES, bPack, bBase + 2 * packedN);
                DoubleVector b3 = DoubleVector.fromArray(SPECIES, bPack, bBase + 3 * packedN);

                if (p + PREFETCH_DISTANCE < kBlock) {
                    DoubleVector.fromArray(SPECIES, bPack, (p + PREFETCH_DISTANCE) * packedN + j);
                }

                double a00 = aPack[p];
                double a01 = aPack[p + 1];
                double a02 = aPack[p + 2];
                double a03 = aPack[p + 3];
                c0 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a00), c0);
                c0 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a01), c0);
                c0 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a02), c0);
                c0 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a03), c0);
                if (rows > 1) {
                    double a10 = aPack[row1Offset + p];
                    double a11 = aPack[row1Offset + p + 1];
                    double a12 = aPack[row1Offset + p + 2];
                    double a13 = aPack[row1Offset + p + 3];
                    c1 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a10), c1);
                    c1 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a11), c1);
                    c1 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a12), c1);
                    c1 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a13), c1);
                }
                if (rows > 2) {
                    double a20 = aPack[row2Offset + p];
                    double a21 = aPack[row2Offset + p + 1];
                    double a22 = aPack[row2Offset + p + 2];
                    double a23 = aPack[row2Offset + p + 3];
                    c2 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a20), c2);
                    c2 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a21), c2);
                    c2 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a22), c2);
                    c2 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a23), c2);
                }
                if (rows > 3) {
                    double a30 = aPack[row3Offset + p];
                    double a31 = aPack[row3Offset + p + 1];
                    double a32 = aPack[row3Offset + p + 2];
                    double a33 = aPack[row3Offset + p + 3];
                    c3 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a30), c3);
                    c3 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a31), c3);
                    c3 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a32), c3);
                    c3 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a33), c3);
                }
            }
            for (; p < kBlock; p++) {
                DoubleVector b0 = DoubleVector.fromArray(SPECIES, bPack, p * packedN + j);
                double a0 = aPack[p];
                c0 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a0), c0);
                if (rows > 1) {
                    double a1 = aPack[row1Offset + p];
                    c1 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a1), c1);
                }
                if (rows > 2) {
                    double a2 = aPack[row2Offset + p];
                    c2 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a2), c2);
                }
                if (rows > 3) {
                    double a3 = aPack[row3Offset + p];
                    c3 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a3), c3);
                }
            }

            c0.intoArray(c, cBase);
            if (rows > 1) {
                c1.intoArray(c, cBase + ldc);
            }
            if (rows > 2) {
                c2.intoArray(c, cBase + 2 * ldc);
            }
            if (rows > 3) {
                c3.intoArray(c, cBase + 3 * ldc);
            }
        }

        if (!aligned) {
            int j = jLimit;
            VectorMask<Double> mask = SPECIES.indexInRange(j, blockWidth);
            int cBase = cOffset + j;
            DoubleVector c0 = DoubleVector.fromArray(SPECIES, c, cBase, mask);
            DoubleVector c1 = rows > 1 ? DoubleVector.fromArray(SPECIES, c, cBase + ldc, mask) : DoubleVector.zero(SPECIES);
            DoubleVector c2 = rows > 2 ? DoubleVector.fromArray(SPECIES, c, cBase + 2 * ldc, mask) : DoubleVector.zero(SPECIES);
            DoubleVector c3 = rows > 3 ? DoubleVector.fromArray(SPECIES, c, cBase + 3 * ldc, mask) : DoubleVector.zero(SPECIES);

            for (int p = 0; p < kBlock; p++) {
                DoubleVector b0 = DoubleVector.fromArray(SPECIES, bPack, p * packedN + j, mask);
                double a0 = aPack[p];
                c0 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a0), c0);
                if (rows > 1) {
                    double a1 = aPack[row1Offset + p];
                    c1 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a1), c1);
                }
                if (rows > 2) {
                    double a2 = aPack[row2Offset + p];
                    c2 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a2), c2);
                }
                if (rows > 3) {
                    double a3 = aPack[row3Offset + p];
                    c3 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a3), c3);
                }
            }

            c0.intoArray(c, cBase, mask);
            if (rows > 1) {
                c1.intoArray(c, cBase + ldc, mask);
            }
            if (rows > 2) {
                c2.intoArray(c, cBase + 2 * ldc, mask);
            }
            if (rows > 3) {
                c3.intoArray(c, cBase + 3 * ldc, mask);
            }
        }
    }

    private static void packA(double[] ad, int k, int rowStart, int rows, int kk,
                              int kBlock, double alpha, double[] aPack) {
        int offset = 0;
        if (alpha == 1.0) {
            for (int r = 0; r < rows; r++) {
                int src = (rowStart + r) * k + kk;
                System.arraycopy(ad, src, aPack, offset, kBlock);
                offset += kBlock;
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int src = (rowStart + r) * k + kk;
                for (int p = 0; p < kBlock; p++) {
                    aPack[offset + p] = ad[src + p] * alpha;
                }
                offset += kBlock;
            }
        }
    }

    private static void packB(double[] bd, int n, int kk, int kBlock, int jj,
                              int blockWidth, int packedN, double[] bPack) {
        int dst = 0;
        for (int p = 0; p < kBlock; p++) {
            int src = (kk + p) * n + jj;
            System.arraycopy(bd, src, bPack, dst, blockWidth);
            if (packedN > blockWidth) {
                java.util.Arrays.fill(bPack, dst + blockWidth, dst + packedN, 0.0);
            }
            dst += packedN;
        }
    }

    private static int microRowsForVector(int vecLen) {
        if (vecLen >= 8) {
            return 4;
        }
        if (vecLen >= 4) {
            return 2;
        }
        return 1;
    }

    private static int kUnrollForVector(int vecLen) {
        if (vecLen >= 16) {
            return 8;
        }
        return DEFAULT_K_UNROLL;
    }

    private static int roundUp(int value, int multiple) {
        if (multiple <= 0) {
            return value;
        }
        int rem = value % multiple;
        return rem == 0 ? value : value + multiple - rem;
    }

    private static double[] ensureCapacity(ThreadLocal<double[]> buffer, int size) {
        double[] data = buffer.get();
        if (data == null || data.length < size) {
            data = new double[size];
            buffer.set(data);
        }
        return data;
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
        ForkJoinPool pool = new ForkJoinPool(threads);
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
            shutdownPool(pool);
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

    private static void shutdownPool(ForkJoinPool pool) {
        if (pool == null) {
            return;
        }
        pool.shutdown();
        try {
            if (!pool.awaitTermination(30, TimeUnit.SECONDS)) {
                pool.shutdownNow();
            }
        } catch (InterruptedException ex) {
            pool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
