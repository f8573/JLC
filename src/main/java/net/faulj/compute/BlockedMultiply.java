package net.faulj.compute;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;
import java.util.function.IntConsumer;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Implements cache-optimized blocked matrix multiplication algorithms.
 */
public class BlockedMultiply {
    private static final double ZERO_EPS = 1e-15;
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final ConcurrentHashMap<Integer, ForkJoinPool> POOLS = new ConcurrentHashMap<>();
    private static final int PACK_THRESHOLD = 4096;
    private static final int DENSITY_SAMPLE_LIMIT = 4096;
    private static final double DENSE_THRESHOLD = 0.90;
    private static final long TRANSPOSE_THRESHOLD = 16_384L;
    private static final ThreadLocal<double[]> PACK_BUFFER = new ThreadLocal<>();

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(BlockedMultiply::shutdownPools, "faulj-blockedmultiply-shutdown"));
    }

    private BlockedMultiply() {
    }

    /**
     * Multiply two matrices using the default dispatch policy.
     *
     * @param a left matrix
     * @param b right matrix
     * @return product matrix
     */
    public static Matrix multiply(Matrix a, Matrix b) {
        return multiply(a, b, DispatchPolicy.defaultPolicy());
    }

    /**
     * Multiply two matrices using a specific dispatch policy.
     *
     * @param a left matrix
     * @param b right matrix
     * @param policy dispatch policy
     * @return product matrix
     */
    public static Matrix multiply(Matrix a, Matrix b, DispatchPolicy policy) {
        RuntimeProfile.applyConfiguredProfile();
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
            return (a instanceof OffHeapMatrix || b instanceof OffHeapMatrix)
                ? new OffHeapMatrix(m, n)
                : net.faulj.matrix.Matrix.zero(m, n);
        }
        if (n == 1) {
            return multiplyVector(a, b);
        }
        DispatchPolicy active = policy == null ? DispatchPolicy.defaultPolicy() : policy;
        DispatchPolicy.Algorithm algorithm = active.selectForMultiply(m, n, k);
        boolean parallel = active.shouldParallelize(m, n, k);
        int blockSize = active.blockSize(m, n, k);
        int parallelism = active.getParallelism();
        try {
            if (algorithm == DispatchPolicy.Algorithm.CUDA) {
                Matrix cuda = CudaGemm.multiply(a, b);
                if (cuda != null) {
                    return cuda;
                }
                algorithm = active.selectCpuAlgorithm(m, n, k);
            }
            switch (algorithm) {
                case SIMD:
                    return multiplySimdFlat(a, b, blockSize);
                case BLAS3:
                    return BLAS3Kernels.gemm(a, b, active);
                case PARALLEL:
                    return multiplyParallel(a, b, blockSize, parallelism);
                case BLOCKED, STRASSEN:
                    if (parallel) {
                        return multiplyParallel(a, b, blockSize, parallelism);
                    }
                    return multiplyBlocked(a, b, blockSize);
                case NAIVE, SPECIALIZED:
                default:
                    if (parallel) {
                        return multiplyNaiveParallel(a, b, parallelism);
                    }
                    return multiplyNaive(a, b);
                }
        } catch (OutOfMemoryError oom) {
            // Release thread-local scratch and force a low-overhead fallback path.
            PACK_BUFFER.remove();
            System.gc();
            return multiplyNaive(a, b);
        }
    }

    private static Matrix multiplySimdFlat(Matrix a, Matrix b, int blockSize) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();

        if (a instanceof OffHeapMatrix && b instanceof OffHeapMatrix) {
            return multiplySimdFlatOffHeap((OffHeapMatrix) a, (OffHeapMatrix) b, m, k, n, blockSize);
        }

        Matrix result = new Matrix(m, n);
        double[] c = result.getRawData();
        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] ai = a.getRawImagData();
        double[] bi = b.getRawImagData();
        if (ai != null || bi != null) {
            return multiplyNaive(a, b);
        }
        int bs = Math.max(1, blockSize);
        BLAS3Kernels.dgemmSimdPacked(ad, bd, c, m, k, n, 1.0, 0.0, bs);

        return result;
    }

    private static Matrix multiplySimdFlatOffHeap(OffHeapMatrix a, OffHeapMatrix b, int m, int k, int n, int blockSize) {
        OffHeapMatrix result = new OffHeapMatrix(m, n);
        if (a.getRawImagData() != null || b.getRawImagData() != null) {
            return multiplyNaive(a, b);
        }

        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] cd = result.getRawData();
        int bs = Math.max(1, blockSize);
        BLAS3Kernels.dgemmSimdPacked(ad, bd, cd, m, k, n, 1.0, 0.0, bs);
        result.syncToOffHeap();
        return result;
    }

    private static Matrix multiplyVector(Matrix a, Matrix b) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();
        int threads = Math.max(1, Runtime.getRuntime().availableProcessors());

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
        boolean skipZeros = shouldSkipZeros(ad, ai);
        if (!complex && !skipZeros && (long) k * n >= TRANSPOSE_THRESHOLD) {
            double[] bt = transposeToArray(bd, k, n);
            ForkJoinPool pool = getPool(threads);
            try {
                runChunked(pool, m, rowStart -> {
                    int aRowOffset = rowStart * k;
                    int cRowOffset = rowStart * n;
                    for (int j = 0; j < n; j++) {
                        int btRowOffset = j * k;
                        double sum = 0.0;
                        for (int kk = 0; kk < k; kk++) {
                            sum = Math.fma(ad[aRowOffset + kk], bt[btRowOffset + kk], sum);
                        }
                        c[cRowOffset + j] = sum;
                    }
                });
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                return multiplyNaive(a, b);
            } catch (ExecutionException ex) {
                return multiplyNaive(a, b);
            }
            return result;
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

    /**
     * Multiply using a naive triple-loop implementation.
     *
     * @param a left matrix
     * @param b right matrix
     * @return product matrix
     */
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
        boolean skipZeros = shouldSkipZeros(ad, ai);
        if (!complex && !skipZeros && (long) k * n >= TRANSPOSE_THRESHOLD) {
            double[] bt = transposeToArray(bd, k, n);
            for (int i = 0; i < m; i++) {
                int aRowOffset = i * k;
                int cRowOffset = i * n;
                for (int j = 0; j < n; j++) {
                    int btRowOffset = j * k;
                    double sum = 0.0;
                    for (int kk = 0; kk < k; kk++) {
                        sum = Math.fma(ad[aRowOffset + kk], bt[btRowOffset + kk], sum);
                    }
                    c[cRowOffset + j] = sum;
                }
            }
            return result;
        }
        if (skipZeros) {
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
        } else {
            for (int i = 0; i < m; i++) {
                int aRowOffset = i * k;
                int cRowOffset = i * n;
                for (int kk = 0; kk < k; kk++) {
                    double aVal = ad[aRowOffset + kk];
                    double aImg = ai == null ? 0.0 : ai[aRowOffset + kk];
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
        }

        return result;
    }

    /**
     * Multiply using a cache-blocked algorithm.
     *
     * @param a left matrix
     * @param b right matrix
     * @param blockSize block size
     * @return product matrix
     */
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
        boolean skipZeros = shouldSkipZeros(ad, ai);

        if (!complex && !skipZeros) {
            BLAS3Kernels.dgemmSimdPacked(ad, bd, c, m, k, n, 1.0, 0.0, bs);
            return result;
        }

        int vecLen = SPECIES.length();
        for (int ii = 0; ii < m; ii += bs) {
            int iMax = Math.min(ii + bs, m);
            for (int kk = 0; kk < k; kk += bs) {
                int kMax = Math.min(kk + bs, k);
                for (int jj = 0; jj < n; jj += bs) {
                    int jMax = Math.min(jj + bs, n);
                    int jBlock = jMax - jj;
                    int loopBound = SPECIES.loopBound(jBlock) + jj;
                    double[] bPack = null;
                    if (!complex && (long) (kMax - kk) * jBlock >= PACK_THRESHOLD) {
                        bPack = getPackBuffer((kMax - kk) * jBlock);
                        int packRow = 0;
                        for (int kIndex = kk; kIndex < kMax; kIndex++) {
                            int bRowOffset = kIndex * n + jj;
                            System.arraycopy(bd, bRowOffset, bPack, packRow, jBlock);
                            packRow += jBlock;
                        }
                    }
                    for (int i = ii; i < iMax; i++) {
                        int aRowOffset = i * k;
                        int cRowOffset = i * n;
                        for (int kIndex = kk; kIndex < kMax; kIndex++) {
                            double aVal = ad[aRowOffset + kIndex];
                            double aImg = ai == null ? 0.0 : ai[aRowOffset + kIndex];
                            if (skipZeros && Math.abs(aVal) <= ZERO_EPS && Math.abs(aImg) <= ZERO_EPS) {
                                continue;
                            }
                            if (!complex) {
                                DoubleVector av = DoubleVector.broadcast(SPECIES, aVal);
                                if (bPack == null) {
                                    int bRowOffset = kIndex * n;
                                    int j = jj;
                                    for (; j < loopBound; j += vecLen) {
                                        int bIndex = bRowOffset + j;
                                        int cIndex = cRowOffset + j;
                                        DoubleVector bv = DoubleVector.fromArray(SPECIES, bd, bIndex);
                                        DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cIndex);
                                        cv = bv.lanewise(VectorOperators.FMA, av, cv);
                                        cv.intoArray(c, cIndex);
                                    }
                                    if (j < jMax) {
                                        VectorMask<Double> mask = SPECIES.indexInRange(j, jMax);
                                        int bIndex = bRowOffset + j;
                                        int cIndex = cRowOffset + j;
                                        DoubleVector bv = DoubleVector.fromArray(SPECIES, bd, bIndex, mask);
                                        DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cIndex, mask);
                                        cv = bv.lanewise(VectorOperators.FMA, av, cv);
                                        cv.intoArray(c, cIndex, mask);
                                    }
                                } else {
                                    int packRowOffset = (kIndex - kk) * jBlock;
                                    int jLocal = 0;
                                    int packLoopBound = SPECIES.loopBound(jBlock);
                                    for (; jLocal < packLoopBound; jLocal += vecLen) {
                                        int bIndex = packRowOffset + jLocal;
                                        int cIndex = cRowOffset + jj + jLocal;
                                        DoubleVector bv = DoubleVector.fromArray(SPECIES, bPack, bIndex);
                                        DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cIndex);
                                        cv = bv.lanewise(VectorOperators.FMA, av, cv);
                                        cv.intoArray(c, cIndex);
                                    }
                                    if (jLocal < jBlock) {
                                        VectorMask<Double> mask = SPECIES.indexInRange(jLocal, jBlock);
                                        int bIndex = packRowOffset + jLocal;
                                        int cIndex = cRowOffset + jj + jLocal;
                                        DoubleVector bv = DoubleVector.fromArray(SPECIES, bPack, bIndex, mask);
                                        DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cIndex, mask);
                                        cv = bv.lanewise(VectorOperators.FMA, av, cv);
                                        cv.intoArray(c, cIndex, mask);
                                    }
                                }
                            } else {
                                int bRowOffset = kIndex * n;
                                int cIndex = cRowOffset + jj;
                                int bIndex = bRowOffset + jj;
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

    /**
     * Multiply using parallel blocked multiplication.
     *
     * @param a left matrix
     * @param b right matrix
     * @param blockSize block size
     * @param parallelism number of threads
     * @return product matrix
     */
    public static Matrix multiplyParallel(Matrix a, Matrix b, int blockSize, int parallelism) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();
        int bs = Math.max(1, blockSize);
        int vecLen = SPECIES.length();

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
        boolean skipZeros = shouldSkipZeros(ad, ai);

        if (!complex && !skipZeros && threads <= 1) {
            BLAS3Kernels.dgemmSimdPacked(ad, bd, c, m, k, n, 1.0, 0.0, bs);
            return result;
        }

        ForkJoinPool pool = getPool(threads);
        try {
            runChunked(pool, blockRows, blockRow -> {
                int ii = blockRow * bs;
                int iMax = Math.min(ii + bs, m);
                for (int kk = 0; kk < k; kk += bs) {
                    int kMax = Math.min(kk + bs, k);
                    for (int jj = 0; jj < n; jj += bs) {
                        int jMax = Math.min(jj + bs, n);
                        int jBlock = jMax - jj;
                        int loopBound = SPECIES.loopBound(jBlock) + jj;
                        double[] bPack = null;
                        if (!complex && (long) (kMax - kk) * jBlock >= PACK_THRESHOLD) {
                            bPack = getPackBuffer((kMax - kk) * jBlock);
                            int packRow = 0;
                            for (int kIndex = kk; kIndex < kMax; kIndex++) {
                                int bRowOffset = kIndex * n + jj;
                                System.arraycopy(bd, bRowOffset, bPack, packRow, jBlock);
                                packRow += jBlock;
                            }
                        }
                        for (int i = ii; i < iMax; i++) {
                            int aRowOffset = i * k;
                            int cRowOffset = i * n;
                            for (int kIndex = kk; kIndex < kMax; kIndex++) {
                                double aVal = ad[aRowOffset + kIndex];
                                double aImg = ai == null ? 0.0 : ai[aRowOffset + kIndex];
                                if (skipZeros && Math.abs(aVal) <= ZERO_EPS && Math.abs(aImg) <= ZERO_EPS) {
                                    continue;
                                }
                                if (!complex) {
                                    DoubleVector av = DoubleVector.broadcast(SPECIES, aVal);
                                    if (bPack == null) {
                                        int bRowOffset = kIndex * n;
                                        int j = jj;
                                        for (; j < loopBound; j += vecLen) {
                                            int bIndex = bRowOffset + j;
                                            int cIndex = cRowOffset + j;
                                            DoubleVector bv = DoubleVector.fromArray(SPECIES, bd, bIndex);
                                            DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cIndex);
                                            cv = bv.lanewise(VectorOperators.FMA, av, cv);
                                            cv.intoArray(c, cIndex);
                                        }
                                        if (j < jMax) {
                                            VectorMask<Double> mask = SPECIES.indexInRange(j, jMax);
                                            int bIndex = bRowOffset + j;
                                            int cIndex = cRowOffset + j;
                                            DoubleVector bv = DoubleVector.fromArray(SPECIES, bd, bIndex, mask);
                                            DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cIndex, mask);
                                            cv = bv.lanewise(VectorOperators.FMA, av, cv);
                                            cv.intoArray(c, cIndex, mask);
                                        }
                                    } else {
                                        int packRowOffset = (kIndex - kk) * jBlock;
                                        int jLocal = 0;
                                        int packLoopBound = SPECIES.loopBound(jBlock);
                                        for (; jLocal < packLoopBound; jLocal += vecLen) {
                                            int bIndex = packRowOffset + jLocal;
                                            int cIndex = cRowOffset + jj + jLocal;
                                            DoubleVector bv = DoubleVector.fromArray(SPECIES, bPack, bIndex);
                                            DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cIndex);
                                            cv = bv.lanewise(VectorOperators.FMA, av, cv);
                                            cv.intoArray(c, cIndex);
                                        }
                                        if (jLocal < jBlock) {
                                            VectorMask<Double> mask = SPECIES.indexInRange(jLocal, jBlock);
                                            int bIndex = packRowOffset + jLocal;
                                            int cIndex = cRowOffset + jj + jLocal;
                                            DoubleVector bv = DoubleVector.fromArray(SPECIES, bPack, bIndex, mask);
                                            DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cIndex, mask);
                                            cv = bv.lanewise(VectorOperators.FMA, av, cv);
                                            cv.intoArray(c, cIndex, mask);
                                        }
                                    }
                                } else {
                                    int bRowOffset = kIndex * n;
                                    int cIndex = cRowOffset + jj;
                                    int bIndex = bRowOffset + jj;
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
            });
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            return multiplyBlocked(a, b, bs);
        } catch (ExecutionException ex) {
            return multiplyBlocked(a, b, bs);
        }

        return result;
    }

    /**
     * Multiply using a naive algorithm with parallel row partitioning.
     *
     * @param a left matrix
     * @param b right matrix
     * @param parallelism number of threads
     * @return product matrix
     */
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
        boolean skipZeros = shouldSkipZeros(ad, ai);

        ForkJoinPool pool = getPool(threads);
        try {
            double[] finalCi = ci;
            boolean finalComplex = complex;
            boolean finalSkipZeros = skipZeros;
            runChunked(pool, m, i -> {
                int aRowOffset = i * k;
                int cRowOffset = i * n;
                for (int kk = 0; kk < k; kk++) {
                    double aVal = ad[aRowOffset + kk];
                    double aImg = ai == null ? 0.0 : ai[aRowOffset + kk];
                    if (finalSkipZeros && Math.abs(aVal) <= ZERO_EPS && Math.abs(aImg) <= ZERO_EPS) {
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
            });
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            return multiplyNaive(a, b);
        } catch (ExecutionException ex) {
            return multiplyNaive(a, b);
        }

        return result;
    }

    private static boolean shouldSkipZeros(double[] real, double[] imag) {
        double density = estimateDensity(real, imag);
        return density < DENSE_THRESHOLD;
    }

    private static double estimateDensity(double[] real, double[] imag) {
        int len = real == null ? 0 : real.length;
        if (len == 0) {
            return 1.0;
        }
        int samples = Math.min(DENSITY_SAMPLE_LIMIT, len);
        int step = Math.max(1, len / samples);
        int nonZero = 0;
        int seen = 0;
        for (int idx = 0; idx < len && seen < samples; idx += step, seen++) {
            double re = real[idx];
            double im = imag == null ? 0.0 : imag[idx];
            if (Math.abs(re) > ZERO_EPS || Math.abs(im) > ZERO_EPS) {
                nonZero++;
            }
        }
        return seen == 0 ? 1.0 : (double) nonZero / (double) seen;
    }

    private static double[] transposeToArray(double[] data, int rows, int cols) {
        double[] out = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            int rowOffset = r * cols;
            for (int c = 0; c < cols; c++) {
                out[c * rows + r] = data[rowOffset + c];
            }
        }
        return out;
    }

    private static void runChunked(ForkJoinPool pool, int total, IntConsumer body)
            throws InterruptedException, ExecutionException {
        int parallelism = Math.max(1, pool.getParallelism());
        int chunk = Math.max(1, total / (parallelism * 4));
        if (total <= chunk) {
            for (int i = 0; i < total; i++) {
                body.accept(i);
            }
            return;
        }
        List<ForkJoinTask<?>> tasks = new ArrayList<>();
        for (int start = 0; start < total; start += chunk) {
            int end = Math.min(total, start + chunk);
            tasks.add(new RangeTask(start, end, body));
        }
        pool.submit(() -> ForkJoinTask.invokeAll(tasks)).get();
    }

    private static final class RangeTask extends RecursiveAction {
        private final int start;
        private final int end;
        private final IntConsumer body;

        private RangeTask(int start, int end, IntConsumer body) {
            this.start = start;
            this.end = end;
            this.body = body;
        }

        @Override
        protected void compute() {
            for (int i = start; i < end; i++) {
                body.accept(i);
            }
        }
    }

    private static double[] getPackBuffer(int size) {
        double[] buf = PACK_BUFFER.get();
        if (buf == null || buf.length < size) {
            buf = new double[size];
            PACK_BUFFER.set(buf);
        }
        return buf;
    }

    private static ForkJoinPool getPool(int parallelism) {
        int maxThreads = Math.max(1, Runtime.getRuntime().availableProcessors());
        int threads = Math.max(1, Math.min(parallelism, maxThreads));
        return POOLS.computeIfAbsent(threads, ForkJoinPool::new);
    }

    static void shutdownPools() {
        for (ForkJoinPool pool : POOLS.values()) {
            if (pool == null) {
                continue;
            }
            pool.shutdownNow();
        }
        POOLS.clear();
    }
}
