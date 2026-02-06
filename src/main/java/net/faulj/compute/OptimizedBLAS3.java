package net.faulj.compute;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixView;
import net.faulj.matrix.OffHeapMatrix;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Refactored BLAS Level-3 kernels with unified dispatch and optimal kernel selection.
 *
 * Key improvements over previous implementation:
 * - Single code path: all variants route through unified microkernel
 * - Cache-aware blocking (MC/KC/NC based on hardware)
 * - Specialized fast paths (matvec, tiny, small)
 * - Persistent workspace and CUDA resources
 * - Proper parallel scheduling without overhead
 *
 * Performance targets:
 * - CPU: 60-80% of peak GFLOPS (150-200 GF on AVX-512)
 * - GPU: Effective use for large problems (>200M FLOPs)
 * - Memory: Minimize cache misses via optimal blocking
 */
public final class OptimizedBLAS3 {

    // Shared pool for parallel GEMM - avoids expensive pool creation per call
    private static final AtomicReference<ForkJoinPool> SHARED_POOL = new AtomicReference<>();

    private static ForkJoinPool getSharedPool(int threads) {
        ForkJoinPool pool = SHARED_POOL.get();
        // Create new pool if none exists or parallelism changed significantly
        if (pool == null || pool.isShutdown() || pool.getParallelism() != threads) {
            ForkJoinPool newPool = new ForkJoinPool(threads);
            if (SHARED_POOL.compareAndSet(pool, newPool)) {
                // Shut down old pool if we successfully replaced it
                if (pool != null && !pool.isShutdown()) {
                    pool.shutdown();
                }
                pool = newPool;
            } else {
                // Another thread beat us, shut down our new pool
                newPool.shutdown();
                pool = SHARED_POOL.get();
            }
        }
        return pool;
    }

    private OptimizedBLAS3() {}

    /**
     * Main GEMM entry point: C = alpha * A * B + beta * C.
     *
     * Automatically selects optimal kernel based on problem size and hardware.
     */
    public static void gemm(Matrix a, Matrix b, Matrix c,
                           double alpha, double beta, DispatchPolicy policy) {
        if (a == null || b == null || c == null) {
            throw new IllegalArgumentException("Matrices must not be null");
        }

        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();

        if (k != b.getRowCount()) {
            throw new IllegalArgumentException("Inner dimensions must agree");
        }
        if (m != c.getRowCount() || n != c.getColumnCount()) {
            throw new IllegalArgumentException("Output dimensions must match");
        }

        // Handle complex matrices (fallback to naive)
        if (a.getRawImagData() != null || b.getRawImagData() != null || c.getRawImagData() != null) {
            gemmNaiveComplex(a, b, c, alpha, beta);
            return;
        }

        // Get raw arrays
        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] cd = c.getRawData();

        // Dispatch to optimal kernel
        GemmWorkspace ws = GemmWorkspace.get();
        int threads = policy == null ? 1 : policy.getParallelism();

        GemmDispatch.Kernel kernel = GemmDispatch.selectKernel(m, n, k,
            ws.getCudaContext() != null && ws.getCudaContext().isAvailable(), threads);

        switch (kernel) {
            case TINY:
                SpecializedKernels.tinyGemm(ad, k, bd, n, cd, n, m, k, n, alpha, beta);
                break;

            case SMALL:
                SpecializedKernels.smallGemm(ad, k, bd, n, cd, n, m, k, n, alpha, beta);
                break;

            case MATVEC:
                SpecializedKernels.matvec(ad, k, bd, cd, m, k, alpha, beta);
                break;

            case CUDA:
                if (!gemmCuda(a, b, c, alpha, beta, ws)) {
                    // CUDA failed, fallback to microkernel
                    gemmMicrokernel(ad, k, bd, n, cd, n, m, k, n, alpha, beta, ws);
                }
                break;

            case PARALLEL_MICRO:
                gemmParallelMicrokernel(ad, k, bd, n, cd, n, m, k, n, alpha, beta, threads, ws);
                break;

            case MICROKERNEL:
            default:
                gemmMicrokernel(ad, k, bd, n, cd, n, m, k, n, alpha, beta, ws);
                break;
        }

        // Sync back if OffHeapMatrix
        if (c instanceof OffHeapMatrix) {
            ((OffHeapMatrix) c).syncToOffHeap();
        }
    }

    /**
     * GEMM with strided access (for MatrixView support).
     */
    public static void gemmStrided(double[] a, int aOffset, int lda,
                                  double[] b, int bOffset, int ldb,
                                  double[] c, int cOffset, int ldc,
                                  int m, int k, int n,
                                  double alpha, double beta) {
        GemmWorkspace ws = GemmWorkspace.get();
        GemmDispatch.Kernel kernel = GemmDispatch.selectKernel(m, n, k, false, 1);

        switch (kernel) {
            case TINY:
            case SMALL:
                gemmStridedNaive(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc,
                               m, k, n, alpha, beta);
                break;

            case MATVEC:
                // Extract x vector (B column)
                double[] x = new double[k];
                for (int i = 0; i < k; i++) {
                    x[i] = b[bOffset + i * ldb];
                }
                // Extract y vector (C column)
                double[] y = new double[m];
                for (int i = 0; i < m; i++) {
                    y[i] = c[cOffset + i * ldc];
                }
                // Compute A offset array
                double[] aMat = new double[m * k];
                for (int i = 0; i < m; i++) {
                    System.arraycopy(a, aOffset + i * lda, aMat, i * k, k);
                }
                SpecializedKernels.matvec(aMat, k, x, y, m, k, alpha, beta);
                // Write back
                for (int i = 0; i < m; i++) {
                    c[cOffset + i * ldc] = y[i];
                }
                break;

            case MICROKERNEL:
            default:
                gemmMicrokernelStrided(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc,
                                     m, k, n, alpha, beta, ws);
                break;
        }
    }

    /**
     * Single-threaded microkernel GEMM with optimal blocking.
     */
    private static void gemmMicrokernel(double[] a, int lda, double[] b, int ldb,
                                       double[] c, int ldc, int m, int k, int n,
                                       double alpha, double beta, GemmWorkspace ws) {
        // Apply beta to C
        PackingUtils.scaleCPanel(c, 0, m, n, ldc, beta);

        if (alpha == 0.0 || k == 0 || m == 0 || n == 0) {
            return;
        }

        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        int vecLen = blocks.nr;
        int mr = blocks.mr;

        // Three-level blocking: NC -> KC -> MC
        for (int jj = 0; jj < n; jj += blocks.nc) {
            int colEnd = Math.min(jj + blocks.nc, n);
            int nBlock = colEnd - jj;
            int packedN = PackingUtils.roundUp(nBlock, vecLen);

            for (int kk = 0; kk < k; kk += blocks.kc) {
                int kEnd = Math.min(kk + blocks.kc, k);
                int kBlock = kEnd - kk;

                // Pack B panel
                double[] bPack = ws.getPackB(kBlock * packedN);
                PackingUtils.packB(b, ldb, kk, kBlock, jj, nBlock, packedN, bPack);

                for (int ii = 0; ii < m; ii += blocks.mc) {
                    int rowEnd = Math.min(ii + blocks.mc, m);

                    // Process MC block in MR strips
                    for (int i = ii; i < rowEnd; i += mr) {
                        int mBlock = Math.min(mr, rowEnd - i);

                        // Pack A panel
                        double[] aPack = ws.getPackA(mBlock * kBlock);
                        PackingUtils.packA(a, lda, i, mBlock, kk, kBlock, alpha, aPack);

                        // Call microkernel
                        int cOffset = i * ldc + jj;
                        MicroKernel.compute(mBlock, kBlock, packedN, nBlock,
                                          aPack, bPack, c, cOffset, ldc);
                    }
                }
            }
        }
    }

    /**
     * Strided microkernel GEMM.
     */
    private static void gemmMicrokernelStrided(double[] a, int aOffset, int lda,
                                              double[] b, int bOffset, int ldb,
                                              double[] c, int cOffset, int ldc,
                                              int m, int k, int n,
                                              double alpha, double beta, GemmWorkspace ws) {
        // Apply beta to C
        if (beta == 0.0) {
            for (int i = 0; i < m; i++) {
                java.util.Arrays.fill(c, cOffset + i * ldc, cOffset + i * ldc + n, 0.0);
            }
        } else if (beta != 1.0) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    c[cOffset + i * ldc + j] *= beta;
                }
            }
        }

        if (alpha == 0.0 || k == 0 || m == 0 || n == 0) {
            return;
        }

        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        int vecLen = blocks.nr;
        int mr = blocks.mr;

        for (int jj = 0; jj < n; jj += blocks.nc) {
            int colEnd = Math.min(jj + blocks.nc, n);
            int nBlock = colEnd - jj;
            int packedN = PackingUtils.roundUp(nBlock, vecLen);

            for (int kk = 0; kk < k; kk += blocks.kc) {
                int kEnd = Math.min(kk + blocks.kc, k);
                int kBlock = kEnd - kk;

                double[] bPack = ws.getPackB(kBlock * packedN);
                PackingUtils.packB(b, ldb, bOffset + kk * ldb, kBlock, jj, nBlock, packedN, bPack);

                for (int ii = 0; ii < m; ii += blocks.mc) {
                    int rowEnd = Math.min(ii + blocks.mc, m);

                    for (int i = ii; i < rowEnd; i += mr) {
                        int mBlock = Math.min(mr, rowEnd - i);
                        double[] aPack = ws.getPackA(mBlock * kBlock);
                        PackingUtils.packA(a, lda, aOffset + i * lda, mBlock, kk, kBlock, alpha, aPack);

                        int cOff = cOffset + i * ldc + jj;
                        MicroKernel.compute(mBlock, kBlock, packedN, nBlock,
                                          aPack, bPack, c, cOff, ldc);
                    }
                }
            }
        }
    }

    /**
     * Parallel tiled microkernel GEMM.
     * Tiles C by (MC × NC) blocks and processes tiles in parallel.
     */
    private static void gemmParallelMicrokernel(double[] a, int lda, double[] b, int ldb,
                                               double[] c, int ldc, int m, int k, int n,
                                               double alpha, double beta, int threads,
                                               GemmWorkspace ws) {
        // Apply beta to C
        PackingUtils.scaleCPanel(c, 0, m, n, ldc, beta);

        if (alpha == 0.0 || k == 0 || m == 0 || n == 0) {
            return;
        }

        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        int actualThreads = GemmDispatch.optimalParallelism(m, n, k, threads, blocks);

        if (actualThreads <= 1) {
            gemmMicrokernel(a, lda, b, ldb, c, ldc, m, k, n, alpha, beta, ws);
            return;
        }

        int blockRows = (m + blocks.mc - 1) / blocks.mc;
        int blockCols = (n + blocks.nc - 1) / blocks.nc;

        ForkJoinPool pool = getSharedPool(actualThreads);
        try {
            List<RecursiveAction> tasks = new ArrayList<>();

            for (int br = 0; br < blockRows; br++) {
                for (int bc = 0; bc < blockCols; bc++) {
                    final int blockRow = br;
                    final int blockCol = bc;

                    tasks.add(new RecursiveAction() {
                        @Override
                        protected void compute() {
                            int ii = blockRow * blocks.mc;
                            int jj = blockCol * blocks.nc;
                            int rowEnd = Math.min(ii + blocks.mc, m);
                            int colEnd = Math.min(jj + blocks.nc, n);

                            // Get thread-local workspace
                            GemmWorkspace localWs = GemmWorkspace.get();
                            computeTile(a, lda, b, ldb, c, ldc, k,
                                      ii, rowEnd, jj, colEnd,
                                      alpha, blocks, localWs);
                        }
                    });
                }
            }

            // Invoke all tasks
            ForkJoinTask.invokeAll(tasks);
        } catch (Exception e) {
            // Fallback to sequential
            gemmMicrokernel(a, lda, b, ldb, c, ldc, m, k, n, alpha, beta, ws);
        }
        // Note: Do NOT shutdown the shared pool - it's reused across calls
    }

    /**
     * Compute one tile of C for parallel GEMM.
     */
    private static void computeTile(double[] a, int lda, double[] b, int ldb,
                                   double[] c, int ldc, int k,
                                   int rowStart, int rowEnd,
                                   int colStart, int colEnd,
                                   double alpha, GemmDispatch.BlockSizes blocks,
                                   GemmWorkspace ws) {
        int nBlock = colEnd - colStart;
        int packedN = PackingUtils.roundUp(nBlock, blocks.nr);

        for (int kk = 0; kk < k; kk += blocks.kc) {
            int kEnd = Math.min(kk + blocks.kc, k);
            int kBlock = kEnd - kk;

            // Pack B panel (shared across rows of this tile)
            double[] bPack = ws.getPackB(kBlock * packedN);
            PackingUtils.packB(b, ldb, kk, kBlock, colStart, nBlock, packedN, bPack);

            // Process rows in MR strips
            for (int i = rowStart; i < rowEnd; i += blocks.mr) {
                int mBlock = Math.min(blocks.mr, rowEnd - i);

                double[] aPack = ws.getPackA(mBlock * kBlock);
                PackingUtils.packA(a, lda, i, mBlock, kk, kBlock, alpha, aPack);

                int cOffset = i * ldc + colStart;
                MicroKernel.compute(mBlock, kBlock, packedN, nBlock,
                                  aPack, bPack, c, cOffset, ldc);
            }
        }
    }

    /**
     * CUDA GEMM with persistent resources.
     */
    private static boolean gemmCuda(Matrix a, Matrix b, Matrix c,
                                   double alpha, double beta, GemmWorkspace ws) {
        CudaContext ctx = ws.getCudaContext();
        if (ctx == null || !ctx.isAvailable()) {
            return false;
        }

        try {
            // Use standard gemm (context is checked above but not passed to gemm)
            return CudaGemm.gemm(a, b, c, alpha, beta);
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Naive strided GEMM for small/tiny cases.
     */
    private static void gemmStridedNaive(double[] a, int aOffset, int lda,
                                        double[] b, int bOffset, int ldb,
                                        double[] c, int cOffset, int ldc,
                                        int m, int k, int n,
                                        double alpha, double beta) {
        // Apply beta
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = cOffset + i * ldc + j;
                c[idx] = (beta == 0.0) ? 0.0 : c[idx] * beta;
            }
        }

        if (alpha == 0.0 || k == 0) {
            return;
        }

        // Compute C += alpha * A * B
        for (int i = 0; i < m; i++) {
            for (int kk = 0; kk < k; kk++) {
                double aVal = a[aOffset + i * lda + kk] * alpha;
                for (int j = 0; j < n; j++) {
                    int cIdx = cOffset + i * ldc + j;
                    int bIdx = bOffset + kk * ldb + j;
                    c[cIdx] = Math.fma(aVal, b[bIdx], c[cIdx]);
                }
            }
        }
    }

    /**
     * Fallback for complex matrices (not optimized).
     */
    private static void gemmNaiveComplex(Matrix a, Matrix b, Matrix c,
                                        double alpha, double beta) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();

        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] cd = c.getRawData();
        double[] ai = a.getRawImagData();
        double[] bi = b.getRawImagData();
        double[] ci = c.ensureImagData();

        // Apply beta to C
        for (int idx = 0; idx < cd.length; idx++) {
            cd[idx] *= beta;
            ci[idx] *= beta;
        }

        if (alpha == 0.0 || k == 0) {
            return;
        }

        // Complex multiply
        for (int i = 0; i < m; i++) {
            int aRowOffset = i * k;
            int cRowOffset = i * n;
            for (int kk = 0; kk < k; kk++) {
                double aRe = ad[aRowOffset + kk];
                double aIm = (ai == null) ? 0.0 : ai[aRowOffset + kk];
                int bRowOffset = kk * n;

                for (int j = 0; j < n; j++) {
                    double bRe = bd[bRowOffset + j];
                    double bIm = (bi == null) ? 0.0 : bi[bRowOffset + j];

                    // (aRe + i*aIm) * (bRe + i*bIm) * alpha
                    double prodRe = (aRe * bRe - aIm * bIm) * alpha;
                    double prodIm = (aRe * bIm + aIm * bRe) * alpha;

                    int cIdx = cRowOffset + j;
                    cd[cIdx] += prodRe;
                    ci[cIdx] += prodIm;
                }
            }
        }
    }
}
