package net.faulj.compute;

import net.faulj.matrix.Matrix;

/**
 * Provides optimized Level-3 BLAS (Basic Linear Algebra Subprograms) kernel operations.
 * <p>
 * Level-3 BLAS operations perform matrix-matrix operations with O(n³) complexity, making them
 * excellent candidates for cache-optimized and vectorized implementations. This class encapsulates
 * high-performance kernels for operations such as matrix multiplication (GEMM), triangular solve
 * (TRSM), and symmetric rank-k update (SYRK).
 * </p>
 *
 * <h2>Key Features:</h2>
 * <ul>
 *   <li>Cache-friendly blocked algorithms for improved data locality</li>
 *   <li>SIMD-optimized inner loops for modern CPU architectures</li>
 *   <li>Support for various matrix storage layouts (row-major, column-major)</li>
 *   <li>Thread-safe implementations suitable for parallel execution</li>
 * </ul>
 *
 * <h2>Performance Characteristics:</h2>
 * <p>
 * BLAS-3 operations achieve high arithmetic intensity (flops per memory access), making them
 * significantly more efficient than BLAS-1 or BLAS-2 operations. Typical performance:
 * </p>
 * <ul>
 *   <li>Matrix multiplication (GEMM): O(n³) operations, O(n²) memory accesses</li>
 *   <li>Optimal block sizes typically range from 32×32 to 256×256 depending on cache architecture</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see BlockedMultiply
 * @see DispatchPolicy
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
        if (policy == null) {
            policy = DispatchPolicy.getGlobalPolicy();
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
        if (policy == null) {
            policy = DispatchPolicy.getGlobalPolicy();
        }
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int k2 = b.getRowCount();
        int n = b.getColumnCount();
        if (k != k2) {
            throw new IllegalArgumentException("Inner dimensions must agree for multiplication: " + k + " != " + k2);
        }
        if (c.getRowCount() != m || c.getColumnCount() != n) {
            throw new IllegalArgumentException("Output matrix has invalid dimensions");
        }
        if (m == 0 || n == 0 || k == 0) {
            scale(c, beta);
            return;
        }

        DispatchPolicy.Algorithm algorithm = policy.selectForMultiply(m, n, k);
        int blockSize = policy.blockSize(m, n, k);
        switch (algorithm) {
            case NAIVE:
                dgemmNaive(a, b, c, alpha, beta);
                break;
            case CUDA:
                dgemm(a, b, c, alpha, beta, blockSize);
                break;
            case PARALLEL:
                dgemmParallel(a, b, c, alpha, beta, blockSize, policy.getParallelism());
                break;
            case BLOCKED:
            case STRASSEN:
            case SPECIALIZED:
            default:
                dgemm(a, b, c, alpha, beta, blockSize);
                break;
        }
    }

    static void dgemm(Matrix a, Matrix b, Matrix c, double alpha, double beta, int blockSize) {
        if (blockSize <= 0) {
            dgemmNaive(a, b, c, alpha, beta);
            return;
        }
        scale(c, beta);

        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();
        int colBlock = blockSize;

        for (int jBlock = 0; jBlock < n; jBlock += colBlock) {
            int jEnd = Math.min(n, jBlock + colBlock);
            dgemmBlockedRange(a, b, c, alpha, blockSize, jBlock, jEnd);
        }
    }

    static void dgemmParallel(Matrix a, Matrix b, Matrix c, double alpha, double beta, int blockSize, int parallelism) {
        if (parallelism <= 1 || blockSize <= 0) {
            dgemm(a, b, c, alpha, beta, blockSize);
            return;
        }
        scale(c, beta);

        int n = b.getColumnCount();
        int colBlock = blockSize;
        int blocks = (n + colBlock - 1) / colBlock;
        if (blocks <= 1) {
            dgemmBlockedRange(a, b, c, alpha, blockSize, 0, n);
            return;
        }

        java.util.concurrent.ForkJoinPool pool = new java.util.concurrent.ForkJoinPool(parallelism);
        try {
            pool.submit(() -> java.util.stream.IntStream.range(0, blocks).parallel().forEach(block -> {
                int jStart = block * colBlock;
                int jEnd = Math.min(n, jStart + colBlock);
                dgemmBlockedRange(a, b, c, alpha, blockSize, jStart, jEnd);
            })).join();
        } finally {
            pool.shutdown();
        }
    }

    private static void dgemmNaive(Matrix a, Matrix b, Matrix c, double alpha, double beta) {
        scale(c, beta);
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();

        net.faulj.vector.Vector[] aCols = a.getData();
        net.faulj.vector.Vector[] bCols = b.getData();
        net.faulj.vector.Vector[] cCols = c.getData();
        boolean scale = alpha != 1.0;

        for (int j = 0; j < n; j++) {
            double[] cCol = cCols[j].getData();
            double[] bCol = bCols[j].getData();
            for (int kk = 0; kk < k; kk++) {
                double bVal = bCol[kk];
                if (Math.abs(bVal) <= ZERO_EPS) {
                    continue;
                }
                if (scale) {
                    bVal *= alpha;
                }
                double[] aCol = aCols[kk].getData();
                for (int i = 0; i < m; i++) {
                    cCol[i] += aCol[i] * bVal;
                }
            }
        }
    }

    private static void dgemmBlockedRange(Matrix a, Matrix b, Matrix c, double alpha, int blockSize,
                                          int jStart, int jEnd) {
        int m = a.getRowCount();
        int k = a.getColumnCount();

        net.faulj.vector.Vector[] aCols = a.getData();
        net.faulj.vector.Vector[] bCols = b.getData();
        net.faulj.vector.Vector[] cCols = c.getData();
        boolean scale = alpha != 1.0;

        for (int kBlock = 0; kBlock < k; kBlock += blockSize) {
            int kEnd = Math.min(k, kBlock + blockSize);
            for (int iBlock = 0; iBlock < m; iBlock += blockSize) {
                int iEnd = Math.min(m, iBlock + blockSize);
                for (int j = jStart; j < jEnd; j++) {
                    double[] cCol = cCols[j].getData();
                    double[] bCol = bCols[j].getData();
                    for (int kk = kBlock; kk < kEnd; kk++) {
                        double bVal = bCol[kk];
                        if (Math.abs(bVal) <= ZERO_EPS) {
                            continue;
                        }
                        if (scale) {
                            bVal *= alpha;
                        }
                        double[] aCol = aCols[kk].getData();
                        for (int i = iBlock; i < iEnd; i++) {
                            cCol[i] += aCol[i] * bVal;
                        }
                    }
                }
            }
        }
    }

    private static void scale(Matrix c, double beta) {
        if (beta == 1.0) {
            return;
        }
        net.faulj.vector.Vector[] cCols = c.getData();
        if (beta == 0.0) {
            for (net.faulj.vector.Vector col : cCols) {
                java.util.Arrays.fill(col.getData(), 0.0);
            }
            return;
        }
        for (net.faulj.vector.Vector col : cCols) {
            double[] data = col.getData();
            for (int i = 0; i < data.length; i++) {
                data[i] *= beta;
            }
        }
    }
}
