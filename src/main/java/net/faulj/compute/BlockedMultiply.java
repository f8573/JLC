package net.faulj.compute;

import net.faulj.matrix.Matrix;

/**
 * Implements cache-optimized blocked matrix multiplication algorithms.
 * <p>
 * This class provides high-performance matrix multiplication using block decomposition to maximize
 * cache utilization and minimize memory bandwidth requirements. The blocked approach subdivides
 * large matrices into smaller blocks that fit in cache, dramatically improving performance for
 * large-scale matrix operations.
 * </p>
 *
 * <h2>Algorithm Overview:</h2>
 * <p>
 * For matrices A (m×k) and B (k×n), blocked multiplication computes C = A × B by:
 * </p>
 * <ol>
 *   <li>Partitioning A, B, and C into blocks of size b×b (where b is the block size)</li>
 *   <li>Computing each block of C using standard matrix multiplication on blocks</li>
 *   <li>Accumulating results for blocks that share dimensions</li>
 * </ol>
 *
 * <h2>Performance Optimization:</h2>
 * <ul>
 *   <li><b>Block Size Selection:</b> Dynamically chosen based on cache hierarchy (L1, L2, L3)</li>
 *   <li><b>Data Locality:</b> Maximizes temporal and spatial locality for cache efficiency</li>
 *   <li><b>Memory Bandwidth:</b> Reduces main memory traffic by reusing cached blocks</li>
 *   <li><b>Vectorization:</b> Inner loops structured for SIMD instruction utilization</li>
 * </ul>
 *
 * <h2>Complexity:</h2>
 * <ul>
 *   <li><b>Time:</b> O(mnk) arithmetic operations</li>
 *   <li><b>Space:</b> O(1) auxiliary space (in-place when possible)</li>
 *   <li><b>Cache Behavior:</b> O(mnk/b + mk + kn + mn) cache misses for block size b</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * Matrix A = Matrix.random(1000, 500);
 * Matrix B = Matrix.random(500, 800);
 * Matrix C = BlockedMultiply.multiply(A, B);
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see BLAS3Kernels
 * @see net.faulj.matrix.Matrix
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
        if (m == 0 || n == 0 || k == 0) {
            return net.faulj.matrix.Matrix.zero(m, n);
        }

        DispatchPolicy.Algorithm algorithm = policy.selectForMultiply(m, n, k);
        int blockSize = policy.blockSize(m, n, k);

        switch (algorithm) {
            case NAIVE:
                return multiplyNaive(a, b);
            case CUDA:
                return multiplyBlocked(a, b, blockSize);
            case PARALLEL:
                return multiplyParallel(a, b, blockSize, policy.getParallelism());
            case BLOCKED:
            case STRASSEN:
            case SPECIALIZED:
            default:
                return multiplyBlocked(a, b, blockSize);
        }
    }

    public static Matrix multiplyNaive(Matrix a, Matrix b) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();

        net.faulj.vector.Vector[] aCols = a.getData();
        net.faulj.vector.Vector[] bCols = b.getData();
        net.faulj.vector.Vector[] resultCols = new net.faulj.vector.Vector[n];

        for (int j = 0; j < n; j++) {
            double[] cCol = new double[m];
            double[] bCol = bCols[j].getData();
            for (int kk = 0; kk < k; kk++) {
                double bVal = bCol[kk];
                if (Math.abs(bVal) <= ZERO_EPS) {
                    continue;
                }
                double[] aCol = aCols[kk].getData();
                for (int i = 0; i < m; i++) {
                    cCol[i] += aCol[i] * bVal;
                }
            }
            resultCols[j] = new net.faulj.vector.Vector(cCol);
        }

        return new Matrix(resultCols);
    }

    public static Matrix multiplyBlocked(Matrix a, Matrix b, int blockSize) {
        int m = a.getRowCount();
        int n = b.getColumnCount();
        Matrix c = net.faulj.matrix.Matrix.zero(m, n);
        BLAS3Kernels.dgemm(a, b, c, 1.0, 0.0, blockSize);
        return c;
    }

    public static Matrix multiplyParallel(Matrix a, Matrix b, int blockSize, int parallelism) {
        int m = a.getRowCount();
        int n = b.getColumnCount();
        Matrix c = net.faulj.matrix.Matrix.zero(m, n);
        BLAS3Kernels.dgemmParallel(a, b, c, 1.0, 0.0, blockSize, parallelism);
        return c;
    }
}
