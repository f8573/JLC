package net.faulj.compute;

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
}
