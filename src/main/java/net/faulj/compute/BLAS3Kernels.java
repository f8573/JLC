package net.faulj.compute;

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
}
