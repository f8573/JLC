package net.faulj.compute;

/**
 * Defines algorithmic dispatch policies for selecting optimal computational strategies.
 * <p>
 * This class implements a policy-based framework for choosing between different computational
 * approaches based on problem characteristics such as matrix size, sparsity, structure, and
 * hardware capabilities. The dispatch policy ensures that the most efficient algorithm is
 * automatically selected for each operation.
 * </p>
 *
 * <h2>Dispatch Criteria:</h2>
 * <ul>
 *   <li><b>Matrix Dimensions:</b> Small matrices use naive algorithms; large matrices use blocked/strassen</li>
 *   <li><b>Matrix Structure:</b> Symmetric, triangular, or banded matrices use specialized routines</li>
 *   <li><b>Sparsity:</b> Sparse matrices dispatch to sparse-optimized kernels</li>
 *   <li><b>Hardware:</b> CPU architecture, cache sizes, and available instruction sets</li>
 *   <li><b>Precision:</b> Single vs. double precision considerations</li>
 * </ul>
 *
 * <h2>Policy Types:</h2>
 * <ul>
 *   <li><b>NAIVE:</b> Direct implementation, used for small matrices (&lt; 64×64)</li>
 *   <li><b>BLOCKED:</b> Cache-optimized blocked algorithms for medium/large matrices</li>
 *   <li><b>STRASSEN:</b> Strassen's algorithm for very large dense matrices (n &gt; 1024)</li>
 *   <li><b>PARALLEL:</b> Multi-threaded implementation for matrices exceeding parallelization threshold</li>
 *   <li><b>SPECIALIZED:</b> Structure-aware algorithms (triangular, symmetric, banded)</li>
 * </ul>
 *
 * <h2>Threshold Configuration:</h2>
 * <p>
 * Dispatch thresholds can be configured based on empirical performance profiling:
 * </p>
 * <pre>{@code
 * DispatchPolicy policy = DispatchPolicy.builder()
 *     .naiveThreshold(64)
 *     .blockedThreshold(256)
 *     .strassenThreshold(1024)
 *     .parallelThreshold(2048)
 *     .build();
 * }</pre>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Policies are evaluated at runtime with minimal overhead</li>
 *   <li>Default thresholds are optimized for typical x86-64 architectures</li>
 *   <li>Custom policies can be defined for specialized hardware</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see BLAS3Kernels
 * @see BlockedMultiply
 */
public class DispatchPolicy {
}
