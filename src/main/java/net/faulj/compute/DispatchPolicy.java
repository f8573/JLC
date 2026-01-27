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
 *   <li><b>BLAS3:</b> Level-3 BLAS-style GEMM kernels for large dense matrices</li>
 *   <li><b>STRASSEN:</b> Strassen's algorithm for very large dense matrices (n &gt; 1024)</li>
 *   <li><b>PARALLEL:</b> Multi-threaded execution applied as a modifier when enabled</li>
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
    public enum Algorithm {
        NAIVE,
        BLOCKED,
        BLAS3,
        SIMD,
        CUDA,
        PARALLEL,
        STRASSEN,
        SPECIALIZED
    }

    @FunctionalInterface
    public interface GpuDetector {
        boolean isCudaAvailable();
    }

    @FunctionalInterface
    public interface SimdDetector {
        boolean isSimdAvailable();
    }

    private static volatile DispatchPolicy globalPolicy = builder().build();

    private final int naiveThreshold;
    private final int blockedThreshold;
    private final int strassenThreshold;
    private final int parallelThreshold;
    private final int baseBlockSize;
    private final int minBlockSize;
    private final int maxBlockSize;
    private final int parallelism;
    private final boolean enableParallel;
    private final boolean enableBlas3;
    private final int blas3Threshold;
    private final boolean enableSimd;
    private final int simdThreshold;
    private final boolean enableStrassen;
    private final boolean enableCuda;
    private final int cudaMinDim;
    private final long cudaMinElements;
    private final long cudaMinFlops;
    private final GpuDetector cudaDetector;
    private final SimdDetector simdDetector;

    private DispatchPolicy(Builder builder) {
        this.naiveThreshold = builder.naiveThreshold;
        this.blockedThreshold = builder.blockedThreshold;
        this.strassenThreshold = builder.strassenThreshold;
        this.parallelThreshold = builder.parallelThreshold;
        this.baseBlockSize = builder.baseBlockSize;
        this.minBlockSize = builder.minBlockSize;
        this.maxBlockSize = builder.maxBlockSize;
        this.parallelism = builder.parallelism;
        this.enableParallel = builder.enableParallel;
        this.enableBlas3 = builder.enableBlas3;
        this.blas3Threshold = builder.blas3Threshold;
        this.enableSimd = builder.enableSimd;
        this.simdThreshold = builder.simdThreshold;
        this.enableStrassen = builder.enableStrassen;
        this.enableCuda = builder.enableCuda;
        this.cudaMinDim = builder.cudaMinDim;
        this.cudaMinElements = builder.cudaMinElements;
        this.cudaMinFlops = builder.cudaMinFlops;
        this.cudaDetector = builder.cudaDetector;
        this.simdDetector = builder.simdDetector;
    }

    /**
     * Create a new dispatch policy builder.
     *
     * @return builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Get the default global policy.
     *
     * @return global policy
     */
    public static DispatchPolicy defaultPolicy() {
        return globalPolicy;
    }

    /**
     * Get the current global policy.
     *
     * @return global policy
     */
    public static DispatchPolicy getGlobalPolicy() {
        return globalPolicy;
    }

    /**
     * Replace the global policy.
     *
     * @param policy new policy
     */
    public static void setGlobalPolicy(DispatchPolicy policy) {
        if (policy == null) {
            throw new IllegalArgumentException("Dispatch policy must not be null");
        }
        globalPolicy = policy;
    }

    /**
     * Reset the global policy to defaults.
     */
    public static void resetGlobalPolicy() {
        globalPolicy = builder().build();
    }

    /**
     * Select an algorithm for matrix multiplication.
     *
     * @param m rows of A
     * @param n columns of B
     * @param k shared dimension
     * @return algorithm selection
     */
    public Algorithm selectForMultiply(int m, int n, int k) {
        return selectBaseAlgorithm(m, n, k, true);
    }

    /**
     * Select a CPU-only algorithm for multiplication.
     *
     * @param m rows of A
     * @param n columns of B
     * @param k shared dimension
     * @return algorithm selection
     */
    public Algorithm selectCpuAlgorithm(int m, int n, int k) {
        return selectBaseAlgorithm(m, n, k, false);
    }

    /**
     * Determine whether to parallelize based on matrix sizes.
     *
     * @param m rows of A
     * @param n columns of B
     * @param k shared dimension
     * @return true if parallel execution should be used
     */
    public boolean shouldParallelize(int m, int n, int k) {
        if (!enableParallel || parallelism <= 1) {
            return false;
        }
        int maxDim = Math.max(m, Math.max(n, k));
        return maxDim >= parallelThreshold;
    }

    /**
     * Determine whether to offload multiplication to CUDA.
     *
     * @param m rows of A
     * @param n columns of B
     * @param k shared dimension
     * @return true if CUDA offload is recommended
     */
    public boolean shouldOffloadToCuda(int m, int n, int k) {
        if (!enableCuda) {
            return false;
        }
        if (cudaDetector == null || !cudaDetector.isCudaAvailable()) {
            return false;
        }
        int minDim = Math.min(m, Math.min(n, k));
        if (minDim < cudaMinDim) {
            return false;
        }
        long elements = (long) m * k + (long) k * n + (long) m * n;
        if (elements < cudaMinElements) {
            return false;
        }
        long flops = 2L * m * n * k;
        return flops >= cudaMinFlops;
    }

    /**
     * Determine whether SIMD kernels should be used.
     *
     * @param m rows of A
     * @param n columns of B
     * @param k shared dimension
     * @return true if SIMD kernels are recommended
     */
    public boolean shouldUseSimd(int m, int n, int k) {
        if (!enableSimd || simdDetector == null || !simdDetector.isSimdAvailable()) {
            return false;
        }
        int maxDim = Math.max(m, Math.max(n, k));
        return maxDim < simdThreshold;
    }

    private Algorithm selectBaseAlgorithm(int m, int n, int k, boolean allowCuda) {
        int maxDim = Math.max(m, Math.max(n, k));
        if (maxDim <= naiveThreshold) {
            if (enableSimd && maxDim < simdThreshold && simdDetector != null && simdDetector.isSimdAvailable()) {
                return Algorithm.SIMD;
            }
            return Algorithm.NAIVE;
        }
        if (allowCuda && shouldOffloadToCuda(m, n, k)) {
            return Algorithm.CUDA;
        }
        boolean square = (m == n && n == k);
        if (enableStrassen && square && maxDim >= strassenThreshold) {
            return Algorithm.STRASSEN;
        }
        if (enableBlas3 && maxDim >= blas3Threshold) {
            return Algorithm.BLAS3;
        }
        if (enableSimd && maxDim < simdThreshold && simdDetector != null && simdDetector.isSimdAvailable()) {
            return Algorithm.SIMD;
        }
        if (maxDim >= blockedThreshold) {
            return Algorithm.BLOCKED;
        }
        return Algorithm.NAIVE;
    }

    /**
     * Compute an appropriate block size for blocked multiplication.
     *
     * @param m rows of A
     * @param n columns of B
     * @param k shared dimension
     * @return block size
     */
    public int blockSize(int m, int n, int k) {
        int limit = Math.min(m, Math.min(n, k));
        if (limit <= 0) {
            return 0;
        }
        int size = Math.min(baseBlockSize, limit);
        size = Math.min(size, maxBlockSize);
        if (size < minBlockSize) {
            size = limit;
        }
        int aligned = size - (size % 8);
        if (aligned > 0) {
            size = aligned;
        }
        return Math.max(1, size);
    }

    /**
     * Get configured thread parallelism level.
     *
     * @return parallelism
     */
    public int getParallelism() {
        return parallelism;
    }

    /**
     * @return dimension threshold for naive algorithm
     */
    public int getNaiveThreshold() {
        return naiveThreshold;
    }

    /**
     * @return dimension threshold for blocked algorithm
     */
    public int getBlockedThreshold() {
        return blockedThreshold;
    }

    /**
     * @return dimension threshold for Strassen algorithm
     */
    public int getStrassenThreshold() {
        return strassenThreshold;
    }

    /**
     * @return dimension threshold for parallel execution
     */
    public int getParallelThreshold() {
        return parallelThreshold;
    }

    /**
     * @return true if parallel execution is enabled
     */
    public boolean isParallelEnabled() {
        return enableParallel;
    }

    /**
     * @return true if BLAS3 kernels are enabled
     */
    public boolean isBlas3Enabled() {
        return enableBlas3;
    }

    /**
     * @return true if SIMD kernels are enabled
     */
    public boolean isSimdEnabled() {
        return enableSimd;
    }

    /**
     * @return dimension threshold for SIMD dispatch
     */
    public int getSimdThreshold() {
        return simdThreshold;
    }

    /**
     * @return true if SIMD is available at runtime
     */
    public boolean isSimdAvailable() {
        return simdDetector != null && simdDetector.isSimdAvailable();
    }

    /**
     * @return dimension threshold for BLAS3 dispatch
     */
    public int getBlas3Threshold() {
        return blas3Threshold;
    }

    /**
     * @return true if Strassen algorithm is enabled
     */
    public boolean isStrassenEnabled() {
        return enableStrassen;
    }

    /**
     * @return true if CUDA offload is enabled
     */
    public boolean isCudaEnabled() {
        return enableCuda;
    }

    /**
     * @return minimum dimension for CUDA offload
     */
    public int getCudaMinDim() {
        return cudaMinDim;
    }

    /**
     * @return minimum element count for CUDA offload
     */
    public long getCudaMinElements() {
        return cudaMinElements;
    }

    /**
     * @return minimum FLOP threshold for CUDA offload
     */
    public long getCudaMinFlops() {
        return cudaMinFlops;
    }

    /**
     * Builder for configuring a DispatchPolicy.
     */
    public static final class Builder {
        private int naiveThreshold = 64;
        private int blockedThreshold = 256;
        private int strassenThreshold = 1024;
        private int parallelThreshold = 2048;
        private int baseBlockSize = 128;
        private int minBlockSize = 128;
        private int maxBlockSize = 256;
        private int parallelism = Math.max(1, Runtime.getRuntime().availableProcessors());
        private boolean enableParallel = true;
        private boolean enableBlas3 = true;
        private int blas3Threshold = 256;
        private boolean enableSimd = true;
        private int simdThreshold = 256;
        private boolean enableStrassen = false;
        private boolean enableCuda = true;
        private int cudaMinDim = 256;
        private long cudaMinElements = 1_000_000L;
        private long cudaMinFlops = 1_000_000_000L;
        private GpuDetector cudaDetector = CudaSupport::isCudaAvailable;
        private SimdDetector simdDetector = VectorSupport::isVectorApiAvailable;

        /**
         * Set the dimension threshold for naive multiplication.
         *
         * @param naiveThreshold threshold
         * @return builder
         */
        public Builder naiveThreshold(int naiveThreshold) {
            this.naiveThreshold = Math.max(1, naiveThreshold);
            return this;
        }

        /**
         * Set the dimension threshold for blocked multiplication.
         *
         * @param blockedThreshold threshold
         * @return builder
         */
        public Builder blockedThreshold(int blockedThreshold) {
            this.blockedThreshold = Math.max(1, blockedThreshold);
            return this;
        }

        /**
         * Set the dimension threshold for Strassen multiplication.
         *
         * @param strassenThreshold threshold
         * @return builder
         */
        public Builder strassenThreshold(int strassenThreshold) {
            this.strassenThreshold = Math.max(1, strassenThreshold);
            return this;
        }

        /**
         * Set the dimension threshold for parallel execution.
         *
         * @param parallelThreshold threshold
         * @return builder
         */
        public Builder parallelThreshold(int parallelThreshold) {
            this.parallelThreshold = Math.max(1, parallelThreshold);
            return this;
        }

        /**
         * Set the base block size.
         *
         * @param blockSize block size
         * @return builder
         */
        public Builder blockSize(int blockSize) {
            this.baseBlockSize = Math.max(1, blockSize);
            return this;
        }

        /**
         * Set the minimum block size.
         *
         * @param minBlockSize minimum block size
         * @return builder
         */
        public Builder minBlockSize(int minBlockSize) {
            this.minBlockSize = Math.max(1, minBlockSize);
            return this;
        }

        /**
         * Set the maximum block size.
         *
         * @param maxBlockSize maximum block size
         * @return builder
         */
        public Builder maxBlockSize(int maxBlockSize) {
            this.maxBlockSize = Math.max(1, maxBlockSize);
            return this;
        }

        /**
         * Set the number of threads for parallel execution.
         *
         * @param parallelism thread count
         * @return builder
         */
        public Builder parallelism(int parallelism) {
            this.parallelism = Math.max(1, parallelism);
            return this;
        }

        /**
         * Enable or disable parallel execution.
         *
         * @param enableParallel flag
         * @return builder
         */
        public Builder enableParallel(boolean enableParallel) {
            this.enableParallel = enableParallel;
            return this;
        }

        /**
         * Enable or disable BLAS3 kernels.
         *
         * @param enableBlas3 flag
         * @return builder
         */
        public Builder enableBlas3(boolean enableBlas3) {
            this.enableBlas3 = enableBlas3;
            return this;
        }

        /**
         * Enable or disable SIMD kernels.
         *
         * @param enableSimd flag
         * @return builder
         */
        public Builder enableSimd(boolean enableSimd) {
            this.enableSimd = enableSimd;
            return this;
        }

        /**
         * Set the dimension threshold for BLAS3.
         *
         * @param blas3Threshold threshold
         * @return builder
         */
        public Builder blas3Threshold(int blas3Threshold) {
            this.blas3Threshold = Math.max(1, blas3Threshold);
            return this;
        }

        /**
         * Set the dimension threshold for SIMD dispatch.
         *
         * @param simdThreshold threshold
         * @return builder
         */
        public Builder simdThreshold(int simdThreshold) {
            this.simdThreshold = Math.max(1, simdThreshold);
            return this;
        }

        /**
         * Enable or disable Strassen multiplication.
         *
         * @param enableStrassen flag
         * @return builder
         */
        public Builder enableStrassen(boolean enableStrassen) {
            this.enableStrassen = enableStrassen;
            return this;
        }

        /**
         * Enable or disable CUDA offload.
         *
         * @param enableCuda flag
         * @return builder
         */
        public Builder enableCuda(boolean enableCuda) {
            this.enableCuda = enableCuda;
            return this;
        }

        /**
         * Set the minimum dimension for CUDA offload.
         *
         * @param cudaMinDim minimum dimension
         * @return builder
         */
        public Builder cudaMinDim(int cudaMinDim) {
            this.cudaMinDim = Math.max(1, cudaMinDim);
            return this;
        }

        /**
         * Set the minimum element count for CUDA offload.
         *
         * @param cudaMinElements minimum elements
         * @return builder
         */
        public Builder cudaMinElements(long cudaMinElements) {
            this.cudaMinElements = Math.max(1L, cudaMinElements);
            return this;
        }

        /**
         * Set the minimum FLOP threshold for CUDA offload.
         *
         * @param cudaMinFlops minimum FLOPs
         * @return builder
         */
        public Builder cudaMinFlops(long cudaMinFlops) {
            this.cudaMinFlops = Math.max(1L, cudaMinFlops);
            return this;
        }

        /**
         * Provide a custom CUDA detector.
         *
         * @param cudaDetector detector implementation
         * @return builder
         */
        public Builder cudaDetector(GpuDetector cudaDetector) {
            this.cudaDetector = cudaDetector;
            return this;
        }

        /**
         * Provide a custom SIMD detector.
         *
         * @param simdDetector detector implementation
         * @return builder
         */
        public Builder simdDetector(SimdDetector simdDetector) {
            this.simdDetector = simdDetector;
            return this;
        }

        /**
         * Build a DispatchPolicy instance.
         *
         * @return policy
         */
        public DispatchPolicy build() {
            if (minBlockSize > maxBlockSize) {
                int temp = minBlockSize;
                minBlockSize = maxBlockSize;
                maxBlockSize = temp;
            }
            return new DispatchPolicy(this);
        }
    }
}
