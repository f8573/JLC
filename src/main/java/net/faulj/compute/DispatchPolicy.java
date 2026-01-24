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
        CUDA,
        PARALLEL,
        STRASSEN,
        SPECIALIZED
    }

    @FunctionalInterface
    public interface GpuDetector {
        boolean isCudaAvailable();
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
    private final boolean enableStrassen;
    private final boolean enableCuda;
    private final int cudaMinDim;
    private final long cudaMinElements;
    private final long cudaMinFlops;
    private final GpuDetector cudaDetector;

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
        this.enableStrassen = builder.enableStrassen;
        this.enableCuda = builder.enableCuda;
        this.cudaMinDim = builder.cudaMinDim;
        this.cudaMinElements = builder.cudaMinElements;
        this.cudaMinFlops = builder.cudaMinFlops;
        this.cudaDetector = builder.cudaDetector;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static DispatchPolicy defaultPolicy() {
        return globalPolicy;
    }

    public static DispatchPolicy getGlobalPolicy() {
        return globalPolicy;
    }

    public static void setGlobalPolicy(DispatchPolicy policy) {
        if (policy == null) {
            throw new IllegalArgumentException("Dispatch policy must not be null");
        }
        globalPolicy = policy;
    }

    public static void resetGlobalPolicy() {
        globalPolicy = builder().build();
    }

    public Algorithm selectForMultiply(int m, int n, int k) {
        return selectBaseAlgorithm(m, n, k, true);
    }

    public Algorithm selectCpuAlgorithm(int m, int n, int k) {
        return selectBaseAlgorithm(m, n, k, false);
    }

    public boolean shouldParallelize(int m, int n, int k) {
        if (!enableParallel || parallelism <= 1) {
            return false;
        }
        int maxDim = Math.max(m, Math.max(n, k));
        return maxDim >= parallelThreshold;
    }

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

    private Algorithm selectBaseAlgorithm(int m, int n, int k, boolean allowCuda) {
        int maxDim = Math.max(m, Math.max(n, k));
        if (maxDim <= naiveThreshold) {
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
        if (maxDim >= blockedThreshold) {
            return Algorithm.BLOCKED;
        }
        return Algorithm.NAIVE;
    }

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

    public int getParallelism() {
        return parallelism;
    }

    public int getNaiveThreshold() {
        return naiveThreshold;
    }

    public int getBlockedThreshold() {
        return blockedThreshold;
    }

    public int getStrassenThreshold() {
        return strassenThreshold;
    }

    public int getParallelThreshold() {
        return parallelThreshold;
    }

    public boolean isParallelEnabled() {
        return enableParallel;
    }

    public boolean isBlas3Enabled() {
        return enableBlas3;
    }

    public int getBlas3Threshold() {
        return blas3Threshold;
    }

    public boolean isStrassenEnabled() {
        return enableStrassen;
    }

    public boolean isCudaEnabled() {
        return enableCuda;
    }

    public int getCudaMinDim() {
        return cudaMinDim;
    }

    public long getCudaMinElements() {
        return cudaMinElements;
    }

    public long getCudaMinFlops() {
        return cudaMinFlops;
    }

    public static final class Builder {
        private int naiveThreshold = 64;
        private int blockedThreshold = 256;
        private int strassenThreshold = 1024;
        private int parallelThreshold = 2048;
        private int baseBlockSize = 64;
        private int minBlockSize = 16;
        private int maxBlockSize = 256;
        private int parallelism = Math.max(1, Runtime.getRuntime().availableProcessors());
        private boolean enableParallel = true;
        private boolean enableBlas3 = false;
        private int blas3Threshold = 512;
        private boolean enableStrassen = false;
        private boolean enableCuda = true;
        private int cudaMinDim = 256;
        private long cudaMinElements = 1_000_000L;
        private long cudaMinFlops = 1_000_000_000L;
        private GpuDetector cudaDetector = CudaSupport::isCudaAvailable;

        public Builder naiveThreshold(int naiveThreshold) {
            this.naiveThreshold = Math.max(1, naiveThreshold);
            return this;
        }

        public Builder blockedThreshold(int blockedThreshold) {
            this.blockedThreshold = Math.max(1, blockedThreshold);
            return this;
        }

        public Builder strassenThreshold(int strassenThreshold) {
            this.strassenThreshold = Math.max(1, strassenThreshold);
            return this;
        }

        public Builder parallelThreshold(int parallelThreshold) {
            this.parallelThreshold = Math.max(1, parallelThreshold);
            return this;
        }

        public Builder blockSize(int blockSize) {
            this.baseBlockSize = Math.max(1, blockSize);
            return this;
        }

        public Builder minBlockSize(int minBlockSize) {
            this.minBlockSize = Math.max(1, minBlockSize);
            return this;
        }

        public Builder maxBlockSize(int maxBlockSize) {
            this.maxBlockSize = Math.max(1, maxBlockSize);
            return this;
        }

        public Builder parallelism(int parallelism) {
            this.parallelism = Math.max(1, parallelism);
            return this;
        }

        public Builder enableParallel(boolean enableParallel) {
            this.enableParallel = enableParallel;
            return this;
        }

        public Builder enableBlas3(boolean enableBlas3) {
            this.enableBlas3 = enableBlas3;
            return this;
        }

        public Builder blas3Threshold(int blas3Threshold) {
            this.blas3Threshold = Math.max(1, blas3Threshold);
            return this;
        }

        public Builder enableStrassen(boolean enableStrassen) {
            this.enableStrassen = enableStrassen;
            return this;
        }

        public Builder enableCuda(boolean enableCuda) {
            this.enableCuda = enableCuda;
            return this;
        }

        public Builder cudaMinDim(int cudaMinDim) {
            this.cudaMinDim = Math.max(1, cudaMinDim);
            return this;
        }

        public Builder cudaMinElements(long cudaMinElements) {
            this.cudaMinElements = Math.max(1L, cudaMinElements);
            return this;
        }

        public Builder cudaMinFlops(long cudaMinFlops) {
            this.cudaMinFlops = Math.max(1L, cudaMinFlops);
            return this;
        }

        public Builder cudaDetector(GpuDetector cudaDetector) {
            this.cudaDetector = cudaDetector;
            return this;
        }

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
