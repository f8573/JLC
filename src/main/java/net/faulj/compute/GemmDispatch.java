package net.faulj.compute;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Intelligent GEMM kernel dispatcher with cache-aware blocking.
 *
 * Selects optimal algorithm based on:
 * - Problem size (m, n, k)
 * - Hardware (CPU cache sizes, GPU availability, SIMD width)
 * - Data characteristics (sparsity, alignment)
 * - Parallelism potential
 *
 * Blocking strategy follows BLIS/GotoBLAS2 design:
 * - KC: sized to L2 cache (panel of B)
 * - MC: sized to L3 cache (panel of A)
 * - NC: sized to stay in L3 with MC × NC block of C
 * - MR × NR: microkernel register block
 */
public final class GemmDispatch {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    // Cache size estimates (bytes)
    private static final int L1_CACHE = 32 * 1024;      // 32 KB typical L1D
    private static final int L2_CACHE = 256 * 1024;     // 256 KB typical L2
    private static final int L3_CACHE = 8 * 1024 * 1024; // 8 MB typical L3

    // Size thresholds (FLOPs)
    private static final long TINY_THRESHOLD = 8 * 8 * 8;          // 512 FLOPs
    private static final long SMALL_THRESHOLD = 50_000;            // 50k FLOPs
    private static final long CUDA_THRESHOLD = 200_000_000;        // 200M FLOPs
    private static final long PARALLEL_THRESHOLD = 5_000_000;      // 5M FLOPs

    private GemmDispatch() {}

    /**
     * Kernel selection result.
     */
    public enum Kernel {
        TINY,           // Scalar loop, register-only
        SMALL,          // Scalar loop with FMA, no packing
        MATVEC,         // Specialized matrix-vector
        MICROKERNEL,    // Packed MR×NR microkernel
        CUDA,           // GPU offload
        PARALLEL_MICRO  // Parallel tiled microkernel
    }

    /**
     * Blocking parameters for cache optimization.
     */
    public static final class BlockSizes {
        public final int mc;  // M blocking (rows of A)
        public final int kc;  // K blocking (columns of A / rows of B)
        public final int nc;  // N blocking (columns of B)
        public final int mr;  // Microkernel rows
        public final int nr;  // Microkernel columns (SIMD width)

        public BlockSizes(int mc, int kc, int nc, int mr, int nr) {
            this.mc = mc;
            this.kc = kc;
            this.nc = nc;
            this.mr = mr;
            this.nr = nr;
        }

        @Override
        public String toString() {
            return String.format("MC=%d, KC=%d, NC=%d, MR=%d, NR=%d", mc, kc, nc, mr, nr);
        }
    }

    /**
     * Select optimal kernel for given problem size.
     */
    public static Kernel selectKernel(int m, int n, int k, boolean cudaAvailable, int threads) {
        long flops = 2L * m * n * k;

        // Matvec special case
        if (n == 1) {
            return Kernel.MATVEC;
        }

        // Tiny matrices (fit in registers)
        if (flops <= TINY_THRESHOLD && m <= 8 && n <= 8 && k <= 8) {
            return Kernel.TINY;
        }

        // Small matrices (overhead of packing dominates)
        if (flops <= SMALL_THRESHOLD) {
            return Kernel.SMALL;
        }

        // CUDA for very large problems (if available)
        if (cudaAvailable && flops >= CUDA_THRESHOLD) {
            return Kernel.CUDA;
        }

        // Parallel microkernel for large problems
        if (threads > 1 && flops >= PARALLEL_THRESHOLD) {
            return Kernel.PARALLEL_MICRO;
        }

        // Default: single-threaded microkernel
        return Kernel.MICROKERNEL;
    }

    /**
     * Compute optimal blocking parameters based on hardware.
     *
     * Strategy:
     * 1. NR = SIMD width (4 for AVX2, 8 for AVX-512)
     * 2. MR chosen so MR × NR ≈ 32-48 (fits in register file)
     * 3. KC sized so MR × KC fits in L1 (A panel)
     * 4. NC sized so KC × NC fits in L2 (B panel)
     * 5. MC sized so MC × KC + KC × NC fits in L3
     */
    public static BlockSizes computeBlockSizes() {
        int vecLen = SPECIES.length();
        int nr = vecLen;
        int mr = MicroKernel.optimalMR(vecLen);

        // KC: size B panel (KC × NC) to fit in L2
        // Goal: KC × NC × 8 bytes ≈ L2_CACHE / 2 (leave room for A and C)
        // Also constraint: MR × KC fits in L1
        int l1Constraint = L1_CACHE / (8 * mr * 2);  // Half L1 for A panel
        int l2Constraint = (int) Math.sqrt(L2_CACHE / (8.0 * 2));  // Square root heuristic

        int kc = Math.min(l1Constraint, l2Constraint);
        kc = PackingUtils.roundUp(kc, 8);  // Round to multiple of 8
        kc = Math.max(kc, 64);  // Minimum KC
        kc = Math.min(kc, 512); // Maximum KC

        // NC: size B panel to fit in L2 with KC
        int nc = (L2_CACHE / 2) / (kc * 8);
        nc = PackingUtils.roundUp(nc, nr);  // Round to SIMD width
        nc = Math.max(nc, nr * 4);  // Minimum 4 SIMD vectors
        nc = Math.min(nc, 4096);    // Maximum NC

        // MC: size to fit in L3 with B panel and C block
        // MC × KC (A) + KC × NC (B) + MC × NC (C) ≈ L3
        // Simplify: MC × (KC + NC) ≈ L3 / 8
        int mc = L3_CACHE / (8 * (kc + nc));
        mc = PackingUtils.roundUp(mc, mr);  // Round to MR
        mc = Math.max(mc, mr * 4);   // Minimum 4 microtiles
        mc = Math.min(mc, 2048);     // Maximum MC

        return new BlockSizes(mc, kc, nc, mr, nr);
    }

    /**
     * Estimate whether CUDA will be faster than CPU for this problem.
     * Factors: data transfer overhead, kernel launch latency, compute intensity.
     */
    public static boolean preferCuda(int m, int n, int k, CudaContext ctx) {
        if (ctx == null || !ctx.isAvailable()) {
            return false;
        }

        long flops = 2L * m * n * k;

        // CUDA overhead model (very rough):
        // - H2D + D2H: ~5 GB/s effective for PCIe 3.0 x16
        // - Data volume: (m*k + k*n + m*n) * 8 bytes
        // - Kernel launch: ~10 microseconds
        // - CPU: ~100 GFLOPS (conservative for single-thread microkernel)
        // - GPU: ~1000 GFLOPS (conservative for 3060/3090)

        long dataBytes = (long) (m * k + k * n + m * n) * 8L;
        double transferTime = dataBytes / (5.0e9);  // seconds
        double launchOverhead = 10e-6;              // seconds
        double cpuTime = flops / 100e9;             // seconds
        double gpuComputeTime = flops / 1000e9;     // seconds
        double gpuTotalTime = transferTime + launchOverhead + gpuComputeTime;

        // Prefer GPU if total time is at least 2x faster (amortize noise)
        return gpuTotalTime * 2.0 < cpuTime && flops >= CUDA_THRESHOLD;
    }

    /**
     * Determine optimal parallelism level based on problem size and threads.
     */
    public static int optimalParallelism(int m, int n, int k, int maxThreads, BlockSizes blocks) {
        if (maxThreads <= 1) {
            return 1;
        }

        long flops = 2L * m * n * k;
        if (flops < PARALLEL_THRESHOLD) {
            return 1;  // Too small for parallelism
        }

        // Estimate work per thread
        int blockRows = (m + blocks.mc - 1) / blocks.mc;
        int blockCols = (n + blocks.nc - 1) / blocks.nc;
        int totalTiles = blockRows * blockCols;

        // Each tile should have enough work to amortize thread overhead
        long minFlopsPerThread = 500_000;  // 500k FLOPs minimum
        int maxUsefulThreads = (int) (flops / minFlopsPerThread);

        // Don't use more threads than tiles (leads to idle threads)
        maxUsefulThreads = Math.min(maxUsefulThreads, totalTiles);

        // Use power-of-2 threads for better load balance
        int threads = Math.min(maxThreads, maxUsefulThreads);
        return Math.max(1, Integer.highestOneBit(threads));
    }

    /**
     * Check if matrix is effectively sparse (density < 10%).
     * Only meaningful for large matrices where scan cost is acceptable.
     */
    public static boolean isEffectivelySparse(double[] data, int sampleSize) {
        if (data.length < 1000) {
            return false;  // Too small to matter
        }

        int samples = Math.min(sampleSize, data.length);
        int step = Math.max(1, data.length / samples);
        int nonZero = 0;

        final double ZERO_THRESHOLD = 1e-15;

        for (int i = 0; i < data.length && nonZero * 10 >= samples; i += step) {
            if (Math.abs(data[i]) > ZERO_THRESHOLD) {
                nonZero++;
                if (nonZero * 10 >= samples) {
                    return false;  // Early exit if density > 10%
                }
            }
            if (i / step >= samples) {
                break;
            }
        }

        return nonZero * 10 < samples;  // density < 10%
    }
}
