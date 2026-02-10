package net.faulj.benchmark;

import net.faulj.matrix.Matrix;
import net.faulj.compute.DispatchPolicy;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.svd.SVDecomposition;
import net.faulj.decomposition.polar.PolarDecomposition;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.decomposition.result.*;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

import org.junit.Test;

/**
 * Comprehensive performance benchmark measuring theoretical peak FLOPs
 * and efficiency across all major algorithms.
 *
 * Compares:
 * - CUDA GEMM vs CPU GEMM
 * - Optimized vs Blocked GEMM
 * - Decomposition algorithms (QR, Hessenberg, Bidiag, LU, SVD, Polar, Schur)
 * - Efficiency vs theoretical hardware peak
 */
public class ComprehensivePerfBenchmark {

    // Benchmark configuration
    private static final int WARMUP_ITERATIONS = 10;
    private static final int MEASUREMENT_ITERATIONS = 20;
    private static final int[] TEST_SIZES = {64, 128, 256, 512, 1024};
    private static final String OUTPUT_FILE = "comprehensive_performance_results.csv";

    // GEMM baseline performance (measured, not theoretical)
    private static final Map<Integer, Double> GEMM_BASELINE = new HashMap<>();

    // Hardware theoretical limits (GFLOPS)
    // Based on: cores * clock * FLOPs_per_cycle
    // AVX2: 4 doubles * 2 FMA ops = 8 FLOPs/cycle/core
    // Example: 16 cores * 3.5 GHz * 8 = 448 GFLOPS peak
    private static double CPU_THEORETICAL_PEAK = 0.0;
    private static double GPU_THEORETICAL_PEAK = 0.0;

    // Configurable hardware specs (override with system properties if needed)
    private static final int CPU_CORES = Integer.getInteger("bench.cores",
        Runtime.getRuntime().availableProcessors());
    private static final double CPU_CLOCK_GHZ = Double.parseDouble(
        System.getProperty("bench.clock", "3.5"));
    private static final int FLOPS_PER_CYCLE = Integer.getInteger("bench.fpc", 8); // AVX2 FMA

    @Test
    public void runBenchmark() {
        main(new String[0]);
    }

    public static void main(String[] args) {
        System.out.println("=== Comprehensive Performance Benchmark ===");
        System.out.println("Testing CPU and GPU performance across all algorithms");
        System.out.println();

        // Estimate theoretical hardware peaks (for reference)
        estimateHardwareLimits();

        List<BenchmarkResult> results = new ArrayList<>();

        // Test GEMM variants FIRST to establish baseline
        System.out.println("=== GEMM Benchmarks (Establishing Baseline) ===");
        List<BenchmarkResult> gemmResults = benchmarkGemmVariants();
        results.addAll(gemmResults);

        // Store parallel GEMM results as baseline for each size
        for (BenchmarkResult r : gemmResults) {
            if (r.name.equals("GEMM (parallel)")) {
                GEMM_BASELINE.put(r.size, r.gflops);
                System.out.printf("  Baseline for n=%d: %.2f GFLOPS\n", r.size, r.gflops);
            }
        }

        // Test decomposition algorithms (compare against GEMM baseline, not theoretical peak)
        System.out.println("\n=== Decomposition Benchmarks ===");
        results.addAll(benchmarkDecompositions());

        // Print results table
        printResultsTable(results);

        // Export to CSV
        exportToCSV(results);

        System.out.println("\nResults written to: " + OUTPUT_FILE);
        System.out.println("\nSummary:");
        printEfficiencySummary(results);
    }

    /**
     * Estimate theoretical CPU and GPU peak FLOPs.
     */
    private static void estimateHardwareLimits() {
        System.out.println("Computing hardware theoretical limits...");
        System.out.printf("  CPU Cores: %d\n", CPU_CORES);
        System.out.printf("  CPU Clock: %.2f GHz\n", CPU_CLOCK_GHZ);
        System.out.printf("  FLOPs/cycle/core: %d (AVX2 FMA)\n", FLOPS_PER_CYCLE);

        // Theoretical peak: cores * clock * FLOPs_per_cycle
        CPU_THEORETICAL_PEAK = CPU_CORES * CPU_CLOCK_GHZ * FLOPS_PER_CYCLE;
        System.out.printf("CPU Theoretical Peak: %.2f GFLOPS\n", CPU_THEORETICAL_PEAK);

        // Also do empirical measurement for reference
        double empirical = estimateCpuPeak();
        System.out.printf("CPU Empirical Estimate: %.2f GFLOPS (single-threaded FMA loop)\n", empirical);

        // GPU peak estimation (if available)
        GPU_THEORETICAL_PEAK = estimateGpuPeak();
        if (GPU_THEORETICAL_PEAK > 0) {
            System.out.printf("GPU Theoretical Peak: %.2f GFLOPS\n", GPU_THEORETICAL_PEAK);
        } else {
            System.out.println("GPU not available or CUDA disabled");
        }
        System.out.println();
    }

    /**
     * Estimate CPU peak with tight vectorized FMA loop.
     */
    private static double estimateCpuPeak() {
        int n = 4096;
        double[] a = new double[n];
        double[] b = new double[n];
        double[] c = new double[n];
        Arrays.fill(a, 1.0);
        Arrays.fill(b, 2.0);
        Arrays.fill(c, 0.0);

        // Warmup
        for (int iter = 0; iter < 100; iter++) {
            for (int i = 0; i < n; i++) {
                c[i] = Math.fma(a[i], b[i], c[i]);
            }
        }

        // Measure
        long ops = (long) n * 1000L;
        long start = System.nanoTime();
        for (int iter = 0; iter < 1000; iter++) {
            for (int i = 0; i < n; i++) {
                c[i] = Math.fma(a[i], b[i], c[i]);
            }
        }
        long end = System.nanoTime();
        double seconds = (end - start) / 1e9;
        return (ops / seconds) / 1e9;
    }

    /**
     * Estimate GPU peak (placeholder - CUDA GEMM is package-private).
     * For now, return 0 to indicate GPU testing is skipped.
     */
    private static double estimateGpuPeak() {
        // CudaGemm is package-private, so we can't directly call it from tests
        // Return 0 to skip GPU benchmarks
        return 0.0;
    }

    /**
     * Benchmark all GEMM variants.
     */
    private static List<BenchmarkResult> benchmarkGemmVariants() {
        List<BenchmarkResult> results = new ArrayList<>();

        for (int n : TEST_SIZES) {
            System.out.printf("Testing GEMM n=%d...\n", n);

            // Single-threaded microkernel GEMM
            results.add(benchmarkOptimizedGemmSingleThread(n));

            // Parallel microkernel GEMM (via default policy)
            results.add(benchmarkOptimizedGemmParallel(n));

            // Skip BLAS3 Direct for now (redundant with above, and has index calculation issues)
            // results.add(benchmarkBlas3SimdDirect(n));

            // CUDA GEMM
            results.add(benchmarkCudaGemm(n));
        }

        return results;
    }

    /**
     * Benchmark decomposition algorithms.
     */
    private static List<BenchmarkResult> benchmarkDecompositions() {
        List<BenchmarkResult> results = new ArrayList<>();

        for (int n : TEST_SIZES) {
            System.out.printf("Testing decompositions n=%d...\n", n);

            results.add(benchmarkQR(n));
            results.add(benchmarkHessenberg(n));
            results.add(benchmarkBidiagonal(n));
            results.add(benchmarkLU(n));

            // Skip expensive algorithms for large sizes
            if (n <= 512) {
                results.add(benchmarkSVD(n));
                results.add(benchmarkPolar(n));
            }
            if (n <= 256) {
                results.add(benchmarkSchur(n));
            }
        }

        return results;
    }

    private static BenchmarkResult benchmarkOptimizedGemmSingleThread(int n) {
        Matrix a = Matrix.randomMatrix(n, n);
        Matrix b = Matrix.randomMatrix(n, n);
        Matrix c = new Matrix(n, n);

        // Single-threaded policy
        DispatchPolicy singleThread = DispatchPolicy.builder()
            .parallelism(1)
            .enableParallel(false)
            .build();

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            Gemm.gemm(a, b, c, 1.0, 0.0, singleThread);
        }

        // Measure
        long ops = 2L * n * n * n;
        long start = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            Gemm.gemm(a, b, c, 1.0, 0.0, singleThread);
        }
        long end = System.nanoTime();

        double seconds = (end - start) / 1e9 / MEASUREMENT_ITERATIONS;
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("GEMM (1 thread)", n, gflops, ops, CPU_THEORETICAL_PEAK);
    }

    private static BenchmarkResult benchmarkOptimizedGemmParallel(int n) {
        Matrix a = Matrix.randomMatrix(n, n);
        Matrix b = Matrix.randomMatrix(n, n);
        Matrix c = new Matrix(n, n);

        // Default policy (parallel)
        DispatchPolicy parallel = DispatchPolicy.defaultPolicy();

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            Gemm.gemm(a, b, c, 1.0, 0.0, parallel);
        }

        // Measure
        long ops = 2L * n * n * n;
        long start = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            Gemm.gemm(a, b, c, 1.0, 0.0, parallel);
        }
        long end = System.nanoTime();

        double seconds = (end - start) / 1e9 / MEASUREMENT_ITERATIONS;
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("GEMM (parallel)", n, gflops, ops, CPU_THEORETICAL_PEAK);
    }

    private static BenchmarkResult benchmarkBlas3SimdDirect(int n) {
        Matrix a = Matrix.randomMatrix(n, n);
        Matrix b = Matrix.randomMatrix(n, n);
        Matrix c = new Matrix(n, n);

        double[] ad = a.getRawData();
        double[] bd = b.getRawData();
        double[] cd = c.getRawData();

        // Direct BLAS3 kernel call (bypasses dispatch overhead)
        int blockSize = 64;

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            java.util.Arrays.fill(cd, 0.0);
            Gemm.gemmStrided(ad, 0, n, bd, 0, n, cd, 0, n, n, n, n, 1.0, 0.0, blockSize);
        }

        // Measure
        long ops = 2L * n * n * n;
        long start = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            java.util.Arrays.fill(cd, 0.0);
            Gemm.gemmStrided(ad, 0, n, bd, 0, n, cd, 0, n, n, n, n, 1.0, 0.0, blockSize);
        }
        long end = System.nanoTime();

        double seconds = (end - start) / 1e9 / MEASUREMENT_ITERATIONS;
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("BLAS3 Direct", n, gflops, ops, CPU_THEORETICAL_PEAK);
    }

    private static BenchmarkResult benchmarkCudaGemm(int n) {
        // CudaGemm is package-private, so we skip CUDA testing
        // Return a zero result
        return new BenchmarkResult("CUDA GEMM", n, 0.0, 0, GPU_THEORETICAL_PEAK);
    }

    private static BenchmarkResult benchmarkQR(int n) {
        Matrix a = Matrix.randomMatrix(n, n);

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            HouseholderQR.decompose(a);
        }

        // Measure
        long ops = (2L * n * n * n) / 3; // QR is O(2n³/3)
        long start = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            HouseholderQR.decompose(a);
        }
        long end = System.nanoTime();

        double seconds = (end - start) / 1e9 / MEASUREMENT_ITERATIONS;
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("Householder QR", n, gflops, ops, GEMM_BASELINE.getOrDefault(n, CPU_THEORETICAL_PEAK));
    }

    private static BenchmarkResult benchmarkHessenberg(int n) {
        Matrix a = Matrix.randomMatrix(n, n);

        // Warmup
        for (int i = 0; i < Math.min(3, WARMUP_ITERATIONS); i++) {
            HessenbergReduction.decompose(a);
        }

        // Measure
        long ops = (10L * n * n * n) / 3; // Hessenberg is O(10n³/3)
        long start = System.nanoTime();
        for (int i = 0; i < Math.min(5, MEASUREMENT_ITERATIONS); i++) {
            HessenbergReduction.decompose(a);
        }
        long end = System.nanoTime();

        double seconds = (end - start) / 1e9 / Math.min(5, MEASUREMENT_ITERATIONS);
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("Hessenberg", n, gflops, ops, GEMM_BASELINE.getOrDefault(n, CPU_THEORETICAL_PEAK));
    }

    private static BenchmarkResult benchmarkBidiagonal(int n) {
        Matrix a = Matrix.randomMatrix(n, n);
        Bidiagonalization bidiag = new Bidiagonalization();

        // Warmup
        for (int i = 0; i < Math.min(3, WARMUP_ITERATIONS); i++) {
            bidiag.decompose(a);
        }

        // Measure
        long ops = (4L * n * n * n) / 3; // Bidiagonal is O(4n³/3)
        long start = System.nanoTime();
        for (int i = 0; i < Math.min(5, MEASUREMENT_ITERATIONS); i++) {
            bidiag.decompose(a);
        }
        long end = System.nanoTime();

        double seconds = (end - start) / 1e9 / Math.min(5, MEASUREMENT_ITERATIONS);
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("Bidiagonal", n, gflops, ops, GEMM_BASELINE.getOrDefault(n, CPU_THEORETICAL_PEAK));
    }

    private static BenchmarkResult benchmarkLU(int n) {
        Matrix a = Matrix.randomMatrix(n, n);
        LUDecomposition lu = new LUDecomposition();

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            lu.decompose(a);
        }

        // Measure
        long ops = (2L * n * n * n) / 3; // LU is O(2n³/3)
        long start = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            lu.decompose(a);
        }
        long end = System.nanoTime();

        double seconds = (end - start) / 1e9 / MEASUREMENT_ITERATIONS;
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("LU", n, gflops, ops, GEMM_BASELINE.getOrDefault(n, CPU_THEORETICAL_PEAK));
    }

    private static BenchmarkResult benchmarkSVD(int n) {
        Matrix a = Matrix.randomMatrix(n, n);
        SVDecomposition svd = new SVDecomposition();

        // Warmup
        for (int i = 0; i < 2; i++) {
            svd.decompose(a);
        }

        // Measure
        long ops = 11L * n * n * n; // SVD is expensive, O(11n³) roughly
        long start = System.nanoTime();
        for (int i = 0; i < 3; i++) {
            svd.decompose(a);
        }
        long end = System.nanoTime();

        double seconds = (end - start) / 1e9 / 3;
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("SVD", n, gflops, ops, GEMM_BASELINE.getOrDefault(n, CPU_THEORETICAL_PEAK));
    }

    private static BenchmarkResult benchmarkPolar(int n) {
        Matrix a = Matrix.randomMatrix(n, n);
        PolarDecomposition polar = new PolarDecomposition();

        // Warmup
        for (int i = 0; i < 2; i++) {
            polar.decompose(a);
        }

        // Measure
        long ops = 15L * n * n * n; // Polar uses SVD, expensive
        long start = System.nanoTime();
        for (int i = 0; i < 3; i++) {
            polar.decompose(a);
        }
        long end = System.nanoTime();

        double seconds = (end - start) / 1e9 / 3;
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("Polar", n, gflops, ops, GEMM_BASELINE.getOrDefault(n, CPU_THEORETICAL_PEAK));
    }

    private static BenchmarkResult benchmarkSchur(int n) {
        Matrix a = Matrix.randomMatrix(n, n);
        RealSchurDecomposition schur = new RealSchurDecomposition();

        // Warmup
        for (int i = 0; i < 2; i++) {
            try {
                schur.decompose(a);
            } catch (Exception e) {
                // May fail for some matrices
            }
        }

        // Measure
        long ops = 25L * n * n * n; // Schur with QR iterations is expensive
        long start = System.nanoTime();
        int successful = 0;
        for (int i = 0; i < 3; i++) {
            try {
                schur.decompose(a);
                successful++;
            } catch (Exception e) {
                // Skip
            }
        }
        long end = System.nanoTime();

        if (successful == 0) {
            return new BenchmarkResult("Schur", n, 0.0, ops, GEMM_BASELINE.getOrDefault(n, CPU_THEORETICAL_PEAK));
        }

        double seconds = (end - start) / 1e9 / successful;
        double gflops = (ops / seconds) / 1e9;
        return new BenchmarkResult("Schur (Implicit QR)", n, gflops, ops, GEMM_BASELINE.getOrDefault(n, CPU_THEORETICAL_PEAK));
    }

    private static void printResultsTable(List<BenchmarkResult> results) {
        System.out.println("\n=== Performance Results ===");
        System.out.println();
        System.out.printf("%-25s %8s %12s %12s %10s\n",
            "Algorithm", "Size", "GFLOPS", "GEMM Ref", "% of GEMM");
        System.out.println("-".repeat(75));

        for (BenchmarkResult r : results) {
            System.out.printf("%-25s %8d %12.2f %12.2f %9.1f%%\n",
                r.name, r.size, r.gflops,
                r.theoreticalPeak, r.efficiency() * 100);
        }
    }

    private static void exportToCSV(List<BenchmarkResult> results) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(OUTPUT_FILE))) {
            writer.println("Algorithm,Size,GFLOPS,GEMMReference,EfficiencyVsGEMM,FLOPs");

            for (BenchmarkResult r : results) {
                writer.printf("%s,%d,%.4f,%.4f,%.4f,%d\n",
                    r.name, r.size, r.gflops, r.theoreticalPeak,
                    r.efficiency(), r.flops);
            }
        } catch (Exception e) {
            System.err.println("Failed to write CSV: " + e.getMessage());
        }
    }

    private static void printEfficiencySummary(List<BenchmarkResult> results) {
        Map<String, Double> avgEfficiency = new HashMap<>();
        Map<String, Integer> counts = new HashMap<>();

        for (BenchmarkResult r : results) {
            if (r.gflops > 0) {
                avgEfficiency.merge(r.name, r.efficiency(), Double::sum);
                counts.merge(r.name, 1, Integer::sum);
            }
        }

        System.out.println("\n=== Average Efficiency vs GEMM Baseline ===");
        avgEfficiency.entrySet().stream()
            .sorted((a, b) -> Double.compare(
                b.getValue() / counts.get(b.getKey()),
                a.getValue() / counts.get(a.getKey())))
            .forEach(entry -> {
                String name = entry.getKey();
                double avg = entry.getValue() / counts.get(name);
                System.out.printf("%-25s: %5.1f%%\n", name, avg * 100);
            });
    }

    static class BenchmarkResult {
        final String name;
        final int size;
        final double gflops;
        final long flops;
        final double theoreticalPeak;

        BenchmarkResult(String name, int size, double gflops, long flops, double theoreticalPeak) {
            this.name = name;
            this.size = size;
            this.gflops = gflops;
            this.flops = flops;
            this.theoreticalPeak = theoreticalPeak;
        }

        double efficiency() {
            if (theoreticalPeak <= 0) return 0.0;
            return gflops / theoreticalPeak;
        }
    }
}
