package net.faulj.benchmark;

import net.faulj.matrix.Matrix;
import net.faulj.compute.OptimizedBLAS3;
import net.faulj.compute.BlockedMultiply;
import net.faulj.decomposition.lu.LUDecomposition;
import org.junit.Test;

/**
 * Diagnostic benchmark to debug FLOP counting and performance measurements.
 * Prints detailed timing information to help identify issues.
 */
public class DiagnosticBench {

    @Test
    public void diagnosePerformance() {
        System.out.println("=== Diagnostic Performance Benchmark ===\n");

        int[] sizes = {128, 256, 512, 1024};

        for (int n : sizes) {
            System.out.printf("=== Testing n=%d ===\n", n);

            // Test 1: Optimized GEMM (direct call)
            {
                Matrix a = Matrix.randomMatrix(n, n);
                Matrix b = Matrix.randomMatrix(n, n);
                Matrix c = new Matrix(n, n);

                // Warmup
                for (int i = 0; i < 3; i++) {
                    OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, null);
                }

                // Measure
                long start = System.nanoTime();
                for (int i = 0; i < 5; i++) {
                    OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, null);
                }
                long end = System.nanoTime();

                double timeMs = (end - start) / 1e6 / 5.0;
                long flops = 2L * n * n * n;
                double gflops = (flops / (timeMs / 1000.0)) / 1e9;

                System.out.printf("  Optimized GEMM (direct):  %8.3f ms  |  %8.2f GFLOPS\n",
                    timeMs, gflops);
            }

            // Test 2: Blocked GEMM (old path)
            {
                Matrix a = Matrix.randomMatrix(n, n);
                Matrix b = Matrix.randomMatrix(n, n);

                // Warmup
                for (int i = 0; i < 3; i++) {
                    BlockedMultiply.multiply(a, b);
                }

                // Measure
                long start = System.nanoTime();
                for (int i = 0; i < 5; i++) {
                    BlockedMultiply.multiply(a, b);
                }
                long end = System.nanoTime();

                double timeMs = (end - start) / 1e6 / 5.0;
                long flops = 2L * n * n * n;
                double gflops = (flops / (timeMs / 1000.0)) / 1e9;

                System.out.printf("  Blocked GEMM (old):        %8.3f ms  |  %8.2f GFLOPS\n",
                    timeMs, gflops);
            }

            // Test 3: Matrix.multiply() (should use optimized now)
            {
                Matrix a = Matrix.randomMatrix(n, n);
                Matrix b = Matrix.randomMatrix(n, n);

                // Warmup
                for (int i = 0; i < 3; i++) {
                    a.multiply(b);
                }

                // Measure
                long start = System.nanoTime();
                for (int i = 0; i < 5; i++) {
                    a.multiply(b);
                }
                long end = System.nanoTime();

                double timeMs = (end - start) / 1e6 / 5.0;
                long flops = 2L * n * n * n;
                double gflops = (flops / (timeMs / 1000.0)) / 1e9;

                System.out.printf("  Matrix.multiply():         %8.3f ms  |  %8.2f GFLOPS\n",
                    timeMs, gflops);
            }

            // Test 4: LU Decomposition
            {
                Matrix a = Matrix.randomMatrix(n, n);
                LUDecomposition lu = new LUDecomposition();

                // Warmup
                for (int i = 0; i < 3; i++) {
                    lu.decompose(a);
                }

                // Measure
                long start = System.nanoTime();
                for (int i = 0; i < 5; i++) {
                    lu.decompose(a);
                }
                long end = System.nanoTime();

                double timeMs = (end - start) / 1e6 / 5.0;

                // Try multiple FLOP formulas to see which matches
                long flopsTheoretical = (2L * n * n * n) / 3;
                long flopsActual = estimateLUFlops(n); // More accurate count

                double gflopsTheoretical = (flopsTheoretical / (timeMs / 1000.0)) / 1e9;
                double gflopsActual = (flopsActual / (timeMs / 1000.0)) / 1e9;

                System.out.printf("  LU Decomposition:          %8.3f ms  |  %8.2f GFLOPS (theoretical 2n³/3)\n",
                    timeMs, gflopsTheoretical);
                System.out.printf("                                        |  %8.2f GFLOPS (actual count)\n",
                    gflopsActual);
            }

            System.out.println();
        }

        System.out.println("=== Analysis ===");
        System.out.println("If Matrix.multiply() matches Optimized GEMM:");
        System.out.println("  ✅ Decompositions are using optimized path");
        System.out.println();
        System.out.println("If Matrix.multiply() matches Blocked GEMM:");
        System.out.println("  ❌ Decompositions are still using old slow path");
        System.out.println("  → Check Matrix.multiply() implementation");
        System.out.println();
        System.out.println("If LU appears faster than GEMM:");
        System.out.println("  ⚠️  FLOP counting is incorrect");
        System.out.println("  → LU theoretical formula may not match implementation");
    }

    /**
     * More accurate FLOP count for LU decomposition.
     * Accounts for actual operations in typical implementation.
     */
    private long estimateLUFlops(int n) {
        // LU with partial pivoting:
        // For each column k=0..n-1:
        //   - Find pivot: O(n-k) comparisons (negligible)
        //   - Swap rows: O(n-k) moves (negligible)
        //   - Compute multipliers: (n-k-1) divides
        //   - Update submatrix: (n-k-1) × (n-k) FMAs

        long totalFlops = 0;
        for (int k = 0; k < n; k++) {
            int remaining = n - k - 1;
            if (remaining > 0) {
                // Multipliers: remaining divisions (count as 1 FLOP each)
                totalFlops += remaining;

                // Update: remaining rows × (n-k) columns, each needs multiply + add
                totalFlops += 2L * remaining * (n - k);
            }
        }

        return totalFlops;
    }
}
