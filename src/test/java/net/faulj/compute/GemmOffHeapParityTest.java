package net.faulj.compute;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class GemmOffHeapParityTest {
    private static final DispatchPolicy CPU_POLICY = DispatchPolicy.builder()
        .enableCuda(false)
        .enableParallel(false)
        .parallelism(1)
        .naiveThreshold(1)
        .blockedThreshold(1)
        .blas3Threshold(1)
        .enableBlas3(true)
        .build();

    @Test
    public void blas3KernelsOffHeapFacadeMatchesReference() {
        OffHeapMatrix a = new OffHeapMatrix(24, 16);
        OffHeapMatrix b = new OffHeapMatrix(16, 10);
        fillOffHeap(a, 5101L);
        fillOffHeap(b, 5102L);

        Matrix product = BLAS3Kernels.gemm(a, b, CPU_POLICY);
        assertTrue(product instanceof OffHeapMatrix);

        double[] expected = GemmReference.gemm(a.getRawData(), b.getRawData(), new double[24 * 10], 24, 16, 10, 1.0, 0.0);
        GemmReference.assertParity(
            "BLAS3Kernels.gemm off-heap", 24, 16, 10, 1.0, 0.0, 5100L,
            expected, product.getRawData(), 1e-12, GemmReference.cpuAbsTolerance(expected, 16)
        );
    }

    @Test
    public void optimizedBlas3OffHeapOutputSyncMatchesReference() {
        OffHeapMatrix a = new OffHeapMatrix(18, 12);
        OffHeapMatrix b = new OffHeapMatrix(12, 14);
        OffHeapMatrix c = new OffHeapMatrix(18, 14);
        fillOffHeap(a, 5201L);
        fillOffHeap(b, 5202L);
        fillOffHeap(c, 5203L);

        double[] expected = GemmReference.gemm(a.getRawData(), b.getRawData(), c.getRawData(), 18, 12, 14, -0.5, 2.5);
        OptimizedBLAS3.gemm(a, b, c, -0.5, 2.5, CPU_POLICY);

        GemmReference.assertParity(
            "OptimizedBLAS3.gemm off-heap", 18, 12, 14, -0.5, 2.5, 5200L,
            expected, c.getRawData(), 1e-12, GemmReference.cpuAbsTolerance(expected, 12)
        );
        assertOffHeapSampleMatches(c, expected, 18, 14);
    }

    private static void fillOffHeap(OffHeapMatrix matrix, long seed) {
        double[] data = GemmReference.seededArray(matrix.getRowCount() * matrix.getColumnCount(), seed);
        for (int i = 0; i < matrix.getRowCount(); i++) {
            for (int j = 0; j < matrix.getColumnCount(); j++) {
                matrix.set(i, j, data[i * matrix.getColumnCount() + j]);
            }
        }
        matrix.syncToOffHeap();
    }

    private static void assertOffHeapSampleMatches(OffHeapMatrix matrix, double[] expected, int rows, int cols) {
        int[][] samples = {
            {0, 0},
            {rows / 2, cols / 2},
            {rows - 1, cols - 1}
        };
        for (int[] sample : samples) {
            int row = sample[0];
            int col = sample[1];
            double expectedValue = expected[row * cols + col];
            assertTrue(Math.abs(matrix.getOffHeap(row, col) - expectedValue) <= 1e-12);
        }
    }
}
