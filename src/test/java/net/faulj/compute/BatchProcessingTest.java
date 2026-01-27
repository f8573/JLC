package net.faulj.compute;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import net.faulj.matrix.Matrix;

/**
 * Tests for batch and accelerated matrix multiplication utilities.
 */
public class BatchProcessingTest {

    /**
     * Validate naive multiplication output on a small matrix.
     */
    @Test
    public void testMultiplyProducesCorrectResult() {
        Matrix a = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});
        Matrix expected = new Matrix(new double[][]{{19, 22}, {43, 50}});

        Matrix result = a.multiply(b);

        double eps = 1e-9;
        for (int r = 0; r < expected.getRowCount(); r++) {
            for (int c = 0; c < expected.getColumnCount(); c++) {
                assertEquals(expected.get(r, c), result.get(r, c), eps);
            }
        }
    }

    /**
     * Validate parallel dispatch and multiplication consistency.
     */
    @Test
    public void testParallelDispatchAndParallelMultiply() {
        DispatchPolicy policy = DispatchPolicy.builder()
            .naiveThreshold(1)
            .enableCuda(false)
            .enableParallel(true)
            .parallelThreshold(1)
            .parallelism(Math.max(2, Runtime.getRuntime().availableProcessors()))
            .build();

        // ensure the policy will parallelize these dimensions
        assertTrue(policy.shouldParallelize(16, 16, 16));

        Matrix a = Matrix.randomMatrix(16, 16);
        Matrix b = Matrix.randomMatrix(16, 16);

        Matrix baseline = BlockedMultiply.multiplyNaive(a, b);
        Matrix parallel = BlockedMultiply.multiply(a, b, policy);

        // compare by Frobenius norm of the difference
        Matrix diff = baseline.subtract(parallel);
        double err = diff.frobeniusNorm();
        assertTrue("Parallel multiply should match naive multiply", err < 1e-8);
    }

    /**
     * Validate CUDA availability probe runs without error.
     */
    @Test
    public void testCudaSupportRefreshAndDetection() {
        // Refresh any cached detection
        CudaSupport.refresh();
        Boolean available = CudaSupport.isCudaAvailable();
        assertNotNull(available);
    }
}
