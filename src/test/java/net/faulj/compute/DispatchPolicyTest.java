package net.faulj.compute;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * Tests for dispatch policy selection and thresholds.
 */
public class DispatchPolicyTest {

    /**
     * Ensure CUDA is preferred when eligible and available.
     */
    @Test
    public void testSelectionPrefersCudaWhenAvailable() {
        DispatchPolicy policy = DispatchPolicy.builder()
            .naiveThreshold(1)
            .blockedThreshold(4)
            .blas3Threshold(8)
            .parallelThreshold(16)
            .strassenThreshold(32)
            .enableBlas3(true)
            .enableParallel(true)
            .enableStrassen(true)
            .enableCuda(true)
            .cudaMinDim(2)
            .cudaMinElements(1L)
            .cudaMinFlops(1L)
            .cudaDetector(() -> true)
            .build();

        assertEquals(DispatchPolicy.Algorithm.CUDA, policy.selectForMultiply(64, 64, 64));
    }

    /**
     * Ensure Strassen is selected when enabled and thresholds are met.
     */
    @Test
    public void testSelectionUsesStrassenWhenEnabled() {
        DispatchPolicy policy = DispatchPolicy.builder()
            .naiveThreshold(1)
            .blockedThreshold(4)
            .blas3Threshold(8)
            .parallelThreshold(16)
            .strassenThreshold(32)
            .enableBlas3(true)
            .enableParallel(true)
            .enableStrassen(true)
            .enableCuda(false)
            .build();

        assertEquals(DispatchPolicy.Algorithm.STRASSEN, policy.selectForMultiply(64, 64, 64));
    }

    /**
     * Validate block size alignment within policy bounds.
     */
    @Test
    public void testBlockSizeAlignment() {
        DispatchPolicy policy = DispatchPolicy.builder()
            .blockSize(70)
            .minBlockSize(16)
            .maxBlockSize(128)
            .build();

        assertEquals(64, policy.blockSize(100, 100, 100));
    }

    /**
     * Validate CUDA offload thresholds.
     */
    @Test
    public void testShouldOffloadToCudaThresholds() {
        DispatchPolicy policy = DispatchPolicy.builder()
            .enableCuda(true)
            .cudaDetector(() -> true)
            .cudaMinDim(4)
            .cudaMinElements(1000L)
            .cudaMinFlops(100000L)
            .build();

        assertFalse(policy.shouldOffloadToCuda(4, 4, 4));
        assertTrue(policy.shouldOffloadToCuda(40, 40, 40));
    }

    /**
     * Validate parallelization threshold logic.
     */
    @Test
    public void testShouldParallelizeUsesThreshold() {
        DispatchPolicy policy = DispatchPolicy.builder()
            .enableParallel(true)
            .parallelThreshold(8)
            .parallelism(4)
            .build();

        assertFalse(policy.shouldParallelize(4, 4, 4));
        assertTrue(policy.shouldParallelize(8, 8, 8));
    }

    /**
     * Ensure CPU algorithm selection excludes CUDA.
     */
    @Test
    public void testCpuAlgorithmSkipsCuda() {
        DispatchPolicy policy = DispatchPolicy.builder()
            .naiveThreshold(1)
            .enableCuda(true)
            .enableBlas3(true)
            .blas3Threshold(4)
            .cudaDetector(() -> true)
            .cudaMinDim(2)
            .cudaMinElements(1L)
            .cudaMinFlops(1L)
            .build();

        assertEquals(DispatchPolicy.Algorithm.CUDA, policy.selectForMultiply(8, 8, 8));
        assertEquals(DispatchPolicy.Algorithm.BLAS3, policy.selectCpuAlgorithm(8, 8, 8));
    }

    /**
     * Validate global policy reset logic.
     */
    @Test
    public void testGlobalPolicyReset() {
        DispatchPolicy original = DispatchPolicy.getGlobalPolicy();
        DispatchPolicy custom = DispatchPolicy.builder().enableCuda(false).build();
        DispatchPolicy.setGlobalPolicy(custom);
        try {
            assertSame(custom, DispatchPolicy.defaultPolicy());
        } finally {
            DispatchPolicy.setGlobalPolicy(original);
        }
    }
}
