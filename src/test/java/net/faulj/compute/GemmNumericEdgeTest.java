package net.faulj.compute;

import net.faulj.matrix.Matrix;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class GemmNumericEdgeTest {
    private static final DispatchPolicy BLAS3_CPU_POLICY = DispatchPolicy.builder()
        .enableCuda(false)
        .enableParallel(false)
        .parallelism(1)
        .enableSimd(false)
        .naiveThreshold(8)
        .blockedThreshold(1024)
        .blas3Threshold(1024)
        .enableBlas3(false)
        .build();

    private static final DispatchPolicy OPT_CPU_POLICY = DispatchPolicy.builder()
        .enableCuda(false)
        .enableParallel(false)
        .parallelism(1)
        .build();

    // No -ffast-math flags are present in the current build.gradle GEMM test/runtime wiring.
    // If future native backends add aggressive math flags, reevaluate these edge-case assertions.

    @Test
    public void betaZeroBehaviorDiffersByCpuPathWhenCStartsAsNan() {
        Matrix a = Matrix.wrap(new double[]{1.0, 2.0, 3.0, 4.0}, 2, 2);
        Matrix b = Matrix.wrap(new double[]{5.0, 6.0, 7.0, 8.0}, 2, 2);

        Matrix blasOut = Matrix.wrap(new double[]{Double.NaN, Double.NaN, Double.NaN, Double.NaN}, 2, 2);
        BLAS3Kernels.gemm(a, b, blasOut, 1.0, 0.0, BLAS3_CPU_POLICY);
        assertFinite(blasOut.getRawData());
        GemmReference.assertParity(
            "BLAS3Kernels beta=0 clears C", 2, 2, 2, 1.0, 0.0, 6001L,
            new double[]{19.0, 22.0, 43.0, 50.0}, blasOut.getRawData(), 1e-12, 1e-12
        );

        Matrix optOut = Matrix.wrap(new double[]{Double.NaN, Double.NaN, Double.NaN, Double.NaN}, 2, 2);
        OptimizedBLAS3.gemm(a, b, optOut, 1.0, 0.0, OPT_CPU_POLICY);
        assertTrue("Current OptimizedBLAS3 tiny path retains NaN when beta=0 and C is NaN-seeded",
            containsNaN(optOut.getRawData()));
    }

    @Test
    public void alphaZeroScalesOnlyCForBothCpuPathsWithFiniteInputs() {
        Matrix a = Matrix.wrap(new double[]{3.0, -1.0, 5.0, 2.0}, 2, 2);
        Matrix b = Matrix.wrap(new double[]{8.0, -4.0, 7.0, 9.0}, 2, 2);
        double[] initial = {2.0, -6.0, 10.0, -14.0};
        double[] expected = {5.0, -15.0, 25.0, -35.0};

        Matrix blasOut = Matrix.wrap(initial.clone(), 2, 2);
        BLAS3Kernels.gemm(a, b, blasOut, 0.0, 2.5, BLAS3_CPU_POLICY);
        GemmReference.assertParity("BLAS3Kernels alpha=0 finite", 2, 2, 2, 0.0, 2.5, 6101L, expected, blasOut.getRawData(), 1e-12, 1e-12);

        Matrix optOut = Matrix.wrap(initial.clone(), 2, 2);
        OptimizedBLAS3.gemm(a, b, optOut, 0.0, 2.5, OPT_CPU_POLICY);
        GemmReference.assertParity("OptimizedBLAS3 alpha=0 finite", 2, 2, 2, 0.0, 2.5, 6102L, expected, optOut.getRawData(), 1e-12, 1e-12);
    }

    @Test
    public void blas3KernelsZeroSkipTreatsZeroTimesInfinityAsSkippedContribution() {
        Matrix a = Matrix.wrap(new double[]{0.0}, 1, 1);
        Matrix b = Matrix.wrap(new double[]{Double.POSITIVE_INFINITY}, 1, 1);
        Matrix c = Matrix.wrap(new double[]{7.0}, 1, 1);

        BLAS3Kernels.gemm(a, b, c, 1.0, 1.0, BLAS3_CPU_POLICY);
        assertEquals("Current BLAS3Kernels zero-skip keeps C unchanged for 0 * Infinity", 7.0, c.get(0, 0), 0.0);
        assertFalse(Double.isNaN(c.get(0, 0)));
    }

    @Test
    public void optimizedBlas3DoesNotApplyZeroSkipToZeroTimesInfinityInTinyPath() {
        Matrix a = Matrix.wrap(new double[]{0.0}, 1, 1);
        Matrix b = Matrix.wrap(new double[]{Double.POSITIVE_INFINITY}, 1, 1);
        Matrix c = Matrix.wrap(new double[]{7.0}, 1, 1);

        OptimizedBLAS3.gemm(a, b, c, 1.0, 1.0, OPT_CPU_POLICY);
        assertTrue("Current OptimizedBLAS3 tiny path propagates 0 * Infinity to NaN", Double.isNaN(c.get(0, 0)));
    }

    @Test
    public void optimizedBlas3AlphaZeroIgnoresNanInputs() {
        Matrix a = Matrix.wrap(new double[]{Double.NaN}, 1, 1);
        Matrix b = Matrix.wrap(new double[]{Double.NaN}, 1, 1);
        Matrix c = Matrix.wrap(new double[]{4.0}, 1, 1);

        OptimizedBLAS3.gemm(a, b, c, 0.0, -0.5, OPT_CPU_POLICY);
        assertEquals(-2.0, c.get(0, 0), 0.0);
        assertFalse(Double.isNaN(c.get(0, 0)));
    }

    private static void assertFinite(double[] values) {
        for (double value : values) {
            assertTrue("Expected finite value but saw " + value, Double.isFinite(value));
        }
    }

    private static boolean containsNaN(double[] values) {
        for (double value : values) {
            if (Double.isNaN(value)) {
                return true;
            }
        }
        return false;
    }
}
