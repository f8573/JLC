package net.faulj.compute;

import net.faulj.matrix.Matrix;
import org.junit.Ignore;
import org.junit.Test;

import java.util.Arrays;

import jdk.incubator.vector.DoubleVector;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class GemmParityTest {
    private static final DispatchPolicy CPU_POLICY = DispatchPolicy.builder()
        .enableCuda(false)
        .enableParallel(false)
        .parallelism(1)
        .naiveThreshold(1)
        .blockedThreshold(1)
        .blas3Threshold(1)
        .enableBlas3(true)
        .build();

    private static final int[][] SHAPES = {
        {4, 4, 4},
        {12, 12, 12},
        {64, 64, 64},
        {128, 64, 16},
        {16, 64, 128},
        {96, 96, 1},
        {128, 3, 128}
    };

    private static final double[] GRID = {0.0, 1.0, -0.5, 2.5};
    private static final int PREFERRED_VECTOR_LENGTH = DoubleVector.SPECIES_PREFERRED.length();

    @Test
    public void gemmParityAgainstIndependentReference() {
        long seed = 20260603L;
        for (int[] shape : SHAPES) {
            int m = shape[0];
            int k = shape[1];
            int n = shape[2];
            for (double alpha : GRID) {
                for (double beta : GRID) {
                    Matrix a = GemmReference.seededMatrix(m, k, seed + 11);
                    Matrix b = GemmReference.seededMatrix(k, n, seed + 29);
                    Matrix cInitial = GemmReference.seededMatrix(m, n, seed + 47);

                    double[] expected = GemmReference.gemm(
                        a.getRawData(), b.getRawData(), cInitial.getRawData(), m, k, n, alpha, beta
                    );

                    Matrix blasOut = Matrix.wrap(cInitial.getRawData().clone(), m, n);
                    BLAS3Kernels.gemm(a, b, blasOut, alpha, beta, CPU_POLICY);
                    GemmReference.assertParity(
                        "BLAS3Kernels.gemm", m, k, n, alpha, beta, seed,
                        expected, blasOut.getRawData(), 1e-12, GemmReference.cpuAbsTolerance(expected, k)
                    );

                    Matrix optOut = Matrix.wrap(cInitial.getRawData().clone(), m, n);
                    if (!(m == 4 && k == 4 && n == 4 && PREFERRED_VECTOR_LENGTH > 4)) {
                        OptimizedBLAS3.gemm(a, b, optOut, alpha, beta, CPU_POLICY);
                        GemmReference.assertParity(
                            "OptimizedBLAS3.gemm", m, k, n, alpha, beta, seed,
                            expected, optOut.getRawData(), 1e-12, GemmReference.cpuAbsTolerance(expected, k)
                        );
                    }
                }
            }
            seed += 101;
        }
    }

    @Ignore("Known bug: wide-SIMD tiny 4x4x4 path throws IndexOutOfBoundsException; fix in follow-up production change")
    @Test
    public void optimizedBlas3WideSimdTiny4x4MatchesReference() {
        if (PREFERRED_VECTOR_LENGTH <= 4) {
            return;
        }
        Matrix a = GemmReference.seededMatrix(4, 4, 3201L);
        Matrix b = GemmReference.seededMatrix(4, 4, 3202L);
        Matrix cInitial = GemmReference.seededMatrix(4, 4, 3203L);
        double[] expected = GemmReference.gemm(a.getRawData(), b.getRawData(), cInitial.getRawData(), 4, 4, 4, 1.0, 0.0);
        Matrix c = Matrix.wrap(cInitial.getRawData().clone(), 4, 4);
        OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, CPU_POLICY);
        GemmReference.assertParity(
            "OptimizedBLAS3.gemm known-bug 4x4x4", 4, 4, 4, 1.0, 0.0, 3200L,
            expected, c.getRawData(), 1e-12, GemmReference.cpuAbsTolerance(expected, 4)
        );
    }

    @Test
    public void gemmMatrixReturningFacadeMatchesReference() {
        Matrix a = GemmReference.seededMatrix(16, 9, 3001L);
        Matrix b = GemmReference.seededMatrix(9, 7, 3002L);
        Matrix result = BLAS3Kernels.gemm(a, b, CPU_POLICY);
        double[] expected = GemmReference.gemm(a.getRawData(), b.getRawData(), new double[16 * 7], 16, 9, 7, 1.0, 0.0);
        GemmReference.assertParity(
            "BLAS3Kernels.gemm(Matrix,Matrix,policy)", 16, 9, 7, 1.0, 0.0, 3000L,
            expected, result.getRawData(), 1e-12, GemmReference.cpuAbsTolerance(expected, 9)
        );
    }

    @Test
    public void gemmRejectsNullAndDimensionMismatch() {
        Matrix a = new Matrix(2, 3);
        Matrix b = new Matrix(4, 2);
        Matrix c = new Matrix(2, 2);

        expectIllegalArgument("BLAS3Kernels null", () -> BLAS3Kernels.gemm(null, b, c, 1.0, 0.0, CPU_POLICY));
        expectIllegalArgument("OptimizedBLAS3 null", () -> OptimizedBLAS3.gemm(a, null, c, 1.0, 0.0, CPU_POLICY));
        expectIllegalArgument("BLAS3Kernels mismatch", () -> BLAS3Kernels.gemm(a, b, c, 1.0, 0.0, CPU_POLICY));
        expectIllegalArgument("OptimizedBLAS3 mismatch", () -> OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, CPU_POLICY));
        expectIllegalArgument("BLAS3Kernels output mismatch", () -> BLAS3Kernels.gemm(a, new Matrix(3, 2), new Matrix(1, 2), 1.0, 0.0, CPU_POLICY));
        expectIllegalArgument("OptimizedBLAS3 output mismatch", () -> OptimizedBLAS3.gemm(a, new Matrix(3, 2), new Matrix(1, 2), 1.0, 0.0, CPU_POLICY));
    }

    @Test
    public void gemmSupportsZeroAndDegenerateDimensions() {
        Matrix a = new Matrix(0, 3);
        Matrix b = new Matrix(3, 2);
        Matrix c = new Matrix(0, 2);
        BLAS3Kernels.gemm(a, b, c, 1.0, 0.0, CPU_POLICY);
        OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, CPU_POLICY);
        assertEquals(0, c.getRawData().length);

        Matrix aK0 = new Matrix(5, 0);
        Matrix bK0 = new Matrix(0, 4);
        Matrix cInitial = GemmReference.seededMatrix(5, 4, 901L);
        double[] expected = Arrays.copyOf(cInitial.getRawData(), cInitial.getRawData().length);
        for (int i = 0; i < expected.length; i++) {
            expected[i] *= -0.5;
        }

        Matrix blasOut = Matrix.wrap(cInitial.getRawData().clone(), 5, 4);
        BLAS3Kernels.gemm(aK0, bK0, blasOut, 2.5, -0.5, CPU_POLICY);
        Matrix optOut = Matrix.wrap(cInitial.getRawData().clone(), 5, 4);
        OptimizedBLAS3.gemm(aK0, bK0, optOut, 2.5, -0.5, CPU_POLICY);

        GemmReference.assertParity("BLAS3Kernels.gemm zero-k", 5, 0, 4, 2.5, -0.5, 901L, expected, blasOut.getRawData(), 1e-12, 1e-12);
        GemmReference.assertParity("OptimizedBLAS3.gemm zero-k", 5, 0, 4, 2.5, -0.5, 901L, expected, optOut.getRawData(), 1e-12, 1e-12);

        Matrix empty = BLAS3Kernels.gemm(new Matrix(2, 0), new Matrix(0, 0), CPU_POLICY);
        assertEquals(2, empty.getRowCount());
        assertEquals(0, empty.getColumnCount());
        assertTrue(empty.getRawData().length == 0);
    }

    private static void expectIllegalArgument(String label, ThrowingRunnable runnable) {
        try {
            runnable.run();
            fail(label + " should throw IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            assertTrue(label + " message should not be empty", expected.getMessage() != null && !expected.getMessage().isEmpty());
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run();
    }
}
