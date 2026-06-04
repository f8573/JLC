package net.faulj.compute;

import net.faulj.matrix.Matrix;
import org.junit.Assume;
import org.junit.Test;

public class GemmLargeParityTest {
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
    public void largeParityCases() {
        Assume.assumeTrue("Set -Dfaulj.gemm.slow=true to run slow GEMM parity coverage", Boolean.getBoolean("faulj.gemm.slow"));
        runCase(512, 512, 512, 8001L);
        runCase(768, 512, 256, 8101L);
    }

    private static void runCase(int m, int k, int n, long seed) {
        Matrix a = GemmReference.seededMatrix(m, k, seed + 1);
        Matrix b = GemmReference.seededMatrix(k, n, seed + 2);
        Matrix cInitial = GemmReference.seededMatrix(m, n, seed + 3);
        double[] expected = GemmReference.gemm(a.getRawData(), b.getRawData(), cInitial.getRawData(), m, k, n, 1.0, -0.5);

        Matrix blasOut = Matrix.wrap(cInitial.getRawData().clone(), m, n);
        BLAS3Kernels.gemm(a, b, blasOut, 1.0, -0.5, CPU_POLICY);
        GemmReference.assertParity(
            "BLAS3Kernels.gemm large", m, k, n, 1.0, -0.5, seed,
            expected, blasOut.getRawData(), 1e-12, GemmReference.cpuAbsTolerance(expected, k)
        );

        Matrix optOut = Matrix.wrap(cInitial.getRawData().clone(), m, n);
        OptimizedBLAS3.gemm(a, b, optOut, 1.0, -0.5, CPU_POLICY);
        GemmReference.assertParity(
            "OptimizedBLAS3.gemm large", m, k, n, 1.0, -0.5, seed,
            expected, optOut.getRawData(), 1e-12, GemmReference.cpuAbsTolerance(expected, k)
        );
    }
}
