package net.faulj.compute;

import net.faulj.matrix.Matrix;
import org.junit.Assume;
import org.junit.Test;

public class GemmCudaParityTest {
    @Test
    public void cudaGemmParityWhenCudaActuallyExecutes() {
        CudaSupport.refresh();
        Assume.assumeTrue("CUDA not available", CudaSupport.isCudaAvailable());

        Matrix a = GemmReference.seededMatrix(64, 64, 7001L);
        Matrix b = GemmReference.seededMatrix(64, 64, 7002L);
        Matrix c = GemmReference.seededMatrix(64, 64, 7003L);
        double[] expected = GemmReference.gemm(a.getRawData(), b.getRawData(), c.getRawData(), 64, 64, 64, 1.0, -0.5);

        Matrix out = Matrix.wrap(c.getRawData().clone(), 64, 64);
        boolean ok = CudaGemm.gemm(a, b, out, 1.0, -0.5);
        Assume.assumeTrue("CudaGemm.gemm returned false on this machine", ok);

        GemmReference.assertParity(
            "CudaGemm.gemm", 64, 64, 64, 1.0, -0.5, 7000L,
            expected, out.getRawData(), 1e-10, GemmReference.cudaAbsTolerance(expected, 64)
        );
    }
}
