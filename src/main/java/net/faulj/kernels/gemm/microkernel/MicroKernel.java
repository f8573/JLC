package net.faulj.kernels.gemm.microkernel;

/**
 * Register microkernel facade.
 */
public final class MicroKernel {
    private MicroKernel() {
    }

    public static int optimalKUnroll(int vecLen) {
        return net.faulj.compute.MicroKernel.optimalKUnroll(vecLen);
    }

    public static void compute(int mr, int kBlock, int packedN, int actualN,
                               double[] aPack, double[] bPack, double[] c,
                               int cOffset, int ldc) {
        net.faulj.compute.MicroKernel.compute(mr, kBlock, packedN, actualN, aPack, bPack, c, cOffset, ldc);
    }
}
