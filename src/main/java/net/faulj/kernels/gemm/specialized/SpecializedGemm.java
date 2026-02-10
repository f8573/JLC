package net.faulj.kernels.gemm.specialized;

/**
 * Specialized tiny/small GEMM and matvec facade.
 */
public final class SpecializedGemm {
    private SpecializedGemm() {
    }

    public static void matvec(double[] a, int lda, double[] x, double[] y,
                              int m, int k, double alpha, double beta) {
        net.faulj.compute.SpecializedKernels.matvec(a, lda, x, y, m, k, alpha, beta);
    }

    public static void smallGemm(double[] a, int lda, double[] b, int ldb,
                                 double[] c, int ldc, int m, int k, int n,
                                 double alpha, double beta) {
        net.faulj.compute.SpecializedKernels.smallGemm(a, lda, b, ldb, c, ldc, m, k, n, alpha, beta);
    }

    public static void tinyGemm(double[] a, int lda, double[] b, int ldb,
                                double[] c, int ldc, int m, int k, int n,
                                double alpha, double beta) {
        net.faulj.compute.SpecializedKernels.tinyGemm(a, lda, b, ldb, c, ldc, m, k, n, alpha, beta);
    }
}
