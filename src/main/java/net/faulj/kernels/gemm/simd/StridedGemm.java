package net.faulj.kernels.gemm.simd;

import net.faulj.compute.BLAS3Kernels;

/**
 * SIMD packed strided GEMM adapter surface.
 */
public final class StridedGemm {
    private StridedGemm() {
    }

    public static void gemm(double[] ad, int aOff, int lda,
                            double[] bd, int bOff, int ldb,
                            double[] cd, int cOff, int ldc,
                            int m, int k, int n,
                            double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStrided(ad, aOff, lda, bd, bOff, ldb, cd, cOff, ldc, m, k, n, alpha, beta, blockSize);
    }

    public static void gemmTransA(double[] ad, int aOff, int lda,
                                  double[] bd, int bOff, int ldb,
                                  double[] cd, int cOff, int ldc,
                                  int m, int k, int n,
                                  double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStridedTransA(ad, aOff, lda, bd, bOff, ldb, cd, cOff, ldc, m, k, n, alpha, beta, blockSize);
    }

    public static void gemmColMajorA(double[] ad, int aOff, int lda,
                                     double[] bd, int bOff, int ldb,
                                     double[] cd, int cOff, int ldc,
                                     int m, int k, int n,
                                     double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStridedColMajorA(ad, aOff, lda, bd, bOff, ldb, cd, cOff, ldc, m, k, n, alpha, beta, blockSize);
    }

    public static void gemmColMajorB(double[] ad, int aOff, int lda,
                                     double[] bd, int bOff, int ldb,
                                     double[] cd, int cOff, int ldc,
                                     int m, int k, int n,
                                     double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStridedColMajorB(ad, aOff, lda, bd, bOff, ldb, cd, cOff, ldc, m, k, n, alpha, beta, blockSize);
    }
}
