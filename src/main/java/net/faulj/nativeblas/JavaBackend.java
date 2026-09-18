package net.faulj.nativeblas;

import net.faulj.compute.BLAS3Kernels;
import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.OptimizedBLAS3;
import net.faulj.matrix.Matrix;

/**
 * Current in-process Java backend.
 */
final class JavaBackend implements ComputeBackend {
    @Override
    public String backendId() {
        return "java";
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public void gemm(Matrix a, Matrix b, Matrix c, double alpha, double beta, DispatchPolicy policy) {
        OptimizedBLAS3.gemm(a, b, c, alpha, beta, policy);
    }

    @Override
    public void gemmStrided(double[] a, int aOffset, int lda,
                            double[] b, int bOffset, int ldb,
                            double[] c, int cOffset, int ldc,
                            int m, int k, int n,
                            double alpha, double beta) {
        OptimizedBLAS3.gemmStrided(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta);
    }

    @Override
    public void gemmStrided(boolean transposeA,
                            double[] a, int aOffset, int lda,
                            double[] b, int bOffset, int ldb,
                            double[] c, int cOffset, int ldc,
                            int m, int k, int n,
                            double alpha, double beta) {
        OptimizedBLAS3.gemmStrided(transposeA, a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta);
    }

    @Override
    public void gemmStrided(double[] a, int aOffset, int lda,
                            double[] b, int bOffset, int ldb,
                            double[] c, int cOffset, int ldc,
                            int m, int k, int n,
                            double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStrided(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
    }

    @Override
    public void gemmStridedTransA(double[] a, int aOffset, int lda,
                                  double[] b, int bOffset, int ldb,
                                  double[] c, int cOffset, int ldc,
                                  int m, int k, int n,
                                  double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStridedTransA(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
    }

    @Override
    public void gemmStridedColMajorA(double[] a, int aOffset, int lda,
                                     double[] b, int bOffset, int ldb,
                                     double[] c, int cOffset, int ldc,
                                     int m, int k, int n,
                                     double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStridedColMajorA(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
    }

    @Override
    public void gemmStridedColMajorB(double[] a, int aOffset, int lda,
                                     double[] b, int bOffset, int ldb,
                                     double[] c, int cOffset, int ldc,
                                     int m, int k, int n,
                                     double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStridedColMajorB(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
    }

    @Override
    public void gemmStridedBatched(double[] a, int aOffset, int lda, int aStride,
                                   double[] b, int bOffset, int ldb, int bStride,
                                   double[] c, int cOffset, int ldc, int cStride,
                                   int m, int k, int n,
                                   int batchCount,
                                   double alpha, double beta) {
        if (batchCount <= 0) {
            return;
        }
        for (int batch = 0; batch < batchCount; batch++) {
            int aBatchOffset = aOffset + batch * aStride;
            int bBatchOffset = bOffset + batch * bStride;
            int cBatchOffset = cOffset + batch * cStride;
            OptimizedBLAS3.gemmStrided(a, aBatchOffset, lda, b, bBatchOffset, ldb, c, cBatchOffset, ldc, m, k, n, alpha, beta);
        }
    }
}
