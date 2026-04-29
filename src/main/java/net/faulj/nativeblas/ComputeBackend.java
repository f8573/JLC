package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;
import net.faulj.matrix.Matrix;

/**
 * Narrow dense-compute backend interface used by the canonical GEMM facade.
 */
public interface ComputeBackend {
    String backendId();

    boolean isAvailable();

    void gemm(Matrix a, Matrix b, Matrix c, double alpha, double beta, DispatchPolicy policy);

    void gemmStrided(double[] a, int aOffset, int lda,
                     double[] b, int bOffset, int ldb,
                     double[] c, int cOffset, int ldc,
                     int m, int k, int n,
                     double alpha, double beta);

    void gemmStrided(boolean transposeA,
                     double[] a, int aOffset, int lda,
                     double[] b, int bOffset, int ldb,
                     double[] c, int cOffset, int ldc,
                     int m, int k, int n,
                     double alpha, double beta);

    void gemmStrided(double[] a, int aOffset, int lda,
                     double[] b, int bOffset, int ldb,
                     double[] c, int cOffset, int ldc,
                     int m, int k, int n,
                     double alpha, double beta, int blockSize);

    void gemmStridedTransA(double[] a, int aOffset, int lda,
                           double[] b, int bOffset, int ldb,
                           double[] c, int cOffset, int ldc,
                           int m, int k, int n,
                           double alpha, double beta, int blockSize);

    void gemmStridedColMajorA(double[] a, int aOffset, int lda,
                              double[] b, int bOffset, int ldb,
                              double[] c, int cOffset, int ldc,
                              int m, int k, int n,
                              double alpha, double beta, int blockSize);

    void gemmStridedColMajorB(double[] a, int aOffset, int lda,
                              double[] b, int bOffset, int ldb,
                              double[] c, int cOffset, int ldc,
                              int m, int k, int n,
                              double alpha, double beta, int blockSize);

    void gemmStridedBatched(double[] a, int aOffset, int lda, int aStride,
                            double[] b, int bOffset, int ldb, int bStride,
                            double[] c, int cOffset, int ldc, int cStride,
                            int m, int k, int n,
                            int batchCount,
                            double alpha, double beta);
}
