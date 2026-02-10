package net.faulj.kernels.gemm;

import net.faulj.compute.BLAS3Kernels;
import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.OptimizedBLAS3;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

/**
 * Canonical GEMM facade for the library.
 *
 * <p>This class is the primary entry point for dense matrix-matrix multiply
 * and related strided GEMM variants. Existing compute implementations are
 * preserved and routed through this facade to keep behavior stable while
 * making GEMM explicitly discoverable.</p>
 */
public final class Gemm {
    private Gemm() {
    }

    /**
     * Compute C = A * B using the default dispatch policy.
     */
    public static Matrix multiply(Matrix a, Matrix b) {
        return multiply(a, b, DispatchPolicy.defaultPolicy());
    }

    /**
     * Compute C = A * B using the provided dispatch policy.
     */
    public static Matrix multiply(Matrix a, Matrix b, DispatchPolicy policy) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Matrices must not be null");
        }
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int k2 = b.getRowCount();
        int n = b.getColumnCount();
        if (k != k2) {
            throw new IllegalArgumentException("Inner dimensions must agree for multiplication: " + k + " != " + k2);
        }
        Matrix c = (a instanceof OffHeapMatrix || b instanceof OffHeapMatrix)
            ? new OffHeapMatrix(m, n)
            : Matrix.zero(m, n);
        gemm(a, b, c, 1.0, 0.0, policy);
        return c;
    }

    /**
     * Compute C = alpha * A * B + beta * C.
     */
    public static void gemm(Matrix a, Matrix b, Matrix c,
                            double alpha, double beta, DispatchPolicy policy) {
        OptimizedBLAS3.gemm(a, b, c, alpha, beta, policy);
    }

    /**
     * Strided GEMM facade (auto-kernel selection).
     */
    public static void gemmStrided(double[] a, int aOffset, int lda,
                                   double[] b, int bOffset, int ldb,
                                   double[] c, int cOffset, int ldc,
                                   int m, int k, int n,
                                   double alpha, double beta) {
        OptimizedBLAS3.gemmStrided(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta);
    }

    /**
     * Strided GEMM with optional transpose of A.
     */
    public static void gemmStrided(boolean transposeA,
                                   double[] a, int aOffset, int lda,
                                   double[] b, int bOffset, int ldb,
                                   double[] c, int cOffset, int ldc,
                                   int m, int k, int n,
                                   double alpha, double beta) {
        OptimizedBLAS3.gemmStrided(transposeA, a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta);
    }

    /**
     * Strided packed-SIMD GEMM with explicit block size.
     */
    public static void gemmStrided(double[] ad, int aOff, int lda,
                                   double[] bd, int bOff, int ldb,
                                   double[] cd, int cOff, int ldc,
                                   int m, int k, int n,
                                   double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStrided(ad, aOff, lda, bd, bOff, ldb, cd, cOff, ldc, m, k, n, alpha, beta, blockSize);
    }

    /**
     * Strided GEMM for C = alpha * A^T * B + beta * C.
     */
    public static void gemmStridedTransA(double[] ad, int aOff, int lda,
                                         double[] bd, int bOff, int ldb,
                                         double[] cd, int cOff, int ldc,
                                         int m, int k, int n,
                                         double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStridedTransA(ad, aOff, lda, bd, bOff, ldb, cd, cOff, ldc, m, k, n, alpha, beta, blockSize);
    }

    /**
     * Strided GEMM where A is column-major and B/C are row-major.
     */
    public static void gemmStridedColMajorA(double[] ad, int aOff, int lda,
                                            double[] bd, int bOff, int ldb,
                                            double[] cd, int cOff, int ldc,
                                            int m, int k, int n,
                                            double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStridedColMajorA(ad, aOff, lda, bd, bOff, ldb, cd, cOff, ldc, m, k, n, alpha, beta, blockSize);
    }

    /**
     * Strided GEMM where B is column-major and A/C are row-major.
     */
    public static void gemmStridedColMajorB(double[] ad, int aOff, int lda,
                                            double[] bd, int bOff, int ldb,
                                            double[] cd, int cOff, int ldc,
                                            int m, int k, int n,
                                            double alpha, double beta, int blockSize) {
        BLAS3Kernels.gemmStridedColMajorB(ad, aOff, lda, bd, bOff, ldb, cd, cOff, ldc, m, k, n, alpha, beta, blockSize);
    }
}
