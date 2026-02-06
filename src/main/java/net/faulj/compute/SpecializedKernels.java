package net.faulj.compute;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Specialized fast kernels for common GEMM cases.
 */
public final class SpecializedKernels {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private SpecializedKernels() {}

    /**
     * Matrix-vector multiplication: y = alpha * A * x + beta * y.
     * Optimized for n=1 case with SIMD dot products.
     *
     * @param a matrix A (m × k, row-major)
     * @param lda leading dimension of A
     * @param x vector x (length k)
     * @param y vector y (length m, output)
     * @param alpha scalar alpha
     * @param beta scalar beta
     */
    public static void matvec(double[] a, int lda, double[] x, double[] y,
                             int m, int k, double alpha, double beta) {
        int vecLen = SPECIES.length();

        // Apply beta to y
        if (beta == 0.0) {
            java.util.Arrays.fill(y, 0, m, 0.0);
        } else if (beta != 1.0) {
            DoubleVector betaVec = DoubleVector.broadcast(SPECIES, beta);
            int i = 0;
            int loopBound = SPECIES.loopBound(m);
            for (; i < loopBound; i += vecLen) {
                DoubleVector yv = DoubleVector.fromArray(SPECIES, y, i);
                yv.mul(betaVec).intoArray(y, i);
            }
            for (; i < m; i++) {
                y[i] *= beta;
            }
        }

        if (alpha == 0.0 || k == 0) {
            return;
        }

        // Compute y += alpha * A * x with SIMD dot products
        for (int i = 0; i < m; i++) {
            int rowOffset = i * lda;
            DoubleVector sum = DoubleVector.zero(SPECIES);

            int kk = 0;
            int loopBound = SPECIES.loopBound(k);
            for (; kk < loopBound; kk += vecLen) {
                DoubleVector av = DoubleVector.fromArray(SPECIES, a, rowOffset + kk);
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, kk);
                sum = av.lanewise(VectorOperators.FMA, xv, sum);
            }

            double total = sum.reduceLanes(VectorOperators.ADD);
            for (; kk < k; kk++) {
                total = Math.fma(a[rowOffset + kk], x[kk], total);
            }

            y[i] += alpha * total;
        }
    }

    /**
     * Small matrix multiply using scalar loops with FMA.
     * Optimal for matrices where m*n*k < ~50000 (overhead dominates for packed kernel).
     *
     * Loop order: i -> k -> j (AXPY style, B rows accessed contiguously).
     */
    public static void smallGemm(double[] a, int lda, double[] b, int ldb,
                                double[] c, int ldc, int m, int k, int n,
                                double alpha, double beta) {
        // Apply beta to C
        PackingUtils.scaleCPanel(c, 0, m, n, ldc, beta);

        if (alpha == 0.0 || k == 0) {
            return;
        }

        // Compute C += alpha * A * B
        for (int i = 0; i < m; i++) {
            int aRowOffset = i * lda;
            int cRowOffset = i * ldc;
            for (int kk = 0; kk < k; kk++) {
                double aVal = a[aRowOffset + kk] * alpha;
                int bRowOffset = kk * ldb;
                for (int j = 0; j < n; j++) {
                    c[cRowOffset + j] = Math.fma(aVal, b[bRowOffset + j], c[cRowOffset + j]);
                }
            }
        }
    }

    /**
     * Tiny matrix multiply for very small sizes (up to 8×8).
     * Uses register-only computation when possible.
     * Dispatches to specialized kernels for 2x2, 3x3, 4x4.
     */
    public static void tinyGemm(double[] a, int lda, double[] b, int ldb,
                               double[] c, int ldc, int m, int k, int n,
                               double alpha, double beta) {
        if (m == 1 && n == 1) {
            // Dot product
            double sum = 0.0;
            for (int kk = 0; kk < k; kk++) {
                sum = Math.fma(a[kk], b[kk * ldb], sum);
            }
            c[0] = Math.fma(alpha, sum, beta * c[0]);
            return;
        }

        if (n == 1) {
            // Matrix-vector
            double[] y = new double[m];
            if (beta != 0.0) {
                for (int i = 0; i < m; i++) {
                    y[i] = c[i * ldc] * beta;
                }
            }
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                for (int kk = 0; kk < k; kk++) {
                    sum = Math.fma(a[i * lda + kk], b[kk * ldb], sum);
                }
                y[i] += alpha * sum;
            }
            for (int i = 0; i < m; i++) {
                c[i * ldc] = y[i];
            }
            return;
        }

        // Specialized kernels for square small matrices with matching strides
        if (m == 2 && k == 2 && n == 2 && lda == 2 && ldb == 2 && ldc == 2) {
            gemm2x2(a, b, c, alpha, beta);
            return;
        }
        if (m == 3 && k == 3 && n == 3 && lda == 3 && ldb == 3 && ldc == 3) {
            gemm3x3(a, b, c, alpha, beta);
            return;
        }
        if (m == 4 && k == 4 && n == 4 && lda == 4 && ldb == 4 && ldc == 4) {
            gemm4x4(a, b, c, alpha, beta);
            return;
        }

        // General tiny case
        smallGemm(a, lda, b, ldb, c, ldc, m, k, n, alpha, beta);
    }

    /**
     * Fully unrolled 2x2 GEMM: C = alpha*A*B + beta*C
     * All 4 elements computed without loops for maximum performance.
     */
    public static void gemm2x2(double[] a, double[] b, double[] c,
                              double alpha, double beta) {
        // Load and scale C
        double c00 = beta * c[0];
        double c01 = beta * c[1];
        double c10 = beta * c[2];
        double c11 = beta * c[3];

        // Load A with alpha scaling
        double a00 = a[0] * alpha;
        double a01 = a[1] * alpha;
        double a10 = a[2] * alpha;
        double a11 = a[3] * alpha;

        // Load B
        double b00 = b[0], b01 = b[1];
        double b10 = b[2], b11 = b[3];

        // Compute C += A * B (all FMAs)
        c00 = Math.fma(a00, b00, Math.fma(a01, b10, c00));
        c01 = Math.fma(a00, b01, Math.fma(a01, b11, c01));
        c10 = Math.fma(a10, b00, Math.fma(a11, b10, c10));
        c11 = Math.fma(a10, b01, Math.fma(a11, b11, c11));

        // Store C
        c[0] = c00; c[1] = c01;
        c[2] = c10; c[3] = c11;
    }

    /**
     * Fully unrolled 3x3 GEMM: C = alpha*A*B + beta*C
     * All 9 elements computed with 27 FMAs.
     */
    public static void gemm3x3(double[] a, double[] b, double[] c,
                              double alpha, double beta) {
        // Load and scale C
        double c00 = beta * c[0], c01 = beta * c[1], c02 = beta * c[2];
        double c10 = beta * c[3], c11 = beta * c[4], c12 = beta * c[5];
        double c20 = beta * c[6], c21 = beta * c[7], c22 = beta * c[8];

        // Load B
        double b00 = b[0], b01 = b[1], b02 = b[2];
        double b10 = b[3], b11 = b[4], b12 = b[5];
        double b20 = b[6], b21 = b[7], b22 = b[8];

        // k=0: A[row, 0] * B[0, col]
        double a00 = a[0] * alpha, a10 = a[3] * alpha, a20 = a[6] * alpha;
        c00 = Math.fma(a00, b00, c00); c01 = Math.fma(a00, b01, c01); c02 = Math.fma(a00, b02, c02);
        c10 = Math.fma(a10, b00, c10); c11 = Math.fma(a10, b01, c11); c12 = Math.fma(a10, b02, c12);
        c20 = Math.fma(a20, b00, c20); c21 = Math.fma(a20, b01, c21); c22 = Math.fma(a20, b02, c22);

        // k=1: A[row, 1] * B[1, col]
        double a01 = a[1] * alpha, a11 = a[4] * alpha, a21 = a[7] * alpha;
        c00 = Math.fma(a01, b10, c00); c01 = Math.fma(a01, b11, c01); c02 = Math.fma(a01, b12, c02);
        c10 = Math.fma(a11, b10, c10); c11 = Math.fma(a11, b11, c11); c12 = Math.fma(a11, b12, c12);
        c20 = Math.fma(a21, b10, c20); c21 = Math.fma(a21, b11, c21); c22 = Math.fma(a21, b12, c22);

        // k=2: A[row, 2] * B[2, col]
        double a02 = a[2] * alpha, a12 = a[5] * alpha, a22 = a[8] * alpha;
        c00 = Math.fma(a02, b20, c00); c01 = Math.fma(a02, b21, c01); c02 = Math.fma(a02, b22, c02);
        c10 = Math.fma(a12, b20, c10); c11 = Math.fma(a12, b21, c11); c12 = Math.fma(a12, b22, c12);
        c20 = Math.fma(a22, b20, c20); c21 = Math.fma(a22, b21, c21); c22 = Math.fma(a22, b22, c22);

        // Store C
        c[0] = c00; c[1] = c01; c[2] = c02;
        c[3] = c10; c[4] = c11; c[5] = c12;
        c[6] = c20; c[7] = c21; c[8] = c22;
    }

    /**
     * SIMD-vectorized 4x4 GEMM: C = alpha*A*B + beta*C
     * Uses AVX2 to compute 4 outputs per row in one vector.
     */
    public static void gemm4x4(double[] a, double[] b, double[] c,
                              double alpha, double beta) {
        int vecLen = SPECIES.length();

        // For AVX2 (vecLen=4), each row of C fits in one vector
        if (vecLen >= 4) {
            DoubleVector betaVec = DoubleVector.broadcast(SPECIES, beta);

            // Load C rows
            DoubleVector c0 = DoubleVector.fromArray(SPECIES, c, 0).mul(betaVec);
            DoubleVector c1 = DoubleVector.fromArray(SPECIES, c, 4).mul(betaVec);
            DoubleVector c2 = DoubleVector.fromArray(SPECIES, c, 8).mul(betaVec);
            DoubleVector c3 = DoubleVector.fromArray(SPECIES, c, 12).mul(betaVec);

            // For each k, load B row and broadcast A elements
            for (int kk = 0; kk < 4; kk++) {
                DoubleVector bRow = DoubleVector.fromArray(SPECIES, b, kk * 4);

                c0 = bRow.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a[kk] * alpha), c0);
                c1 = bRow.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a[4 + kk] * alpha), c1);
                c2 = bRow.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a[8 + kk] * alpha), c2);
                c3 = bRow.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, a[12 + kk] * alpha), c3);
            }

            // Store C rows
            c0.intoArray(c, 0);
            c1.intoArray(c, 4);
            c2.intoArray(c, 8);
            c3.intoArray(c, 12);
        } else {
            // Fall back to scalar for non-AVX2
            smallGemm(a, 4, b, 4, c, 4, 4, 4, 4, alpha, beta);
        }
    }

    /**
     * Outer-product update: C += alpha * x * y^T.
     * Highly optimized for rank-1 updates common in decompositions.
     */
    public static void outerProduct(double[] x, double[] y, double[] c, int ldc,
                                   int m, int n, double alpha) {
        int vecLen = SPECIES.length();

        for (int i = 0; i < m; i++) {
            int cRowOffset = i * ldc;
            double xi = x[i] * alpha;
            DoubleVector xiVec = DoubleVector.broadcast(SPECIES, xi);

            int j = 0;
            int loopBound = SPECIES.loopBound(n);
            for (; j < loopBound; j += vecLen) {
                DoubleVector yv = DoubleVector.fromArray(SPECIES, y, j);
                DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cRowOffset + j);
                yv.lanewise(VectorOperators.FMA, xiVec, cv).intoArray(c, cRowOffset + j);
            }
            for (; j < n; j++) {
                c[cRowOffset + j] = Math.fma(xi, y[j], c[cRowOffset + j]);
            }
        }
    }
}
