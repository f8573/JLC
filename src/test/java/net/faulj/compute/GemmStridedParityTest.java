package net.faulj.compute;

import org.junit.Test;

public class GemmStridedParityTest {
    @Test
    public void stridedRowMajorParity() {
        int m = 9;
        int k = 7;
        int n = 11;
        int aOff = 13;
        int bOff = 13;
        int cOff = 13;
        int lda = k + 5;
        int ldb = n + 7;
        int ldc = n + 3;
        long seed = 4101L;

        double[] a = GemmReference.seededArray(aOff + lda * m + 17, seed + 1);
        double[] b = GemmReference.seededArray(bOff + ldb * k + 17, seed + 2);
        double[] c = GemmReference.seededArray(cOff + ldc * m + 17, seed + 3);
        double[] expected = GemmReference.gemmStrided(a, aOff, lda, b, bOff, ldb, c, cOff, ldc, m, k, n, 2.5, -0.5);

        BLAS3Kernels.gemmStrided(a, aOff, lda, b, bOff, ldb, c, cOff, ldc, m, k, n, 2.5, -0.5, 19);
        GemmReference.assertParity(
            "BLAS3Kernels.gemmStrided", m, k, n, 2.5, -0.5, seed,
            expected, GemmReference.extractRowMajor(c, cOff, ldc, m, n), 1e-12, GemmReference.cpuAbsTolerance(expected, k)
        );
    }

    @Test
    public void stridedTransposedAParity() {
        int m = 8;
        int k = 5;
        int n = 6;
        int aOff = 13;
        int bOff = 13;
        int cOff = 13;
        int lda = k + 7;
        int ldb = n + 5;
        int ldc = n + 3;
        long seed = 4201L;

        double[] a = GemmReference.seededArray(aOff + lda * m + 9, seed + 1);
        double[] b = GemmReference.seededArray(bOff + ldb * m + 9, seed + 2);
        double[] c = GemmReference.seededArray(cOff + ldc * k + 9, seed + 3);
        double[] expected = GemmReference.gemmStridedTransA(a, aOff, lda, b, bOff, ldb, c, cOff, ldc, m, k, n, 1.0, 2.5);

        BLAS3Kernels.gemmStridedTransA(a, aOff, lda, b, bOff, ldb, c, cOff, ldc, m, k, n, 1.0, 2.5, 17);
        GemmReference.assertParity(
            "BLAS3Kernels.gemmStridedTransA", k, m, n, 1.0, 2.5, seed,
            expected, GemmReference.extractRowMajor(c, cOff, ldc, k, n), 1e-12, GemmReference.cpuAbsTolerance(expected, m)
        );
    }

    @Test
    public void stridedColumnMajorVariantsParity() {
        int m = 10;
        int k = 6;
        int n = 7;
        int rowMajorLdA = k + 4;
        int rowMajorLdB = n + 7;
        int colMajorLdA = m + 5;
        int colMajorLdB = k + 3;
        int ldc = n + 5;
        int off = 13;
        long seed = 4301L;

        double[] aRowMajor = GemmReference.seededArray(m * k, seed + 1);
        double[] bRowMajor = GemmReference.seededArray(k * n, seed + 2);
        double[] cA = GemmReference.seededArray(off + ldc * m + 9, seed + 3);
        double[] cB = GemmReference.seededArray(off + ldc * m + 9, seed + 4);

        double[] aColMajor = GemmReference.toColumnMajor(aRowMajor, m, k, colMajorLdA);
        double[] bColMajor = GemmReference.toColumnMajor(bRowMajor, k, n, colMajorLdB);
        double[] bRowMajorStrided = new double[off + rowMajorLdB * k + 9];
        double[] aRowMajorStrided = new double[off + rowMajorLdA * m + 9];

        for (int i = 0; i < m; i++) {
            System.arraycopy(aRowMajor, i * k, aRowMajorStrided, off + i * rowMajorLdA, k);
        }
        for (int i = 0; i < k; i++) {
            System.arraycopy(bRowMajor, i * n, bRowMajorStrided, off + i * rowMajorLdB, n);
        }

        double[] expectedA = GemmReference.gemmStridedColMajorA(
            aColMajor, 0, colMajorLdA, bRowMajorStrided, off, rowMajorLdB, cA, off, ldc, m, k, n, -0.5, 1.0
        );
        BLAS3Kernels.gemmStridedColMajorA(
            aColMajor, 0, colMajorLdA, bRowMajorStrided, off, rowMajorLdB, cA, off, ldc, m, k, n, -0.5, 1.0, 15
        );
        GemmReference.assertParity(
            "BLAS3Kernels.gemmStridedColMajorA", m, k, n, -0.5, 1.0, seed,
            expectedA, GemmReference.extractRowMajor(cA, off, ldc, m, n), 1e-12, GemmReference.cpuAbsTolerance(expectedA, k)
        );

        double[] expectedB = GemmReference.gemmStridedColMajorB(
            aRowMajorStrided, off, rowMajorLdA, bColMajor, 0, colMajorLdB, cB, off, ldc, m, k, n, 2.5, 0.0
        );
        BLAS3Kernels.gemmStridedColMajorB(
            aRowMajorStrided, off, rowMajorLdA, bColMajor, 0, colMajorLdB, cB, off, ldc, m, k, n, 2.5, 0.0, 15
        );
        GemmReference.assertParity(
            "BLAS3Kernels.gemmStridedColMajorB", m, k, n, 2.5, 0.0, seed,
            expectedB, GemmReference.extractRowMajor(cB, off, ldc, m, n), 1e-12, GemmReference.cpuAbsTolerance(expectedB, k)
        );
    }

    @Test
    public void optimizedStridedParity() {
        int m = 13;
        int k = 9;
        int n = 8;
        int off = 13;
        int lda = k + 7;
        int ldb = n + 5;
        int ldc = n + 1;
        long seed = 4401L;

        double[] a = GemmReference.seededArray(off + lda * m + 5, seed + 1);
        double[] b = GemmReference.seededArray(off + ldb * k + 5, seed + 2);
        double[] c = GemmReference.seededArray(off + ldc * m + 5, seed + 3);
        double[] expected = GemmReference.gemmStrided(a, off, lda, b, off, ldb, c, off, ldc, m, k, n, 1.0, -0.5);

        OptimizedBLAS3.gemmStrided(a, off, lda, b, off, ldb, c, off, ldc, m, k, n, 1.0, -0.5);
        GemmReference.assertParity(
            "OptimizedBLAS3.gemmStrided", m, k, n, 1.0, -0.5, seed,
            expected, GemmReference.extractRowMajor(c, off, ldc, m, n), 1e-12, GemmReference.cpuAbsTolerance(expected, k)
        );
    }
}
