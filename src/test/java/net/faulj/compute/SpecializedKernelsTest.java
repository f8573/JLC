package net.faulj.compute;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;

public class SpecializedKernelsTest {
    @Test
    public void tinyGemm4x4DoesNotDependOnPreferredVectorWidth() {
        double[] a = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        double[] b = {
            2, 0, 1, 3,
            1, 4, 0, 2,
            3, 1, 2, 0,
            0, 2, 4, 1
        };
        double[] actual = new double[16];
        double[] expected = new double[16];

        SpecializedKernels.tinyGemm(a, 4, b, 4, actual, 4, 4, 4, 4, 1.0, 0.0);
        referenceGemm(a, 4, b, 4, expected, 4, 4, 4, 4, 1.0, 0.0);

        assertArrayEquals(expected, actual, 1.0e-12);
    }

    @Test
    public void tinyGemmUsesGenericPathForNonCompactSquareStorage() {
        double[] a = {
            1, 2, 0,
            3, 4, 0
        };
        double[] b = {
            5, 6, 0,
            7, 8, 0
        };
        double[] actual = {10, 20, 0, 30, 40, 0};
        double[] expected = actual.clone();

        SpecializedKernels.tinyGemm(a, 3, b, 3, actual, 3, 2, 2, 2, 1.0, 0.5);
        referenceGemm(a, 3, b, 3, expected, 3, 2, 2, 2, 1.0, 0.5);

        assertArrayEquals(expected, actual, 1.0e-12);
    }

    @Test
    public void tinyGemmRejectsImpossibleBackingStorageBeforeFixedKernelSelection() {
        double[] tooSmall = new double[9];

        assertThrows(IllegalArgumentException.class, () ->
            SpecializedKernels.tinyGemm(tooSmall, 4, tooSmall, 4, tooSmall, 4, 4, 4, 4, 1.0, 0.0));
    }

    private static void referenceGemm(double[] a, int lda, double[] b, int ldb,
                                      double[] c, int ldc, int m, int k, int n,
                                      double alpha, double beta) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int kk = 0; kk < k; kk++) {
                    sum = Math.fma(a[i * lda + kk], b[kk * ldb + j], sum);
                }
                c[i * ldc + j] = Math.fma(alpha, sum, beta * c[i * ldc + j]);
            }
        }
    }
}
