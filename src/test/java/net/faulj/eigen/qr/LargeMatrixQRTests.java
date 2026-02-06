package net.faulj.eigen.qr;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * Tests ExplicitQRIteration (Schur) and HouseholderQR across a range of matrix sizes.
 *
 * These tests validate:
 * - reconstruction accuracy (relative Frobenius error)
 * - orthogonality of Q factors
 * - Hessenberg / quasi-triangular structure of Schur T
 */
public class LargeMatrixQRTests {

    private static final double TOL_RECON = 1e-8;
    private static final double TOL_ORTH = 1e-10;
    private static final double TOL_HESSEN = 1e-8;

    private Matrix randomDense(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        return new Matrix(a);
    }

    private Matrix randomSymmetric(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double v = rnd.nextDouble() * 2.0 - 1.0;
                a[i][j] = v;
                a[j][i] = v;
            }
        }
        return new Matrix(a);
    }

    private void assertQuasiTriangular(Matrix T) {
        int n = T.getRowCount();
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                double v = T.get(i, j);
                assertEquals("T is not quasi-triangular at (" + i + "," + j + ")", 0.0, v, TOL_HESSEN);
            }
        }
    }

    @Test
    public void testSchurDecomposeVariousSizes() {
        int[] sizes = new int[]{2, 3, 4, 8, 16, 32, 64, 128, 256};
        long baseSeed = 0xC0FFEE;

        for (int n : sizes) {
            Matrix A = randomDense(n, baseSeed + n);

            // Compute Schur (T, Q)
            Matrix[] res = ExplicitQRIteration.decompose(A);
            assertNotNull("Schur result should not be null for n=" + n, res);
            assertEquals("Schur must return two matrices", 2, res.length);

            Matrix T = res[0];
            Matrix Q = res[1];

            // Reconstruction: A == Q*T*Q^T
            Matrix recon = Q.multiply(T).multiply(Q.transpose());
            double rel = MatrixUtils.relativeError(A, recon);
            assertTrue("Relative reconstruction error too large for Schur n=" + n + ": " + rel, rel < TOL_RECON);

            // Orthogonality of Q
            double ort = MatrixUtils.orthogonalityError(Q);
            assertTrue("Q is not orthogonal for Schur n=" + n + ": " + ort, ort < TOL_ORTH);

            // Quasi-triangular (Hessenberg below first subdiagonal must be zero)
            assertQuasiTriangular(T);
        }
    }

    @Test
    public void testSchurOnSymmetricMatrices() {
        int[] sizes = new int[]{2, 3, 4, 8, 16, 32, 64};
        long baseSeed = 0xDEADBEEF;

        for (int n : sizes) {
            Matrix A = randomSymmetric(n, baseSeed + n);

            Matrix[] res = ExplicitQRIteration.decompose(A);
            Matrix T = res[0];
            Matrix Q = res[1];

            // For symmetric matrices, the Schur form should be (nearly) upper triangular and Q orthogonal
            double ort = MatrixUtils.orthogonalityError(Q);
            assertTrue("Q not orthogonal for symmetric n=" + n, ort < TOL_ORTH);

            // Since symmetric, T should be upper triangular (no 2x2 blocks with imaginary parts)
            assertQuasiTriangular(T);
        }
    }

    @Test
    public void testHouseholderQRVariousSizes() {
        int[] sizes = new int[]{2, 3, 4, 8, 16, 32, 64, 128, 256};
        long baseSeed = 0xBEEFCAFE;

        for (int n : sizes) {
            Matrix A = randomDense(n, baseSeed + n);

            QRResult qr = HouseholderQR.decompose(A);
            assertNotNull("QR result null for n=" + n, qr);

            double resid = qr.residualNorm();
            assertTrue("Householder QR residual too large for n=" + n + ": " + resid, resid < TOL_RECON);

            double ort = MatrixUtils.orthogonalityError(qr.getQ());
            assertTrue("Q not orthogonal for QR n=" + n + ": " + ort, ort < TOL_ORTH);
        }
    }
}
