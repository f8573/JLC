package net.faulj.stress;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * Unit tests for Householder QR decomposition.
 */
public class HouseholderQRTest {

    private static final double RECON_TOL = 1e-12;
    private static final double ORTHO_TOL = 1e-12;

    private static final int[] SIZES = {
            1, 2, 3, 5, 10, 20, 50, 100
    };

    private static final long SEED = 12345L;

    @Test
    public void testQRReconstruction() {
        Random rnd = new Random(SEED);

        for (int n : SIZES) {
            Matrix A = randomMatrix(rnd, n);
            Matrix A0 = A.copy();   // deep copy for residual

            QRResult qr = HouseholderQR.decompose(A);

            Matrix Q = qr.getQ();
            Matrix R = qr.getR();

            Matrix QR = Q.multiply(R);

            double relErr = QR.subtract(A0).frobeniusNorm() / A0.frobeniusNorm();

            assertTrue("Reconstruction error too large for n=" + n + ": " + relErr,
                    relErr < RECON_TOL);
        }
    }

    @Test
    public void testQOrthogonality() {
        Random rnd = new Random(SEED + 1);

        for (int n : SIZES) {
            Matrix A = randomMatrix(rnd, n);

            QRResult qr = HouseholderQR.decompose(A);

            Matrix Q = qr.getQ();
            Matrix QtQ = Q.transpose().multiply(Q);
            Matrix I = Matrix.Identity(QtQ.getRowCount());

            double orthoErr = QtQ.subtract(I).frobeniusNorm();

            assertTrue("Q not orthogonal for n=" + n + ": " + orthoErr,
                    orthoErr < ORTHO_TOL);
        }
    }

    @Test
    public void testRUpperTriangular() {
        Random rnd = new Random(SEED + 2);

        for (int n : SIZES) {
            Matrix A = randomMatrix(rnd, n);

            QRResult qr = HouseholderQR.decompose(A);
            Matrix R = qr.getR();

            double maxBelow = 0.0;

            for (int i = 0; i < R.getRowCount(); i++) {
                for (int j = 0; j < Math.min(i, R.getColumnCount()); j++) {
                    maxBelow = Math.max(maxBelow, Math.abs(R.get(i, j)));
                }
            }

            assertTrue("R has large entries below diagonal for n=" + n + ": " + maxBelow,
                    maxBelow < 1e-12);
        }
    }

    @Test
    public void testThinQR() {
        Random rnd = new Random(SEED + 3);

        int m = 100;
        int n = 40;

        Matrix A = randomMatrixRectangular(rnd, m, n);
        Matrix A0 = A.copy();

        QRResult qr = HouseholderQR.decomposeThin(A);

        Matrix Q = qr.getQ();   // m x n
        Matrix R = qr.getR();   // n x n

        Matrix QR = Q.multiply(R);

        double relErr = QR.subtract(A0).frobeniusNorm() / A0.frobeniusNorm();

        assertTrue("Thin QR reconstruction error too large: " + relErr,
                relErr < RECON_TOL);

        // Orthogonality of thin Q: Q^T Q = I
        Matrix QtQ = Q.transpose().multiply(Q);
        Matrix I = Matrix.Identity(n);

        double orthoErr = QtQ.subtract(I).frobeniusNorm();

        assertTrue("Thin Q not orthogonal: " + orthoErr,
                orthoErr < ORTHO_TOL);
    }

    /* ---------------- Helpers ---------------- */

    private static Matrix randomMatrix(Random rnd, int n) {
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                d[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        return new Matrix(d);
    }

    private static Matrix randomMatrixRectangular(Random rnd, int m, int n) {
        double[][] d = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                d[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        return new Matrix(d);
    }
}
