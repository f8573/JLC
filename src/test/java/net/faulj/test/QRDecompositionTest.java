package net.faulj.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import org.junit.Test;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

public class QRDecompositionTest {

    private static final double TOLERANCE = 1e-9;

    /**
     * Tests the Householder QR Decomposition implementation.
     * Verifies:
     * 1. Reconstruction: Q * R = A
     * 2. Orthogonality: Q^T * Q = I
     * 3. Upper Triangularity: R is upper triangular
     */
    @Test
    public void testHouseholderQR() {
        int n = 5;
        Matrix A = Matrix.randomMatrix(n, n);

        // Perform Decomposition
        QRResult result = HouseholderQR.decompose(A);
        Matrix Q = result.getQ();
        Matrix R = result.getR();

        // 1. Check Reconstruction (A = QR)
        Matrix reconstructed = Q.multiply(R);
        assertMatrixEquals("Reconstruction A = QR failed", A, reconstructed, TOLERANCE);

        // 2. Check Orthogonality of Q (Q^T * Q = I)
        Matrix Q_T_Q = Q.transpose().multiply(Q);
        Matrix Identity = Matrix.Identity(n);
        assertMatrixEquals("Orthogonality Q^T * Q = I failed", Identity, Q_T_Q, TOLERANCE);

        // 3. Check R is Upper Triangular (entries below diagonal should be 0)
        assertTrue("R should be upper triangular", isUpperTriangular(R, TOLERANCE));

        System.out.println("HouseholderQR Test Passed");
    }

    @Test
    public void testHouseholderQROnIdentity() {
        int n = 4;
        Matrix I = Matrix.Identity(n);
        QRResult result = HouseholderQR.decompose(I);

        assertMatrixEquals("Q of Identity should be Identity", I, result.getQ(), TOLERANCE);
        assertMatrixEquals("R of Identity should be Identity", I, result.getR(), TOLERANCE);
    }

    // --- Helper Methods ---

    private void assertMatrixEquals(String message, Matrix expected, Matrix actual, double tol) {
        assertEquals(message + ": Row counts differ", expected.getRowCount(), actual.getRowCount());
        assertEquals(message + ": Column counts differ", expected.getColumnCount(), actual.getColumnCount());

        for (int i = 0; i < expected.getRowCount(); i++) {
            for (int j = 0; j < expected.getColumnCount(); j++) {
                double val1 = expected.get(i, j);
                double val2 = actual.get(i, j);
                if (Math.abs(val1 - val2) > tol) {
                    fail(String.format("%s: Mismatch at (%d, %d). Expected %.8f but got %.8f",
                            message, i, j, val1, val2));
                }
            }
        }
    }

    private boolean isUpperTriangular(Matrix M, double tol) {
        for (int c = 0; c < M.getColumnCount(); c++) {
            for (int r = c + 1; r < M.getRowCount(); r++) {
                if (Math.abs(M.get(r, c)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    private boolean isHessenberg(Matrix M, double tol) {
        for (int c = 0; c < M.getColumnCount(); c++) {
            for (int r = c + 2; r < M.getRowCount(); r++) {
                if (Math.abs(M.get(r, c)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }
}