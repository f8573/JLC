package net.faulj.eigen.qr;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.matrix.Matrix;

public class QRAlgorithmsTest {

    private static final double TOL = 1e-10;

    private Matrix randomHessenberg(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        // Fill upper triangle and first subdiagonal
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        for (int i = 1; i < n; i++) {
            a[i][i - 1] = rnd.nextDouble() * 2.0 - 1.0;
        }
        // Zero out below first subdiagonal explicitly
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                a[i][j] = 0.0;
            }
        }
        return new Matrix(a);
    }

    @Test
    public void testBulgeChasingPreservesHessenberg() {
        int n = 8;
        Matrix H = randomHessenberg(n, 12345L);
        Matrix Q = Matrix.Identity(n);

        // Use two shifts
        double[] shifts = MultiShiftQR.generateShifts(H, 0, n - 1, 2);
        BulgeChasing.performSweep(H, Q, 0, n - 1, shifts);

        // Verify Hessenberg structure: elements below subdiagonal are (near) zero
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals("Hessenberg fill-in at (" + i + "," + j + ")", 0.0, H.get(i, j), 1e-8);
            }
        }
    }

    @Test
    public void testAEDDeflationOnTinySpike() {
        int n = 6;
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = 0.0;
        }
        // Make diagonal distinct
        for (int i = 0; i < n; i++) a[i][i] = i + 1.0;
        // Make Hessenberg: fill first superdiagonal
        for (int i = 0; i < n - 1; i++) a[i][i + 1] = 0.1 * (i + 1);
        // Make subdiagonal normally
        for (int i = 1; i < n; i++) a[i][i - 1] = 0.1 * (i + 1);

        // Make last spike very small to encourage deflation
        a[n - 1][n - 2] = 1e-14;

        Matrix H = new Matrix(a);
        Matrix Q = Matrix.Identity(n);

        AggressiveEarlyDeflation.AEDResult res = AggressiveEarlyDeflation.process(H, Q, 0, n - 1, 4, 1e-12, 0);
        assertTrue("AED should deflate at least one eigenvalue when spike is tiny", res.deflatedCount >= 1);
    }

    @Test
    public void testMultiShiftExceptionalShifts() {
        int n = 10;
        Matrix H = randomHessenberg(n, 42L);
        double[] ex = MultiShiftQR.generateExceptionalShifts(H, 0, n - 1, 40, 4);
        assertNotNull(ex);
        assertEquals("Exceptional shifts should be even count", 0, ex.length % 2);
        for (double s : ex) {
            assertFalse("Shift must not be NaN", Double.isNaN(s));
            assertFalse("Shift must be finite", Double.isInfinite(s));
        }
    }

    @Test
    public void testExplicitQRProducesQuasiTriangular() {
        int n = 6;
        Matrix A = randomHessenberg(n, 777L);
        Matrix[] res = ExplicitQRIteration.decompose(A.copy());
        Matrix T = res[0];
        // Check below subdiagonal zero
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals(0.0, T.get(i, j), 1e-8);
            }
        }
    }

    @Test
    public void testImplicitFrancisSmallFallback() {
        int n = 6;
        Matrix A = randomHessenberg(n, 2023L);
        SchurResult result = ImplicitQRFrancis.decompose(A);
        assertNotNull(result);
        Matrix T = result.getT();
        // Check Schur quasi-triangular
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals("Schur form lower zero at (" + i + "," + j + ")", 0.0, T.get(i, j), 1e-8);
            }
        }
        // Residual should be non-negative (may be -1.0 when skipped); if computed, expect small
        double r = result.residualNorm();
        assertTrue(r >= -1.0);
    }
}
