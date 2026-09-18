package net.faulj.decomposition.qr.caqr;

import net.faulj.decomposition.qr.QRFactory;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class CommunicationAvoidingQRAccuracyTest {

    private String previousStrategy;

    @Before
    public void saveStrategy() {
        previousStrategy = System.getProperty("la.qr.strategy");
    }

    @After
    public void restoreStrategy() {
        if (previousStrategy == null) System.clearProperty("la.qr.strategy");
        else System.setProperty("la.qr.strategy", previousStrategy);
    }

    private Matrix randomMatrix(int rows, int cols, long seed) {
        Random rng = new Random(seed);
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rng.nextGaussian();
            }
        }
        return new Matrix(data);
    }

    @Test
    public void caqrMatchesHouseholder_small() {
        int m = 256;
        int n = 128;
        Matrix A = randomMatrix(m, n, 42L);

        // Run CAQR
        System.setProperty("la.qr.strategy", "CAQR");
        QRResult caqr = QRFactory.decompose(A, true);

        // Run Householder
        System.setProperty("la.qr.strategy", "HOUSEHOLDER");
        QRResult hh = QRFactory.decompose(A, true);

        // Residuals
        double resCA = caqr.residualNorm();
        double resHH = hh.residualNorm();

        // Orthogonality errors
        double orthoCA = caqr.verifyOrthogonality(caqr.getQ())[0];
        double orthoHH = hh.verifyOrthogonality(hh.getQ())[0];

        // CAQR should be comparable to Householder within reasonable tolerance
        double tol = 1e-9;
        assertTrue("CAQR residual too large: " + resCA, resCA <= 1e-6 || resCA / resHH < 1e1);
        assertTrue("CAQR orthogonality too large: " + orthoCA, orthoCA <= 1e-6 || orthoCA / orthoHH < 1e1);
    }

    @Test
    public void caqrDeterministic_reproducible() {
        int m = 200;
        int n = 100;
        Matrix A1 = randomMatrix(m, n, 123L);
        Matrix A2 = randomMatrix(m, n, 123L);

        System.setProperty("la.qr.strategy", "CAQR");
        QRResult r1 = QRFactory.decompose(A1, true);
        QRResult r2 = QRFactory.decompose(A2, true);

        double res1 = r1.residualNorm();
        double res2 = r2.residualNorm();

        assertTrue(Math.abs(res1 - res2) < 1e-12);
    }
}
