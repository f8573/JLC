package net.faulj.stress;

import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.decomposition.result.QRResult;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;
import net.faulj.scalar.Complex;
import org.junit.Assume;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class StressAccuracyTest {

    private static final int TRIALS = 2000;
    private static final int MIN_SIZE = 2;
    private static final int MAX_SIZE = 8;
    private static final int[] LARGE_SIZES = {50, 100, 200, 500, 1000};
    private static final boolean RUN_LARGE = Boolean.getBoolean("stressLarge");
    private static final double QR_RESIDUAL_LIMIT = 1e-2;
    private static final double HESS_RESIDUAL_LIMIT = 1e-2;
    private static final double SCHUR_RESIDUAL_LIMIT = 5e-2;
    private static final double LARGE_QR_RESIDUAL_LIMIT = 5e-2;
    private static final double LARGE_HESS_RESIDUAL_LIMIT = 5e-2;
    private static final double TRACE_TOL = 1e-5;
    // How often to print progress during the main stress loop (in trials)
    private static final int PROGRESS_INTERVAL = 100;

    @Test
    public void stressDecompositionAndEigenAccuracy() {
        Random rnd = new Random(1234567L);

        for (int t = 0; t < TRIALS; t++) {
            int n = MIN_SIZE + (t % (MAX_SIZE - MIN_SIZE + 1));
            Matrix A = randomMatrix(rnd, n);
            QRResult qr = HouseholderQR.decompose(A);
            double qrRes = qr.residualNorm();
            assertTrue("QR residual too large at size " + n + " trial " + t + ": " + qrRes,
                qrRes < QR_RESIDUAL_LIMIT);

            HessenbergResult hess = HessenbergReduction.decompose(A);
            double hessRes = hess.residualNorm();
            assertTrue("Hessenberg residual too large at size " + n + " trial " + t + ": " + hessRes,
                hessRes < HESS_RESIDUAL_LIMIT);

            SchurResult schur = RealSchurDecomposition.decompose(A);
            double schurRes = schur.residualNorm();
            assertTrue("Schur residual too large at size " + n + " trial " + t + ": " + schurRes,
                schurRes < SCHUR_RESIDUAL_LIMIT);

            if (t % PROGRESS_INTERVAL == 0) {
            System.out.printf("[stress] trial=%d size=%d QR=%.3e HESS=%.3e SCHUR=%.3e trace=%.6f%n",
                t, n, qrRes, hessRes, schurRes, A.trace());
            }

            Complex[] eigenvalues = schur.getEigenvalues();
            double sumReal = 0.0;
            double sumImag = 0.0;
            for (Complex eigenvalue : eigenvalues) {
                sumReal += eigenvalue.real;
                sumImag += eigenvalue.imag;
            }
                if (t % (PROGRESS_INTERVAL * 5) == 0) {
                System.out.printf("[eigs] trial=%d size=%d sumReal=%.6f sumImag=%.6f%n",
                    t, n, sumReal, sumImag);
                }

                assertEquals("Trace mismatch at size " + n + " trial " + t,
                    A.trace(), sumReal, TRACE_TOL);
                assertEquals("Imaginary sum should be near zero at size " + n + " trial " + t,
                    0.0, sumImag, TRACE_TOL);
        }
    }

    @Test
    public void stressLargeMatrixDecompositions() {
        //Assume.assumeTrue("Run with -DstressLarge=true to enable large-matrix stress test", RUN_LARGE);
        Random rnd = new Random(7654321L);

        for (int size : LARGE_SIZES) {
            System.out.printf("[large] starting size=%d%n", size);
            Matrix A = randomMatrix(rnd, size);

            QRResult qr = HouseholderQR.decompose(A);
            double qrRes = qr.residualNorm();
            System.out.printf("[large] size=%d QR=%.3e%n", size, qrRes);
            assertTrue("QR residual too large at size " + size + ": " + qrRes,
                qrRes < LARGE_QR_RESIDUAL_LIMIT);

            HessenbergResult hess = HessenbergReduction.decompose(A);
            double hessRes = hess.residualNorm();
            System.out.printf("[large] size=%d HESS=%.3e%n", size, hessRes);
            assertTrue("Hessenberg residual too large at size " + size + ": " + hessRes,
                hessRes < LARGE_HESS_RESIDUAL_LIMIT);
        }
    }

    private static Matrix randomMatrix(Random rnd, int n) {
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        return new Matrix(data);
    }
}
