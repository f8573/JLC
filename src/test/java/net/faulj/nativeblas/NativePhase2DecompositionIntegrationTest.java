package net.faulj.nativeblas;

import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.CholeskyResult;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.decomposition.result.LUResult;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class NativePhase2DecompositionIntegrationTest {

    @After
    public void cleanup() {
        System.clearProperty("jlc.backend");
        System.clearProperty("jlc.native.hessenberg.provider");
        System.clearProperty("jlc.algorithm.lu.backend");
        System.clearProperty("jlc.algorithm.qr.backend");
        System.clearProperty("jlc.algorithm.cholesky.backend");
        System.clearProperty("jlc.algorithm.hessenberg.backend");
        System.clearProperty("jlc.algorithm.calibration.path");
        System.clearProperty("net.faulj.decomposition.lu.blockThreshold");
        System.clearProperty("net.faulj.decomposition.lu.blockSize");
        System.clearProperty("net.faulj.decomposition.hessenberg.blockSize");
        System.clearProperty("net.faulj.eigen.qr.BlockedHessenbergQR.blockSize");
        System.clearProperty("net.faulj.eigen.qr.blockSize");
        BackendRegistry.resetForTests();
    }

    @Test
    public void blockedLuRemainsAccurateWithNativeBackend() {
        assumeNativeBackendReady();
        System.setProperty("net.faulj.decomposition.lu.blockThreshold", "8");
        System.setProperty("net.faulj.decomposition.lu.blockSize", "8");

        Matrix a = randomDiagonallyDominant(48, 41L);
        LUResult result = new LUDecomposition().decompose(a);

        assertFalse(result.isSingular());
        assertTrue("LU residual too large: " + result.residualNorm(), result.residualNorm() < 1e-10);
    }

    @Test
    public void blockedHessenbergRemainsAccurateWithNativeBackend() {
        assumeNativeBackendReady();

        Matrix a = randomMatrix(40, 73L);
        HessenbergResult result = HessenbergReduction.decompose(a);

        assertTrue("Hessenberg residual too large: " + result.residualNorm(), result.residualNorm() < 1e-8);

        Matrix h = result.getH();
        for (int row = 2; row < h.getRowCount(); row++) {
            for (int col = 0; col < row - 1; col++) {
                assertEquals("Unexpected fill below subdiagonal", 0.0, h.get(row, col), 1e-8);
            }
        }
    }

    @Test
    public void publicLapackSelectionNoLongerRoutesHessenberg() {
        assumeNativeBackendReady();

        Matrix a = randomMatrix(72, 91L);

        System.setProperty("jlc.algorithm.hessenberg.backend", "java");
        HessenbergResult javaResult = HessenbergReduction.decompose(a);

        System.setProperty("jlc.native.hessenberg.provider", "vendor");
        HessenbergResult ignoredLegacyProviderResult = HessenbergReduction.decompose(a);

        assertTrue("Hessenberg residual too large: " + ignoredLegacyProviderResult.residualNorm(),
            ignoredLegacyProviderResult.residualNorm() < 1e-8);
        assertTrue("Hessenberg H mismatch too large", relativeDifference(javaResult.getH(), ignoredLegacyProviderResult.getH()) < 1e-10);
        assertTrue("Hessenberg Q mismatch too large", relativeDifference(javaResult.getQ(), ignoredLegacyProviderResult.getQ()) < 1e-10);
    }

    @Test
    public void nativeBuiltinLuMatchesJava() {
        assumeNativeBackendReady();

        Matrix a = randomDiagonallyDominant(72, 121L);

        System.setProperty("jlc.algorithm.lu.backend", "java");
        LUResult javaResult = new LUDecomposition().decompose(a);

        System.setProperty("jlc.algorithm.lu.backend", "cpp");
        LUResult nativeResult = new LUDecomposition().decompose(a);

        assertFalse(nativeResult.isSingular());
        assertTrue("Native LU residual too large: " + nativeResult.residualNorm(), nativeResult.residualNorm() < 1e-10);
        assertTrue("LU L mismatch too large", relativeDifference(javaResult.getL(), nativeResult.getL()) < 1e-10);
        assertTrue("LU U mismatch too large", relativeDifference(javaResult.getU(), nativeResult.getU()) < 1e-10);
        for (int i = 0; i < javaResult.getP().size(); i++) {
            assertEquals("LU permutation mismatch at " + i, javaResult.getP().get(i), nativeResult.getP().get(i));
        }
    }

    @Test
    public void nativeBuiltinQrMatchesJava() {
        assumeNativeBackendReady();

        Matrix a = randomRectangularMatrix(96, 64, 157L);

        System.setProperty("jlc.algorithm.qr.backend", "java");
        QRResult javaResult = HouseholderQR.decompose(a);

        System.setProperty("jlc.algorithm.qr.backend", "cpp");
        QRResult nativeResult = HouseholderQR.decompose(a);

        assertTrue("Native QR residual too large: " + nativeResult.residualNorm(), nativeResult.residualNorm() < 1e-10);
        assertTrue("Native Q orthogonality too large: " + nativeResult.verifyOrthogonality(nativeResult.getQ())[0],
            nativeResult.verifyOrthogonality(nativeResult.getQ())[0] < 1e-10);
        assertTrue("QR reconstruction mismatch too large", relativeDifference(javaResult.reconstruct(), nativeResult.reconstruct()) < 1e-10);
    }

    @Test
    public void nativeBuiltinQrThinAndFullSupportCoreShapes() {
        assumeNativeBackendReady();
        System.setProperty("jlc.algorithm.qr.backend", "cpp");

        validateQr(HouseholderQR.decomposeThin(randomRectangularMatrix(96, 48, 301L)), 96, 48, 48, 48);
        validateQr(HouseholderQR.decompose(randomRectangularMatrix(96, 48, 302L)), 96, 48, 96, 96);
        validateQr(HouseholderQR.decomposeThin(randomRectangularMatrix(48, 96, 303L)), 48, 96, 48, 48);
        validateQr(HouseholderQR.decompose(randomRectangularMatrix(64, 64, 304L)), 64, 64, 64, 64);
    }

    @Test
    public void nativeBuiltinQrHandlesNearSingularAndIllConditionedInputs() {
        assumeNativeBackendReady();
        System.setProperty("jlc.algorithm.qr.backend", "cpp");

        validateQr(HouseholderQR.decompose(nearSingularMatrix(64)), 64, 64, 64, 64);
        validateQr(HouseholderQR.decomposeThin(illConditionedTallMatrix(96, 48)), 96, 48, 48, 48);
    }

    @Test
    public void qrFallsBackToJavaWhenNativeBackendIsDisabled() {
        System.setProperty("jlc.backend", "java");
        System.setProperty("jlc.algorithm.qr.backend", "cpp");
        BackendRegistry.resetForTests();

        QRResult result = HouseholderQR.decomposeThin(randomRectangularMatrix(40, 24, 401L));

        validateQr(result, 40, 24, 24, 24);
        assertEquals("java", BackendRegistry.snapshot().activeBackend());
    }

    @Test
    public void nativeBuiltinCholeskyMatchesJava() {
        assumeNativeBackendReady();

        Matrix a = randomPositiveDefinite(72, 211L);

        System.setProperty("jlc.algorithm.cholesky.backend", "java");
        CholeskyResult javaResult = new CholeskyDecomposition().decompose(a);

        System.setProperty("jlc.algorithm.cholesky.backend", "cpp");
        CholeskyResult nativeResult = new CholeskyDecomposition().decompose(a);

        assertTrue("Native Cholesky residual too large: " + nativeResult.residualNorm(), nativeResult.residualNorm() < 1e-10);
        assertTrue("Cholesky L mismatch too large", relativeDifference(javaResult.getL(), nativeResult.getL()) < 1e-10);
    }

    private static void assumeNativeBackendReady() {
        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        Assume.assumeTrue("Native backend unavailable: " + snapshot.nativeContext().getMessage(),
            "native".equals(snapshot.activeBackend()));
    }

    private static Matrix randomMatrix(int n, long seed) {
        Random random = new Random(seed);
        double[] data = new double[n * n];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return Matrix.wrap(data, n, n);
    }

    private static Matrix randomRectangularMatrix(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return Matrix.wrap(data, rows, cols);
    }

    private static Matrix randomDiagonallyDominant(int n, long seed) {
        Random random = new Random(seed);
        double[] data = new double[n * n];
        for (int row = 0; row < n; row++) {
            double rowSum = 0.0;
            int rowOffset = row * n;
            for (int col = 0; col < n; col++) {
                if (row == col) {
                    continue;
                }
                double value = random.nextDouble() * 0.4 - 0.2;
                data[rowOffset + col] = value;
                rowSum += Math.abs(value);
            }
            data[rowOffset + row] = rowSum + 1.0 + random.nextDouble();
        }
        return Matrix.wrap(data, n, n);
    }

    private static Matrix randomPositiveDefinite(int n, long seed) {
        Matrix base = randomMatrix(n, seed);
        Matrix spd = base.transpose().multiply(base);
        double[] data = spd.getRawData().clone();
        for (int i = 0; i < n; i++) {
            data[i * n + i] += n;
        }
        return Matrix.wrap(data, n, n);
    }

    private static Matrix nearSingularMatrix(int n) {
        double[] data = randomMatrix(n, 501L).getRawData().clone();
        for (int col = 0; col < n; col++) {
            data[(n - 1) * n + col] = data[(n - 2) * n + col] * (1.0 + 1e-12);
        }
        return Matrix.wrap(data, n, n);
    }

    private static Matrix illConditionedTallMatrix(int rows, int cols) {
        double[] data = randomRectangularMatrix(rows, cols, 601L).getRawData().clone();
        for (int col = 0; col < cols; col++) {
            double scale = Math.pow(10.0, -8.0 * col / Math.max(1, cols - 1));
            for (int row = 0; row < rows; row++) {
                data[row * cols + col] *= scale;
            }
        }
        return Matrix.wrap(data, rows, cols);
    }

    private static void validateQr(QRResult result, int rows, int cols, int qCols, int rRows) {
        assertEquals("Q row count", rows, result.getQ().getRowCount());
        assertEquals("Q column count", qCols, result.getQ().getColumnCount());
        assertEquals("R row count", rRows, result.getR().getRowCount());
        assertEquals("R column count", cols, result.getR().getColumnCount());
        assertTrue("QR residual too large: " + result.residualNorm(), result.residualNorm() < 1e-8);
        assertTrue("Q orthogonality too large: " + result.verifyOrthogonality(result.getQ())[0],
            result.verifyOrthogonality(result.getQ())[0] < 1e-8);
    }

    private static double relativeDifference(Matrix a, Matrix b) {
        double diff = 0.0;
        double ref = 0.0;
        for (int row = 0; row < a.getRowCount(); row++) {
            for (int col = 0; col < a.getColumnCount(); col++) {
                double av = a.get(row, col);
                double bv = b.get(row, col);
                diff = Math.fma(av - bv, av - bv, diff);
                ref = Math.fma(av, av, ref);
            }
        }
        return Math.sqrt(diff) / Math.max(1.0, Math.sqrt(ref));
    }
}
