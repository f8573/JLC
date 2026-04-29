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
        System.clearProperty("jlc.native.lu.provider");
        System.clearProperty("jlc.native.lu.minSize");
        System.clearProperty("jlc.native.qr.provider");
        System.clearProperty("jlc.native.qr.minSize");
        System.clearProperty("jlc.native.cholesky.provider");
        System.clearProperty("jlc.native.cholesky.minSize");
        System.clearProperty("jlc.native.hessenberg.provider");
        System.clearProperty("jlc.native.hessenberg.minSize");
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
    public void vendorLapackHessenbergMatchesJavaWhenAvailable() {
        assumeNativeBackendReady();
        Assume.assumeTrue("Vendor LAPACK unavailable", NativeBindings.nativeVendorLapackAvailable());

        Matrix a = randomMatrix(72, 91L);

        System.setProperty("jlc.native.hessenberg.provider", "java");
        HessenbergResult javaResult = HessenbergReduction.decompose(a);

        System.setProperty("jlc.native.hessenberg.provider", "vendor");
        System.setProperty("jlc.native.hessenberg.minSize", "1");
        HessenbergResult vendorResult = HessenbergReduction.decompose(a);

        assertTrue("Vendor Hessenberg residual too large: " + vendorResult.residualNorm(), vendorResult.residualNorm() < 1e-8);
        assertTrue("Hessenberg H mismatch too large", relativeDifference(javaResult.getH(), vendorResult.getH()) < 1e-10);
        assertTrue("Hessenberg Q mismatch too large", relativeDifference(javaResult.getQ(), vendorResult.getQ()) < 1e-10);
    }

    @Test
    public void nativeBuiltinLuMatchesJava() {
        assumeNativeBackendReady();

        Matrix a = randomDiagonallyDominant(72, 121L);

        System.setProperty("jlc.native.lu.provider", "java");
        LUResult javaResult = new LUDecomposition().decompose(a);

        System.setProperty("jlc.native.lu.provider", "builtin");
        System.setProperty("jlc.native.lu.minSize", "1");
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

        System.setProperty("jlc.native.qr.provider", "java");
        QRResult javaResult = HouseholderQR.decompose(a);

        System.setProperty("jlc.native.qr.provider", "builtin");
        System.setProperty("jlc.native.qr.minSize", "1");
        QRResult nativeResult = HouseholderQR.decompose(a);

        assertTrue("Native QR residual too large: " + nativeResult.residualNorm(), nativeResult.residualNorm() < 1e-10);
        assertTrue("Native Q orthogonality too large: " + nativeResult.verifyOrthogonality(nativeResult.getQ())[0],
            nativeResult.verifyOrthogonality(nativeResult.getQ())[0] < 1e-10);
        assertTrue("QR reconstruction mismatch too large", relativeDifference(javaResult.reconstruct(), nativeResult.reconstruct()) < 1e-10);
    }

    @Test
    public void nativeBuiltinCholeskyMatchesJava() {
        assumeNativeBackendReady();

        Matrix a = randomPositiveDefinite(72, 211L);

        System.setProperty("jlc.native.cholesky.provider", "java");
        CholeskyResult javaResult = new CholeskyDecomposition().decompose(a);

        System.setProperty("jlc.native.cholesky.provider", "builtin");
        System.setProperty("jlc.native.cholesky.minSize", "1");
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
