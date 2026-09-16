package net.faulj.nativeblas;

import net.faulj.decomposition.result.SVDResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import net.faulj.matrix.OffHeapMatrix;
import net.faulj.svd.SVDecomposition;
import net.faulj.svd.ThinSVD;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Adversarial and contract coverage for the native bidiagonal-QR SVD path.
 */
public class NativeSvdProductionTest {
    private static final double EPS = Math.ulp(1.0);

    @Before
    public void useNativeBidiagonalQr() {
        System.setProperty("jlc.backend", "native");
        System.setProperty("jlc.algorithm.svd.backend", "cpp");
        System.setProperty("jlc.algorithm.svd.nativeAlgorithm", "bidiag_qr");
        BackendRegistry.resetForTests();

        Assume.assumeTrue("Native library path not configured",
            System.getProperty("jlc.native.lib.path") != null);
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        Assume.assumeTrue("Native backend is unavailable: " + snapshot.nativeContext().getMessage(),
            snapshot.nativeContext().isAvailable());
    }

    @After
    public void restoreBackendProperties() {
        System.clearProperty("jlc.backend");
        System.clearProperty("jlc.algorithm.svd.backend");
        System.clearProperty("jlc.algorithm.svd.nativeAlgorithm");
        System.clearProperty("jlc.algorithm.svd.jacobiMaxDimension");
        BackendRegistry.resetForTests();
    }

    @Test
    public void zeroMatrix() {
        validate("zero", Matrix.zero(7, 5), false);
    }

    @Test
    public void identity() {
        validate("identity", Matrix.Identity(8), false);
    }

    @Test
    public void descendingDiagonal() {
        SVDResult result = validate("descending-diagonal", diagonal(9.0, 5.0, 2.0, 0.25), false);
        assertSingularValues(result, new double[] {9.0, 5.0, 2.0, 0.25});
    }

    @Test
    public void ascendingDiagonalIsSorted() {
        SVDResult result = validate("ascending-diagonal", diagonal(0.25, 2.0, 5.0, 9.0), false);
        assertSingularValues(result, new double[] {9.0, 5.0, 2.0, 0.25});
    }

    @Test
    public void denseRankOneAdversary() {
        int rows = 96;
        int cols = 73;
        double[] left = new double[rows];
        double[] right = new double[cols];
        for (int i = 0; i < rows; i++) {
            left[i] = (i & 1) == 0 ? 1.0 + i / 31.0 : -0.75 - i / 37.0;
        }
        for (int j = 0; j < cols; j++) {
            right[j] = (j % 3) == 0 ? 1.0 - j / 43.0 : -0.5 - j / 29.0;
        }
        validate("dense-rank-one-adversary", outerProduct(left, right), false);
    }

    @Test
    public void exactRankDeficiency() {
        double[][] data = new double[18][9];
        for (int row = 0; row < data.length; row++) {
            double x = row + 1.0;
            for (int col = 0; col < data[row].length; col++) {
                double y = (col + 1.0) * (col + 1.0);
                data[row][col] = x * y + (row % 2 == 0 ? 0.25 : -0.5) * (col + 1.0);
            }
        }
        Matrix a = new Matrix(data);
        // The construction has only two independent column patterns.
        SVDResult result = validate("exact-rank-deficient", a, false);
        double[] sigma = result.getSingularValues();
        assertTrue("Expected a small trailing singular value: " + Arrays.toString(sigma),
            sigma[sigma.length - 1] <= 1.0e-12 * sigma[0]);
    }

    @Test
    public void repeatedSingularValues() {
        Matrix a = rotatedDiagonal(new double[] {8.0, 8.0, 3.0, 3.0, 1.0}, 0.37, -0.61);
        SVDResult result = validate("repeated", a, false);
        assertSingularValues(result, new double[] {8.0, 8.0, 3.0, 3.0, 1.0});
    }

    @Test
    public void tightlyClusteredSingularValues() {
        Matrix a = rotatedDiagonal(new double[] {8.0, 8.0 - 1.0e-11, 8.0 - 2.0e-11, 1.0, 1.0 - 1.0e-12},
            0.19, -0.47);
        SVDResult result = validate("clustered", a, false);
        double[] sigma = result.getSingularValues();
        assertEquals(8.0, sigma[0], 1.0e-10);
        assertEquals(8.0 - 1.0e-11, sigma[1], 1.0e-10);
        assertEquals(1.0, sigma[3], 1.0e-10);
    }

    @Test
    public void oneVerySmallSingularValue() {
        Matrix a = rotatedDiagonal(new double[] {4.0, 0.5, 1.0e-12, 2.0e-13}, 0.31, -0.52);
        double[] directU = new double[16];
        double[] directSigma = new double[4];
        double[] directV = new double[16];
        NativeBindings.nativeSvdDecomposeWithAlgorithm(
            a.getRawData(), 4, 4, directU, directSigma, directV, 2);
        assertTrue("Direct native SVD lost the small singular value: "
                + Arrays.toString(directSigma), directSigma[3] > 0.0);
        SVDResult result = validate("one-small", a, false);
        assertTrue("Small singular value was lost: " + Arrays.toString(result.getSingularValues()),
            result.getSingularValues()[3] > 0.0);
    }

    @Test
    public void largeScaleDisparity() {
        Matrix a = rotatedDiagonal(new double[] {1.0e12, 3.0, 1.0e-3, 2.0e-3}, 0.41, -0.29);
        SVDResult result = validate("scale-disparity", a, false);
        assertEquals(1.0e12, result.getSingularValues()[0], 1.0e-3);
        assertTrue(result.getSingularValues()[3] > 0.0);
    }

    @Test
    public void tallMatrix() {
        validate("tall", randomMatrix(37, 11, 1901), false);
        validate("tall-thin", randomMatrix(37, 11, 1901), true);
    }

    @Test
    public void wideMatrix() {
        validate("wide", randomMatrix(11, 37, 1902), false);
        validate("wide-thin", randomMatrix(11, 37, 1902), true);
    }

    @Test
    public void nearlyDependentColumns() {
        double[][] data = new double[40][12];
        for (int row = 0; row < data.length; row++) {
            double base = Math.sin(0.17 * row);
            for (int col = 0; col < data[row].length; col++) {
                data[row][col] = base * (1.0 + 0.01 * col)
                    + 1.0e-10 * Math.cos((row + 1.0) * (col + 2.0));
            }
        }
        validate("nearly-dependent-columns", new Matrix(data), false);
    }

    @Test
    public void nearlyDependentRows() {
        double[][] data = new double[12][40];
        for (int row = 0; row < data.length; row++) {
            for (int col = 0; col < data[row].length; col++) {
                data[row][col] = Math.cos(0.11 * col) * (1.0 + 0.01 * row)
                    + 1.0e-10 * Math.sin((row + 2.0) * (col + 1.0));
            }
        }
        validate("nearly-dependent-rows", new Matrix(data), false);
    }

    @Test
    public void verySmallEntries() {
        validate("very-small-entries", scale(randomMatrix(14, 9, 1903), 1.0e-100), false);
    }

    @Test
    public void veryLargeFiniteEntries() {
        validate("very-large-entries", scale(randomMatrix(14, 9, 1904), 1.0e100), false);
    }

    @Test
    public void offHeapRowMajorInput() {
        assertOffHeapInput(OffHeapMatrix.Order.ROW_MAJOR, "offheap-row-major");
    }

    @Test
    public void offHeapColumnMajorInput() {
        assertOffHeapInput(OffHeapMatrix.Order.COL_MAJOR, "offheap-column-major");
    }

    @Test
    public void fullAndThinHaveCompatibleContracts() {
        Matrix a = randomMatrix(19, 7, 1905);
        SVDResult full = validate("full-contract", a, false);
        SVDResult thin = validate("thin-contract", a, true);
        for (int i = 0; i < full.getSingularValues().length; i++) {
            assertEquals(full.getSingularValues()[i], thin.getSingularValues()[i],
                1.0e-11 * Math.max(1.0, full.getSingularValues()[i]));
        }
    }

    @Test
    public void explicitJacobiOverrideStillWorks() {
        Matrix a = randomMatrix(24, 17, 1906);
        System.setProperty("jlc.algorithm.svd.nativeAlgorithm", "jacobi");
        BackendRegistry.resetForTests();
        validate("explicit-jacobi", a, false);
        System.setProperty("jlc.algorithm.svd.nativeAlgorithm", "bidiag_qr");
        BackendRegistry.resetForTests();
        validate("explicit-bidiag-qr", a, false);
    }

    @Test
    public void largeSquareUsesProductionPath() {
        Matrix a = randomMatrix(256, 256, 1907);
        System.clearProperty("jlc.algorithm.svd.nativeAlgorithm");
        BackendRegistry.resetForTests();
        long start = System.nanoTime();
        validate("large-square", a, true);
        long elapsedMillis = (System.nanoTime() - start) / 1_000_000L;
        assertTrue("Bidiagonal QR path did not converge in a reasonable time: " + elapsedMillis + " ms",
            elapsedMillis < 5000L);
    }

    private static void assertOffHeapInput(OffHeapMatrix.Order order, String name) {
        Matrix source = randomMatrix(17, 8, 1908 + order.ordinal());
        try (OffHeapMatrix a = new OffHeapMatrix(17, 8, order, 64)) {
            for (int row = 0; row < a.getRowCount(); row++) {
                for (int col = 0; col < a.getColumnCount(); col++) {
                    a.setOffHeap(row, col, source.get(row, col));
                }
            }
            SVDResult result = validate(name, a, true);
            assertTrue(name + " reconstruction changed the input values",
                MatrixUtils.relativeError(source, result.reconstruct()) < 1.0e-11);
        }
    }

    private static SVDResult validate(String name, Matrix a, boolean thin) {
        SVDResult result = thin ? new ThinSVD().decompose(a) : new SVDecomposition().decompose(a);
        int m = a.getRowCount();
        int n = a.getColumnCount();
        int rank = Math.min(m, n);
        assertEquals(name + " singular-value count", rank, result.getSingularValues().length);
        assertEquals(name + " U rows", m, result.getU().getRowCount());
        assertEquals(name + " V rows", n, result.getV().getRowCount());
        assertEquals(name + " U columns", thin ? rank : m, result.getU().getColumnCount());
        assertEquals(name + " V columns", thin ? rank : n, result.getV().getColumnCount());

        double tolerance = 8192.0 * EPS * Math.max(1, Math.max(m, n));
        double reconstruction = MatrixUtils.relativeError(a, result.reconstruct());
        double uOrthogonality = MatrixUtils.orthogonalityError(result.getU());
        double vOrthogonality = MatrixUtils.orthogonalityError(result.getV());
        assertTrue(name + " reconstruction=" + reconstruction + " tolerance=" + tolerance,
            reconstruction <= tolerance * 4.0);
        assertTrue(name + " U orthogonality=" + uOrthogonality + " tolerance=" + tolerance,
            uOrthogonality <= tolerance * 4.0);
        assertTrue(name + " V orthogonality=" + vOrthogonality + " tolerance=" + tolerance,
            vOrthogonality <= tolerance * 4.0);

        double previous = Double.POSITIVE_INFINITY;
        for (double sigma : result.getSingularValues()) {
            assertTrue(name + " non-finite singular value", Double.isFinite(sigma));
            assertTrue(name + " negative singular value", sigma >= 0.0);
            assertTrue(name + " singular values are not descending", previous >= sigma);
            previous = sigma;
        }
        assertFinite(name + " U", result.getU().getRawData());
        assertFinite(name + " V", result.getV().getRawData());
        return result;
    }

    private static void assertSingularValues(SVDResult result, double[] expected) {
        assertEquals(expected.length, result.getSingularValues().length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result.getSingularValues()[i],
                4096.0 * EPS * Math.max(1.0, Math.abs(expected[i])));
        }
    }

    private static void assertFinite(String name, double[] values) {
        for (double value : values) {
            assertTrue(name + " contains non-finite data", Double.isFinite(value));
        }
    }

    private static Matrix diagonal(double... values) {
        double[][] data = new double[values.length][values.length];
        for (int i = 0; i < values.length; i++) {
            data[i][i] = values[i];
        }
        return new Matrix(data);
    }

    private static Matrix rotatedDiagonal(double[] values, double leftAngle, double rightAngle) {
        Matrix d = diagonal(values);
        Matrix left = rotation(leftAngle, 0, 1, values.length)
            .multiply(rotation(-0.23, 2, values.length - 1, values.length));
        Matrix right = rotation(rightAngle, 1, 2, values.length)
            .multiply(rotation(0.17, 0, values.length - 1, values.length));
        return left.multiply(d).multiply(right.transpose());
    }

    private static Matrix rotation(double angle, int first, int second, int size) {
        Matrix result = Matrix.Identity(size);
        double c = Math.cos(angle);
        double s = Math.sin(angle);
        result.set(first, first, c);
        result.set(second, second, c);
        result.set(first, second, -s);
        result.set(second, first, s);
        return result;
    }

    private static Matrix outerProduct(double[] left, double[] right) {
        double[][] data = new double[left.length][right.length];
        for (int row = 0; row < left.length; row++) {
            for (int col = 0; col < right.length; col++) {
                data[row][col] = left[row] * right[col];
            }
        }
        return new Matrix(data);
    }

    private static Matrix scale(Matrix input, double factor) {
        double[][] data = new double[input.getRowCount()][input.getColumnCount()];
        for (int row = 0; row < input.getRowCount(); row++) {
            for (int col = 0; col < input.getColumnCount(); col++) {
                data[row][col] = input.get(row, col) * factor;
            }
        }
        return new Matrix(data);
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[][] data = new double[rows][cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                data[row][col] = 2.0 * random.nextDouble() - 1.0;
            }
        }
        return new Matrix(data);
    }
}
