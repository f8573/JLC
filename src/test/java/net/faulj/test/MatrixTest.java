// File: src/test/java/net/faulj/test/MatrixTest.java
package net.faulj.test;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.Matrix.LUResult;
import net.faulj.matrix.Matrix.QRResult;
import net.faulj.matrix.Matrix.EigenResult;
import net.faulj.matrix.Matrix.SVDResult;
import org.junit.Test;
import static org.junit.Assert.*;

public class MatrixTest {

    private static final float TOL = 1e-4f;

    // Check equality of two float values using a relative error.
    private void assertRelativeEquals(float expected, float actual, String message) {
        float diff = Math.abs(expected - actual);
        float relError = (Math.abs(expected) > TOL) ? diff / Math.abs(expected) : diff;
        if (relError > TOL) {
            fail(message + ": expected " + expected + " but got " + actual);
        }
    }

    // Compare matrices (only for real matrices).
    private void assertMatrixEquals(Matrix expected, Matrix actual) {
        assertEquals("Row count mismatch", expected.rows(), actual.rows());
        assertEquals("Column count mismatch", expected.columns(), actual.columns());
        for (int i = 0; i < expected.rows(); i++) {
            for (int j = 0; j < expected.columns(); j++) {
                assertRelativeEquals(expected.get(i, j), actual.get(i, j),
                        "Mismatch at (" + i + ", " + j + ")");
            }
        }
    }

    @Test
    public void testMatrixAdditionReal() {
        Matrix A = Matrix.of(new float[][] { { 1, 2 }, { 3, 4 } });
        Matrix B = Matrix.of(new float[][] { { 5, 6 }, { 7, 8 } });
        Matrix expected = Matrix.of(new float[][] { { 6, 8 }, { 10, 12 } });
        Matrix result = A.add(B);
        assertMatrixEquals(expected, result);
    }

    @Test
    public void testMatrixAdditionComplex() {
        // For complex matrices, data layout is [real, imag, ...]
        Matrix A = new Matrix(2, 2, true);
        Matrix B = new Matrix(2, 2, true);
        // Initialize A and B with some complex numbers.
        // A = [ (1+2i) (3+4i); (5+6i) (7+8i) ]
        A.getData()[0] = 1; A.getData()[1] = 2;
        A.getData()[2] = 3; A.getData()[3] = 4;
        A.getData()[4] = 5; A.getData()[5] = 6;
        A.getData()[6] = 7; A.getData()[7] = 8;
        // B = [ (8+7i) (6+5i); (4+3i) (2+1i) ]
        B.getData()[0] = 8; B.getData()[1] = 7;
        B.getData()[2] = 6; B.getData()[3] = 5;
        B.getData()[4] = 4; B.getData()[5] = 3;
        B.getData()[6] = 2; B.getData()[7] = 1;

        Matrix result = A.add(B);
        Matrix expected = new Matrix(2, 2, true);
        expected.getData()[0] = 9;  expected.getData()[1] = 9;
        expected.getData()[2] = 9;  expected.getData()[3] = 9;
        expected.getData()[4] = 9;  expected.getData()[5] = 9;
        expected.getData()[6] = 9;  expected.getData()[7] = 9;
        // Compare component-wise.
        for (int i = 0; i < expected.getData().length; i++) {
            assertRelativeEquals(expected.getData()[i], result.getData()[i], "Complex addition mismatch at index " + i);
        }
    }

    @Test
    public void testMatrixSubtractionReal() {
        Matrix A = Matrix.of(new float[][] { { 5, 6 }, { 7, 8 } });
        Matrix B = Matrix.of(new float[][] { { 1, 2 }, { 3, 4 } });
        Matrix expected = Matrix.of(new float[][] { { 4, 4 }, { 4, 4 } });
        Matrix result = A.subtract(B);
        assertMatrixEquals(expected, result);
    }

    @Test
    public void testScalarMultiplicationReal() {
        Matrix A = Matrix.of(new float[][] { { 1, 2 }, { 3, 4 } });
        float scalar = 2.0f;
        Matrix expected = Matrix.of(new float[][] { { 2, 4 }, { 6, 8 } });
        Matrix result = A.scalarMultiply(scalar);
        assertMatrixEquals(expected, result);
    }

    @Test
    public void testMatrixMultiplicationReal() {
        Matrix A = Matrix.of(new float[][] { { 1, 2 }, { 3, 4 } });
        Matrix B = Matrix.of(new float[][] { { 2, 0 }, { 1, 2 } });
        Matrix expected = Matrix.of(new float[][] { { 4, 4 }, { 10, 8 } });
        Matrix result = A.multiply(B);
        assertMatrixEquals(expected, result);
    }

    @Test
    public void testTransposeReal() {
        Matrix A = Matrix.of(new float[][] { { 1, 2, 3 }, { 4, 5, 6 } });
        Matrix expected = Matrix.of(new float[][] { { 1, 4 }, { 2, 5 }, { 3, 6 } });
        Matrix result = A.transpose();
        assertMatrixEquals(expected, result);
    }

    @Test
    public void testInverseReal() {
        Matrix A = Matrix.of(new float[][] { { 4, 7 }, { 2, 6 } });
        Matrix expected = Matrix.of(new float[][] { { 0.6f, -0.7f }, { -0.2f, 0.4f } });
        Matrix result = A.inverse();
        assertMatrixEquals(expected, result);
    }

    @Test
    public void testDeterminantReal() {
        Matrix A = Matrix.of(new float[][] { { 4, 7 }, { 2, 6 } });
        float expected = 10f;
        float det = A.determinant();
        assertRelativeEquals(expected, det, "Determinant mismatch");
    }

    @Test
    public void testAdjugateReal() {
        Matrix A = Matrix.of(new float[][] { { 4, 7 }, { 2, 6 } });
        Matrix expected = Matrix.of(new float[][] { { 6, -7 }, { -2, 4 } });
        Matrix result = A.adjugate();
        assertMatrixEquals(expected, result);
    }

    @Test
    public void testLUDecompositionReal() {
        Matrix A = Matrix.of(new float[][] { { 4, 3 }, { 6, 3 } });
        LUResult result = A.lu();
        Matrix L = result.getL();
        Matrix U = result.getU();
        Matrix recomposed = L.multiply(U);
        assertMatrixEquals(A, recomposed);
    }

    @Test
    public void testQRDecompositionReal() {
        Matrix A = Matrix.of(new float[][] {
                { 12, -51, 4 },
                { 6, 167, -68 },
                { -4, 24, -41 }
        });
        QRResult result = A.qr();
        Matrix Q = result.getQ();
        Matrix R = result.getR();
        Matrix recomposed = Q.multiply(R);
        assertMatrixEquals(A, recomposed);
    }

    @Test
    public void testEigenDecompositionReal() {
        // Placeholder: once is supported, implement tests accordingly.
        Matrix A = Matrix.of(new float[][] { { 5, 4 }, { 1, 2 } });
        try {
            A.eigen();
            fail("Expected UnsupportedOperationException as this operation is malfunctioning.");
        } catch (UnsupportedOperationException e) {
            // expected
        }

    }

    @Test
    public void testSVDReal() {
        Matrix A = Matrix.of(new float[][] { { 1, 2 }, { 3, 4 } });
        SVDResult result = A.svd();
        Matrix U = result.getU();
        Matrix S = result.getS();
        Matrix Vt = result.getVt();
        Matrix recomposed = U.multiply(S).multiply(Vt);
        assertMatrixEquals(A, recomposed);
    }

    @Test
    public void testSVDComplex() {
        // Placeholder: once complex SVD is supported, implement tests accordingly.
        // For now, simply check that the method throws an exception.
        Matrix A = new Matrix(2, 2, true);
        try {
            A.svd();
            fail("Expected UnsupportedOperationException for complex SVD.");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }
}