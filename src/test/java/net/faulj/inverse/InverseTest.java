package net.faulj.inverse;

import net.faulj.matrix.Matrix;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Tests adjugate- and LU-based matrix inversion implementations.
 */
public class InverseTest {

    private static final double TOL = 1e-8;

    // ========== AdjugateInverse Tests ==========

    /**
     * Verifies adjugate inverse on a $2\times2$ matrix.
     */
    @Test
    public void adjugateInverse2x2Matrix() {
        Matrix A = new Matrix(new double[][]{
                {4, 7},
                {2, 6}
        });

        Matrix Ainv = AdjugateInverse.compute(A);
        Matrix I = A.multiply(Ainv);

        // Check A * A^(-1) = I
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("A * A^(-1) at (" + i + "," + j + ")",
                        expected, I.get(i, j), TOL);
            }
        }
    }

    /**
     * Verifies adjugate inverse on a $3\times3$ matrix.
     */
    @Test
    public void adjugateInverse3x3Matrix() {
        Matrix A = new Matrix(new double[][]{
                {1, 2, 3},
                {0, 1, 4},
                {5, 6, 0}
        });

        Matrix Ainv = AdjugateInverse.compute(A);
        Matrix I = A.multiply(Ainv);

        // Check A * A^(-1) = I
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("A * A^(-1) at (" + i + "," + j + ")",
                        expected, I.get(i, j), TOL);
            }
        }
    }

    /**
     * Cross-checks adjugate inverse against LU inverse for consistency.
     */
    @Test
    public void adjugateInverseComparisonWithLUInverse() {
        Matrix A = new Matrix(new double[][]{
                {3, 2, 1},
                {1, 4, 2},
                {2, 1, 5}
        });

        Matrix adjInv = AdjugateInverse.compute(A);
        Matrix luInv = LUInverse.compute(A);

        // Both methods should give same result
        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals("Both methods should give same result at (" + i + "," + j + ")",
                        luInv.get(i, j), adjInv.get(i, j), TOL);
            }
        }
    }

    /**
     * Ensures singular matrices trigger an exception for adjugate inverse.
     */
    @Test(expected = ArithmeticException.class)
    public void adjugateInverseSingularMatrixThrowsException() {
        Matrix A = new Matrix(new double[][]{
                {1, 2, 3},
                {2, 4, 6},
                {1, 1, 1}
        });

        AdjugateInverse.compute(A);
    }

    /**
     * Confirms adjugate inverse returns reciprocal diagonal entries.
     */
    @Test
    public void adjugateInverseOfDiagonalMatrix() {
        Matrix D = new Matrix(new double[][]{
                {3, 0, 0},
                {0, 4, 0},
                {0, 0, 2}
        });

        Matrix Dinv = AdjugateInverse.compute(D);

        assertEquals(1.0 / 3.0, Dinv.get(0, 0), TOL);
        assertEquals(0.25, Dinv.get(1, 1), TOL);
        assertEquals(0.5, Dinv.get(2, 2), TOL);
    }

    /**
     * Confirms adjugate inverse preserves identity.
     */
    @Test
    public void adjugateInverseOfIdentityIsIdentity() {
        Matrix I = Matrix.Identity(3);
        Matrix Iinv = AdjugateInverse.compute(I);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals("I^(-1) at (" + i + "," + j + ")",
                        I.get(i, j), Iinv.get(i, j), TOL);
            }
        }
    }

    // ========== LUInverse Tests ==========

    /**
     * Verifies $A A^{-1} = I$ for LU-based inverse.
     */
    @Test
    public void luInverseMultiplicationGivesIdentity() {
        Matrix A = new Matrix(new double[][]{
                {4, 3, 2},
                {2, 5, 1},
                {1, 2, 3}
        });

        Matrix Ainv = LUInverse.compute(A);
        Matrix I = A.multiply(Ainv);

        // Check A * A^(-1) = I
        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("A * A^(-1) at (" + i + "," + j + ")",
                        expected, I.get(i, j), TOL);
            }
        }
    }

    /**
     * Checks that inverting an inverse returns the original matrix.
     */
    @Test
    public void luInverseOfInverseIsOriginal() {
        Matrix A = new Matrix(new double[][]{
                {5, 2, 1},
                {1, 4, 3},
                {2, 1, 6}
        });

        Matrix Ainv = LUInverse.compute(A);
        Matrix AinvInv = LUInverse.compute(Ainv);

        // (A^(-1))^(-1) = A
        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals("(A^(-1))^(-1) at (" + i + "," + j + ")",
                        A.get(i, j), AinvInv.get(i, j), TOL);
            }
        }
    }

    /**
     * Confirms LU inverse preserves identity.
     */
    @Test
    public void luInverseOfIdentityIsIdentity() {
        Matrix I = Matrix.Identity(4);
        Matrix Iinv = LUInverse.compute(I);

        // I^(-1) = I
        int n = I.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals("I^(-1) at (" + i + "," + j + ")",
                        I.get(i, j), Iinv.get(i, j), TOL);
            }
        }
    }

    /**
     * Verifies LU inverse on diagonal matrices yields reciprocal diagonals.
     */
    @Test
    public void luInverseOfDiagonalMatrix() {
        Matrix D = new Matrix(new double[][]{
                {2, 0, 0},
                {0, 3, 0},
                {0, 0, 5}
        });

        Matrix Dinv = LUInverse.compute(D);

        // For diagonal matrix, inverse elements are 1/d_ii
        assertEquals(0.5, Dinv.get(0, 0), TOL);
        assertEquals(1.0 / 3.0, Dinv.get(1, 1), TOL);
        assertEquals(0.2, Dinv.get(2, 2), TOL);
    }

    /**
     * Ensures LU inverse rejects singular matrices.
     */
    @Test(expected = ArithmeticException.class)
    public void luInverseSingularMatrixThrowsException() {
        // Singular matrix (row 2 = row 1)
        Matrix A = new Matrix(new double[][]{
                {1, 2, 3},
                {1, 2, 3},
                {4, 5, 6}
        });

        LUInverse.compute(A);
    }

    /**
     * Ensures LU inverse rejects the zero matrix.
     */
    @Test(expected = ArithmeticException.class)
    public void luInverseZeroMatrixThrowsException() {
        Matrix zero = new Matrix(new double[][]{
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
        });

        LUInverse.compute(zero);
    }

    /**
     * Ensures LU inverse rejects non-square matrices.
     */
    @Test(expected = IllegalArgumentException.class)
    public void luInverseNonSquareMatrixThrowsException() {
        Matrix A = new Matrix(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        LUInverse.compute(A);
    }

    /**
     * Verifies $A^{-1} A = I$ for LU inverse.
     */
    @Test
    public void luInverseTimesOriginalIsIdentity() {
        Matrix A = new Matrix(new double[][]{
                {3, 0, 2},
                {2, 0, -2},
                {0, 1, 1}
        });

        Matrix Ainv = LUInverse.compute(A);
        Matrix I = Ainv.multiply(A);

        // Check A^(-1) * A = I
        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("A^(-1) * A at (" + i + "," + j + ")",
                        expected, I.get(i, j), TOL);
            }
        }
    }

    /**
     * Confirms symmetry preservation for inverse of symmetric matrices.
     */
    @Test
    public void luInverseSymmetricMatrixInverse() {
        Matrix A = new Matrix(new double[][]{
                {4, 1, 2},
                {1, 3, 1},
                {2, 1, 5}
        });

        Matrix Ainv = LUInverse.compute(A);

        // For symmetric matrix, inverse should also be symmetric
        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals("Inverse should be symmetric",
                        Ainv.get(i, j), Ainv.get(j, i), TOL);
            }
        }
    }

    /**
     * Validates LU inverse accuracy on a random matrix.
     */
    @Test
    public void luInverseRandomMatrixInverse() {
        Matrix A = Matrix.randomMatrix(5, 5);

        Matrix Ainv = LUInverse.compute(A);
        Matrix I = A.multiply(Ainv);

        // Verify it's close to identity
        double maxDeviation = 0;
        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                maxDeviation = Math.max(maxDeviation, Math.abs(I.get(i, j) - expected));
            }
        }

        assertTrue("Max deviation should be small: " + maxDeviation, maxDeviation < TOL);
    }

    /**
     * Verifies LU inverse for a larger fixed $4\times4$ matrix.
     */
    @Test
    public void luInverseLargerMatrix() {
        Matrix A = new Matrix(new double[][]{
                {5, 7, 2, 1},
                {1, 4, 3, 2},
                {2, 5, 8, 3},
                {3, 1, 2, 6}
        });

        Matrix Ainv = LUInverse.compute(A);
        Matrix I = A.multiply(Ainv);

        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("Identity check at (" + i + "," + j + ")",
                        expected, I.get(i, j), TOL);
            }
        }
    }
}
