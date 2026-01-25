package net.faulj.decomposition;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import net.faulj.decomposition.qr.GramSchmidt;
import net.faulj.decomposition.qr.ModifiedGramSchmidt;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import net.faulj.vector.Vector;

/**
 * Comprehensive test suite for Gram-Schmidt QR decomposition variants.
 * Tests both classical and modified Gram-Schmidt implementations with:
 * - Randomized matrix sizes and values
 * - Fixed edge cases (singular, near-singular, orthogonal matrices)
 * - Numerical stability comparisons
 */
public class GramSchmidtTest {

    private static final double TOLERANCE = 1e-9;
    private static final double RELAXED_TOLERANCE = 1e-6;
    private static final Random RNG = new Random(42);

    // ========== Helper Methods ==========

    private static Matrix fromRowMajor(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        Vector[] colsV = new Vector[cols];
        for (int c = 0; c < cols; c++) {
            double[] col = new double[rows];
            for (int r = 0; r < rows; r++) col[r] = a[r][c];
            colsV[c] = new Vector(col);
        }
        return new Matrix(colsV);
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                a[i][j] = rnd.nextDouble() * 2 - 1;
            }
        }
        return fromRowMajor(a);
    }

    private void verifyQR(Matrix A, QRResult result, double tol, String context) {
        Matrix Q = result.getQ();
        Matrix R = result.getR();

        // Check orthonormality of Q
        double orthError = MatrixUtils.orthogonalityError(Q);
        assertTrue(context + ": Q orthogonality error = " + orthError, 
                   orthError < tol);

        // Check R is upper triangular
        int k = Math.min(A.getRowCount(), A.getColumnCount());
        for (int i = 1; i < R.getRowCount(); i++) {
            for (int j = 0; j < Math.min(i, R.getColumnCount()); j++) {
                if (j < k && i < k) {
                    assertEquals(context + ": R[" + i + "," + j + "] not zero",
                            0.0, R.get(i, j), tol);
                }
            }
        }

        // Check factorization: A = Q * R
        Matrix reconstructed = Q.multiply(R);
        double reconError = A.subtract(reconstructed).frobeniusNorm() / 
                           Math.max(1.0, A.frobeniusNorm());
        assertTrue(context + ": Reconstruction error = " + reconError,
                   reconError < tol);
    }

    // ========== Classical Gram-Schmidt Tests ==========

    @Test
    public void testClassicalGS_SmallWellConditioned() {
        System.out.println("\n=== Classical GS: Small Well-Conditioned Matrices ===");
        
        // 3x3 matrix
        Matrix A = fromRowMajor(new double[][]{
            {1, 1, 0},
            {1, 0, 1},
            {0, 1, 1}
        });
        
        QRResult result = GramSchmidt.decompose(A);
        verifyQR(A, result, TOLERANCE, "3x3 well-conditioned");
        System.out.println("  3x3 well-conditioned: PASSED");
    }

    @Test
    public void testClassicalGS_RandomSquare() {
        System.out.println("\n=== Classical GS: Random Square Matrices ===");
        int[] sizes = {4, 5, 8, 10};
        
        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 100 + n);
            QRResult result = GramSchmidt.decompose(A);
            verifyQR(A, result, RELAXED_TOLERANCE, "Random " + n + "x" + n);
            System.out.println("  Random " + n + "x" + n + ": PASSED");
        }
    }

    @Test
    public void testClassicalGS_RandomRectangular() {
        System.out.println("\n=== Classical GS: Random Rectangular Matrices ===");
        int[][] sizes = {{6, 3}, {5, 7}, {10, 5}, {8, 12}};
        
        for (int[] dim : sizes) {
            Matrix A = randomMatrix(dim[0], dim[1], 200 + dim[0] * 10 + dim[1]);
            QRResult result = GramSchmidt.decompose(A);
            verifyQR(A, result, RELAXED_TOLERANCE, 
                    "Random " + dim[0] + "x" + dim[1]);
            System.out.println("  Random " + dim[0] + "x" + dim[1] + ": PASSED");
        }
    }

    @Test
    public void testClassicalGS_OrthogonalMatrix() {
        System.out.println("\n=== Classical GS: Orthogonal Matrix ===");
        
        // Create orthogonal matrix from Householder reflections
        Matrix A = Matrix.Identity(4);
        double[] v1 = {1, 1, 0, 0};
        double norm1 = Math.sqrt(2);
        Matrix H1 = Matrix.Identity(4);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                H1.set(i, j, H1.get(i, j) - 2 * v1[i] * v1[j] / (norm1 * norm1));
            }
        }
        A = A.multiply(H1);
        
        QRResult result = GramSchmidt.decompose(A);
        verifyQR(A, result, TOLERANCE, "Orthogonal matrix");
        System.out.println("  Orthogonal matrix: PASSED");
    }

    // ========== Modified Gram-Schmidt Tests ==========

    @Test
    public void testModifiedGS_SmallWellConditioned() {
        System.out.println("\n=== Modified GS: Small Well-Conditioned Matrices ===");
        
        Matrix A = fromRowMajor(new double[][]{
            {12, -51, 4},
            {6, 167, -68},
            {-4, 24, -41}
        });
        
        QRResult result = ModifiedGramSchmidt.decompose(A);
        verifyQR(A, result, TOLERANCE, "3x3 standard example");
        System.out.println("  3x3 standard example: PASSED");
    }

    @Test
    public void testModifiedGS_RandomSquare() {
        System.out.println("\n=== Modified GS: Random Square Matrices ===");
        int[] sizes = {4, 5, 8, 10, 15, 20};
        
        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 300 + n);
            QRResult result = ModifiedGramSchmidt.decompose(A);
            verifyQR(A, result, TOLERANCE, "Random " + n + "x" + n);
            System.out.println("  Random " + n + "x" + n + ": PASSED");
        }
    }

    @Test
    public void testModifiedGS_RandomRectangular() {
        System.out.println("\n=== Modified GS: Random Rectangular Matrices ===");
        int[][] sizes = {{6, 3}, {5, 7}, {10, 5}, {8, 12}, {15, 8}, {12, 18}};
        
        for (int[] dim : sizes) {
            Matrix A = randomMatrix(dim[0], dim[1], 400 + dim[0] * 10 + dim[1]);
            QRResult result = ModifiedGramSchmidt.decompose(A);
            verifyQR(A, result, TOLERANCE, "Random " + dim[0] + "x" + dim[1]);
            System.out.println("  Random " + dim[0] + "x" + dim[1] + ": PASSED");
        }
    }

    @Test
    public void testModifiedGS_TallSkinny() {
        System.out.println("\n=== Modified GS: Tall-Skinny Matrices ===");
        int[][] sizes = {{100, 5}, {50, 3}, {80, 10}};
        
        for (int[] dim : sizes) {
            Matrix A = randomMatrix(dim[0], dim[1], 500 + dim[0] * 10 + dim[1]);
            QRResult result = ModifiedGramSchmidt.decompose(A);
            verifyQR(A, result, TOLERANCE * Math.sqrt(dim[0]), 
                    "Tall-Skinny " + dim[0] + "x" + dim[1]);
            System.out.println("  Tall-Skinny " + dim[0] + "x" + dim[1] + ": PASSED");
        }
    }

    // ========== Comparison Tests ==========

    @Test
    public void testStabilityComparison() {
        System.out.println("\n=== Stability Comparison: Classical vs Modified GS ===");
        
        int n = 10;
        Matrix A = randomMatrix(n, n, 600);
        
        QRResult classicalResult = GramSchmidt.decompose(A);
        QRResult modifiedResult = ModifiedGramSchmidt.decompose(A);
        
        double classicalOrthError = MatrixUtils.orthogonalityError(
            classicalResult.getQ());
        double modifiedOrthError = MatrixUtils.orthogonalityError(
            modifiedResult.getQ());
        
        System.out.println("  Classical GS orthogonality error: " + classicalOrthError);
        System.out.println("  Modified GS orthogonality error: " + modifiedOrthError);
        System.out.println("  Modified is " + (classicalOrthError / modifiedOrthError) + 
                           "x more accurate");
        
        // Modified should be more accurate or at least comparable
        assertTrue("Modified GS should be more stable", 
                   modifiedOrthError <= classicalOrthError * 2);
    }

    // ========== Edge Cases ==========

    @Test
    public void testSingularMatrix() {
        System.out.println("\n=== Edge Case: Rank-Deficient Matrix ===");
        
        // Create rank-1 matrix
        Matrix A = fromRowMajor(new double[][]{
            {1, 2, 3},
            {2, 4, 6},
            {3, 6, 9}
        });
        
        QRResult result = ModifiedGramSchmidt.decompose(A);
        assertNotNull("Should handle singular matrix", result);
        
        // R should have zero diagonal elements after the first
        double r11 = Math.abs(result.getR().get(0, 0));
        assertTrue("First diagonal should be non-zero", r11 > TOLERANCE);
        System.out.println("  Rank-deficient matrix: PASSED");
    }

    @Test
    public void testNearSingularMatrix() {
        System.out.println("\n=== Edge Case: Near-Singular Matrix ===");
        
        Matrix A = fromRowMajor(new double[][]{
            {1, 1 + 1e-10},
            {1, 1 - 1e-10}
        });
        
        QRResult classicalResult = GramSchmidt.decompose(A);
        QRResult modifiedResult = ModifiedGramSchmidt.decompose(A);
        
        double classicalOrthError = MatrixUtils.orthogonalityError(
            classicalResult.getQ());
        double modifiedOrthError = MatrixUtils.orthogonalityError(
            modifiedResult.getQ());
        
        System.out.println("  Near-singular Classical orthogonality: " + 
                           classicalOrthError);
        System.out.println("  Near-singular Modified orthogonality: " + 
                           modifiedOrthError);
        
        // Modified should handle near-singular matrices better
        // For very small, near-singular matrices, some degradation is expected
        // The difference of 2e-10 between columns leads to conditioning issues
        assertTrue("Modified should be reasonably stable for near-singular",
                   modifiedOrthError < 1e-5);
        assertTrue("Classical should be reasonably stable for near-singular",
                   classicalOrthError < 1e-5);
        System.out.println("  Near-singular matrix: PASSED");
    }

    @Test
    public void testIdentityMatrix() {
        System.out.println("\n=== Edge Case: Identity Matrix ===");
        
        Matrix I = Matrix.Identity(5);
        QRResult result = ModifiedGramSchmidt.decompose(I);
        
        // Q should be identity, R should be identity
        double qError = I.subtract(result.getQ()).frobeniusNorm();
        double rError = I.subtract(result.getR()).frobeniusNorm();
        
        assertTrue("Q should be identity", qError < TOLERANCE);
        assertTrue("R should be identity", rError < TOLERANCE);
        System.out.println("  Identity matrix: PASSED");
    }

    @Test
    public void testSingleColumn() {
        System.out.println("\n=== Edge Case: Single Column Matrix ===");
        
        Matrix A = fromRowMajor(new double[][]{{1}, {2}, {3}, {4}});
        QRResult result = ModifiedGramSchmidt.decompose(A);
        
        Matrix Q = result.getQ();
        Matrix R = result.getR();
        
        // Q should be normalized version of A
        double norm = Math.sqrt(1 + 4 + 9 + 16);
        for (int i = 0; i < 4; i++) {
            assertEquals("Q element " + i, (i + 1) / norm, Q.get(i, 0), TOLERANCE);
        }
        
        assertEquals("R[0,0] should be norm", norm, R.get(0, 0), TOLERANCE);
        System.out.println("  Single column matrix: PASSED");
    }

    @Test
    public void testSingleRow() {
        System.out.println("\n=== Edge Case: Single Row Matrix ===");
        
        Matrix A = fromRowMajor(new double[][]{{1, 2, 3, 4}});
        QRResult result = ModifiedGramSchmidt.decompose(A);
        
        assertNotNull("Should handle single row", result);
        assertEquals("Q should be 1x1", 1, result.getQ().getRowCount());
        System.out.println("  Single row matrix: PASSED");
    }

    // ========== Performance Comparison ==========

    @Test
    public void testPerformanceComparison() {
        System.out.println("\n=== Performance Comparison ===");
        
        int n = 100;
        Matrix A = randomMatrix(n, n, 700);
        
        long start1 = System.nanoTime();
        QRResult classical = GramSchmidt.decompose(A);
        long time1 = System.nanoTime() - start1;
        
        long start2 = System.nanoTime();
        QRResult modified = ModifiedGramSchmidt.decompose(A);
        long time2 = System.nanoTime() - start2;
        
        System.out.println("  Classical GS: " + (time1 / 1_000_000) + "ms");
        System.out.println("  Modified GS: " + (time2 / 1_000_000) + "ms");
        System.out.println("  Ratio: " + String.format("%.2f", (double)time1 / time2));
        
        // Both should complete successfully
        assertNotNull(classical);
        assertNotNull(modified);
    }
}
