package net.faulj.givens;

import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import net.faulj.vector.Vector;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * Comprehensive test suite for Givens utility classes:
 * - GivensBidiagonal
 * - GivensTridiagonal
 * - QRUpdateDowndate
 */
public class GivensUtilitiesTest {

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

    private static Matrix randomSymmetric(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double val = rnd.nextDouble() * 2 - 1;
                a[i][j] = val;
                a[j][i] = val;
            }
        }
        return fromRowMajor(a);
    }

    private boolean isBidiagonal(Matrix B, double tol) {
        int m = B.getRowCount();
        int n = B.getColumnCount();
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j || (i == j - 1 && j < n)) {
                    // Diagonal or superdiagonal - OK
                    continue;
                }
                if (Math.abs(B.get(i, j)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    private boolean isTridiagonal(Matrix T, double tol) {
        int n = T.getRowCount();
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(i - j) <= 1) {
                    // Main diagonal or first sub/super diagonal - OK
                    continue;
                }
                if (Math.abs(T.get(i, j)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    // ========== GivensBidiagonal Tests ==========

    @Test
    public void testBidiagonal_SmallMatrix() {
        System.out.println("\n=== Givens Bidiagonal: Small Matrix ===");
        
        Matrix A = fromRowMajor(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
        
        GivensBidiagonal gb = new GivensBidiagonal();
        BidiagonalizationResult result = gb.bidiagonalize(A);
        
        Matrix U = result.getU();
        Matrix B = result.getB();
        Matrix V = result.getV();
        
        // Check orthogonality
        double uOrthError = MatrixUtils.orthogonalityError(U);
        double vOrthError = MatrixUtils.orthogonalityError(V);
        
        assertTrue("U should be orthogonal", uOrthError < TOLERANCE);
        assertTrue("V should be orthogonal", vOrthError < TOLERANCE);
        
        // Check bidiagonal form
        assertTrue("B should be bidiagonal", isBidiagonal(B, TOLERANCE));
        
        // Check factorization: A = U * B * V^T
        Matrix reconstructed = result.reconstruct();
        double reconError = A.subtract(reconstructed).frobeniusNorm() / 
                           A.frobeniusNorm();
        assertTrue("Reconstruction error = " + reconError, 
                   reconError < RELAXED_TOLERANCE);
        
        System.out.println("  4x3 matrix: PASSED");
    }

    @Test
    public void testBidiagonal_RandomRectangular() {
        System.out.println("\n=== Givens Bidiagonal: Random Rectangular ===");
        int[][] sizes = {{5, 3}, {6, 4}, {10, 5}, {8, 6}};
        
        for (int[] dim : sizes) {
            Matrix A = randomMatrix(dim[0], dim[1], 100 + dim[0] * 10 + dim[1]);
            
            GivensBidiagonal gb = new GivensBidiagonal();
            BidiagonalizationResult result = gb.bidiagonalize(A);
            
            assertTrue("B should be bidiagonal", 
                      isBidiagonal(result.getB(), RELAXED_TOLERANCE));
            
            double reconError = A.subtract(result.reconstruct()).frobeniusNorm() / 
                               A.frobeniusNorm();
            assertTrue("Reconstruction error for " + dim[0] + "x" + dim[1],
                      reconError < RELAXED_TOLERANCE * 10);
            
            System.out.println("  " + dim[0] + "x" + dim[1] + ": PASSED");
        }
    }

    @Test
    public void testBidiagonal_SquareMatrix() {
        System.out.println("\n=== Givens Bidiagonal: Square Matrix ===");
        
        int n = 5;
        Matrix A = randomMatrix(n, n, 200);
        
        GivensBidiagonal gb = new GivensBidiagonal();
        BidiagonalizationResult result = gb.bidiagonalize(A);
        
        assertTrue("B should be bidiagonal", 
                  isBidiagonal(result.getB(), RELAXED_TOLERANCE));
        
        double reconError = A.subtract(result.reconstruct()).frobeniusNorm() / 
                           A.frobeniusNorm();
        assertTrue("Square matrix reconstruction", 
                  reconError < RELAXED_TOLERANCE * 10);
        
        System.out.println("  5x5 matrix: PASSED");
    }

    // ========== GivensTridiagonal Tests ==========

    @Test
    public void testTridiagonal_SmallSymmetric() {
        System.out.println("\n=== Givens Tridiagonal: Small Symmetric ===");
        
        Matrix A = fromRowMajor(new double[][]{
            {4, 1, 0, 0},
            {1, 4, 1, 0},
            {0, 1, 4, 1},
            {0, 0, 1, 4}
        });
        
        Matrix[] result = GivensTridiagonal.tridiagonalize(A);
        Matrix T = result[0];
        Matrix Q = result[1];
        
        // Check Q is orthogonal
        double orthError = MatrixUtils.orthogonalityError(Q);
        assertTrue("Q should be orthogonal", orthError < TOLERANCE);
        
        // Check T is tridiagonal
        assertTrue("T should be tridiagonal", isTridiagonal(T, TOLERANCE));
        
        // Check similarity: A = Q * T * Q^T
        Matrix reconstructed = Q.multiply(T).multiply(Q.transpose());
        double reconError = A.subtract(reconstructed).frobeniusNorm() / 
                           A.frobeniusNorm();
        assertTrue("Reconstruction error = " + reconError,
                  reconError < RELAXED_TOLERANCE);
        
        System.out.println("  4x4 tridiagonal: PASSED");
    }

    @Test
    public void testTridiagonal_RandomSymmetric() {
        System.out.println("\n=== Givens Tridiagonal: Random Symmetric ===");
        int[] sizes = {3, 4, 5, 6, 8};
        
        for (int n : sizes) {
            Matrix A = randomSymmetric(n, 300 + n);
            
            Matrix[] result = GivensTridiagonal.tridiagonalize(A);
            Matrix T = result[0];
            Matrix Q = result[1];
            
            assertTrue("T should be tridiagonal for " + n + "x" + n,
                      isTridiagonal(T, RELAXED_TOLERANCE));
            
            double orthError = MatrixUtils.orthogonalityError(Q);
            assertTrue("Q orthogonality for " + n + "x" + n,
                      orthError < RELAXED_TOLERANCE);
            
            Matrix reconstructed = Q.multiply(T).multiply(Q.transpose());
            double reconError = A.subtract(reconstructed).frobeniusNorm() / 
                               A.frobeniusNorm();
            assertTrue("Reconstruction for " + n + "x" + n,
                      reconError < RELAXED_TOLERANCE * 10);
            
            System.out.println("  " + n + "x" + n + ": PASSED");
        }
    }

    @Test
    public void testTridiagonal_AlreadyTridiagonal() {
        System.out.println("\n=== Givens Tridiagonal: Already Tridiagonal ===");
        
        int n = 5;
        double[][] a = new double[n][n];
        Random rnd = new Random(400);
        
        for (int i = 0; i < n; i++) {
            a[i][i] = rnd.nextDouble() * 2 - 1;
            if (i < n - 1) {
                double offDiag = rnd.nextDouble() * 2 - 1;
                a[i][i + 1] = offDiag;
                a[i + 1][i] = offDiag;
            }
        }
        
        Matrix A = fromRowMajor(a);
        Matrix[] result = GivensTridiagonal.tridiagonalize(A);
        Matrix T = result[0];
        
        // Should preserve tridiagonal form
        assertTrue("Should remain tridiagonal", isTridiagonal(T, TOLERANCE));
        
        double reconError = A.subtract(result[1].multiply(T)
            .multiply(result[1].transpose())).frobeniusNorm() / A.frobeniusNorm();
        assertTrue("Should preserve values", reconError < RELAXED_TOLERANCE);
        
        System.out.println("  Already tridiagonal: PASSED");
    }

    // ========== QRUpdateDowndate Tests ==========

    @Test
    public void testQRUpdate_AppendRow() {
        System.out.println("\n=== QR Update: Append Row ===");
        
        // Start with small QR
        Matrix A = fromRowMajor(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        
        QRResult initial = net.faulj.decomposition.qr.GivensQR.decompose(A);
        Matrix Q = initial.getQ();
        Matrix R = initial.getR();
        
        // Append a row
        Vector newRow = new Vector(new double[]{10, 11, 12});
        QRResult updated = QRUpdateDowndate.appendRow(Q, R, newRow);
        
        assertNotNull("Updated QR should exist", updated);
        assertEquals("Q should have 4 rows", 4, updated.getQ().getRowCount());
        assertEquals("R should have 4 rows", 4, updated.getR().getRowCount());
        
        // Check orthogonality
        double orthError = MatrixUtils.orthogonalityError(updated.getQ());
        assertTrue("Updated Q should be orthogonal", 
                  orthError < RELAXED_TOLERANCE * 10);
        
        System.out.println("  Append row: PASSED");
    }

    @Test
    public void testQRUpdate_RankOneUpdate() {
        System.out.println("\n=== QR Update: Rank-One Update ===");
        
        Matrix A = fromRowMajor(new double[][]{
            {1, 2},
            {3, 4},
            {5, 6}
        });
        
        QRResult initial = net.faulj.decomposition.qr.GivensQR.decompose(A);
        
        Vector u = new Vector(new double[]{1, 0, 0});
        Vector v = new Vector(new double[]{0, 1});
        double alpha = 0.5;
        
        QRResult updated = QRUpdateDowndate.rankOneUpdate(
            initial.getQ(), initial.getR(), u, v, alpha);
        
        assertNotNull("Updated QR should exist", updated);
        
        // Check that R is upper triangular
        int m = updated.getR().getRowCount();
        int n = updated.getR().getColumnCount();
        for (int i = 1; i < Math.min(m, n); i++) {
            for (int j = 0; j < i; j++) {
                assertEquals("R should be upper triangular",
                           0.0, updated.getR().get(i, j), RELAXED_TOLERANCE);
            }
        }
        
        System.out.println("  Rank-one update: PASSED");
    }

    @Test
    public void testQRUpdate_MultipleUpdates() {
        System.out.println("\n=== QR Update: Multiple Sequential Updates ===");
        
        Matrix A = randomMatrix(5, 3, 500);
        QRResult current = net.faulj.decomposition.qr.GivensQR.decompose(A);
        
        // Apply several rank-one updates
        for (int k = 0; k < 3; k++) {
            Vector u = new Vector(new double[]{
                RNG.nextDouble(), RNG.nextDouble(), RNG.nextDouble(),
                RNG.nextDouble(), RNG.nextDouble()
            });
            Vector v = new Vector(new double[]{
                RNG.nextDouble(), RNG.nextDouble(), RNG.nextDouble()
            });
            
            current = QRUpdateDowndate.rankOneUpdate(
                current.getQ(), current.getR(), u, v, 0.1);
            
            assertNotNull("Update " + k + " should succeed", current);
        }
        
        // Check final result maintains properties
        double orthError = MatrixUtils.orthogonalityError(current.getQ());
        assertTrue("Final Q should maintain orthogonality",
                  orthError < RELAXED_TOLERANCE * 50);
        
        System.out.println("  Multiple updates: PASSED");
    }

    // ========== Edge Cases ==========

    @Test
    public void testBidiagonal_SingleColumn() {
        System.out.println("\n=== Edge Case: Single Column Bidiagonal ===");
        
        Matrix A = fromRowMajor(new double[][]{{1}, {2}, {3}});
        
        GivensBidiagonal gb = new GivensBidiagonal();
        BidiagonalizationResult result = gb.bidiagonalize(A);
        
        assertNotNull("Should handle single column", result);
        assertTrue("Result should be bidiagonal",
                  isBidiagonal(result.getB(), TOLERANCE));
        
        System.out.println("  Single column bidiagonal: PASSED");
    }

    @Test
    public void testTridiagonal_TwoByTwo() {
        System.out.println("\n=== Edge Case: 2x2 Tridiagonal ===");
        
        Matrix A = fromRowMajor(new double[][]{
            {1, 0.5},
            {0.5, 2}
        });
        
        Matrix[] result = GivensTridiagonal.tridiagonalize(A);
        
        assertNotNull("Should handle 2x2", result);
        assertTrue("Should be tridiagonal", isTridiagonal(result[0], TOLERANCE));
        
        System.out.println("  2x2 tridiagonal: PASSED");
    }

    // ========== Performance Tests ==========

    @Test
    public void testBidiagonal_Performance() {
        System.out.println("\n=== Bidiagonal Performance ===");
        
        int[][] sizes = {{10, 5}, {20, 10}, {30, 15}};
        
        for (int[] dim : sizes) {
            Matrix A = randomMatrix(dim[0], dim[1], 600 + dim[0]);
            
            GivensBidiagonal gb = new GivensBidiagonal();
            long start = System.nanoTime();
            BidiagonalizationResult result = gb.bidiagonalize(A);
            long elapsed = System.nanoTime() - start;
            
            assertNotNull(result);
            System.out.println("  " + dim[0] + "x" + dim[1] + ": " + 
                             (elapsed / 1_000_000) + "ms");
        }
    }

    @Test
    public void testTridiagonal_Performance() {
        System.out.println("\n=== Tridiagonal Performance ===");
        
        int[] sizes = {10, 20, 30};
        
        for (int n : sizes) {
            Matrix A = randomSymmetric(n, 700 + n);
            
            long start = System.nanoTime();
            Matrix[] result = GivensTridiagonal.tridiagonalize(A);
            long elapsed = System.nanoTime() - start;
            
            assertNotNull(result);
            System.out.println("  " + n + "x" + n + ": " + 
                             (elapsed / 1_000_000) + "ms");
        }
    }
}
