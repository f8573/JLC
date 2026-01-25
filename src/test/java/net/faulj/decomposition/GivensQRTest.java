package net.faulj.decomposition;

import net.faulj.decomposition.qr.GivensQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import net.faulj.vector.Vector;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

public class GivensQRTest {

    private static final double TOLERANCE = 1e-9;

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

        // Check orthogonality of Q
        double orthError = MatrixUtils.orthogonalityError(Q);
        assertTrue(context + ": Q orthogonality error = " + orthError,
                orthError < tol);

        // Check R is upper triangular
        int k = Math.min(A.getRowCount(), A.getColumnCount());
        for (int i = 1; i < Math.min(R.getRowCount(), k); i++) {
            for (int j = 0; j < Math.min(i, R.getColumnCount()); j++) {
                assertEquals(context + ": R[" + i + "," + j + "] not zero",
                        0.0, Math.abs(R.get(i, j)), tol);
            }
        }

        // Check factorization: A = Q * R
        Matrix reconstructed = Q.multiply(R);
        double reconError = A.subtract(reconstructed).frobeniusNorm() /
                Math.max(1.0, A.frobeniusNorm());
        assertTrue(context + ": Reconstruction error = " + reconError,
                reconError < tol);
    }

    @Test
    public void testGivensQR_NearZeroElements() {
        System.out.println("\n=== Edge Case: Matrix with Near-Zero Elements ===");
        
        Matrix A = fromRowMajor(new double[][]{
            {1.0, 1e-15, 1e-14},
            {1e-15, 2.0, 1e-15},
            {1e-14, 1e-15, 3.0}
        });
        
        QRResult result = GivensQR.decompose(A);
        verifyQR(A, result, TOLERANCE, "Near-zero elements");
        System.out.println("  Near-zero elements: PASSED");
    }

    // ========== Performance Tests ==========

    @Test
    public void testGivensQR_Performance() {
        System.out.println("\n=== Performance Test ===");
        
        int[] sizes = {10, 20, 30, 50};
        
        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 900 + n);
            
            long start = System.nanoTime();
            QRResult result = GivensQR.decompose(A);
            long elapsed = System.nanoTime() - start;
            
            assertNotNull(result);
            System.out.println("  " + n + "x" + n + ": " + 
                             (elapsed / 1_000_000) + "ms");
        }
    }

    @Test
    public void testGivensQR_LargeMatrix() {
        System.out.println("\n=== Large Matrix Test ===");
        
        int n = 100;
        Matrix A = randomMatrix(n, n, 1000);
        
        long start = System.nanoTime();
        QRResult result = GivensQR.decompose(A);
        long elapsed = System.nanoTime() - start;
        
        verifyQR(A, result, TOLERANCE * Math.sqrt(n), "Random 100x100");
        System.out.println("  100x100 completed in " + (elapsed / 1_000_000) + "ms");
    }
}
