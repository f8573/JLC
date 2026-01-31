package net.faulj.stress;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.result.HessenbergResult;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class HessenbergDebugTest {
    
    @Test
    public void debugSize4() {
        Random rnd = new Random(1234567L);
        // Generate matrices for trial 0, 1, 2 (size 2, 3, 4)
        for (int trial = 0; trial <= 2; trial++) {
            int n = 2 + trial;
            double[][] data = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    data[i][j] = rnd.nextDouble() * 2.0 - 1.0;
                }
            }
            Matrix A = new Matrix(data);
            
            System.out.println("\n=== Trial " + trial + ", size " + n + " ===");
            System.out.println("A = ");
            printMatrix(A);
            
            HessenbergResult hess = HessenbergReduction.decompose(A);
            Matrix H = hess.getH();
            Matrix Q = hess.getQ();
            
            System.out.println("\nH = ");
            printMatrix(H);
            
            System.out.println("\nQ = ");
            printMatrix(Q);
            
            Matrix QtQ = Q.transpose().multiply(Q);
            System.out.println("\nQ'*Q (should be I) = ");
            printMatrix(QtQ);
            
            Matrix reconstructed = Q.multiply(H).multiply(Q.transpose());
            System.out.println("\nQ*H*Q' (should be A) = ");
            printMatrix(reconstructed);
            
            double residual = hess.residualNorm();
            System.out.println("\nResidual: " + residual);
            
            // Check orthogonality
            double orthError = orthogonalityError(QtQ, n);
            System.out.println("Orthogonality error: " + orthError);
            
            assertTrue("Hessenberg residual too large at size " + n + " trial " + trial + ": " + residual,
                residual < 0.01);
        }
    }
    
    private static double orthogonalityError(Matrix QtQ, int n) {
        double[] data = QtQ.getRawData();
        double err = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = Math.abs(data[i * n + j] - expected);
                err = Math.max(err, diff);
            }
        }
        return err;
    }
    
    private static void printMatrix(Matrix m) {
        double[] data = m.getRawData();
        int rows = m.getRowCount();
        int cols = m.getColumnCount();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%12.8f ", data[i * cols + j]);
            }
            System.out.println();
        }
    }
}
