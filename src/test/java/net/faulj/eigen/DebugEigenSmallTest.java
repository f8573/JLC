package net.faulj.eigen;

import org.junit.Test;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;

public class DebugEigenSmallTest {

    @Test
    public void printNearSingular10() {
        int n = 10;
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            data[i][i] = (i == 0) ? 1e-10 : i + 1;
        }
        // create diagonal in row-major then convert
        double[][] cols = new double[n][n];
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) cols[c][r] = data[r][c];
        }
        net.faulj.vector.Vector[] vecs = new net.faulj.vector.Vector[n];
        for (int c = 0; c < n; c++) vecs[c] = new net.faulj.vector.Vector(cols[c]);
        Matrix D = new Matrix(vecs);

        Matrix Q = HouseholderQR.decompose(randomMatrix(n, 2200 + n)).getQ();
        Matrix A = Q.multiply(D).multiply(Q.transpose());

        System.out.println("D diagonal:");
        for (int i = 0; i < n; i++) {
            System.out.printf("  D[%d] = %.12e\n", i, D.get(i, i));
        }
        // check Q orthogonality
        Matrix QtQ = Q.transpose().multiply(Q);
        System.out.printf("Q orthonorm residual: %.2e\n", QtQ.subtract(Matrix.Identity(n)).frobeniusNorm());

        SchurResult schur = RealSchurDecomposition.decompose(A);
        double[] eigs = schur.getRealEigenvalues();
        System.out.println("Schur eigenvalues:");
        for (int i = 0; i < eigs.length; i++) {
            System.out.printf("  [%d] = %.12e\n", i, eigs[i]);
        }
    }
    
        private static Matrix randomMatrix(int n, long seed) {
            java.util.Random rnd = new java.util.Random(seed);
            double[][] a = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    a[i][j] = rnd.nextDouble() * 2 - 1;
                }
            }
            int rows = a.length;
            int cols = a[0].length;
            net.faulj.vector.Vector[] colsV = new net.faulj.vector.Vector[cols];
            for (int c = 0; c < cols; c++) {
                double[] col = new double[rows];
                for (int r = 0; r < rows; r++) col[r] = a[r][c];
                colsV[c] = new net.faulj.vector.Vector(col);
            }
            return new Matrix(colsV);
        }

}
