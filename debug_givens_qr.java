import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.decomposition.qr.GivensQR;
import net.faulj.decomposition.result.QRResult;

public class debug_givens_qr {
    
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
    
    public static void main(String[] args) {
        System.out.println("=== DEBUGGING GIVENS QR ===\n");
        
        // Test matrix from GivensQRTest
        double[][] data = {
            {12, -51, 4},
            {6, 167, -68},
            {-4, 24, -41}
        };
        
        Matrix A = fromRowMajor(data);
        System.out.println("Original matrix A:");
        System.out.println(A.toString());
        System.out.println();
        
        // Perform decomposition
        QRResult result = GivensQR.decompose(A);
        Matrix Q = result.getQ();
        Matrix R = result.getR();
        
        System.out.println("Q matrix:");
        System.out.println(Q.toString());
        System.out.println();
        
        System.out.println("R matrix:");
        System.out.println(R.toString());
        System.out.println();
        
        // Check Q orthogonality: Q^T * Q should be I
        Matrix QtQ = Q.transpose().multiply(Q);
        System.out.println("Q^T * Q (should be Identity):");
        System.out.println(QtQ.toString());
        double orthError = 0.0;
        for (int i = 0; i < QtQ.getRowCount(); i++) {
            for (int j = 0; j < QtQ.getColumnCount(); j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double actual = QtQ.get(i, j);
                orthError += Math.abs(expected - actual);
            }
        }
        System.out.println("Q orthogonality error (sum of absolute differences): " + orthError);
        System.out.println();
        
        // Check if R is upper triangular
        System.out.println("Checking R upper triangular:");
        boolean isUpperTriangular = true;
        for (int i = 1; i < R.getRowCount(); i++) {
            for (int j = 0; j < Math.min(i, R.getColumnCount()); j++) {
                double val = R.get(i, j);
                if (Math.abs(val) > 1e-10) {
                    System.out.println("  R[" + i + "," + j + "] = " + val + " (NOT ZERO!)");
                    isUpperTriangular = false;
                }
            }
        }
        if (isUpperTriangular) {
            System.out.println("  R is upper triangular: PASS");
        } else {
            System.out.println("  R is NOT upper triangular: FAIL");
        }
        System.out.println();
        
        // Check reconstruction: A = Q * R
        Matrix QR = Q.multiply(R);
        System.out.println("Q * R (should equal A):");
        System.out.println(QR.toString());
        System.out.println();
        
        System.out.println("A - Q*R (reconstruction error):");
        Matrix diff = A.subtract(QR);
        System.out.println(diff.toString());
        double reconError = diff.frobeniusNorm() / Math.max(1.0, A.frobeniusNorm());
        System.out.println("Normalized reconstruction error: " + reconError);
        System.out.println();
        
        // Detailed diagnosis
        System.out.println("=== DIAGNOSIS ===");
        if (orthError > 1e-10) {
            System.out.println("PROBLEM: Q is NOT orthogonal!");
        } else {
            System.out.println("OK: Q is orthogonal");
        }
        
        if (!isUpperTriangular) {
            System.out.println("PROBLEM: R is NOT upper triangular!");
        } else {
            System.out.println("OK: R is upper triangular");
        }
        
        if (reconError > 1e-10) {
            System.out.println("PROBLEM: Reconstruction error too large!");
        } else {
            System.out.println("OK: Reconstruction A = Q*R is accurate");
        }
    }
}
