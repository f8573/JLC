import net.faulj.matrix.Matrix;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.result.HessenbergResult;
import java.util.Random;

public class DebugHess {
    public static void main(String[] args) {
        Random rnd = new Random(1234567L);
        // Skip to trial 2 (size 4)
        rnd.nextDouble(); rnd.nextDouble(); rnd.nextDouble(); rnd.nextDouble(); // skip trial 0
        rnd.nextDouble(); rnd.nextDouble(); rnd.nextDouble(); rnd.nextDouble(); rnd.nextDouble(); rnd.nextDouble(); rnd.nextDouble(); rnd.nextDouble(); rnd.nextDouble(); // skip trial 1
        
        int n = 4;
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        Matrix A = new Matrix(data);
        System.out.println("A = ");
        printMatrix(A);
        
        HessenbergResult hess = HessenbergReduction.decompose(A);
        Matrix H = hess.getH();
        Matrix Q = hess.getQ();
        
        System.out.println("\nH = ");
        printMatrix(H);
        
        System.out.println("\nQ = ");
        printMatrix(Q);
        
        System.out.println("\nQ'*Q (should be I) = ");
        printMatrix(Q.transpose().multiply(Q));
        
        System.out.println("\nQ*H*Q' (should be A) = ");
        Matrix reconstructed = Q.multiply(H).multiply(Q.transpose());
        printMatrix(reconstructed);
        
        double residual = hess.residualNorm();
        System.out.println("\nResidual: " + residual);
    }
    
    static void printMatrix(Matrix m) {
        double[] data = m.getRawData();
        int rows = m.getRowCount();
        int cols = m.getColumnCount();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%10.6f ", data[i * cols + j]);
            }
            System.out.println();
        }
    }
}
