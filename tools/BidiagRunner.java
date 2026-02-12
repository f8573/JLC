import net.faulj.matrix.Matrix;
import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.matrix.MatrixUtils;

public class BidiagRunner {
    public static void main(String[] args) {
        double[][] a = new double[][]{
            {1,2,3},
            {2,4,6},
            {3,6,9}
        };
        Matrix A = new Matrix(a);
        System.out.println("A: " + MatrixUtils.matrixSummary(A, 3, 3));
        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult res = bidiag.decompose(A);
        Matrix U = res.getU();
        Matrix B = res.getB();
        Matrix V = res.getV();
        System.out.println("Bidiagonal B: " + MatrixUtils.matrixSummary(B, 3, 3));
        // Compute orthogonality manually to avoid invoking multiply/other classes
        System.out.println("U orthogonality=" + orthogonalityErrorManual(U));
        System.out.println("V orthogonality=" + orthogonalityErrorManual(V));
        System.out.println("U: " + MatrixUtils.matrixSummary(U, 3, 3));
        System.out.println("V: " + MatrixUtils.matrixSummary(V, 3, 3));
        // Reconstruct using several candidate formulas (manual multiply)
        Matrix recon1 = multiplyManual(multiplyManual(U, B), V.transpose());
        Matrix recon2 = multiplyManual(U, multiplyManual(B, V.transpose()));
        Matrix recon3 = multiplyManual(multiplyManual(U, B.transpose()), V.transpose());
        Matrix recon4 = multiplyManual(multiplyManual(U, B), V);
        System.out.println("Reconstructed A (U*B*V^T): " + MatrixUtils.matrixSummary(recon1, 3, 3));
        System.out.println("recon rel error (U*B*V^T)=" + relativeErrorManual(A, recon1));
        System.out.println("Reconstructed A (U*(B*V^T)): " + MatrixUtils.matrixSummary(recon2, 3, 3));
        System.out.println("recon rel error (U*(B*V^T))=" + relativeErrorManual(A, recon2));
        System.out.println("Reconstructed A (U*B^T*V^T): " + MatrixUtils.matrixSummary(recon3, 3, 3));
        System.out.println("recon rel error (U*B^T*V^T)=" + relativeErrorManual(A, recon3));
        System.out.println("Reconstructed A (U*B*V): " + MatrixUtils.matrixSummary(recon4, 3, 3));
        System.out.println("recon rel error (U*B*V)=" + relativeErrorManual(A, recon4));
    }

    private static double orthogonalityErrorManual(Matrix Q) {
        int n = Q.getColumnCount();
        int rows = Q.getRowCount();
        double[] data = Q.getRawData();
        double err = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dot = 0.0;
                for (int r = 0; r < rows; r++) {
                    dot += data[r * n + i] * data[r * n + j];
                }
                double delta = dot - (i == j ? 1.0 : 0.0);
                err += delta * delta;
            }
        }
        return Math.sqrt(err);
    }

    private static Matrix multiplyManual(Matrix A, Matrix B) {
        int m = A.getRowCount();
        int k = A.getColumnCount();
        int n = B.getColumnCount();
        Matrix out = new Matrix(m, n);
        double[] a = A.getRawData();
        double[] b = B.getRawData();
        double[] c = out.getRawData();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int t = 0; t < k; t++) {
                    sum += a[i * k + t] * b[t * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return out;
    }

    private static double relativeErrorManual(Matrix A, Matrix Ahat) {
        // frobenius norm based
        double[] a = A.getRawData();
        double[] b = Ahat.getRawData();
        double normA = 0.0;
        double normDiff = 0.0;
        for (int i = 0; i < a.length; i++) {
            normA += a[i] * a[i];
            double d = a[i] - b[i];
            normDiff += d * d;
        }
        if (normA <= 0.0) return 0.0;
        return Math.sqrt(normDiff) / Math.sqrt(normA);
    }
}
