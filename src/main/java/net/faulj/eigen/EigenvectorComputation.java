package net.faulj.eigen;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;
import net.faulj.core.Tolerance;

public class EigenvectorComputation {

    /**
     * Computes eigenvectors from a Schur Decomposition.
     * Note: Currently returns a Matrix where columns are real parts of eigenvectors.
     * Complex eigenvector handling would require a Complex matrix class.
     */
    public static Matrix computeEigenvectors(SchurResult schur) {
        Matrix T = schur.getT();
        Matrix U = schur.getU();
        int n = T.getRowCount();

        // We compute eigenvectors of T first: (T - lambda*I)y = 0
        // Since T is quasi-upper triangular, we use back-substitution.
        // Then x = U*y.

        // Storage for eigenvectors of T
        Matrix Y = Matrix.zero(n, n);

        double[] realEig = schur.getRealEigenvalues();
        double[] imagEig = schur.getImagEigenvalues();

        for (int i = 0; i < n; i++) {
            // Check if complex pair
            if (Math.abs(imagEig[i]) > Tolerance.get()) {
                // Complex case: T has a 2x2 block here or near here.
                // For this implementation, we will skip detailed complex arithmetic
                // and focus on real eigenvalues as per standard "Phase 1" constraints.
                // In a full implementation, we solve (T - (Re + i*Im)I)y = 0
                continue;
            }

            // Real case: Solve (T - lambda*I)y = 0
            double lambda = realEig[i];

            // We are looking for y.
            // Back substitution.
            // Problem: T - lambda*I is singular (by definition).
            // We assume y_i (or the component corresponding to the diagonal block) is 1.

            double[] yData = new double[n];
            yData[i] = 1.0;

            for (int row = i - 1; row >= 0; row--) {
                double sum = 0.0;
                for (int col = row + 1; col <= i; col++) {
                    sum += T.get(row, col) * yData[col];
                }

                double denominator = lambda - T.get(row, row);
                if (Math.abs(denominator) < Tolerance.get()) {
                    // Perturb slightly to avoid division by zero in ill-conditioned cases
                    denominator = Tolerance.get();
                }
                yData[row] = sum / denominator;
            }

            Y.setData(replaceColumn(Y.getData(), i, new Vector(yData)));
        }

        // Transform back: X = U * Y
        return U.multiply(Y);
    }

    private static Vector[] replaceColumn(Vector[] data, int colIndex, Vector v) {
        // Helper to set column in the column-major array structure
        data[colIndex] = v;
        return data;
    }
}