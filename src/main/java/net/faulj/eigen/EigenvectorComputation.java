package net.faulj.eigen;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;
import net.faulj.core.Tolerance;

/**
 * Utilities for computing eigenvectors from matrix decompositions.
 * <p>
 * This class focuses on extracting eigenvectors from the Schur form (Real Schur Decomposition).
 * While eigenvalues are immediately apparent on the diagonal of the Schur matrix T,
 * eigenvectors require a back-substitution process followed by a transformation.
 * </p>
 *
 * <h2>Computational Strategy:</h2>
 * <p>
 * To find the eigenvector x associated with eigenvalue λ:
 * </p>
 * <ol>
 * <li><b>Solve for y:</b> Find eigenvector y of the quasi-triangular matrix T.
 * <pre>(T - λI)y = 0</pre>
 * Since T is upper triangular (or quasi-upper), this is solved efficiently via back-substitution.
 * </li>
 * <li><b>Transform to x:</b> Transform y back to the original basis using the Schur vectors U.
 * <pre>x = Uy</pre>
 * Because A = UTU<sup>T</sup>, if Ty = λy, then A(Uy) = U(Ty) = U(λy) = λ(Uy).
 * </li>
 * </ol>
 *
 * <h2>Handling Quasi-Triangular Forms:</h2>
 * <p>
 * The Real Schur form contains 2x2 blocks on the diagonal corresponding to complex conjugate
 * eigenvalue pairs.
 * </p>
 * <ul>
 * <li><b>Real Eigenvalues:</b> Direct back-substitution is used.</li>
 * <li><b>Complex Eigenvalues:</b> Requires complex arithmetic or solving for invariant planes (2D subspaces).
 * <i>Current implementation focuses on Real eigenvalues.</i></li>
 * </ul>
 *
 * <h2>Numerical Stability:</h2>
 * <ul>
 * <li><b>Ill-conditioning:</b> If eigenvalues are very close, the system (T - λI) becomes nearly singular
 * beyond just the rank deficiency expected for an eigenvector.</li>
 * <li><b>Perturbation:</b> Small perturbations are applied to denominators to prevent division by zero
 * during back-substitution in degenerate cases.</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // 1. Compute Schur Decomposition
 * SchurResult schur = realSchurDecomposition.decompose(A);
 *
 * // 2. Extract Eigenvectors
 * Matrix eigenvectors = EigenvectorComputation.computeEigenvectors(schur);
 *
 * // 3. Verify A*v = lambda*v for the i-th eigenpair
 * Vector v = eigenvectors.getColumn(i);
 * double lambda = schur.getRealEigenvalues()[i];
 * Vector Av = A.multiply(v);
 * Vector lv = v.multiply(lambda);
 * // assert Av.approxEquals(lv);
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.result.SchurResult
 */
public class EigenvectorComputation {

    /**
     * Computes eigenvectors from a Schur Decomposition.
     * <p>
     * This method performs back-substitution on the quasi-triangular matrix T to find
     * eigenvectors y, and then computes x = Uy.
     * </p>
     * <p>
     * <b>Note:</b> Currently returns a Matrix where columns are real parts of eigenvectors.
     * Complex eigenvector handling would require a Complex matrix class and is skipped
     * in this implementation version.
     * </p>
     *
     * @param schur The result of a Real Schur Decomposition containing T, U, and eigenvalues.
     * @return A Matrix where the i-th column is the eigenvector corresponding to the i-th eigenvalue.
     * Columns corresponding to complex eigenvalues may be zeroed or incomplete.
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