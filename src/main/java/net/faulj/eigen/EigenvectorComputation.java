package net.faulj.eigen;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.matrix.Matrix;

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
     * Computes eigenvectors from a Schur decomposition result.
     *
     * @param schur The Schur result containing T and U
     * @return Matrix whose columns are eigenvectors (complex-aware)
     */
    public static Matrix computeEigenvectors(SchurResult schur) {
        if (schur == null) {
            throw new IllegalArgumentException("Schur result must not be null");
        }
        Matrix T = schur.getT();
        Matrix U = schur.getU();
        if (T == null || U == null) {
            throw new IllegalArgumentException("Schur result must contain T and U");
        }
        net.faulj.eigen.schur.SchurEigenExtractor extractor =
                new net.faulj.eigen.schur.SchurEigenExtractor(T, U);
        return extractor.getEigenvectors();
    }

    /**
     * Computes eigenvectors by performing a real Schur decomposition first.
     *
     * @param A The input matrix
     * @return Matrix whose columns are eigenvectors (complex-aware)
     */
    public static Matrix computeEigenvectors(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        SchurResult schur = net.faulj.eigen.schur.RealSchurDecomposition.decompose(A);
        return computeEigenvectors(schur);
    }
}
