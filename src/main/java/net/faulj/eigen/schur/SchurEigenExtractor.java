package net.faulj.eigen.schur;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.matrix.Matrix;
import net.faulj.scalar.Complex;

/**
 * Utility class for computing eigenvectors from a Real Schur Decomposition.
 * <p>
 * While the Schur decomposition (A = UTU<sup>T</sup>) reveals eigenvalues on the diagonal blocks of T,
 * it does not explicitly provide the eigenvectors of A. This class bridges that gap by computing
 * the eigenvectors of the quasi-triangular matrix T and transforming them back to the basis of A.
 * </p>
 *
 * <h2>Algorithm:</h2>
 * <ol>
 * <li><b>Solve for y:</b> For each eigenvalue λ, find the eigenvector y of T by solving the system:
 * <pre>(T - λI)y = 0</pre>
 * Since T is quasi-upper triangular, this is solved via back-substitution.
 * </li>
 * <li><b>Transform to x:</b> Convert the eigenvector y (relative to T) to the eigenvector x (relative to A)
 * using the Schur vectors U:
 * <pre>x = Uy</pre>
 * </li>
 * </ol>
 *
 * <h2>Complex Eigenvectors:</h2>
 * <p>
 * For complex eigenvalues coming from 2×2 diagonal blocks, this class handles the complex arithmetic
 * required to compute the corresponding complex conjugate eigenvector pairs.
 * </p>
 *
 * <h2>Usage:</h2>
 * <p>
 * This class is typically used when both eigenvalues and eigenvectors are required, but the
 * stability of the Schur decomposition is preferred over direct eigendecomposition algorithms.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see RealSchurDecomposition
 * @see SchurResult
 */
public class SchurEigenExtractor {
    private Matrix Schur;
    private Complex[] eigenvalues;

    public SchurEigenExtractor(Matrix S) {
        Schur = S;
        int len = S.getColumnCount();
        eigenvalues = new Complex[len];
        initialize();
    }

    private void initialize() {
        eigenvalues = ExplicitQRIteration.getEigenvalues(Schur).toArray(new Complex[0]);
    }

    public Complex[] getEigenvalues() {
        return eigenvalues;
    }
}