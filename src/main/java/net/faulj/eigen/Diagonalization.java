package net.faulj.eigen;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.eigen.schur.SchurEigenExtractor;
import net.faulj.matrix.Matrix;
import net.faulj.scalar.Complex;

/**
 * Handles the diagonalization of square matrices via eigendecomposition.
 */
public class Diagonalization {
    private final Matrix A;
    private final Matrix D;
    private final Matrix P;
    private final Complex[] eigenvalues;

    /**
     * Create a diagonalization container.
     *
     * @param A original matrix
     * @param D diagonal matrix of eigenvalues
     * @param P eigenvector matrix
     * @param eigenvalues eigenvalues as complex numbers
     */
    public Diagonalization(Matrix A, Matrix D, Matrix P, Complex[] eigenvalues) {
        this.A = A;
        this.D = D;
        this.P = P;
        this.eigenvalues = eigenvalues;
    }

    /**
     * @return original matrix
     */
    public Matrix getA() {
        return A;
    }

    /**
     * @return diagonal eigenvalue matrix
     */
    public Matrix getD() {
        return D;
    }

    /**
     * @return eigenvector matrix
     */
    public Matrix getP() {
        return P;
    }

    /**
     * @return eigenvalues as complex numbers
     */
    public Complex[] getEigenvalues() {
        return eigenvalues;
    }

    /**
     * Compute the diagonalization A = P D P^{-1} via Schur decomposition.
     *
     * @param A matrix to diagonalize
     * @return diagonalization result
     */
    public static Diagonalization decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Diagonalization requires a square matrix");
        }
        SchurResult schur = RealSchurDecomposition.decompose(A);
        SchurEigenExtractor extractor = new SchurEigenExtractor(schur.getT(), schur.getU());
        Matrix P = extractor.getEigenvectors();
        Complex[] eigenvalues = extractor.getEigenvalues();
        Matrix D = diagonalMatrix(eigenvalues);
        return new Diagonalization(A, D, P, eigenvalues);
    }

    /**
     * Build a complex diagonal matrix from eigenvalues.
     *
     * @param eigenvalues eigenvalues
     * @return diagonal matrix
     */
    private static Matrix diagonalMatrix(Complex[] eigenvalues) {
        int n = eigenvalues.length;
        Matrix D = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            Complex lambda = eigenvalues[i];
            D.setComplex(i, i, lambda.real, lambda.imag);
        }
        return D;
    }
}
