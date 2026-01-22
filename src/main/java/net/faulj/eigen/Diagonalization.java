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

    public Diagonalization(Matrix A, Matrix D, Matrix P, Complex[] eigenvalues) {
        this.A = A;
        this.D = D;
        this.P = P;
        this.eigenvalues = eigenvalues;
    }

    public Matrix getA() {
        return A;
    }

    public Matrix getD() {
        return D;
    }

    public Matrix getP() {
        return P;
    }

    public Complex[] getEigenvalues() {
        return eigenvalues;
    }

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
