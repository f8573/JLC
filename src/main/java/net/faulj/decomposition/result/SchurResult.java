package net.faulj.decomposition.result;

import net.faulj.matrix.Matrix;
import java.util.Arrays;

public class SchurResult {
    private final Matrix T; // Quasi-upper triangular matrix
    private final Matrix U; // Unitary/Orthogonal matrix (Schur vectors)
    private final double[] realEigenvalues;
    private final double[] imagEigenvalues;

    public SchurResult(Matrix T, Matrix U, double[] realEigenvalues, double[] imagEigenvalues) {
        this.T = T;
        this.U = U;
        this.realEigenvalues = realEigenvalues;
        this.imagEigenvalues = imagEigenvalues;
    }

    public Matrix getT() { return T; }
    public Matrix getU() { return U; }
    public double[] getRealEigenvalues() { return realEigenvalues; }
    public double[] getImagEigenvalues() { return imagEigenvalues; }

    @Override
    public String toString() {
        return "SchurResult{\n" +
                "  T=" + T.getRowCount() + "x" + T.getColumnCount() + "\n" +
                "  Eigenvalues=" + Arrays.toString(realEigenvalues) + " + " + Arrays.toString(imagEigenvalues) + "i\n" +
                "}";
    }
}