package net.faulj.solve;

import net.faulj.determinant.LUDeterminant;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Solves linear systems Ax = b using Cramer's Rule.
 * xi = det(Ai) / det(A)
 * Only practical for very small matrices.
 */
public class CramerSolver {

    public Vector solve(Matrix A, Vector b) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Cramer's rule requires a square matrix");
        }
        if (b.dimension() != A.getRowCount()) {
            throw new IllegalArgumentException("Dimension mismatch");
        }

        double detA = LUDeterminant.compute(A);
        if (Math.abs(detA) < 1e-12) {
            throw new ArithmeticException("Matrix is singular");
        }

        int n = A.getRowCount();
        double[] x = new double[n];

        for (int i = 0; i < n; i++) {
            // Construct Ai: Matrix A with column i replaced by b
            Matrix Ai = replaceColumn(A, i, b);
            x[i] = LUDeterminant.compute(Ai) / detA;
        }

        return new Vector(x);
    }

    private Matrix replaceColumn(Matrix A, int colIndex, Vector b) {
        Vector[] originalData = A.getData();
        Vector[] newData = new Vector[originalData.length];

        for (int i = 0; i < originalData.length; i++) {
            if (i == colIndex) {
                newData[i] = b.copy();
            } else {
                newData[i] = originalData[i].copy();
            }
        }
        return new Matrix(newData);
    }
}