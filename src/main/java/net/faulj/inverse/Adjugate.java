package net.faulj.inverse;

import net.faulj.determinant.LUDeterminant;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Computes the classical adjoint (adjugate) of a matrix.
 * adj(A) is the transpose of the cofactor matrix.
 */
public class Adjugate {

    public static Matrix compute(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Adjugate requires a square matrix");
        }

        int n = A.getRowCount();
        Vector[] cols = new Vector[n];

        for (int j = 0; j < n; j++) {
            double[] colData = new double[n];
            for (int i = 0; i < n; i++) {
                // Cofactor C_ij = (-1)^(i+j) * M_ij
                // Adjugate is the transpose, so entry (i,j) of Result is Cofactor(j,i)
                // adj(A)_ij = C_ji

                Matrix minor = A.minor(j, i); // Minor of element at row j, col i
                double det = LUDeterminant.compute(minor);

                double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
                colData[i] = sign * det;
            }
            cols[j] = new Vector(colData);
        }

        return new Matrix(cols);
    }
}