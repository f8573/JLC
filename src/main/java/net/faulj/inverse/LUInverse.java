package net.faulj.inverse;

import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;
import net.faulj.solve.LUSolver;
import net.faulj.vector.Vector;

/**
 * Computes the matrix inverse using LU Decomposition.
 * Solves AX = I column by column.
 */
public class LUInverse {

    public static Matrix compute(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Matrix inversion requires a square matrix");
        }

        int n = A.getRowCount();
        LUDecomposition decomp = new LUDecomposition();
        LUResult lu = decomp.decompose(A);

        if (lu.isSingular()) {
            throw new ArithmeticException("Matrix is singular (determinant is 0) and cannot be inverted");
        }

        LUSolver solver = new LUSolver();
        Vector[] invColumns = new Vector[n];

        // Solve Ax_i = e_i for each column i
        for (int i = 0; i < n; i++) {
            // Create standard basis vector e_i
            double[] eData = new double[n];
            eData[i] = 1.0;
            Vector e = new Vector(eData);

            // The solution x is the ith column of the inverse
            invColumns[i] = solver.solve(lu, e);
        }

        return new Matrix(invColumns);
    }
}