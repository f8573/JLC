package net.faulj.inverse;

import net.faulj.determinant.LUDeterminant;
import net.faulj.matrix.Matrix;

/**
 * Computes the inverse using the adjugate formula:
 * A^-1 = (1/det(A)) * adj(A)
 * Note: This is computationally inefficient compared to LUInverse.
 */
public class AdjugateInverse {

    public static Matrix compute(Matrix A) {
        double det = LUDeterminant.compute(A);
        if (Math.abs(det) < 1e-12) {
            throw new ArithmeticException("Matrix is singular");
        }

        Matrix adj = Adjugate.compute(A);
        return adj.multiplyScalar(1.0 / det);
    }
}