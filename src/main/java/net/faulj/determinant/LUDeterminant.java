package net.faulj.determinant;

import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;

/**
 * Computes the determinant of a matrix using LU Decomposition.
 * This is the preferred method for dense matrices (O(n^3)).
 */
public class LUDeterminant {

    /**
     * Computes the determinant of the given matrix.
     * @param A the matrix
     * @return the determinant
     */
    public static double compute(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Determinant requires a square matrix");
        }
        // LUResult already computes det(A) = det(P^-1) * det(L) * det(U)
        // where det(L)=1, det(P)=+/-1, det(U)=product of diagonal
        return new LUDecomposition().decompose(A).getDeterminant();
    }
}