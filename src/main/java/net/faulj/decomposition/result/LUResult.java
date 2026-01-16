package net.faulj.decomposition.result;

import net.faulj.core.PermutationVector;
import net.faulj.matrix.Matrix;

/**
 * Result of LU decomposition: PA = LU
 * Contains L (lower triangular), U (upper triangular), P (permutation).
 */
public class LUResult {
    
    private final Matrix L;
    private final Matrix U;
    private final PermutationVector P;
    private final boolean singular;
    private final double determinant;
    
    public LUResult(Matrix L, Matrix U, PermutationVector P, boolean singular) {
        this.L = L;
        this.U = U;
        this.P = P;
        this.singular = singular;
        this.determinant = computeDeterminant();
    }
    
    private double computeDeterminant() {
        if (singular) return 0.0;
        double det = P.sign();
        for (int i = 0; i < U.getRowCount(); i++) {
            det *= U.get(i, i);
        }
        return det;
    }
    
    public Matrix getL() { return L; }
    public Matrix getU() { return U; }
    public PermutationVector getP() { return P; }
    public boolean isSingular() { return singular; }
    public double getDeterminant() { return determinant; }
    
    /**
     * Reconstructs PA from L and U for verification.
     */
    public Matrix reconstruct() {
        return L.multiply(U);
    }
    
    /**
     * Computes reconstruction error ||PA - LU||_F
     */
    public double getResidualNorm(Matrix A) {
        Matrix PA = permuteRows(A);
        Matrix LU = reconstruct();
        return PA.subtract(LU).frobeniusNorm();
    }
    
    private Matrix permuteRows(Matrix A) {
        Matrix result = A.copy();
        for (int i = 0; i < P.size(); i++) {
            if (P.get(i) != i) {
                // Apply permutation to rows
                for (int j = 0; j < A.getColumnCount(); j++) {
                    result.set(i, j, A.get(P.get(i), j));
                }
            }
        }
        return result;
    }
}
