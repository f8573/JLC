package net.faulj.decomposition.lu;

import net.faulj.core.PermutationVector;
import net.faulj.core.Tolerance;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;

/**
 * LU Decomposition with pivoting: PA = LU
 * Primary backend for linear system solving.
 */
public class LUDecomposition {
    
    private final PivotPolicy pivotPolicy;
    
    public LUDecomposition() {
        this(PivotPolicy.PARTIAL);
    }
    
    public LUDecomposition(PivotPolicy pivotPolicy) {
        this.pivotPolicy = pivotPolicy;
    }
    
    /**
     * Computes LU factorization with pivoting.
     * @param A square matrix to factor
     * @return LUResult containing L, U, P, and diagnostics
     */
    public LUResult decompose(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("LU decomposition requires a square matrix");
        }
        
        int n = A.getRowCount();
        Matrix U = A.copy();
        Matrix L = Matrix.Identity(n);
        PermutationVector P = new PermutationVector(n);
        boolean singular = false;
        
        for (int k = 0; k < n - 1; k++) {
            // Select pivot
            int pivotRow = pivotPolicy.selectPivotRow(U, k, k);
            
            // Exchange rows if necessary
            if (pivotRow != k) {
                U.exchangeRows(k, pivotRow);
                P.exchange(k, pivotRow);
                // Also exchange already-computed L entries
                for (int j = 0; j < k; j++) {
                    double temp = L.get(k, j);
                    L.set(k, j, L.get(pivotRow, j));
                    L.set(pivotRow, j, temp);
                }
            }
            
            double pivot = U.get(k, k);
            if (Tolerance.isZero(pivot)) {
                singular = true;
                continue;
            }
            
            // Eliminate below pivot
            for (int i = k + 1; i < n; i++) {
                double factor = U.get(i, k) / pivot;
                L.set(i, k, factor);
                U.set(i, k, 0.0);
                for (int j = k + 1; j < n; j++) {
                    U.set(i, j, U.get(i, j) - factor * U.get(k, j));
                }
            }
        }
        
        // Check last diagonal element
        if (Tolerance.isZero(U.get(n - 1, n - 1))) {
            singular = true;
        }
        
        return new LUResult(L, U, P, singular);
    }
}
