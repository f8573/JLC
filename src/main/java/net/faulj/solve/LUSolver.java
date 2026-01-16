package net.faulj.solve;

import net.faulj.core.PermutationVector;
import net.faulj.core.Tolerance;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Solves linear systems Ax = b via LU decomposition.
 * Uses forward and back substitution.
 */
public class LUSolver {
    
    private final LUDecomposition luDecomposition;
    
    public LUSolver() {
        this.luDecomposition = new LUDecomposition();
    }
    
    public LUSolver(LUDecomposition luDecomposition) {
        this.luDecomposition = luDecomposition;
    }
    
    /**
     * Solves Ax = b using LU decomposition.
     * PA = LU => Ax = b => PAx = Pb => LUx = Pb
     * Let Ux = y, solve Ly = Pb (forward), then Ux = y (backward).
     */
    public Vector solve(Matrix A, Vector b) {
        if (b.dimension() != A.getRowCount()) {
            throw new IllegalArgumentException("Dimension mismatch: b has " + b.dimension() 
                    + " elements but A has " + A.getRowCount() + " rows");
        }
        
        LUResult lu = luDecomposition.decompose(A);
        
        if (lu.isSingular()) {
            throw new ArithmeticException("Matrix is singular; no unique solution exists");
        }
        
        return solve(lu, b);
    }
    
    /**
     * Solves using pre-computed LU factorization.
     */
    public Vector solve(LUResult lu, Vector b) {
        Matrix L = lu.getL();
        Matrix U = lu.getU();
        PermutationVector P = lu.getP();
        int n = L.getRowCount();
        
        // Apply permutation to b: Pb
        double[] pb = new double[n];
        for (int i = 0; i < n; i++) {
            pb[i] = b.get(P.get(i));
        }
        
        // Forward substitution: Ly = Pb
        double[] y = forwardSubstitution(L, pb);
        
        // Back substitution: Ux = y
        double[] x = backSubstitution(U, y);
        
        return new Vector(x);
    }
    
    /**
     * Forward substitution for lower triangular L.
     * L is assumed to have 1s on diagonal.
     */
    private double[] forwardSubstitution(Matrix L, double[] b) {
        int n = L.getRowCount();
        double[] y = new double[n];
        
        for (int i = 0; i < n; i++) {
            double sum = b[i];
            for (int j = 0; j < i; j++) {
                sum -= L.get(i, j) * y[j];
            }
            y[i] = sum; // L[i,i] = 1
        }
        
        return y;
    }
    
    /**
     * Back substitution for upper triangular U.
     */
    private double[] backSubstitution(Matrix U, double[] y) {
        int n = U.getRowCount();
        double[] x = new double[n];
        
        for (int i = n - 1; i >= 0; i--) {
            double sum = y[i];
            for (int j = i + 1; j < n; j++) {
                sum -= U.get(i, j) * x[j];
            }
            double diag = U.get(i, i);
            if (Tolerance.isZero(diag)) {
                throw new ArithmeticException("Zero diagonal encountered in back substitution");
            }
            x[i] = sum / diag;
        }
        
        return x;
    }
}
