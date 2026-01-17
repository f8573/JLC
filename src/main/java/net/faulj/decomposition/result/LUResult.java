package net.faulj.decomposition.result;

import net.faulj.core.PermutationVector;
import net.faulj.matrix.Matrix;

/**
 * Encapsulates the result of LU decomposition with partial pivoting.
 * <p>
 * This class represents the factorization PA = LU where:
 * </p>
 * <ul>
 *   <li><b>P</b> - Permutation matrix representing row exchanges</li>
 *   <li><b>L</b> - Unit lower triangular matrix (ones on diagonal)</li>
 *   <li><b>U</b> - Upper triangular matrix</li>
 *   <li><b>A</b> - Original matrix being factored</li>
 * </ul>
 *
 * <h2>Matrix Properties:</h2>
 * <ul>
 *   <li><b>Lower triangular L:</b> L[i,j] = 0 for j &gt; i, L[i,i] = 1</li>
 *   <li><b>Upper triangular U:</b> U[i,j] = 0 for i &gt; j</li>
 *   <li><b>Permutation P:</b> Represented efficiently as {@link PermutationVector}</li>
 * </ul>
 *
 * <h2>Singularity Detection:</h2>
 * <p>
 * The result tracks whether the matrix is singular (or numerically singular) by checking
 * if any diagonal element of U is effectively zero. A singular matrix:
 * </p>
 * <ul>
 *   <li>Has determinant zero</li>
 *   <li>Cannot be inverted</li>
 *   <li>Represents a rank-deficient linear system</li>
 * </ul>
 *
 * <h2>Determinant Computation:</h2>
 * <p>
 * The determinant is computed efficiently from the factorization:
 * </p>
 * <pre>
 *   det(A) = det(P) × ∏<sub>i</sub> U[i,i]
 * </pre>
 * <p>
 * where det(P) is +1 or -1 depending on whether P represents an even or odd number of row swaps.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * LUDecomposition lu = new LUDecomposition(PivotPolicy.PARTIAL);
 * LUResult result = lu.decompose(A);
 *
 * // Access factors
 * Matrix L = result.getL();
 * Matrix U = result.getU();
 * PermutationVector P = result.getP();
 *
 * // Check singularity
 * if (result.isSingular()) {
 *     System.out.println("Matrix is singular!");
 * }
 *
 * // Get determinant (O(1) after decomposition)
 * double det = result.getDeterminant();
 *
 * // Verify factorization: PA = LU
 * Matrix PA = P.applyTo(A);
 * Matrix LU = result.reconstruct();
 * double error = result.getResidualNorm(A);
 * System.out.println("Factorization error: " + error);
 * }</pre>
 *
 * <h2>Provided Operations:</h2>
 * <ul>
 *   <li><b>Factor access:</b> Get L, U, and P matrices</li>
 *   <li><b>Singularity check:</b> Detect rank deficiency</li>
 *   <li><b>Determinant:</b> O(1) retrieval after factorization</li>
 *   <li><b>Reconstruction:</b> Verify PA = LU for testing</li>
 *   <li><b>Residual norm:</b> Measure factorization accuracy</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Solving linear systems (via forward/back substitution)</li>
 *   <li>Computing matrix determinants</li>
 *   <li>Matrix inversion</li>
 *   <li>Rank determination</li>
 *   <li>Condition number estimation</li>
 * </ul>
 *
 * <h2>Immutability:</h2>
 * <p>
 * This class is immutable - all fields are final and matrices are not defensively copied.
 * Users should not modify returned matrices if factorization integrity is required.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.lu.LUDecomposition
 * @see net.faulj.core.PermutationVector
 * @see net.faulj.solve.LUSolver
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
