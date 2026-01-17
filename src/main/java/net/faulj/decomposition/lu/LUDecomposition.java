package net.faulj.decomposition.lu;

import net.faulj.core.PermutationVector;
import net.faulj.core.Tolerance;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;

/**
 * Computes the LU decomposition of a square matrix with partial or full pivoting.
 * <p>
 * The LU decomposition factors a square matrix A (with row permutation P) into the form:
 * </p>
 * <pre>
 *   PA = LU
 * </pre>
 * <p>
 * where:
 * </p>
 * <ul>
 *   <li><b>P</b> is a permutation matrix representing row exchanges</li>
 *   <li><b>L</b> is a unit lower triangular matrix (ones on diagonal)</li>
 *   <li><b>U</b> is an upper triangular matrix</li>
 * </ul>
 *
 * <h2>Gaussian Elimination:</h2>
 * <p>
 * LU decomposition is equivalent to Gaussian elimination with pivoting. The factorization
 * captures the elimination multipliers in L and the reduced form in U.
 * </p>
 *
 * <h2>Algorithm:</h2>
 * <p>
 * The algorithm performs elimination with row pivoting:
 * </p>
 * <ol>
 *   <li>For column k = 0 to n-2:</li>
 *   <li>Select pivot row using the chosen {@link PivotPolicy}</li>
 *   <li>Exchange rows if necessary and update permutation P</li>
 *   <li>Compute elimination multipliers: L[i,k] = U[i,k] / U[k,k]</li>
 *   <li>Update remaining submatrix: U[i,j] -= L[i,k] * U[k,j]</li>
 * </ol>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Time complexity:</b> O(2n<sup>3</sup>/3) flops for decomposition</li>
 *   <li><b>Solving Ax=b:</b> O(n<sup>2</sup>) additional flops (forward and back substitution)</li>
 *   <li><b>Space complexity:</b> O(n<sup>2</sup>) for L and U (can be stored in-place)</li>
 *   <li><b>Comparison:</b> Twice as fast as Cholesky for non-symmetric matrices</li>
 * </ul>
 *
 * <h2>Pivoting Strategies:</h2>
 * <ul>
 *   <li><b>Partial pivoting (default):</b> Select largest element in current column below diagonal</li>
 *   <li><b>No pivoting:</b> Use natural order (fails for some matrices, less stable)</li>
 *   <li><b>Complete pivoting:</b> Select largest element in entire remaining submatrix (more stable, more expensive)</li>
 * </ul>
 * <p>
 * Partial pivoting provides a good balance between stability and efficiency for most applications.
 * </p>
 *
 * <h2>Numerical Stability:</h2>
 * <ul>
 *   <li><b>With partial pivoting:</b> Backward stable for most practical matrices</li>
 *   <li><b>Growth factor:</b> Measures instability; typically small but can be O(2<sup>n</sup>) worst case</li>
 *   <li><b>Singular detection:</b> Small pivots indicate near-singularity</li>
 *   <li><b>Condition number:</b> Error amplification bounded by κ(A) * machine epsilon</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Solving linear systems Ax = b</li>
 *   <li>Computing determinants: det(A) = det(P) * &prod; U[i,i]</li>
 *   <li>Matrix inversion: A<sup>-1</sup> = U<sup>-1</sup> L<sup>-1</sup> P</li>
 *   <li>Computing matrix rank</li>
 *   <li>Condition number estimation</li>
 *   <li>Backend for many numerical algorithms</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][] {
 *     {2,  1, -1},
 *     {-3, -1, 2},
 *     {-2, 1,  2}
 * });
 *
 * LUDecomposition lu = new LUDecomposition(PivotPolicy.PARTIAL);
 * LUResult result = lu.decompose(A);
 *
 * Matrix L = result.getL();     // Lower triangular with unit diagonal
 * Matrix U = result.getU();     // Upper triangular
 * PermutationVector P = result.getP();
 *
 * // Solve Ax = b
 * Vector b = new Vector(new double[] {8, -11, -3});
 * Vector x = result.solve(b);   // Forward/backward substitution
 *
 * // Compute determinant
 * double det = result.determinant();
 *
 * // Check singularity
 * if (result.isSingular()) {
 *     System.out.println("Matrix is singular or nearly singular");
 * }
 *
 * // Verify: PA = LU
 * Matrix PA = P.applyTo(A);
 * Matrix reconstructed = L.multiply(U);
 * }</pre>
 *
 * <h2>When to Use LU:</h2>
 * <ul>
 *   <li><b>General square systems:</b> LU is the default choice</li>
 *   <li><b>Multiple right-hand sides:</b> Decompose once, solve many times</li>
 *   <li><b>Determinants and inverses:</b> LU provides these efficiently</li>
 * </ul>
 *
 * <h2>When to Use Alternatives:</h2>
 * <ul>
 *   <li><b>Symmetric positive definite:</b> Use Cholesky (2x faster, more stable)</li>
 *   <li><b>Overdetermined systems:</b> Use QR decomposition for least squares</li>
 *   <li><b>Rank-deficient:</b> Use SVD for reliable rank determination</li>
 *   <li><b>Sparse matrices:</b> Use iterative methods or sparse LU</li>
 * </ul>
 *
 * <h2>Singularity Detection:</h2>
 * <p>
 * The decomposition detects singular or nearly singular matrices by checking for
 * small pivots. The result object provides an {@code isSingular()} method that
 * returns true if any diagonal element of U is effectively zero (below tolerance).
 * </p>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Supports configurable pivoting strategies</li>
 *   <li>Tracks row permutations efficiently with {@link PermutationVector}</li>
 *   <li>Detects singular matrices during decomposition</li>
 *   <li>Can be used for in-place factorization to save memory</li>
 *   <li>Uses {@link Tolerance} for zero detection</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see PivotPolicy
 * @see net.faulj.decomposition.result.LUResult
 * @see net.faulj.solve.LUSolver
 * @see net.faulj.decomposition.cholesky.CholeskyDecomposition
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
