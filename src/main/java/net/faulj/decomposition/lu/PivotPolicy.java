package net.faulj.decomposition.lu;

import net.faulj.matrix.Matrix;

/**
 * Defines pivoting strategies for LU decomposition to ensure numerical stability and avoid division by zero.
 * <p>
 * Pivoting is the process of rearranging rows (and sometimes columns) to place larger elements
 * on the diagonal during Gaussian elimination. This improves numerical stability and prevents
 * algorithm failure when encountering zero or small pivot elements.
 * </p>
 *
 * <h2>Why Pivoting Matters:</h2>
 * <p>
 * Without pivoting, LU decomposition can:
 * </p>
 * <ul>
 *   <li>Fail due to division by zero when encountering a zero diagonal element</li>
 *   <li>Produce inaccurate results when dividing by very small numbers (loss of precision)</li>
 *   <li>Amplify rounding errors through large intermediate values (growth factor)</li>
 * </ul>
 *
 * <h2>Pivoting Strategies:</h2>
 *
 * <h3>1. Partial Pivoting (PARTIAL):</h3>
 * <ul>
 *   <li><b>Strategy:</b> Select the row with largest absolute value in current column</li>
 *   <li><b>Cost:</b> O(n) comparisons per elimination step</li>
 *   <li><b>Stability:</b> Excellent for most practical matrices</li>
 *   <li><b>Growth factor:</b> Typically small (usually &lt; 10)</li>
 *   <li><b>Worst case:</b> O(2<sup>n</sup>) growth factor (extremely rare in practice)</li>
 *   <li><b>Recommendation:</b> Default choice for general matrices</li>
 * </ul>
 *
 * <h3>2. No Pivoting (NONE):</h3>
 * <ul>
 *   <li><b>Strategy:</b> Use natural row order, no exchanges</li>
 *   <li><b>Cost:</b> Zero overhead</li>
 *   <li><b>Stability:</b> Poor, can fail or produce inaccurate results</li>
 *   <li><b>Use cases:</b> Diagonally dominant matrices, structured matrices, testing</li>
 *   <li><b>Caution:</b> Only use when matrix properties guarantee stability</li>
 * </ul>
 *
 * <h3>3. Complete Pivoting (not implemented):</h3>
 * <ul>
 *   <li><b>Strategy:</b> Select largest element in entire remaining submatrix</li>
 *   <li><b>Cost:</b> O(n<sup>2</sup>) comparisons per elimination step (prohibitive)</li>
 *   <li><b>Stability:</b> Maximum stability, growth factor bounded by ~n</li>
 *   <li><b>Use cases:</b> Extremely ill-conditioned matrices</li>
 *   <li><b>Note:</b> Rarely used due to high computational cost</li>
 * </ul>
 *
 * <h2>Mathematical Background:</h2>
 * <p>
 * Given the elimination step at column k:
 * </p>
 * <pre>
 * For i = k+1 to n:
 *   multiplier = A[i,k] / A[k,k]  ← Division by pivot
 *   A[i,:] = A[i,:] - multiplier * A[k,:]
 * </pre>
 * <p>
 * If A[k,k] (the pivot) is zero or small, this operation causes:
 * </p>
 * <ul>
 *   <li>Division by zero (algorithm failure)</li>
 *   <li>Large multipliers (error amplification)</li>
 *   <li>Loss of significant digits</li>
 * </ul>
 * <p>
 * Pivoting selects a different row with a larger pivot to avoid these issues.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Use partial pivoting (recommended default)
 * LUDecomposition lu1 = new LUDecomposition(PivotPolicy.PARTIAL);
 * LUResult result1 = lu1.decompose(A);
 *
 * // Use no pivoting for diagonally dominant matrix
 * Matrix diagonallyDominant = ...; // |A[i,i]| > sum(|A[i,j]|) for j != i
 * LUDecomposition lu2 = new LUDecomposition(PivotPolicy.NONE);
 * LUResult result2 = lu2.decompose(diagonallyDominant);
 *
 * // Custom pivoting strategy
 * PivotPolicy custom = (A, k, startRow) -> {
 *     // Custom logic to select pivot row
 *     // Return row index with desired properties
 *     return computeCustomPivotRow(A, k, startRow);
 * };
 * LUDecomposition lu3 = new LUDecomposition(custom);
 * }</pre>
 *
 * <h2>Diagonally Dominant Matrices:</h2>
 * <p>
 * A matrix is strictly diagonally dominant if for each row:
 * </p>
 * <pre>
 *   |A[i,i]| &gt; &Sigma;<sub>j≠i</sub> |A[i,j]|
 * </pre>
 * <p>
 * Such matrices can safely use {@code PivotPolicy.NONE} because:
 * </p>
 * <ul>
 *   <li>Diagonal elements are guaranteed to be non-zero</li>
 *   <li>Elimination multipliers are bounded by 1</li>
 *   <li>No growth in matrix elements occurs</li>
 * </ul>
 *
 * <h2>Growth Factor:</h2>
 * <p>
 * The growth factor measures element growth during elimination:
 * </p>
 * <pre>
 *   ρ = max<sub>i,j,k</sub> |a<sub>ij</sub><sup>(k)</sup>| / max<sub>i,j</sub> |a<sub>ij</sub>|
 * </pre>
 * <p>
 * Bounds:
 * </p>
 * <ul>
 *   <li><b>No pivoting:</b> Unbounded (can be arbitrarily large)</li>
 *   <li><b>Partial pivoting:</b> ≤ 2<sup>n-1</sup> (theoretical worst case, rarely achieved)</li>
 *   <li><b>Complete pivoting:</b> ≤ n<sup>1/2</sup> * (2·3<sup>1/2</sup>·4<sup>1/3</sup>···n<sup>1/(n-1)</sup>)<sup>1/2</sup> ≈ n</li>
 * </ul>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Implemented as a functional interface for easy customization</li>
 *   <li>Provides predefined strategies as static instances</li>
 *   <li>Can be extended with lambda expressions or method references</li>
 *   <li>Zero overhead when NONE is used</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see LUDecomposition
 * @see net.faulj.decomposition.result.LUResult
 */
public interface PivotPolicy {
    
    /**
     * Selects the pivot row for column k.
     * @param A the matrix being factored
     * @param k current column
     * @param startRow first candidate row
     * @return row index of selected pivot
     */
    int selectPivotRow(Matrix A, int k, int startRow);
    
    /**
     * Partial pivoting: select largest magnitude in column.
     */
    PivotPolicy PARTIAL = (A, k, startRow) -> {
        int n = A.getRowCount();
        int maxRow = startRow;
        double maxAbs = Math.abs(A.get(startRow, k));
        for (int r = startRow + 1; r < n; r++) {
            double abs = Math.abs(A.get(r, k));
            if (abs > maxAbs) {
                maxAbs = abs;
                maxRow = r;
            }
        }
        return maxRow;
    };
    
    /**
     * No pivoting (use natural order).
     */
    PivotPolicy NONE = (A, k, startRow) -> startRow;
}
