package net.faulj.core;

import net.faulj.matrix.Matrix;

import java.util.Arrays;

/**
 * Represents a permutation vector for tracking pivoting operations in matrix factorizations.
 * <p>
 * A permutation vector encodes row or column exchanges performed during Gaussian elimination,
 * LU decomposition, QR factorization, and other matrix algorithms that require pivoting for
 * numerical stability. This class efficiently tracks the cumulative effect of swaps and provides
 * utilities for computing permutation signs and applying permutations to vectors and matrices.
 * </p>
 *
 * <h2>Mathematical Representation:</h2>
 * <p>
 * A permutation π of {0, 1, ..., n-1} is represented as an array where π[i] = j indicates that
 * row/column i is mapped to position j. Permutation matrices P can be applied as:
 * </p>
 * <pre>
 * PA = [A[π[0]], A[π[1]], ..., A[π[n-1]]]^T
 * </pre>
 *
 * <h2>Key Operations:</h2>
 * <ul>
 *   <li><b>exchange(i, j):</b> Swaps positions i and j in the permutation</li>
 *   <li><b>sign():</b> Returns +1 for even permutations, -1 for odd permutations</li>
 *   <li><b>apply(vector):</b> Applies the permutation to a vector</li>
 *   <li><b>inverse():</b> Computes the inverse permutation</li>
 * </ul>
 *
 * <h2>Permutation Sign:</h2>
 * <p>
 * The sign (parity) of a permutation is critical for computing determinants:
 * </p>
 * <pre>
 * det(PA) = sign(P) · det(A)
 * </pre>
 * <p>
 * The sign is +1 if the permutation is expressible as an even number of transpositions, and -1
 * if odd. This class efficiently tracks the sign by counting exchanges.
 * </p>
 *
 * <h2>Example Usage in LU Decomposition:</h2>
 * <pre>{@code
 * PermutationVector perm = new PermutationVector(n);
 * for (int k = 0; k < n; k++) {
 *     int pivotRow = findPivot(A, k);
 *     perm.exchange(k, pivotRow);
 *     swapRows(A, k, pivotRow);
 * }
 * int sign = perm.sign(); // For determinant computation
 * double det = sign * productOfDiagonal(U);
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Partial pivoting in LU decomposition</li>
 *   <li>Column pivoting in QR factorization</li>
 *   <li>Complete pivoting in Gaussian elimination</li>
 *   <li>Determinant sign computation</li>
 *   <li>Permuting solutions back to original ordering</li>
 * </ul>
 *
 * <h2>Complexity:</h2>
 * <ul>
 *   <li><b>Construction:</b> O(n)</li>
 *   <li><b>Exchange:</b> O(1)</li>
 *   <li><b>Sign:</b> O(1)</li>
 *   <li><b>Apply to vector:</b> O(n)</li>
 * </ul>
 *
 * <h2>Storage:</h2>
 * <p>
 * Uses O(n) space to store the permutation array plus a single integer for the exchange count.
 * This is more efficient than storing a full n×n permutation matrix.
 * </p>
 *
 * <h2>Thread Safety:</h2>
 * <p>
 * This class is not thread-safe. External synchronization is required if a PermutationVector
 * is shared across threads.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.lu.LUDecomposition
 * @see net.faulj.decomposition.qr.HouseholderQR
 * @see net.faulj.determinant.LUDeterminant
 */
public class PermutationVector {
    private final int[] perm;
    private int exchangeCount;
    /**
     * Create an identity permutation of the given size.
     *
     * @param size permutation size
     */
    public PermutationVector(int size) {
        this.perm = new int[size];
        for (int i = 0; i < size; i++) {
            perm[i] = i;
        }
        this.exchangeCount = 0;
    }
    /**
     * Swap two indices in the permutation.
     *
     * @param i first index
     * @param j second index
     */
    public void exchange(int i, int j) {
        if (i != j) {
            int temp = perm[i];
            perm[i] = perm[j];
            perm[j] = temp;
            exchangeCount++;
        }
    }
    /**
     * Get the mapped index for a position.
     *
     * @param index position
     * @return mapped index
     */
    public int get(int index) {
        return perm[index];
    }
    /**
     * Get permutation size.
     *
     * @return size
     */
    public int size() {
        return perm.length;
    }
    /**
     * Get number of exchanges performed.
     *
     * @return exchange count
     */
    public int getExchangeCount() {
        return exchangeCount;
    }
    /**
     * Returns the sign of the permutation: +1 for even, -1 for odd exchanges.
     */
    public int sign() {
        return (exchangeCount % 2 == 0) ? 1 : -1;
    }
    /**
     * Copy permutation to an array.
     *
     * @return permutation array
     */
    public int[] toArray() {
        return Arrays.copyOf(perm, perm.length);
    }
    /**
     * Create a deep copy of this permutation vector.
     *
     * @return copied permutation vector
     */
    public PermutationVector copy() {
        PermutationVector p = new PermutationVector(perm.length);
        System.arraycopy(this.perm, 0, p.perm, 0, perm.length);
        p.exchangeCount = this.exchangeCount;
        return p;
    }

    /**
     * Builds and returns the permutation matrix P corresponding to this permutation vector.
     * Row i of P has a 1 at column perm[i], so that applying P to a matrix A yields rows
     * (PA)[i,*] = A[perm[i],*], matching the usage in LUResult.
     *
     * @return an n-by-n permutation Matrix P
     */
    public Matrix asMatrix() {
        int n = perm.length;
        Matrix P = new Matrix(n, n);
        // set explicit entries to be clear; assume Matrix initializes to zeros but set anyway
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                P.set(i, j, 0.0);
            }
            P.set(i, perm[i], 1.0);
        }
        return P;
    }
}
