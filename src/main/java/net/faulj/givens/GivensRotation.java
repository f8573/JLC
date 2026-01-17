package net.faulj.givens;

import net.faulj.matrix.Matrix;

/**
 * Represents a Givens rotation and provides methods for its computation and application.
 * <p>
 * A Givens rotation is a linear transformation represented by an orthogonal matrix <b>G</b>
 * that rotates a vector in the plane spanned by two coordinates axes. It is primarily used
 * to introduce zeros into a matrix (annihilation) numerically stably.
 * </p>
 *
 * <h2>Matrix Representation:</h2>
 * <p>
 * The matrix representation G(i, k, θ) differs from the identity matrix only in the
 * intersection of the i-th and k-th rows and columns:
 * </p>
 * <pre>
 * ...  i      k  ...
 * i [ ...  c      s  ... ]
 * k [ ... -s      c  ... ]
 * </pre>
 * <p>
 * Where:
 * </p>
 * <ul>
 * <li><b>c</b> = cos(θ)</li>
 * <li><b>s</b> = sin(θ)</li>
 * <li>c² + s² = 1</li>
 * </ul>
 *
 * <h2>Numerical Computation:</h2>
 * <p>
 * To safely compute c and s such that the second component of a vector [a, b]<sup>T</sup>
 * becomes zero (i.e., mapping [a, b] to [r, 0]), this class uses stable algorithms
 * to avoid overflow or underflow (e.g., using <code>Math.hypot</code>).
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Eliminate b in the vector [a, b]
 * double a = 1.0;
 * double b = 2.0;
 * GivensRotation rot = GivensRotation.compute(a, b);
 *
 * // Apply to a matrix A to zero out A[k, col] using A[i, col]
 * Matrix A = ...;
 * rot.applyLeft(A, i, k, 0, A.getColumnCount() - 1);
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li><b>QR Decomposition:</b> Reducing a matrix to upper triangular form.</li>
 * <li><b>Hessenberg Reduction:</b> Reducing a matrix to Hessenberg form.</li>
 * <li><b>Tridiagonalization:</b> For symmetric eigenvalue problems.</li>
 * <li><b>Bidiagonalization:</b> For SVD computation.</li>
 * <li><b>QR Updates:</b> Updating decompositions after rank-1 modifications.</li>
 * </ul>
 *
 * <h2>Performance Notes:</h2>
 * <p>
 * Givens rotations are more selective than Householder reflections, affecting only two rows
 * or columns at a time. This makes them ideal for parallelization or working with sparse
 * matrices/structures, although they require 4 FLOPS per element applied (vs 2 for Householder).
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.qr.GivensQR
 * @see net.faulj.decomposition.hessenberg.HessenbergReduction
 */
public class GivensRotation {
    /**
     * The cosine component of the rotation.
     */
    public final double c;

    /**
     * The sine component of the rotation.
     */
    public final double s;

    /**
     * The norm of the vector being rotated (r = sqrt(a² + b²)).
     * <p>
     * This field stores the result of the rotation on the target element,
     * often avoiding re-computation during decomposition steps.
     * </p>
     */
    public final double r;

    /**
     * Constructs a Givens rotation with specified components.
     *
     * @param c The cosine component.
     * @param s The sine component.
     * @param r The computed norm (result value).
     */
    public GivensRotation(double c, double s, double r) {
        this.c = c;
        this.s = s;
        this.r = r;
    }

    /**
     * Computes a Givens rotation that annihilates the second component of a vector.
     * <p>
     * Given a vector <code>x = [a, b]<sup>T</sup></code>, this method calculates
     * <code>c</code> and <code>s</code> such that:
     * </p>
     * <pre>
     * [ c  s ] [ a ]   [ r ]
     * [-s  c ] [ b ] = [ 0 ]
     * </pre>
     * <p>
     * The computation is numerically stable, handling cases where a or b are zero
     * or very large/small.
     * </p>
     *
     * @param a The first component of the vector (pivot).
     * @param b The second component of the vector (to be annihilated).
     * @return A {@link GivensRotation} object containing c, s, and r.
     */
    public static GivensRotation compute(double a, double b) {
        double c, s, r;
        if (b == 0) {
            c = 1.0;
            s = 0.0;
            r = a;
        } else if (a == 0) {
            c = 0.0;
            s = 1.0;
            r = b;
        } else {
            // Re-normalize for numerical safety if needed, though the above is stable.
            // Simplified consistent computation:
            double h = Math.hypot(a, b);
            r = h;
            c = a / h;
            s = -b / h;
        }
        return new GivensRotation(c, s, r);
    }

    /**
     * Applies the rotation to rows <code>i</code> and <code>k</code> of matrix <b>A</b> (from the left).
     * <p>
     * Mathematically performs <b>A' = G<sup>T</sup> * A</b>. This affects only rows i and k.
     * </p>
     * <pre>
     * A[i, j] =  c * A[i, j] - s * A[k, j]
     * A[k, j] =  s * A[i, j] + c * A[k, j]
     * </pre>
     *
     * @param A The matrix to transform.
     * @param i The index of the first row (pivot row).
     * @param k The index of the second row (target row).
     * @param colStart The starting column index (inclusive).
     * @param colEnd The ending column index (inclusive).
     */
    public void applyLeft(Matrix A, int i, int k, int colStart, int colEnd) {
        for (int col = colStart; col <= colEnd; col++) {
            double valI = A.get(i, col);
            double valK = A.get(k, col);
            A.set(i, col, c * valI - s * valK);
            A.set(k, col, s * valI + c * valK);
        }
    }

    /**
     * Applies the rotation to columns <code>i</code> and <code>k</code> of matrix <b>A</b> (from the right).
     * <p>
     * Mathematically performs <b>A' = A * G</b>. This affects only columns i and k.
     * </p>
     * <pre>
     * A[j, i] =  c * A[j, i] - s * A[j, k]
     * A[j, k] =  s * A[j, i] + c * A[j, k]
     * </pre>
     *
     * @param A The matrix to transform.
     * @param i The index of the first column.
     * @param k The index of the second column.
     * @param rowStart The starting row index (inclusive).
     * @param rowEnd The ending row index (inclusive).
     */
    public void applyRight(Matrix A, int i, int k, int rowStart, int rowEnd) {
        for (int row = rowStart; row <= rowEnd; row++) {
            double valI = A.get(row, i);
            double valK = A.get(row, k);
            A.set(row, i, c * valI - s * valK);
            A.set(row, k, s * valI + c * valK);
        }
    }

    /**
     * Applies the rotation to a specific column index at rows <code>i</code> and <code>k</code>.
     * <p>
     * This is a specialized version of {@link #applyLeft} for a single column, useful
     * when updating vectors or specific matrix entries.
     * </p>
     *
     * @param A The matrix or vector to transform.
     * @param colIndex The column index to operate on.
     * @param i The index of the first row.
     * @param k The index of the second row.
     */
    public void applyColumn(Matrix A, int colIndex, int i, int k) {
        double valI = A.get(i, colIndex);
        double valK = A.get(k, colIndex);
        A.set(i, colIndex, c * valI - s * valK);
        A.set(k, colIndex, s * valI + c * valK);
    }
}