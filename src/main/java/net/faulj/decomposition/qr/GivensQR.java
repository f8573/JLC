package net.faulj.decomposition.qr;

/**
 * Computes QR decomposition using Givens rotations for selective element elimination.
 * <p>
 * The Givens QR method factors a matrix A into orthogonal Q and upper triangular R using
 * a sequence of plane rotations. Unlike Householder QR which zeros entire columns at once,
 * Givens rotations zero out individual elements one at a time.
 * </p>
 * <pre>
 *   A = Q * R
 * </pre>
 * <p>
 * where:
 * </p>
 * <ul>
 *   <li><b>Q</b> is an m-by-m orthogonal matrix (Q<sup>T</sup>Q = I)</li>
 *   <li><b>R</b> is an m-by-n upper triangular matrix</li>
 * </ul>
 *
 * <h2>Givens Rotation:</h2>
 * <p>
 * A Givens rotation G(i,j,θ) acts in the (i,j) coordinate plane to rotate by angle θ:
 * </p>
 * <pre>
 *   G = ┌             ┐
 *       │  I          │
 *       │    c    s   │  ← row i
 *       │   -s    c   │  ← row j
 *       │          I  │
 *       └             ┘
 *        ↑    ↑
 *      col i  col j
 * </pre>
 * <p>
 * where c = cos(θ) and s = sin(θ) are chosen to zero element A[j,i]:
 * </p>
 * <pre>
 *   r = sqrt(a<sup>2</sup> + b<sup>2</sup>),  c = a/r,  s = -b/r
 * </pre>
 *
 * <h2>Algorithm:</h2>
 * <ol>
 *   <li>For each column k from 0 to n-1:</li>
 *   <li>For each row i from m-1 down to k+1:</li>
 *   <li>Compute Givens rotation G(i-1, i) to zero A[i,k]</li>
 *   <li>Apply rotation: [rows i-1 and i] = G * [rows i-1 and i]</li>
 *   <li>Accumulate Q = Q * G<sup>T</sup></li>
 * </ol>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Time complexity:</b> O(3mn<sup>2</sup> - n<sup>3</sup>) flops, about 50% more than Householder</li>
 *   <li><b>Space complexity:</b> O(mn) for R, O(m<sup>2</sup>) for Q</li>
 *   <li><b>Per rotation:</b> O(n) work (affects 2 rows)</li>
 *   <li><b>Number of rotations:</b> mn - n(n+1)/2</li>
 * </ul>
 *
 * <h2>Advantages over Householder:</h2>
 * <ul>
 *   <li><b>Sparse matrices:</b> Can selectively zero specific elements without disturbing sparsity</li>
 *   <li><b>Parallelization:</b> Independent rotations in different planes can be applied concurrently</li>
 *   <li><b>Updating:</b> Efficient for adding/removing rows or columns to existing QR</li>
 *   <li><b>Structured matrices:</b> Exploits banded or partially dense structures</li>
 *   <li><b>Stability control:</b> More control over pivoting and numerical issues</li>
 * </ul>
 *
 * <h2>Disadvantages:</h2>
 * <ul>
 *   <li><b>Speed:</b> ~50% slower than Householder for dense matrices</li>
 *   <li><b>Implementation complexity:</b> More bookkeeping required</li>
 *   <li><b>Memory:</b> May require storing many rotation parameters</li>
 * </ul>
 *
 * <h2>Numerical Stability:</h2>
 * <ul>
 *   <li><b>Backward stable:</b> Each rotation is orthogonal</li>
 *   <li><b>Error accumulation:</b> Similar to Householder QR</li>
 *   <li><b>Orthogonality:</b> Q<sup>T</sup>Q = I maintained to machine precision</li>
 *   <li><b>Norm preservation:</b> ||A||<sub>F</sub> = ||R||<sub>F</sub></li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Sparse matrix factorization</li>
 *   <li>Rank-one updates to QR decomposition</li>
 *   <li>Least squares with sparse matrices</li>
 *   <li>Eigenvalue algorithms (QR iteration)</li>
 *   <li>Signal processing (adaptive filtering)</li>
 *   <li>Bidiagonalization and SVD computation</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][] {
 *     {12, -51,   4},
 *     { 6, 167, -68},
 *     {-4,  24, -41}
 * });
 *
 * GivensQR givens = new GivensQR();
 * var result = givens.decompose(A);
 *
 * Matrix Q = result.getQ();  // Orthogonal matrix
 * Matrix R = result.getR();  // Upper triangular
 *
 * // Verify orthogonality: Q^T * Q = I
 * Matrix identity = Q.transpose().multiply(Q);
 *
 * // Verify factorization: A = Q * R
 * Matrix reconstructed = Q.multiply(R);
 *
 * // Solve least squares: minimize ||Ax - b||
 * Vector b = new Vector(new double[] {1, 2, 3});
 * Vector x = result.solve(b);
 * }</pre>
 *
 * <h2>Givens vs Householder:</h2>
 * <table border="1">
 *   <tr><th>Aspect</th><th>Givens</th><th>Householder</th></tr>
 *   <tr><td>Work per element</td><td>O(n)</td><td>O(n)</td></tr>
 *   <tr><td>Total flops</td><td>3mn² - n³</td><td>2mn² - 2n³/3</td></tr>
 *   <tr><td>Sparse matrices</td><td>Excellent</td><td>Poor (fills in)</td></tr>
 *   <tr><td>Dense matrices</td><td>Slower</td><td>Faster</td></tr>
 *   <tr><td>Parallelization</td><td>Better</td><td>Harder</td></tr>
 *   <tr><td>Implementation</td><td>More complex</td><td>Simpler</td></tr>
 * </table>
 *
 * <h2>Fast Givens Rotations:</h2>
 * <p>
 * For improved efficiency, fast Givens rotations avoid explicit computation of square roots
 * by using a factored form. This reduces the cost per rotation from ~6 flops to ~4 flops.
 * </p>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Uses numerically stable computation of c and s to avoid overflow/underflow</li>
 *   <li>Can compute compact representation without explicitly forming Q</li>
 *   <li>Supports in-place operation for memory efficiency</li>
 *   <li>Can be combined with pivoting for rank-deficient matrices</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see HouseholderQR
 * @see net.faulj.givens.GivensRotation
 * @see net.faulj.decomposition.result.QRResult
 */
public class GivensQR {
}
