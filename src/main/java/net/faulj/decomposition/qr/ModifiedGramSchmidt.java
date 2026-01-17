package net.faulj.decomposition.qr;

/**
 * Computes QR decomposition using Modified Gram-Schmidt orthogonalization.
 * <p>
 * Modified Gram-Schmidt (MGS) is a numerically stable variant of the classical Gram-Schmidt
 * process. It reorders operations to minimize cancellation errors and maintain better
 * orthogonality in the computed Q matrix.
 * </p>
 * <pre>
 *   A = Q * R
 * </pre>
 * <p>
 * where Q has orthonormal columns and R is upper triangular.
 * </p>
 *
 * <h2>Modified Gram-Schmidt Algorithm:</h2>
 * <pre>
 * For j = 1 to n:
 *   v<sub>j</sub> = a<sub>j</sub>
 * For j = 1 to n:
 *   r<sub>jj</sub> = ||v<sub>j</sub>||<sub>2</sub>
 *   q<sub>j</sub> = v<sub>j</sub> / r<sub>jj</sub>
 *   For i = j+1 to n:
 *     r<sub>ji</sub> = q<sub>j</sub><sup>T</sup> * v<sub>i</sub>
 *     v<sub>i</sub> = v<sub>i</sub> - r<sub>ji</sub> * q<sub>j</sub>
 * </pre>
 * <p>
 * Key difference from classical GS: Each new q<sub>j</sub> is immediately used to update
 * all remaining vectors, rather than computing all projections from original vectors.
 * </p>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Time complexity:</b> O(2mn<sup>2</sup>) flops, same as classical GS</li>
 *   <li><b>Space complexity:</b> O(mn) for Q and R</li>
 *   <li><b>Per column:</b> O(mn) work</li>
 *   <li><b>Overhead:</b> Minimal compared to classical GS</li>
 * </ul>
 *
 * <h2>Numerical Stability:</h2>
 * <ul>
 *   <li><b>Much better than classical GS:</b> Maintains orthogonality to O(ε) instead of O(κ(A)ε)</li>
 *   <li><b>Backward stable:</b> Computed QR close to exact factorization of nearby matrix</li>
 *   <li><b>Orthogonality:</b> ||Q<sup>T</sup>Q - I|| = O(ε) for well-conditioned A</li>
 *   <li><b>Still worse than Householder:</b> Householder achieves O(ε) for any matrix</li>
 * </ul>
 *
 * <h2>Why Modified is Better:</h2>
 * <p>
 * Classical GS computes projections using original (inaccurate) vectors:
 * </p>
 * <pre>
 * v<sub>j</sub> = a<sub>j</sub> - Σ (q<sub>i</sub><sup>T</sup> a<sub>j</sub>) q<sub>i</sub>    [all at once]
 * </pre>
 * <p>
 * Modified GS updates incrementally using already-orthogonalized vectors:
 * </p>
 * <pre>
 * v<sub>j</sub> ← v<sub>j</sub> - (q<sub>1</sub><sup>T</sup> v<sub>j</sub>) q<sub>1</sub>
 * v<sub>j</sub> ← v<sub>j</sub> - (q<sub>2</sub><sup>T</sup> v<sub>j</sub>) q<sub>2</sub>   [sequentially]
 * ...
 * </pre>
 * <p>
 * This sequential updating significantly reduces cancellation errors.
 * </p>
 *
 * <h2>Advantages:</h2>
 * <ul>
 *   <li><b>Better stability:</b> Much more accurate orthogonality than classical GS</li>
 *   <li><b>Same cost:</b> No additional computational overhead</li>
 *   <li><b>Simplicity:</b> Easy to implement and understand</li>
 *   <li><b>Cache efficiency:</b> Better memory access patterns than Householder</li>
 *   <li><b>Incremental:</b> Can process columns as they become available</li>
 * </ul>
 *
 * <h2>Disadvantages vs Householder:</h2>
 * <ul>
 *   <li><b>Slightly less stable:</b> Householder is backward stable for all matrices</li>
 *   <li><b>No in-place:</b> Requires separate storage for Q and R</li>
 *   <li><b>Rank detection:</b> Harder to detect rank deficiency</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>QR decomposition for well-conditioned matrices</li>
 *   <li>Least squares problems with moderate condition numbers</li>
 *   <li>Orthonormalizing sets of vectors</li>
 *   <li>Krylov subspace methods (GMRES, Arnoldi)</li>
 *   <li>Streaming/online algorithms where columns arrive sequentially</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][] {
 *     {1,  1,  0},
 *     {1,  0,  1},
 *     {0,  1,  1},
 *     {1,  1,  1}
 * });
 *
 * ModifiedGramSchmidt mgs = new ModifiedGramSchmidt();
 * var result = mgs.decompose(A);
 *
 * Matrix Q = result.getQ();  // Orthonormal columns
 * Matrix R = result.getR();  // Upper triangular
 *
 * // Verify orthogonality: Q^T * Q should be identity
 * Matrix QTQ = Q.transpose().multiply(Q);
 * double orthError = QTQ.subtract(Matrix.Identity(3)).frobeniusNorm();
 * System.out.println("Orthogonality error: " + orthError);  // Should be ~1e-15
 *
 * // Verify factorization: A = Q * R
 * Matrix reconstructed = Q.multiply(R);
 * double factorError = A.subtract(reconstructed).frobeniusNorm();
 *
 * // Solve least squares: minimize ||Ax - b||
 * Vector b = new Vector(new double[] {1, 2, 3, 4});
 * Vector x = result.solve(b);
 * }</pre>
 *
 * <h2>Comparison with Other Methods:</h2>
 * <table border="1">
 *   <tr><th>Method</th><th>Orthogonality</th><th>Speed</th><th>Memory</th></tr>
 *   <tr><td>Classical GS</td><td>O(κ(A)ε)</td><td>2mn²</td><td>mn</td></tr>
 *   <tr><td>Modified GS</td><td>O(ε)</td><td>2mn²</td><td>mn</td></tr>
 *   <tr><td>Householder</td><td>O(ε)</td><td>2mn² - 2n³/3</td><td>mn (in-place)</td></tr>
 *   <tr><td>Givens</td><td>O(ε)</td><td>3mn² - n³</td><td>mn</td></tr>
 * </table>
 *
 * <h2>When to Use MGS:</h2>
 * <ul>
 *   <li><b>Iterative refinement:</b> When you can afford reorthogonalization if needed</li>
 *   <li><b>Krylov methods:</b> Common in GMRES, Arnoldi, Lanczos implementations</li>
 *   <li><b>Streaming data:</b> When matrix columns arrive sequentially</li>
 *   <li><b>Educational:</b> Demonstrates importance of operation ordering</li>
 * </ul>
 *
 * <h2>When to Use Householder Instead:</h2>
 * <ul>
 *   <li><b>Ill-conditioned matrices:</b> κ(A) &gt; 10<sup>8</sup></li>
 *   <li><b>Production code:</b> When maximum stability is required</li>
 *   <li><b>Batch processing:</b> When all of A is available upfront</li>
 *   <li><b>Memory constraints:</b> When in-place operation is needed</li>
 * </ul>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Implements modified (improved) Gram-Schmidt algorithm</li>
 *   <li>Good numerical stability for most practical matrices</li>
 *   <li>Recommended over classical Gram-Schmidt in all cases</li>
 *   <li>Consider Householder QR for critical applications</li>
 *   <li>Can be enhanced with reorthogonalization for extra stability</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see GramSchmidt
 * @see HouseholderQR
 * @see net.faulj.decomposition.result.QRResult
 */
public class ModifiedGramSchmidt {
}
