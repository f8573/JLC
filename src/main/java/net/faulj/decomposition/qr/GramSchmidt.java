package net.faulj.decomposition.qr;

/**
 * Computes QR decomposition using classical Gram-Schmidt orthogonalization.
 * <p>
 * The Gram-Schmidt process constructs an orthonormal basis from a set of linearly independent
 * vectors by successive projection and normalization. This implementation applies the classical
 * (unmodified) algorithm which processes vectors sequentially.
 * </p>
 * <pre>
 *   A = Q * R
 * </pre>
 * <p>
 * where Q has orthonormal columns and R is upper triangular.
 * </p>
 *
 * <h2>Classical Gram-Schmidt Algorithm:</h2>
 * <pre>
 * For j = 1 to n:
 *   v<sub>j</sub> = a<sub>j</sub>
 *   For i = 1 to j-1:
 *     r<sub>ij</sub> = q<sub>i</sub><sup>T</sup> * a<sub>j</sub>
 *     v<sub>j</sub> = v<sub>j</sub> - r<sub>ij</sub> * q<sub>i</sub>
 *   r<sub>jj</sub> = ||v<sub>j</sub>||<sub>2</sub>
 *   q<sub>j</sub> = v<sub>j</sub> / r<sub>jj</sub>
 * </pre>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Time complexity:</b> O(2mn<sup>2</sup>) flops</li>
 *   <li><b>Space complexity:</b> O(mn) for Q and R</li>
 *   <li><b>Per column:</b> O(mn) work</li>
 * </ul>
 *
 * <h2>Numerical Stability Issues:</h2>
 * <p>
 * Classical Gram-Schmidt is numerically <b>unstable</b> and should generally be avoided:
 * </p>
 * <ul>
 *   <li><b>Loss of orthogonality:</b> Computed Q can deviate significantly from orthogonality</li>
 *   <li><b>Cancellation errors:</b> Subtracting nearly parallel vectors loses precision</li>
 *   <li><b>Error accumulation:</b> Errors compound across iterations</li>
 *   <li><b>Ill-conditioned matrices:</b> Particularly problematic for nearly rank-deficient A</li>
 * </ul>
 * <p>
 * For example, Q<sup>T</sup>Q may differ from identity by O(κ(A) * ε) where κ is condition number.
 * </p>
 *
 * <h2>When to Use:</h2>
 * <ul>
 *   <li><b>Well-conditioned matrices:</b> κ(A) &lt; 10<sup>6</sup> for double precision</li>
 *   <li><b>Educational purposes:</b> Demonstrates basic orthogonalization concept</li>
 *   <li><b>Theoretical analysis:</b> Simpler to analyze than modified version</li>
 *   <li><b>Not recommended for production:</b> Use {@link ModifiedGramSchmidt} or {@link HouseholderQR} instead</li>
 * </ul>
 *
 * <h2>Comparison with Alternatives:</h2>
 * <table border="1">
 *   <tr><th>Method</th><th>Stability</th><th>Speed</th><th>Recommendation</th></tr>
 *   <tr><td>Classical GS</td><td>Poor</td><td>Fast</td><td>Avoid</td></tr>
 *   <tr><td>Modified GS</td><td>Good</td><td>Fast</td><td>Use for well-conditioned</td></tr>
 *   <tr><td>Householder</td><td>Excellent</td><td>Fastest</td><td>Default choice</td></tr>
 *   <tr><td>Givens</td><td>Excellent</td><td>Slower</td><td>Use for sparse/structured</td></tr>
 * </table>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Pedagogical demonstrations of orthogonalization</li>
 *   <li>Theoretical analysis and algorithm development</li>
 *   <li>Quick prototyping with well-conditioned matrices</li>
 *   <li>Krylov subspace methods (with reorthogonalization)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // WARNING: Only use with well-conditioned matrices!
 * Matrix A = Matrix.random(5, 3);  // Well-conditioned random matrix
 *
 * GramSchmidt gs = new GramSchmidt();
 * var result = gs.decompose(A);
 *
 * Matrix Q = result.getQ();
 * Matrix R = result.getR();
 *
 * // Check orthogonality (may be poor for ill-conditioned A)
 * Matrix QTQ = Q.transpose().multiply(Q);
 * double orthogonalityError = QTQ.subtract(Matrix.Identity(3)).frobeniusNorm();
 * System.out.println("Orthogonality error: " + orthogonalityError);
 *
 * // For production, use ModifiedGramSchmidt or HouseholderQR instead
 * }</pre>
 *
 * <h2>Reorthogonalization:</h2>
 * <p>
 * To improve stability, apply Gram-Schmidt twice:
 * </p>
 * <pre>{@code
 * // First pass
 * for (i = 0; i < j; i++) {
 *     r[i][j] = dot(q[i], a[j]);
 *     a[j] = a[j] - r[i][j] * q[i];
 * }
 * // Second pass (reorthogonalization)
 * for (i = 0; i < j; i++) {
 *     s = dot(q[i], a[j]);
 *     r[i][j] += s;
 *     a[j] = a[j] - s * q[i];
 * }
 * q[j] = normalize(a[j]);
 * }</pre>
 * <p>
 * This nearly doubles the cost but significantly improves orthogonality.
 * </p>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Implements classical (unmodified) Gram-Schmidt algorithm</li>
 *   <li>Not recommended for production use due to stability issues</li>
 *   <li>Consider {@link ModifiedGramSchmidt} for better numerical properties</li>
 *   <li>Best used for educational purposes or well-conditioned problems</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see ModifiedGramSchmidt
 * @see HouseholderQR
 * @see net.faulj.decomposition.result.QRResult
 */
public class GramSchmidt {
}
