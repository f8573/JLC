package net.faulj.condition;

/**
 * Computes the reciprocal condition number (RCOND) for efficient stability estimation.
 * <p>
 * The reciprocal condition number is defined as RCOND = 1/κ(A), where κ(A) is the condition
 * number. Computing RCOND directly is often more efficient and numerically stable than computing
 * the full condition number, especially for detecting near-singularity. Values close to zero
 * indicate ill-conditioning, while values close to 1 indicate well-conditioning.
 * </p>
 *
 * <h2>Mathematical Definition:</h2>
 * <pre>
 * RCOND(A) = 1 / (‖A‖ · ‖A⁻¹‖)
 * </pre>
 *
 * <h2>Interpretation Guidelines:</h2>
 * <ul>
 *   <li><b>RCOND ≈ 1:</b> Perfectly conditioned matrix (e.g., orthogonal matrices)</li>
 *   <li><b>RCOND &gt; 10⁻³:</b> Well-conditioned, reliable numerical results</li>
 *   <li><b>10⁻⁶ ≤ RCOND ≤ 10⁻³:</b> Moderately conditioned, use with caution</li>
 *   <li><b>RCOND &lt; 10⁻⁶:</b> Ill-conditioned, results may be unreliable</li>
 *   <li><b>RCOND ≈ 0:</b> Singular or numerically singular matrix</li>
 *   <li><b>RCOND &lt; ε (machine precision):</b> Matrix effectively singular in floating-point arithmetic</li>
 * </ul>
 *
 * <h2>Advantages over Condition Number:</h2>
 * <ul>
 *   <li><b>Numerical Stability:</b> Avoids division by very small numbers</li>
 *   <li><b>Efficiency:</b> Can be estimated during LU or QR decomposition</li>
 *   <li><b>Underflow Prevention:</b> RCOND remains in representable range for ill-conditioned matrices</li>
 *   <li><b>Direct Comparison:</b> Easier to compare against machine precision</li>
 * </ul>
 *
 * <h2>Computational Methods:</h2>
 * <ul>
 *   <li><b>LU-based:</b> Estimates RCOND from LU decomposition triangular factors (O(n²))</li>
 *   <li><b>QR-based:</b> Uses R factor from QR decomposition for estimation</li>
 *   <li><b>Incremental:</b> Updates RCOND during factorization without extra passes</li>
 *   <li><b>1-norm estimation:</b> Typically uses ‖A‖₁ and estimates ‖A⁻¹‖₁</li>
 * </ul>
 *
 * <h2>Typical Usage in LAPACK Style:</h2>
 * <pre>{@code
 * Matrix A = Matrix.create(...);
 * LUResult lu = LUDecomposition.decompose(A);
 * double rcond = ReciprocalCondition.estimateFromLU(lu);
 *
 * if (rcond < 1e-10) {
 *     throw new SingularMatrixException("Matrix is numerically singular");
 * }
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Pre-computation check before solving linear systems</li>
 *   <li>Fast singularity detection in matrix computations</li>
 *   <li>Determining need for iterative refinement</li>
 *   <li>Estimating accuracy loss in numerical algorithms</li>
 * </ul>
 *
 * <h2>Complexity:</h2>
 * <ul>
 *   <li><b>From LU:</b> O(n²) with 1-norm estimation</li>
 *   <li><b>Exact (SVD):</b> O(n³) but rarely needed</li>
 * </ul>
 *
 * <h2>Relation to Expected Accuracy:</h2>
 * <p>
 * The number of accurate digits in a solution is approximately:
 * </p>
 * <pre>
 * accurate_digits ≈ -log₁₀(RCOND)
 * </pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see ConditionNumber
 * @see net.faulj.decomposition.lu.LUDecomposition
 * @see net.faulj.matrix.MatrixNorms
 */
public class ReciprocalCondition {
}
