package net.faulj.condition;

/**
 * Computes the condition number of matrices for numerical stability analysis.
 * <p>
 * The condition number κ(A) measures how sensitive a matrix or linear system is to perturbations
 * in the input data. It is a fundamental metric in numerical analysis, indicating whether numerical
 * solutions are reliable or may be significantly affected by rounding errors.
 * </p>
 *
 * <h2>Mathematical Definition:</h2>
 * <p>
 * For a non-singular matrix A and matrix norm ‖·‖:
 * </p>
 * <pre>
 * κ(A) = ‖A‖ · ‖A⁻¹‖
 * </pre>
 *
 * <h2>Condition Number Interpretations:</h2>
 * <ul>
 *   <li><b>κ(A) = 1:</b> Perfectly conditioned (e.g., orthogonal matrices)</li>
 *   <li><b>κ(A) &lt; 10³:</b> Well-conditioned, stable numerical computation</li>
 *   <li><b>10³ ≤ κ(A) &lt; 10⁶:</b> Moderately conditioned, acceptable with care</li>
 *   <li><b>κ(A) ≥ 10⁶:</b> Ill-conditioned, results may be unreliable</li>
 *   <li><b>κ(A) → ∞:</b> Singular or near-singular matrix</li>
 * </ul>
 *
 * <h2>Norm Types:</h2>
 * <ul>
 *   <li><b>κ₁(A):</b> 1-norm condition number (max absolute column sum)</li>
 *   <li><b>κ₂(A):</b> 2-norm condition number (ratio of largest to smallest singular value)</li>
 *   <li><b>κ∞(A):</b> ∞-norm condition number (max absolute row sum)</li>
 *   <li><b>κF(A):</b> Frobenius norm condition number</li>
 * </ul>
 *
 * <h2>Computational Methods:</h2>
 * <ul>
 *   <li><b>SVD-based:</b> κ₂(A) = σ_max/σ_min (most accurate but expensive)</li>
 *   <li><b>Norm estimation:</b> Uses iterative power method for ‖A⁻¹‖</li>
 *   <li><b>LU-based:</b> Estimates using triangular factors from LU decomposition</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Assessing numerical stability of linear system solvers</li>
 *   <li>Detecting near-singularity in matrix computations</li>
 *   <li>Determining precision requirements for accurate solutions</li>
 *   <li>Validating iterative refinement convergence</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * Matrix A = Matrix.create(...);
 * double kappa = ConditionNumber.compute(A, MatrixNorm.TWO_NORM);
 * if (kappa > 1e6) {
 *     System.out.println("Warning: Matrix is ill-conditioned");
 * }
 * }</pre>
 *
 * <h2>Complexity:</h2>
 * <ul>
 *   <li><b>SVD method:</b> O(n³) for n×n matrix</li>
 *   <li><b>Estimation method:</b> O(n²) with iterative norm computation</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see ReciprocalCondition
 * @see net.faulj.matrix.MatrixNorms
 * @see net.faulj.svd.SVDecomposition
 */
public class ConditionNumber {
}
