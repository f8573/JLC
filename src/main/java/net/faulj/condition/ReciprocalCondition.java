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
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixNorms;
import net.faulj.solve.LUSolver;
import net.faulj.vector.Vector;

/**
 * Utility for estimating the reciprocal condition number from an LU factorization.
 */
public class ReciprocalCondition {

	/**
	 * Estimate RCOND = 1 / (||A||_1 * ||A^{-1}||_1) using a provided LU factorization.
	 * <p>
	 * This implementation builds the inverse by solving n systems A x = e_j using the
	 * provided LU factors and computes the 1-norm of the inverse. The 1-norm of A is
	 * obtained from the LU reconstruction (PA has same column sums as A so norm1 is preserved).
	 * </p>
	 *
	 * @param lu LU factorization result for matrix A (PA = LU)
	 * @return estimated reciprocal condition number in the 1-norm; returns 0.0 if singular
	 */
	public static double estimateFromLU(LUResult lu) {
		if (lu == null) throw new IllegalArgumentException("LU result must not be null");
		if (lu.isSingular()) return 0.0;

		// Norm of A: use norm1 of LU reconstruction (row permutation does not change column sums)
		Matrix LU = lu.reconstruct();
		double normA = MatrixNorms.norm1(LU);

		return estimateFromLU(lu, normA);
	}

	/**
	 * Estimate RCOND given LU factorization and a precomputed norm of A (1-norm).
	 *
	 * @param lu LU factorization result for matrix A (PA = LU)
	 * @param normA precomputed 1-norm of A (must be >= 0)
	 * @return estimated reciprocal condition number; 0.0 if singular or invalid inputs
	 */
	public static double estimateFromLU(LUResult lu, double normA) {
		if (lu == null) throw new IllegalArgumentException("LU result must not be null");
		if (normA < 0) throw new IllegalArgumentException("normA must be non-negative");
		if (lu.isSingular()) return 0.0;
		if (normA == 0.0) return 0.0;

		int n = lu.getL().getRowCount();
		LUSolver solver = new LUSolver();

		// Build inverse column-by-column by solving A x = e_j
		Vector[] invCols = new Vector[n];
		for (int j = 0; j < n; j++) {
			double[] e = new double[n];
			e[j] = 1.0;
			Vector ej = new Vector(e);
			Vector col = solver.solve(lu, ej);
			invCols[j] = col;
		}

		Matrix invA = new Matrix(invCols);
		double normInv = MatrixNorms.norm1(invA);
		if (normInv == 0.0) return 0.0;

		return 1.0 / (normA * normInv);
	}

}
