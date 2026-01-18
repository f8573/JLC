package net.faulj.eigen;

import net.faulj.matrix.Matrix;

/**
 * Handles the diagonalization of square matrices via eigendecomposition.
 * <p>
 * This class represents the decomposition A = PDP<sup>-1</sup> where:
 * </p>
 * <ul>
 * <li><b>D</b> - Diagonal matrix containing eigenvalues (Λ)</li>
 * <li><b>P</b> - Invertible matrix containing eigenvectors as columns</li>
 * <li><b>A</b> - Original square diagonalizable matrix</li>
 * </ul>
 *
 * <h2>Mathematical Concept:</h2>
 * <p>
 * A matrix A is diagonalizable if there exists an invertible matrix P such that
 * P<sup>-1</sup>AP is a diagonal matrix. The diagonal entries of D are the
 * eigenvalues λᵢ of A, and the column vectors of P are the corresponding
 * eigenvectors vᵢ.
 * </p>
 * <pre>
 * A vᵢ = λᵢ vᵢ
 * </pre>
 *
 * <h2>Diagonalizability Conditions:</h2>
 * <p>
 * Not all matrices are diagonalizable. A generic n×n matrix is diagonalizable if and only if:
 * </p>
 * <ul>
 * <li>It has n linearly independent eigenvectors.</li>
 * <li>The algebraic multiplicity equals the geometric multiplicity for every eigenvalue.</li>
 * <li>It has n distinct eigenvalues (sufficient, but not necessary).</li>
 * </ul>
 * <p>
 * Normal matrices (A<sup>H</sup>A = AA<sup>H</sup>) are unitarily diagonalizable
 * (P is unitary/orthogonal). Symmetric real matrices are always diagonalizable.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Assuming we have a decomposer or similar utility
 * Matrix A = ...; // Input matrix
 * Diagonalization result = Diagonalization.decompose(A);
 *
 * Matrix D = result.getD(); // Eigenvalues on diagonal
 * Matrix P = result.getP(); // Eigenvectors
 *
 * // Verify: A * P = P * D
 * Matrix lhs = A.multiply(P);
 * Matrix rhs = P.multiply(D);
 * assert lhs.equals(rhs, 1e-9);
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li><b>Matrix Powers:</b> A<sup>k</sup> = P D<sup>k</sup> P<sup>-1</sup> (computationally efficient for large k)</li>
 * <li><b>Matrix Exponentials:</b> e<sup>A</sup> = P e<sup>D</sup> P<sup>-1</sup></li>
 * <li><b>Decoupling Systems:</b> Converting coupled differential equations into independent ones</li>
 * <li><b>Modal Analysis:</b> Analyzing vibration modes in physics/engineering</li>
 * </ul>
 *
 * <h2>Comparison with Schur Form:</h2>
 * <table border="1">
 * <tr><th>Aspect</th><th>Diagonalization (PDP<sup>-1</sup>)</th><th>Schur (UTU<sup>T</sup>)</th></tr>
 * <tr><td>Target Form</td><td>Diagonal</td><td>Triangular (or Quasi-Triangular)</td></tr>
 * <tr><td>Existence</td><td>Conditioned (must be non-defective)</td><td>Always exists</td></tr>
 * <tr><td>Basis Matrix</td><td>Invertible (often non-orthogonal)</td><td>Orthogonal / Unitary</td></tr>
 * <tr><td>Stability</td><td>Depends on condition number of P</td><td>Always numerically stable</td></tr>
 * </table>
 *
 * <h2>Implementation Note:</h2>
 * <p>
 * This class currently serves as a structural placeholder or base for specific diagonalization
 * algorithms (e.g., Jacobi method for symmetric matrices, or eigenvector extraction from Schur form).
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.eigen.EigenvectorComputation
 * @see net.faulj.decomposition.result.SchurResult
 */
public class Diagonalization {
}