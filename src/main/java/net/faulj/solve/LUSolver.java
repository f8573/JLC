package net.faulj.solve;

import net.faulj.core.PermutationVector;
import net.faulj.core.Tolerance;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Solves linear systems Ax = b using LU Decomposition.
 * <p>
 * This solver uses the factorization PA = LU to transform the system into two
 * triangular systems which are solved via forward and backward substitution.
 * </p>
 *
 * <h2>Algorithm Description:</h2>
 * <p>
 * Given the system Ax = b and the decomposition PA = LU:
 * </p>
 * <ol>
 * <li><b>Permutation:</b> Multiply system by P: PAx = Pb ⇒ LUx = Pb</li>
 * <li><b>Forward Substitution:</b> Solve Ly = Pb for y.
 * <ul><li>L is lower triangular with unit diagonal.</li></ul>
 * </li>
 * <li><b>Backward Substitution:</b> Solve Ux = y for x.
 * <ul><li>U is upper triangular.</li></ul>
 * </li>
 * </ol>
 *
 * <h2>Computational Cost:</h2>
 * <table border="1">
 * <tr><th>Operation</th><th>Complexity</th></tr>
 * <tr><td>Decomposition (PA=LU)</td><td>O(n³/3)</td></tr>
 * <tr><td>Forward Substitution</td><td>O(n²/2)</td></tr>
 * <tr><td>Backward Substitution</td><td>O(n²/2)</td></tr>
 * <tr><td><b>Total Solve (given LU)</b></td><td><b>O(n²)</b></td></tr>
 * </table>
 * <p>
 * Once the decomposition is computed, solving for multiple right-hand side vectors b
 * is very efficient (O(n²)).
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // 1. One-shot solution
 * LUSolver solver = new LUSolver();
 * Vector x = solver.solve(A, b);
 *
 * // 2. Reuse decomposition for multiple vectors
 * LUDecomposition luAlgo = new LUDecomposition();
 * LUResult lu = luAlgo.decompose(A);
 * LUSolver quickSolver = new LUSolver();
 *
 * Vector x1 = quickSolver.solve(lu, b1);
 * Vector x2 = quickSolver.solve(lu, b2); // Very fast
 * }</pre>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li><b>Stability:</b> Generally stable with partial pivoting (handled by {@link LUDecomposition}).</li>
 * <li><b>Requirement:</b> Matrix A must be square and non-singular.</li>
 * <li><b>Memory:</b> Solves can often be performed in-place (though this implementation returns new Vectors).</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.lu.LUDecomposition
 * @see net.faulj.decomposition.result.LUResult
 */
public class LUSolver {

    private final LUDecomposition luDecomposition;

    /**
     * Constructs a solver using the default LU decomposition algorithm.
     */
    public LUSolver() {
        this.luDecomposition = new LUDecomposition();
    }

    /**
     * Constructs a solver using a specific LU decomposition strategy.
     *
     * @param luDecomposition The decomposition algorithm to use.
     */
    public LUSolver(LUDecomposition luDecomposition) {
        this.luDecomposition = luDecomposition;
    }

    /**
     * Solves the system Ax = b from scratch.
     * <p>
     * Performs the LU decomposition of A followed by forward/back substitution.
     * </p>
     *
     * @param A The square coefficient matrix.
     * @param b The right-hand side vector.
     * @return The solution vector x.
     * @throws IllegalArgumentException if dimensions mismatch.
     * @throws ArithmeticException if A is singular.
     */
    public Vector solve(Matrix A, Vector b) {
        if (b.dimension() != A.getRowCount()) {
            throw new IllegalArgumentException("Dimension mismatch: b has " + b.dimension()
                    + " elements but A has " + A.getRowCount() + " rows");
        }

        LUResult lu = luDecomposition.decompose(A);

        if (lu.isSingular()) {
            throw new ArithmeticException("Matrix is singular; no unique solution exists");
        }

        return solve(lu, b);
    }

    /**
     * Solves the system Ax = b using a pre-computed LU factorization.
     * <p>
     * This method avoids the O(n³) cost of decomposition and runs in O(n²).
     * </p>
     *
     * @param lu The result of a previously computed LU decomposition (PA = LU).
     * @param b The right-hand side vector.
     * @return The solution vector x.
     */
    public Vector solve(LUResult lu, Vector b) {
        Matrix L = lu.getL();
        Matrix U = lu.getU();
        PermutationVector P = lu.getP();
        int n = L.getRowCount();

        // Apply permutation to b: Pb
        double[] pb = new double[n];
        for (int i = 0; i < n; i++) {
            pb[i] = b.get(P.get(i));
        }

        // Forward substitution: Ly = Pb
        double[] y = forwardSubstitution(L, pb);

        // Back substitution: Ux = y
        double[] x = backSubstitution(U, y);

        return new Vector(x);
    }

    /**
     * Performs forward substitution to solve Ly = b.
     * <p>
     * Solves for y in:
     * <pre>
     * y₁ = b₁
     * y₂ = b₂ - l₂₁y₁
     * ...
     * </pre>
     * </p>
     * * @param L Lower triangular matrix (assumed unit diagonal).
     * @param b The RHS vector (permuted).
     * @return The intermediate vector y.
     */
    private double[] forwardSubstitution(Matrix L, double[] b) {
        int n = L.getRowCount();
        double[] y = new double[n];

        for (int i = 0; i < n; i++) {
            double sum = b[i];
            for (int j = 0; j < i; j++) {
                sum -= L.get(i, j) * y[j];
            }
            y[i] = sum; // L[i,i] = 1 implicitly
        }

        return y;
    }

    /**
     * Performs back substitution to solve Ux = y.
     * <p>
     * Solves for x starting from the last row:
     * <pre>
     * xₙ = yₙ / uₙₙ
     * xᵢ = (yᵢ - Σ uᵢⱼxⱼ) / uᵢᵢ
     * </pre>
     * </p>
     *
     * @param U Upper triangular matrix.
     * @param y The intermediate vector from forward substitution.
     * @return The solution vector x.
     * @throws ArithmeticException if a zero diagonal element is encountered (singular U).
     */
    private double[] backSubstitution(Matrix U, double[] y) {
        int n = U.getRowCount();
        double[] x = new double[n];

        for (int i = n - 1; i >= 0; i--) {
            double sum = y[i];
            for (int j = i + 1; j < n; j++) {
                sum -= U.get(i, j) * x[j];
            }
            double diag = U.get(i, i);
            if (Tolerance.isZero(diag)) {
                throw new ArithmeticException("Zero diagonal encountered in back substitution");
            }
            x[i] = sum / diag;
        }

        return x;
    }
}