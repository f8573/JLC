package net.faulj.solve;

import net.faulj.core.Tolerance;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Solves triangular linear systems (Lx = b or Ux = b).
 * <p>
 * This solver handles:
 * <ul>
 * <li>Standard square systems (m = n)</li>
 * <li>Overdetermined systems (m > n) by checking consistency</li>
 * <li>Singular/Underdetermined systems by assigning 0 to free variables</li>
 * </ul>
 * </p>
 */
public class TriangularSolver {

    /**
     * Solves Lx = b using forward substitution for lower triangular matrix L.
     *
     * @param L Lower triangular matrix (m x n)
     * @param b Right-hand side vector (length m)
     * @return Solution vector x
     * @throws IllegalArgumentException if dimensions are incompatible
     * @throws ArithmeticException if the system is inconsistent (no solution)
     */
    public static Vector forwardSubstitution(Matrix L, Vector b) {
        int m = L.getRowCount();
        int n = L.getColumnCount();

        if (b.dimension() != m) {
            throw new IllegalArgumentException("Incompatible dimensions: L is " +
                    m + "x" + n + " but b has length " + b.dimension());
        }

        // Check if matrix is overdetermined (more equations than unknowns)
        if (m > n) {
            return handleOverdeterminedForward(L, b, m, n);
        }

        // Standard case: m <= n
        Vector x = new Vector(new double[n]);

        for (int i = 0; i < m; i++) {
            // Check for zero/near-zero diagonal element
            if (Tolerance.isZero(L.get(i, i))) {
                return handleSingularForward(L, b, x, i, m, n);
            }

            double sum = b.get(i);
            for (int j = 0; j < i; j++) {
                sum -= L.get(i, j) * x.get(j);
            }
            x.set(i, sum / L.get(i, i));
        }

        // If underdetermined (m < n), remaining variables are free (set to 0)
        return x;
    }

    /**
     * Solves Ux = b using backward substitution for upper triangular matrix U.
     *
     * @param U Upper triangular matrix (m x n)
     * @param b Right-hand side vector (length m)
     * @return Solution vector x
     * @throws IllegalArgumentException if dimensions are incompatible
     * @throws ArithmeticException if the system is inconsistent (no solution)
     */
    public static Vector backwardSubstitution(Matrix U, Vector b) {
        int m = U.getRowCount();
        int n = U.getColumnCount();

        if (b.dimension() != m) {
            throw new IllegalArgumentException("Incompatible dimensions: U is " +
                    m + "x" + n + " but b has length " + b.dimension());
        }

        // Check if matrix is overdetermined
        if (m > n) {
            return handleOverdeterminedBackward(U, b, m, n);
        }

        // Standard case: m <= n
        Vector x = new Vector(new double[n]);

        // Start from the last equation
        for (int i = m - 1; i >= 0; i--) {
            // Check for zero/near-zero diagonal element
            if (Tolerance.isZero(U.get(i, i))) {
                return handleSingularBackward(U, b, x, i, m, n);
            }

            double sum = b.get(i);
            for (int j = i + 1; j < n; j++) {
                sum -= U.get(i, j) * x.get(j);
            }
            x.set(i, sum / U.get(i, i));
        }

        return x;
    }

    private static Vector handleOverdeterminedForward(Matrix L, Vector b, int m, int n) {
        Vector x = new Vector(new double[n]);

        // Solve the first n equations
        for (int i = 0; i < n; i++) {
            if (Tolerance.isZero(L.get(i, i))) {
                throw new ArithmeticException("Singular matrix: zero diagonal at row " + i);
            }

            double sum = b.get(i);
            for (int j = 0; j < i; j++) {
                sum -= L.get(i, j) * x.get(j);
            }
            x.set(i, sum / L.get(i, i));
        }

        // Check consistency of remaining equations
        for (int i = n; i < m; i++) {
            double sum = b.get(i);
            for (int j = 0; j < n; j++) {
                sum -= L.get(i, j) * x.get(j);
            }
            if (!Tolerance.isZero(sum)) {
                throw new ArithmeticException("Inconsistent system: equation " + i +
                        " has residual " + sum);
            }
        }

        return x;
    }

    private static Vector handleOverdeterminedBackward(Matrix U, Vector b, int m, int n) {
        Vector x = new Vector(new double[n]);

        // Check consistency of first (m-n) equations
        for (int i = 0; i < m - n; i++) {
            double sum = b.get(i);
            for (int j = 0; j < n; j++) {
                sum -= U.get(i, j) * x.get(j);
            }
            if (!Tolerance.isZero(sum)) {
                throw new ArithmeticException("Inconsistent system: equation " + i +
                        " has residual " + sum);
            }
        }

        // Solve the last n equations
        for (int i = m - 1; i >= m - n; i--) {
            int diagIdx = i - (m - n);
            if (Tolerance.isZero(U.get(i, diagIdx))) {
                throw new ArithmeticException("Singular matrix: zero diagonal at row " + i);
            }

            double sum = b.get(i);
            for (int j = diagIdx + 1; j < n; j++) {
                sum -= U.get(i, j) * x.get(j);
            }
            x.set(diagIdx, sum / U.get(i, diagIdx));
        }

        return x;
    }

    private static Vector handleSingularForward(Matrix L, Vector b, Vector x, int singularRow, int m, int n) {
        // Check if the equation is consistent (0 = b[i] - sum)
        double sum = b.get(singularRow);
        for (int j = 0; j < singularRow; j++) {
            sum -= L.get(singularRow, j) * x.get(j);
        }

        if (!Tolerance.isZero(sum)) {
            throw new ArithmeticException("Singular matrix with inconsistent equation at row " +
                    singularRow + ": 0 != " + sum);
        }

        // Solution is not unique (free variable at x[singularRow])
        // Set free variable to 0 and continue
        x.set(singularRow, 0.0);

        for (int i = singularRow + 1; i < m; i++) {
            if (Tolerance.isZero(L.get(i, i))) {
                return handleSingularForward(L, b, x, i, m, n);
            }

            double sum2 = b.get(i);
            for (int j = 0; j < i; j++) {
                sum2 -= L.get(i, j) * x.get(j);
            }
            x.set(i, sum2 / L.get(i, i));
        }

        return x;
    }

    private static Vector handleSingularBackward(Matrix U, Vector b, Vector x, int singularRow, int m, int n) {
        // Check if the equation is consistent
        double sum = b.get(singularRow);
        for (int j = singularRow + 1; j < n; j++) {
            sum -= U.get(singularRow, j) * x.get(j);
        }

        if (!Tolerance.isZero(sum)) {
            throw new ArithmeticException("Singular matrix with inconsistent equation at row " +
                    singularRow + ": 0 != " + sum);
        }

        // Solution is not unique (free variable at x[singularRow])
        // Set free variable to 0 and continue
        x.set(singularRow, 0.0);

        for (int i = singularRow - 1; i >= 0; i--) {
            if (Tolerance.isZero(U.get(i, i))) {
                return handleSingularBackward(U, b, x, i, m, n);
            }

            sum = b.get(i);
            for (int j = i + 1; j < n; j++) {
                sum -= U.get(i, j) * x.get(j);
            }
            x.set(i, sum / U.get(i, i));
        }

        return x;
    }
}