package net.faulj.solve;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Defines a common contract for algorithms capable of solving linear systems.
 * <p>
 * A linear solver finds the vector x such that:
 * </p>
 * <pre>
 * Ax = b
 * </pre>
 * <p>
 * where A is a matrix and b is a vector.
 * </p>
 *
 * <h2>Implementations:</h2>
 * <ul>
 * <li>{@link LUSolver}: Standard Gaussian elimination (O(n³)).</li>
 * <li>{@link CramerSolver}: Determinant-based solution (O(n!)), for small matrices.</li>
 * <li>{@link LeastSquaresSolver}: For overdetermined systems (approximate solution).</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * LinearSolver solver = new LUSolver(); // or new CramerSolver()
 * Vector result = solver.solve(A, b);
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 */
public interface LinearSolver {

    /**
     * Solves the linear system Ax = b.
     *
     * @param A The coefficient matrix.
     * @param b The constant vector.
     * @return The solution vector x.
     * @throws ArithmeticException If the matrix is singular or the system cannot be solved.
     * @throws IllegalArgumentException If dimensions between A and b do not match.
     */
    Vector solve(Matrix A, Vector b);
}