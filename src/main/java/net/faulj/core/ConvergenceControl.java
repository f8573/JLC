package net.faulj.core;

/**
 * Manages convergence criteria and control parameters for iterative algorithms.
 * <p>
 * This class provides a unified framework for monitoring and controlling the convergence behavior
 * of iterative numerical methods such as QR iteration, power methods, Krylov subspace methods,
 * and iterative refinement. It tracks convergence metrics, enforces maximum iteration limits,
 * and determines when algorithms have achieved sufficient accuracy or should terminate.
 * </p>
 *
 * <h2>Convergence Criteria:</h2>
 * <ul>
 *   <li><b>Absolute Tolerance:</b> ‖x_k - x_{k-1}‖ &lt; ε_abs</li>
 *   <li><b>Relative Tolerance:</b> ‖x_k - x_{k-1}‖ / ‖x_k‖ &lt; ε_rel</li>
 *   <li><b>Residual Tolerance:</b> ‖Ax - b‖ &lt; ε_res</li>
 *   <li><b>Maximum Iterations:</b> k &lt; k_max</li>
 *   <li><b>Stagnation Detection:</b> No improvement over N consecutive iterations</li>
 * </ul>
 *
 * <h2>Configurable Parameters:</h2>
 * <ul>
 *   <li><b>maxIterations:</b> Upper bound on iteration count (default: 1000)</li>
 *   <li><b>absoluteTolerance:</b> Threshold for absolute convergence (default: 1e-10)</li>
 *   <li><b>relativeTolerance:</b> Threshold for relative convergence (default: 1e-8)</li>
 *   <li><b>residualTolerance:</b> Threshold for residual norms (default: 1e-12)</li>
 *   <li><b>stagnationWindow:</b> Number of iterations to detect stagnation (default: 50)</li>
 * </ul>
 *
 * <h2>Convergence States:</h2>
 * <ul>
 *   <li><b>CONVERGED:</b> All convergence criteria satisfied</li>
 *   <li><b>MAX_ITERATIONS:</b> Maximum iteration limit reached</li>
 *   <li><b>STAGNATED:</b> No progress over multiple iterations</li>
 *   <li><b>DIVERGED:</b> Residual or error increasing</li>
 *   <li><b>IN_PROGRESS:</b> Still iterating</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * ConvergenceControl control = ConvergenceControl.builder()
 *     .maxIterations(500)
 *     .absoluteTolerance(1e-12)
 *     .relativeTolerance(1e-10)
 *     .build();
 *
 * while (!control.hasConverged()) {
 *     performIteration();
 *     double error = computeError();
 *     control.update(error);
 * }
 *
 * if (control.isConverged()) {
 *     System.out.println("Converged in " + control.getIterationCount() + " iterations");
 * }
 * }</pre>
 *
 * <h2>Advanced Features:</h2>
 * <ul>
 *   <li>Adaptive tolerance adjustment based on progress rate</li>
 *   <li>History tracking for convergence rate estimation</li>
 *   <li>Early exit detection for rapid convergence</li>
 *   <li>Diagnostic reporting for debugging failed convergence</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>QR eigenvalue iteration control</li>
 *   <li>Iterative refinement in linear solvers</li>
 *   <li>Power method and inverse iteration convergence</li>
 *   <li>Krylov subspace method (GMRES, BiCGSTAB) termination</li>
 * </ul>
 *
 * <h2>Thread Safety:</h2>
 * <p>
 * ConvergenceControl instances are not thread-safe by default. Each iteration thread should
 * maintain its own control object.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see Tolerance
 * @see DiagnosticMetrics
 */
public class ConvergenceControl {
}
