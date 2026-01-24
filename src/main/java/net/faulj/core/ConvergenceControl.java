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
    public enum State {
        IN_PROGRESS,
        CONVERGED,
        MAX_ITERATIONS,
        STAGNATED,
        DIVERGED
    }

    private final int maxIterations;
    private final double absoluteTolerance;
    private final double relativeTolerance;
    private final double residualTolerance;
    private final int stagnationWindow;
    private final int divergenceWindow;
    private final double stagnationTolerance;

    private int iterationCount;
    private int stagnationCount;
    private int divergenceCount;
    private double lastError = Double.NaN;
    private double lastResidual = Double.NaN;
    private double bestError = Double.POSITIVE_INFINITY;
    private double bestResidual = Double.POSITIVE_INFINITY;
    private double lastAbsoluteChange = Double.NaN;
    private double lastRelativeChange = Double.NaN;
    private State state = State.IN_PROGRESS;

    private ConvergenceControl(Builder builder) {
        this.maxIterations = builder.maxIterations;
        this.absoluteTolerance = builder.absoluteTolerance;
        this.relativeTolerance = builder.relativeTolerance;
        this.residualTolerance = builder.residualTolerance;
        this.stagnationWindow = builder.stagnationWindow;
        this.divergenceWindow = builder.divergenceWindow;
        this.stagnationTolerance = builder.stagnationTolerance;
    }

    public static Builder builder() {
        return new Builder();
    }

    public State update(double error) {
        return update(error, Double.NaN);
    }

    public State update(double error, double residual) {
        if (state != State.IN_PROGRESS) {
            return state;
        }
        iterationCount++;

        if (!isFinite(error) || (isFinite(residual) == false && !Double.isNaN(residual))) {
            state = State.DIVERGED;
            return state;
        }

        if (!Double.isNaN(lastError)) {
            lastAbsoluteChange = Math.abs(error - lastError);
            double absTolScale = isEnabled(absoluteTolerance) ? absoluteTolerance : 0.0;
            double denom = Math.max(Math.abs(error), absTolScale);
            if (denom == 0.0) {
                denom = 1.0;
            }
            lastRelativeChange = lastAbsoluteChange / denom;
        } else {
            lastAbsoluteChange = Double.NaN;
            lastRelativeChange = Double.NaN;
        }

        boolean improved = false;
        double improvementEps = isEnabled(stagnationTolerance)
            ? stagnationTolerance
            : Math.max(isEnabled(absoluteTolerance) ? absoluteTolerance : 0.0, 1e-15);

        if (error + improvementEps < bestError) {
            bestError = error;
            improved = true;
        }
        if (!Double.isNaN(residual) && residual + improvementEps < bestResidual) {
            bestResidual = residual;
            improved = true;
        }
        if (improved) {
            stagnationCount = 0;
        } else {
            stagnationCount++;
        }

        boolean worse = false;
        if (!Double.isNaN(lastError)) {
            double errIncreaseTol = Math.max(
                isEnabled(absoluteTolerance) ? absoluteTolerance : 0.0,
                Math.abs(lastError) * (isEnabled(relativeTolerance) ? relativeTolerance : 0.0)
            );
            if (error > lastError + errIncreaseTol) {
                worse = true;
            }
        }
        if (!Double.isNaN(residual) && !Double.isNaN(lastResidual)) {
            double resIncreaseTol = Math.max(
                isEnabled(absoluteTolerance) ? absoluteTolerance : 0.0,
                Math.abs(lastResidual) * (isEnabled(relativeTolerance) ? relativeTolerance : 0.0)
            );
            if (residual > lastResidual + resIncreaseTol) {
                worse = true;
            }
        }
        if (worse) {
            divergenceCount++;
        } else {
            divergenceCount = 0;
        }

        lastError = error;
        if (!Double.isNaN(residual)) {
            lastResidual = residual;
        }

        boolean absEnabled = isEnabled(absoluteTolerance);
        boolean relEnabled = isEnabled(relativeTolerance);
        boolean resEnabled = isEnabled(residualTolerance);
        boolean relApplicable = relEnabled && !Double.isNaN(lastRelativeChange);
        boolean resApplicable = resEnabled && !Double.isNaN(residual);

        boolean absSatisfied = !absEnabled || error <= absoluteTolerance;
        boolean relSatisfied = !relEnabled || !relApplicable || lastRelativeChange <= relativeTolerance;
        boolean resSatisfied = !resEnabled || !resApplicable || residual <= residualTolerance;

        boolean anyApplicable = absEnabled || relApplicable || resApplicable;

        if (anyApplicable && absSatisfied && relSatisfied && resSatisfied) {
            state = State.CONVERGED;
        } else if (divergenceWindow > 0 && divergenceCount >= divergenceWindow) {
            state = State.DIVERGED;
        } else if (stagnationWindow > 0 && stagnationCount >= stagnationWindow) {
            state = State.STAGNATED;
        } else if (iterationCount >= maxIterations) {
            state = State.MAX_ITERATIONS;
        } else {
            state = State.IN_PROGRESS;
        }

        return state;
    }

    public void reset() {
        iterationCount = 0;
        stagnationCount = 0;
        divergenceCount = 0;
        lastError = Double.NaN;
        lastResidual = Double.NaN;
        bestError = Double.POSITIVE_INFINITY;
        bestResidual = Double.POSITIVE_INFINITY;
        lastAbsoluteChange = Double.NaN;
        lastRelativeChange = Double.NaN;
        state = State.IN_PROGRESS;
    }

    public boolean hasConverged() {
        return state != State.IN_PROGRESS;
    }

    public boolean shouldStop() {
        return state != State.IN_PROGRESS;
    }

    public boolean isConverged() {
        return state == State.CONVERGED;
    }

    public boolean isStagnated() {
        return state == State.STAGNATED;
    }

    public boolean isDiverged() {
        return state == State.DIVERGED;
    }

    public boolean isMaxIterationsReached() {
        return state == State.MAX_ITERATIONS;
    }

    public State getState() {
        return state;
    }

    public int getIterationCount() {
        return iterationCount;
    }

    public double getLastError() {
        return lastError;
    }

    public double getLastResidual() {
        return lastResidual;
    }

    public double getBestError() {
        return bestError;
    }

    public double getBestResidual() {
        return bestResidual;
    }

    public double getLastAbsoluteChange() {
        return lastAbsoluteChange;
    }

    public double getLastRelativeChange() {
        return lastRelativeChange;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public double getAbsoluteTolerance() {
        return absoluteTolerance;
    }

    public double getRelativeTolerance() {
        return relativeTolerance;
    }

    public double getResidualTolerance() {
        return residualTolerance;
    }

    public int getStagnationWindow() {
        return stagnationWindow;
    }

    public int getDivergenceWindow() {
        return divergenceWindow;
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    private static boolean isEnabled(double value) {
        return isFinite(value);
    }

    public static final class Builder {
        private int maxIterations = 1000;
        private double absoluteTolerance = Tolerance.get();
        private double relativeTolerance = 1e-8;
        private double residualTolerance = 1e-12;
        private int stagnationWindow = 50;
        private int divergenceWindow = 10;
        private double stagnationTolerance = Double.NaN;

        public Builder maxIterations(int maxIterations) {
            this.maxIterations = Math.max(1, maxIterations);
            return this;
        }

        public Builder absoluteTolerance(double absoluteTolerance) {
            this.absoluteTolerance = normalizeTolerance(absoluteTolerance, "absoluteTolerance");
            return this;
        }

        public Builder relativeTolerance(double relativeTolerance) {
            this.relativeTolerance = normalizeTolerance(relativeTolerance, "relativeTolerance");
            return this;
        }

        public Builder residualTolerance(double residualTolerance) {
            this.residualTolerance = normalizeTolerance(residualTolerance, "residualTolerance");
            return this;
        }

        public Builder stagnationWindow(int stagnationWindow) {
            this.stagnationWindow = Math.max(0, stagnationWindow);
            return this;
        }

        public Builder divergenceWindow(int divergenceWindow) {
            this.divergenceWindow = Math.max(0, divergenceWindow);
            return this;
        }

        public Builder stagnationTolerance(double stagnationTolerance) {
            this.stagnationTolerance = normalizeTolerance(stagnationTolerance, "stagnationTolerance");
            return this;
        }

        public Builder disableAbsoluteTolerance() {
            this.absoluteTolerance = Double.NaN;
            return this;
        }

        public Builder disableRelativeTolerance() {
            this.relativeTolerance = Double.NaN;
            return this;
        }

        public Builder disableResidualTolerance() {
            this.residualTolerance = Double.NaN;
            return this;
        }

        public ConvergenceControl build() {
            return new ConvergenceControl(this);
        }

        private static double normalizeTolerance(double value, String label) {
            if (Double.isNaN(value)) {
                return Double.NaN;
            }
            if (value < 0.0) {
                throw new IllegalArgumentException(label + " must be non-negative");
            }
            return value;
        }
    }
}
