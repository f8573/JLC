package net.faulj.core;

/**
 * Provides a global tolerance strategy for floating-point comparisons in numerical algorithms.
 * <p>
 * Due to the inherent imprecision of floating-point arithmetic, direct equality comparisons
 * (a == b) are unreliable for computed values. This class implements a thread-safe singleton
 * pattern for managing numerical tolerance thresholds used throughout the library for:
 * </p>
 * <ul>
 *   <li>Zero testing: determining if a value is effectively zero</li>
 *   <li>Equality testing: comparing two floating-point values</li>
 *   <li>Convergence testing: checking if iterative methods have converged</li>
 *   <li>Rank determination: detecting linear dependence in vectors</li>
 * </ul>
 *
 * <h2>Default Tolerance:</h2>
 * <p>
 * The default tolerance is set to <b>1e-10</b>, which is appropriate for most double-precision
 * computations. This value balances the need to account for rounding errors while remaining
 * strict enough to detect meaningful differences.
 * </p>
 *
 * <h2>Usage Guidelines:</h2>
 * <ul>
 *   <li><b>Standard comparisons:</b> Use {@code Tolerance.isZero(x)} instead of {@code x == 0}</li>
 *   <li><b>Equality checks:</b> Use {@code Tolerance.equals(a, b)} instead of {@code a == b}</li>
 *   <li><b>Custom tolerance:</b> Pass explicit tolerance to overloaded methods when needed</li>
 *   <li><b>Configuration:</b> Call {@code Tolerance.set(tol)} once at application startup</li>
 * </ul>
 *
 * <h2>Choosing Tolerance Values:</h2>
 * <ul>
 *   <li><b>1e-15:</b> Very strict, close to machine epsilon for double (use with caution)</li>
 *   <li><b>1e-12:</b> Tight tolerance for high-precision requirements</li>
 *   <li><b>1e-10:</b> Default, suitable for most numerical linear algebra</li>
 *   <li><b>1e-8:</b> Relaxed tolerance for less sensitive computations</li>
 *   <li><b>1e-6:</b> Loose tolerance for iterative methods or ill-conditioned problems</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Application initialization
 * Tolerance.set(1e-12); // Set stricter tolerance globally
 *
 * // Standard zero check
 * if (Tolerance.isZero(determinant)) {
 *     throw new SingularMatrixException("Matrix is singular");
 * }
 *
 * // Equality comparison
 * if (Tolerance.equals(eigenvalue1, eigenvalue2)) {
 *     System.out.println("Repeated eigenvalue detected");
 * }
 *
 * // Custom tolerance for specific comparison
 * if (Tolerance.isZero(residual, 1e-15)) {
 *     System.out.println("Converged to high precision");
 * }
 * }</pre>
 *
 * <h2>Absolute vs Relative Tolerance:</h2>
 * <p>
 * This class implements <i>absolute tolerance</i> comparisons. For very large or very small
 * numbers, consider using <i>relative tolerance</i>:
 * </p>
 * <pre>{@code
 * double relativeTolerance = 1e-10;
 * boolean equal = Math.abs(a - b) <= relativeTolerance * Math.max(Math.abs(a), Math.abs(b));
 * }</pre>
 *
 * <h2>Thread Safety:</h2>
 * <p>
 * The tolerance value is stored in a {@code volatile} field, ensuring thread-safe reads and
 * writes. However, frequent modification of the global tolerance in multi-threaded environments
 * is discouraged. Set it once during initialization.
 * </p>
 *
 * <h2>Immutability Note:</h2>
 * <p>
 * While the tolerance value can be changed via {@code set()}, most computations should use a
 * consistent tolerance throughout their execution to ensure predictable behavior.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see ConvergenceControl
 * @see net.faulj.matrix.Matrix
 */
public final class Tolerance {
    
    private static volatile double defaultTolerance = 1e-10;
    
    private Tolerance() {}
    
    /**
     * Get the global tolerance value.
     *
     * @return tolerance
     */
    public static double get() {
        return defaultTolerance;
    }
    
    /**
     * Set the global tolerance value.
     *
     * @param tol new tolerance (non-negative)
     */
    public static void set(double tol) {
        if (tol < 0) {
            throw new IllegalArgumentException("Tolerance must be non-negative");
        }
        defaultTolerance = tol;
    }
    
    /**
     * Check if a value is within the global tolerance of zero.
     *
     * @param value value to test
     * @return true if value is approximately zero
     */
    public static boolean isZero(double value) {
        return Math.abs(value) <= defaultTolerance;
    }
    
    /**
     * Check if a value is within a custom tolerance of zero.
     *
     * @param value value to test
     * @param tol tolerance
     * @return true if value is approximately zero
     */
    public static boolean isZero(double value, double tol) {
        return Math.abs(value) <= tol;
    }
    
    /**
     * Compare two values within the global tolerance.
     *
     * @param a first value
     * @param b second value
     * @return true if values are approximately equal
     */
    public static boolean equals(double a, double b) {
        return Math.abs(a - b) <= defaultTolerance;
    }
    
    /**
     * Compare two values within a custom tolerance.
     *
     * @param a first value
     * @param b second value
     * @param tol tolerance
     * @return true if values are approximately equal
     */
    public static boolean equals(double a, double b, double tol) {
        return Math.abs(a - b) <= tol;
    }
}
