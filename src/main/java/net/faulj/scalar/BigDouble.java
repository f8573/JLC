package net.faulj.scalar;

/**
 * Arbitrary-precision real number wrapper.
 * <p>
 * This class serves as a high-precision alternative to the primitive {@code double} type.
 * It is typically implemented as a wrapper around {@link java.math.BigDecimal}, adding
 * mathematical utility functions (sin, cos, sqrt, etc.) computed to arbitrary precision
 * using Taylor series or Newton-Raphson iteration.
 * </p>
 *
 * <h2>Why use BigDouble?</h2>
 * <p>
 * Standard IEEE 754 doubles have limited precision (approx 16 decimal digits).
 * In iterative matrix decompositions (like QR or SVD) or when dealing with matrices
 * having high condition numbers, rounding errors can accumulate rapidly, destroying results.
 * BigDouble allows computations to maintain validity in these extreme cases.
 * </p>
 *
 * <h2>Features:</h2>
 * <ul>
 * <li><b>Configurable Precision:</b> Define context for number of significant digits.</li>
 * <li><b>Mathematical Context:</b> Handles rounding modes (FLOOR, CEILING, HALF_UP).</li>
 * <li><b>Extended Range:</b> Supports exponents far exceeding {@code Double.MAX_VALUE}.</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Example of precision loss in double vs BigDouble
 * double d = 1.0;
 * d += 1e-20; // No change in standard double due to underflow
 *
 * BigDouble bd = new BigDouble("1.0");
 * bd = bd.add(new BigDouble("1e-20")); // Correctly stores 1.000...001
 * }</pre>
 *
 * <h2>Immutability:</h2>
 * <p>
 * This class is immutable. All arithmetic operations return new instances.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see java.math.BigDecimal
 * @see BigComplex
 */
public class BigDouble {
    public BigDouble() {
        throw new RuntimeException("Class not implemented");
    }
}