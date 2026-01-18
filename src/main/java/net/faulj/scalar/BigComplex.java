package net.faulj.scalar;

/**
 * Arbitrary-precision complex number representation.
 * <p>
 * This class mirrors the functionality of {@link Complex} but uses {@link BigDouble}
 * (or standard {@code BigDecimal}) for its internal component storage, allowing for
 * calculations with precision exceeding standard 64-bit IEEE 754 floating point.
 * </p>
 *
 * <h2>Comparison with Complex:</h2>
 * <table border="1">
 * <tr><th>Feature</th><th>Complex</th><th>BigComplex</th></tr>
 * <tr><td>Storage</td><td>{@code double} (64-bit)</td><td>{@code BigDouble} (Arbitrary)</td></tr>
 * <tr><td>Precision</td><td>~15-17 decimal digits</td><td>User configurable (e.g., 100+ digits)</td></tr>
 * <tr><td>Performance</td><td>Very Fast (CPU native)</td><td>Slower (Software emulation)</td></tr>
 * <tr><td>Memory</td><td>Minimal (16 bytes payload)</td><td>Higher (Object overhead)</td></tr>
 * </table>
 *
 * <h2>Usage Scenarios:</h2>
 * <ul>
 * <li><b>Ill-conditioned matrices:</b> Where standard double precision leads to catastrophic cancellation.</li>
 * <li><b>Scientific computing:</b> High-precision physical simulations (e.g., Quantum Mechanics).</li>
 * <li><b>Number theory:</b> Experimental mathematics requiring high decimal accuracy.</li>
 * </ul>
 *
 * <h2>Implementation Note:</h2>
 * <p>
 * Like {@link Complex}, this class is designed to be immutable and thread-safe.
 * Operations return new instances rather than modifying the current object.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see Complex
 * @see BigDouble
 */
public class BigComplex {
    public BigComplex() {
        throw new RuntimeException("Class not implemented");
    }
}