package net.faulj.scalar;

/**
 * Immutable representation of a complex number z = x + iy.
 * <p>
 * This class provides standard arithmetic operations, algebraic functions, and
 * utilities for complex number manipulation using double-precision floating-point
 * arithmetic.
 * </p>
 *
 * <h2>Mathematical Definition:</h2>
 * <p>
 * A complex number z is defined as an ordered pair of real numbers (x, y), often written as:
 * </p>
 * <pre>
 * z = x + i·y
 * </pre>
 * <ul>
 * <li><b>x</b> - The real part (Re(z))</li>
 * <li><b>y</b> - The imaginary part (Im(z))</li>
 * <li><b>i</b> - The imaginary unit, satisfying i² = -1</li>
 * </ul>
 *
 * <h2>Arithmetic Operations:</h2>
 * <p>
 * Given two complex numbers a = (x₁, y₁) and b = (x₂, y₂):
 * </p>
 * <ul>
 * <li><b>Addition:</b> a + b = (x₁ + x₂, y₁ + y₂)</li>
 * <li><b>Subtraction:</b> a - b = (x₁ - x₂, y₁ - y₂)</li>
 * <li><b>Multiplication:</b> a · b = (x₁x₂ - y₁y₂, x₁y₂ + x₂y₁)</li>
 * <li><b>Division:</b> a / b = (x₁x₂ + y₁y₂)/(x₂² + y₂²) + i·(y₁x₂ - x₁y₂)/(x₂² + y₂²)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Complex z1 = new Complex(3.0, 4.0);
 * Complex z2 = Complex.I;
 *
 * // Operations
 * Complex sum = z1.add(z2);        // 3.0 + 5.0i
 * Complex product = z1.multiply(z2); // -4.0 + 3.0i
 * double magnitude = z1.abs();     // 5.0
 *
 * // Chaining
 * Complex result = z1.add(z2).conjugate().divide(Complex.valueOf(2.0));
 * }</pre>
 *
 * <h2>Immutability and Thread Safety:</h2>
 * <p>
 * This class is immutable. Both {@code real} and {@code imag} fields are {@code final}
 * and primitive. Instances of this class are thread-safe and can be shared freely
 * between threads without synchronization.
 * </p>
 *
 * <h2>Precision:</h2>
 * <p>
 * Operations are performed using standard 64-bit IEEE 754 floating point arithmetic ({@code double}).
 * Accuracy is subject to standard machine precision limits and round-off errors.
 * For arbitrary precision, see {@link BigComplex}.
 * </p>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Eigenvalue computation (roots of characteristic polynomials)</li>
 * <li>Signal processing (Fourier Transforms)</li>
 * <li>Control theory (Transfer functions, Bode plots)</li>
 * <li>Fractal generation (Mandelbrot sets)</li>
 * <li>AC circuit analysis (Phasors)</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see BigComplex
 */
public class Complex {
    /** The real part of this complex number. */
    public final double real;
    /** The imaginary part of this complex number. */
    public final double imag;

    /** A complex number representing zero (0 + 0i). */
    public static final Complex ZERO = new Complex(0, 0);
    /** A complex number representing one (1 + 0i). */
    public static final Complex ONE = new Complex(1, 0);
    /** A complex number representing the imaginary unit i (0 + 1i). */
    public static final Complex I = new Complex(0, 1);

    /**
     * Constructs a new Complex number.
     *
     * @param real The real part.
     * @param imag The imaginary part.
     */
    public Complex(double real, double imag) {
        this.real = real;
        this.imag = imag;
    }

    /**
     * Creates a complex number from a real value (imaginary part is 0).
     *
     * @param real The real value.
     * @return A new Complex instance z = real + 0i.
     */
    public static Complex valueOf(double real) {
        return new Complex(real, 0);
    }

    /**
     * Creates a complex number from real and imaginary parts.
     *
     * @param real The real part.
     * @param imag The imaginary part.
     * @return A new Complex instance z = real + i*imag.
     */
    public static Complex valueOf(double real, double imag) {
        return new Complex(real, imag);
    }

    /**
     * Returns the sum of this complex number and another.
     * <p>Formula: (a + bi) + (c + di) = (a+c) + (b+d)i</p>
     *
     * @param other The complex number to add.
     * @return A new Complex representing the sum.
     */
    public Complex add(Complex other) {
        return new Complex(real + other.real, imag + other.imag);
    }

    /**
     * Returns the difference between this complex number and another.
     * <p>Formula: (a + bi) - (c + di) = (a-c) + (b-d)i</p>
     *
     * @param other The complex number to subtract.
     * @return A new Complex representing the difference.
     */
    public Complex subtract(Complex other) {
        return new Complex(real - other.real, imag - other.imag);
    }

    /**
     * Returns the product of this complex number and another.
     * <p>Formula: (a + bi)(c + di) = (ac - bd) + (ad + bc)i</p>
     *
     * @param other The complex number to multiply by.
     * @return A new Complex representing the product.
     */
    public Complex multiply(Complex other) {
        return new Complex(real * other.real - imag * other.imag, real * other.imag + imag * other.real);
    }

    /**
     * Returns the product of this complex number and a scalar double.
     * <p>Formula: s(a + bi) = (sa) + (sb)i</p>
     *
     * @param scalar The real number scalar.
     * @return A new Complex representing the scaled value.
     */
    public Complex multiply(double scalar) {
        return new Complex(real * scalar, imag * scalar);
    }

    /**
     * Returns the result of dividing this complex number by another.
     * <p>
     * This implementation uses the standard algebraic formula:
     * </p>
     * <pre>
     * a + bi    (ac + bd) + (bc - ad)i
     * ------ =  ----------------------
     * c + di          c² + d²
     * </pre>
     *
     * @param other The divisor.
     * @return A new Complex representing the quotient.
     */
    public Complex divide(Complex other) {
        double denominator = other.real * other.real + other.imag * other.imag;
        return new Complex(
                (real * other.real + imag * other.imag) / denominator,
                (imag * other.real - real * other.imag) / denominator
        );
    }

    /**
     * Returns the complex conjugate of this number.
     * <p>Formula: conj(a + bi) = a - bi</p>
     *
     * @return A new Complex number with the sign of the imaginary part negated.
     */
    public Complex conjugate() {
        return new Complex(real, -imag);
    }

    /**
     * Returns the absolute value (magnitude/modulus) of this complex number.
     * <p>Formula: |z| = sqrt(x² + y²)</p>
     * <p>This method uses {@code Math.hypot} to avoid intermediate overflow/underflow.</p>
     *
     * @return The magnitude of the complex number.
     */
    public double abs() {
        return Math.hypot(real, imag);
    }

    /**
     * Returns the principal square root of this complex number.
     * <p>
     * The branch cut is on the negative real axis. The result is always in the
     * right half-plane (Re(sqrt(z)) >= 0).
     * </p>
     * <p>
     * Formula used:
     * <pre>
     * sqrt(z) = ±[ sqrt((|z| + x)/2) + i·sgn(y)sqrt((|z| - x)/2) ]
     * </pre>
     *
     * @return A new Complex representing the square root.
     */
    public Complex sqrt() {
        double r = abs();
        return new Complex(Math.sqrt((r + real) / 2), Math.copySign(Math.sqrt((r - real) / 2), imag));
    }

    /**
     * Checks if this complex number is effectively real (imaginary part is zero).
     *
     * @return true if the imaginary part is exactly 0.0.
     */
    public boolean isReal() {
        return imag == 0;
    }

    /**
     * Returns a string representation of this complex number.
     * <p>Formats:</p>
     * <ul>
     * <li>Real only: "a.0" if imag is 0</li>
     * <li>Standard: "a.0 + b.0i"</li>
     * <li>Negative imag: "a.0 - b.0i"</li>
     * </ul>
     *
     * @return A string representation in Cartesian form.
     */
    @Override
    public String toString() {
        if (imag == 0) return String.valueOf(real);
        if (imag > 0) return real + " + " + imag + "i";
        return real + " - " + (-imag) + "i";
    }
}