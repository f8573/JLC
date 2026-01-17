package net.faulj.scalar;

public class Complex {
    public final double real;
    public final double imag;

    public static final Complex ZERO = new Complex(0, 0);
    public static final Complex ONE = new Complex(1, 0);
    public static final Complex I = new Complex(0, 1);

    public Complex(double real, double imag) {
        this.real = real;
        this.imag = imag;
    }

    public static Complex valueOf(double real) {
        return new Complex(real, 0);
    }

    public static Complex valueOf(double real, double imag) {
        return new Complex(real, imag);
    }

    public Complex add(Complex other) {
        return new Complex(real + other.real, imag + other.imag);
    }

    public Complex subtract(Complex other) {
        return new Complex(real - other.real, imag - other.imag);
    }

    public Complex multiply(Complex other) {
        return new Complex(real * other.real - imag * other.imag, real * other.imag + imag * other.real);
    }

    public Complex multiply(double scalar) {
        return new Complex(real * scalar, imag * scalar);
    }

    public Complex divide(Complex other) {
        double denominator = other.real * other.real + other.imag * other.imag;
        return new Complex(
                (real * other.real + imag * other.imag) / denominator,
                (imag * other.real - real * other.imag) / denominator
        );
    }

    public Complex conjugate() {
        return new Complex(real, -imag);
    }

    public double abs() {
        return Math.hypot(real, imag);
    }

    public Complex sqrt() {
        double r = abs();
        return new Complex(Math.sqrt((r + real) / 2), Math.copySign(Math.sqrt((r - real) / 2), imag));
    }

    public boolean isReal() {
        return imag == 0;
    }

    @Override
    public String toString() {
        if (imag == 0) return String.valueOf(real);
        if (imag > 0) return real + " + " + imag + "i";
        return real + " - " + (-imag) + "i";
    }
}