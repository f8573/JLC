class Rational extends Real<Rational> {
    int numerator;
    int denominator;

    Rational(int numerator, int denominator) {
        if (denominator == 0)
            throw new IllegalArgumentException("Denominator cannot be zero.");

        int gcd = gcd(numerator, denominator);
        this.numerator = numerator / gcd;
        this.denominator = denominator / gcd;

        if (this.denominator < 0) {
            this.numerator = -this.numerator;
            this.denominator = -this.denominator;
        }
    }

    private static int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    @Override
    Rational add(Rational other) {
        int newNumerator = numerator * other.denominator + other.numerator * denominator;
        int newDenominator = denominator * other.denominator;
        return new Rational(newNumerator, newDenominator);
    }

    @Override
    Rational multiply(Rational other) {
        int newNumerator = numerator * other.numerator;
        int newDenominator = denominator * other.denominator;
        return new Rational(newNumerator, newDenominator);
    }

    @Override
    Rational divide(Rational other) {
        if (other.numerator == 0)
            throw new IllegalArgumentException("Cannot divide by zero.");
        return multiply(other.inverse());
    }

    @Override
    Rational inverse() {
        if (numerator == 0)
            throw new IllegalArgumentException("Zero does not have an inverse.");
        return new Rational(denominator, numerator);
    }

    @Override
    Rational negate() {
        return new Rational(-numerator, denominator);
    }

    @Override
    double eval() {
        return (double) numerator / denominator;
    }

    @Override
    public String toString() {
        return numerator+"/"+denominator;
    }
}