public class RationalField extends Field<RationalField> {
    private final int numerator;
    private final int denominator;

    public RationalField(int numerator, int denominator) {
        if (denominator == 0) {
            throw new ArithmeticException("Denominator cannot be zero");
        }

        // Simplify the fraction to its lowest terms
        int gcd = gcd(numerator, denominator);
        this.denominator = denominator / gcd < 0 ? denominator / gcd * -1 : denominator / gcd;
        this.numerator = denominator / gcd < 0 ? numerator / gcd * -1 : numerator / gcd;
    }

    private static int gcd(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    @Override
    public RationalField add(RationalField other) {
        int commonDenominator = this.denominator * other.denominator;
        int newNumerator = this.numerator * other.denominator + other.numerator * this.denominator;
        return new RationalField(newNumerator, commonDenominator).simplify();
    }

    @Override
    public RationalField multiply(RationalField other) {
        return new RationalField(this.numerator * other.numerator, this.denominator * other.denominator).simplify();
    }

    @Override
    public RationalField additiveInverse() {
        return new RationalField(-this.numerator, this.denominator);
    }

    @Override
    public RationalField multiplicativeInverse() {
        if (this.numerator == 0) {
            throw new ArithmeticException("Cannot invert zero");
        }
        return new RationalField(this.denominator, this.numerator);
    }

    public RationalField subtract(RationalField other) {
        return this.add(other.additiveInverse());
    }

    public RationalField divide(RationalField other) {
        return this.multiply(other.multiplicativeInverse());
    }

    private RationalField simplify() {
        int gcd = gcd(this.numerator, this.denominator);
        return new RationalField(this.numerator / gcd, this.denominator / gcd);
    }

    @Override
    public String toString() {
        if (denominator == 1) {
            return String.valueOf(numerator);
        }
        return numerator + "/" + denominator;
    }

    public int getNumerator() {
        return numerator;
    }

    public int getDenominator() {
        return denominator;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        RationalField that = (RationalField) obj;
        return numerator == that.numerator && denominator == that.denominator;
    }

    @Override
    public int hashCode() {
        int result = numerator;
        result = 31 * result + denominator;
        return result;
    }
}
