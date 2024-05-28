package number;

public class Rational extends Field {
    Integer numerator;
    Integer denominator;

    public Rational(Integer numerator) {
        this.numerator = numerator;
        this.denominator = new Integer(1);
        reduce();
    }

    public Rational(Integer numerator, Integer denominator) {
        if (denominator.value == 0) {
            throw new RuntimeException("a/0 is undefined");
        }
        this.numerator = numerator;
        this.denominator = denominator;
        reduce();
    }

    public Rational(int numerator) {
        this(new Integer(numerator));
    }

    public Rational(int numerator, int denominator) {
        this(new Integer(numerator), new Integer(denominator));
    }

    public void reduce() {
        int gcd = gcf(numerator.value, denominator.value);
        numerator = new Integer(numerator.value / gcd);
        denominator = new Integer(denominator.value / gcd);
        if (denominator.value < 0) {
            numerator.value *= -1;
            denominator.value *= -1;
        }
    }

    public boolean isInteger() {
        return denominator.value == 1;
    }

    public Integer toInteger() {
        return denominator.value == 1 ? numerator : null;
    }

    public Ring power(Ring ring) {
        if (ring instanceof Integer) {
            return new Rational((Integer) numerator.power(ring), (Integer) denominator.power(ring));
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public Field additiveInverse() {
        return new Rational((Integer) numerator.multiply(new Integer(-1)), denominator);
    }

    @Override
    public Field multiplicativeInverse() {
        return new Rational(denominator, numerator);
    }

    @Override
    public Ring add(Ring ring) {
        if (ring instanceof Integer) {
            // a/b + c = (a + bc) / b
            int a = numerator.value;
            int b = denominator.value;
            int c = ((Integer) ring).value;
            return new Rational(a + b * c, b);
        } else if (ring instanceof Rational) {
            // a/b + c/d = (ad + bc) / bd
            int a = numerator.value;
            int b = denominator.value;
            int c = ((Rational) ring).numerator.value;
            int d = ((Rational) ring).denominator.value;
            return new Rational(a * d + b * c, b * d);
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public Ring multiply(Ring ring) {
        if (ring instanceof Integer) {
            return new Rational(numerator.value * ((Integer) ring).value, denominator.value);
        } else if (ring instanceof Rational) {
            return new Rational(numerator.value * ((Rational) ring).numerator.value, denominator.value * ((Rational) ring).denominator.value);
        } else if (ring instanceof SquareRoot) {
            return ring.multiply(this);
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isZero() {
        return numerator.isZero();
    }

    public int gcf(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    public double eval() {
        return numerator.eval() / denominator.eval();
    }

    public boolean isPositive() {
        return numerator.value >= 0;
    }

    @Override
    public String toString() {
        if (numerator.eval() / denominator.eval() == Math.round(numerator.eval() / denominator.eval())) {
            return "" + (int) (numerator.eval() / denominator.eval());
        }
        return numerator.toString() + "/" + denominator.toString();
    }
}
