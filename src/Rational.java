public class Rational extends Field {
    Integer numerator;
    Integer denominator;

    public Rational(Integer numerator) {
        this.numerator = numerator;
        this.denominator = new Integer(1);
    }

    public Rational(Integer numerator, Integer denominator) {
        this.numerator = numerator;
        this.denominator = denominator;
    }

    public Rational(int numerator) {
        this(new Integer(numerator));
    }

    public Rational(int numerator, int denominator) {
        this(new Integer(numerator),new Integer(denominator));
    }

    public boolean isInteger() {
        return denominator.value == 1;
    }

    public Integer toInteger() {
        return denominator.value == 1 ? numerator : null;
    }

    @Override
    public Field additiveInverse() {
        return new Rational(numerator, denominator);
    }

    @Override
    public Field multiplicativeInverse() {
        return new Rational(denominator, numerator);
    }

    @Override
    public Ring add(Ring ring) {
        if (ring instanceof Integer) {
            //a/b+c=(cb+a)/b
            int a = numerator.value;
            int b = denominator.value;
            int c = ((Integer) ring).value;
            return new Rational(a+b*c,b);
        } else if (ring instanceof Rational) {
            //a/b+c/d=(ad+cb)/bd
            int a = numerator.value;
            int b = denominator.value;
            int c = ((Rational) ring).numerator.value;
            int d = ((Rational) ring).denominator.value;
            return new Rational(a*d+b*c,b*d);
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public Ring multiply(Ring ring) {
        if (ring instanceof Integer) {
            return new Rational(((Integer) ring).value);
        } else if (ring instanceof Rational) {
            return new Rational(((Rational) ring).numerator.value * numerator.value, ((Rational) ring).denominator.value * denominator.value);
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
        return numerator.toString() + "/" + denominator.toString();
    }
}
