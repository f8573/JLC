package number;

public class SquareRoot extends Field {
    /**
     * form of \sqrt{a}, addition not supported
     * really basic im too stupid to implement addition
     *
     */

    Rational base;
    boolean sign; //true = positive, false = negative

    public SquareRoot(Rational rational, boolean sign) {
        base = rational;
        this.sign = sign;
    }

    public SquareRoot(Rational rational) {
        this.base = rational;
        sign = true;
    }

    public SquareRoot(Integer integer) {
        this.base = integer.toRational();
        sign = true;
    }

    @Override
    public Field additiveInverse() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Field multiplicativeInverse() {
        return new SquareRoot((Rational) base.multiplicativeInverse());
    }

    @Override
    public Ring add(Ring ring) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Ring multiply(Ring ring) {
        if (ring instanceof SquareRoot) {
            boolean sign1 = sign;
            boolean sign2 = ((SquareRoot) ring).sign;
            Rational base1 = base;
            Rational base2 = ((SquareRoot) ring).base;
            return new SquareRoot((Rational) base1.multiply(base2),sign1 && sign2);
        } else if (ring instanceof Integer || ring instanceof Rational) {
            Rational newBase;
            boolean newSign;
            if (ring instanceof Integer) {
                newBase = ((Rational) ((Integer)ring).power(new Integer(2)).multiply(base));
                newSign = ((Integer) ring).value >= 0;
            } else {
                newBase = ((Rational) ((Rational)ring).power(new Integer(2)).multiply(base));
                newSign = ((Rational) ring).isPositive();
            }
            return new SquareRoot(newBase, sign && newSign);
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isZero() {
        return base.isZero();
    }

    @Override
    public String toString() {
        Integer numSimp = null;
        Integer denomSimp = null;
        if (Math.sqrt(base.numerator.value) == Math.round(Math.sqrt(base.numerator.value))) {
            numSimp = new Integer((int)Math.sqrt(base.numerator.value));
        }
        if (Math.sqrt(base.denominator.value) == Math.round(Math.sqrt(base.denominator.value))) {
            denomSimp = new Integer((int)Math.sqrt(base.denominator.value));
        }
        if (numSimp == null && denomSimp == null) {
            return (sign ? "" : "-") +"sqrt("+ base +")";
        } else {
            String numString;
            String denomString;
            if (numSimp != null) {
                numString = numSimp.toString();
            } else {
                numString = "sqrt(" + base.numerator + ")";
            }
            if (denomSimp != null) {
                denomString = denomSimp.toString();
            } else {
                denomString = "sqrt(" + base.denominator + ")";
            }
            return (sign ? "" : "-") + numString + "/" + denomString;
        }
    }

    public double eval() {
        return Math.sqrt(base.eval());
    }
}
