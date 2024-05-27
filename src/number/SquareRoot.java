package number;

public class SquareRoot extends Field {
    /**
     * form of \sqrt{a}, addition not supported
     * really basic im too stupid to implement addition
     *
     */

    Rational base;

    public SquareRoot(Rational rational) {
        this.base = rational;
    }

    public SquareRoot(Integer integer) {
        this.base = integer.toRational();
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

        } else if (ring instanceof Integer || ring instanceof Rational) {
            if (ring instanceof Integer) {
                return new SquareRoot((Rational) ((Integer)ring).power(new Integer(2)).multiply(base));
            } else {
                return new SquareRoot((Rational) ((Rational)ring).power(new Integer(2)).multiply(base));
            }
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isZero() {
        return base.isZero();
    }

    @Override
    public String toString() {
        return "sqrt("+base.toString()+")";
    }

    public double eval() {
        return Math.sqrt(base.eval());
    }
}
