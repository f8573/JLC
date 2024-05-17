import java.util.List;

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
        return new SquareRoot((Rational) base.multiply(ring));
    }
}
