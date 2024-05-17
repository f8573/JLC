public class Integer extends Ring {
    int value;

    public Integer(int val) {
        value = val;
    }

    @Override
    public Ring add(Ring ring) {
        if (ring instanceof Integer) {
            return new Integer(value + ((Integer)ring).value);
        } else if (ring instanceof Rational) {
            return ring.add(this);
        }
        throw new UnsupportedOperationException();
    }

    public Integer additiveInverse() {
        return new Integer(-value);
    }

    @Override
    public Ring multiply(Ring ring) {
        if (ring instanceof Integer) {
            return new Integer(value * ((Integer)ring).value);
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
        return "" + value;
    }

    public Rational toRational() {
        return new Rational(this);
    }
}
