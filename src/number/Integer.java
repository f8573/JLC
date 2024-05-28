package number;

public class Integer extends Ring {
    public int value;

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

    public Ring power(Ring exponent) {
        if (exponent instanceof Integer) {
            Integer result = new Integer(1);
            for (int i = 0; i < ((Integer) exponent).value; i++) {
                result = (Integer) result.multiply(this);
            }
            return result;
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public Ring multiply(Ring ring) {
        if (ring instanceof Integer) {
            return new Integer(value * ((Integer)ring).value);
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isZero() {
        return value == 0;
    }

    @Override
    public String toString() {
        return "" + value;
    }

    public double eval() {
        return value;
    }

    public Rational toRational() {
        return new Rational(this);
    }
}
