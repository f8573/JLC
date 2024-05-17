public class Integer extends Ring {
    int value;

    public Integer(int val) {
        value = val;
    }

    @Override
    public Integer add(Ring ring) {
        if (ring instanceof Integer) {
            return new Integer(value + ((Integer)ring).value);
        }
        throw new UnsupportedOperationException();
    }

    public Integer additiveInverse() {
        return new Integer(-value);
    }

    @Override
    public Integer multiply(Ring ring) {
        if (ring instanceof Integer) {
            return new Integer(value * ((Integer)ring).value);
        }
        throw new UnsupportedOperationException();
    }
}
