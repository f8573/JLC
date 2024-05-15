class Integer extends Rational {
    int value;

    Integer(int value) {
        super(value, 1);
        this.value = value;
    }

    @Override
    Integer add(Rational other) {
        return new Integer(value + other.numerator);
    }

    @Override
    Integer multiply(Rational other) {
        return new Integer(value * other.numerator);
    }

    @Override
    Integer negate() {
        return new Integer(-value);
    }

    @Override
    public String toString() {
        return "" + value;
    }
}