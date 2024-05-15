public class IntegerField extends Field<IntegerField> {
    private final int value;

    public IntegerField(int value) {
        this.value = value;
    }

    @Override
    public IntegerField add(IntegerField other) {
        return new IntegerField(this.value + other.value);
    }

    @Override
    public IntegerField multiply(IntegerField other) {
        return new IntegerField(this.value * other.value);
    }

    @Override
    public IntegerField additiveInverse() {
        return new IntegerField(-this.value);
    }

    @Override
    public IntegerField multiplicativeInverse() {
        if (this.value == 0) {
            throw new ArithmeticException("Division by zero");
        }
        return new IntegerField(1 / this.value);
    }

    public int getValue() {
        return value;
    }

    @Override
    public String toString() {
        return String.valueOf(value);
    }
}
