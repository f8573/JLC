public abstract class Field<T extends Field<T>> extends Ring<T> {
    public abstract T additiveInverse();
    public abstract T multiplicativeInverse();

    public T subtract(T other) {
        return add(other.additiveInverse());
    }

    public T divide(T other) {
        return multiply(other.multiplicativeInverse());
    }
}
