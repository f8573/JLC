abstract class Ring<T extends Ring<T>> {
    abstract T add(T other);
    abstract T multiply(T other);
    abstract T negate();
    T subtract(T other) {
        return add(other.negate());
    }
}