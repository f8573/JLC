public abstract class Ring<T extends Ring<T>> {
    public abstract T add(T other);
    public abstract T multiply(T other);
}