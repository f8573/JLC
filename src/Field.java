abstract class Field<T extends Field<T>> extends Ring<T> {
    abstract T divide(T other);
    abstract T inverse();
}