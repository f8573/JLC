package number;

public abstract class Field extends Ring {
    public abstract Field additiveInverse();
    public abstract Field multiplicativeInverse();

    public Field subtract(Field field) {
        return (Field) add(field.additiveInverse());
    }

    public Field divide(Field field) {
        return (Field) add(field.multiplicativeInverse());
    }
}
