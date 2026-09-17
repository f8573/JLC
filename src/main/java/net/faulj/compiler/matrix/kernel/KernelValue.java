package net.faulj.compiler.matrix.kernel;

/** Immutable SSA-like scalar value declaration. */
public final class KernelValue {
    private final int id;
    private final KernelValueType type;
    private final String name;

    public KernelValue(int id, KernelValueType type, String name) {
        this.id = id;
        this.type = type;
        this.name = name;
    }

    public KernelValue(int id, KernelValueType type) {
        this(id, type, "k" + id);
    }

    public int id() {
        return id;
    }

    public KernelValueType type() {
        return type;
    }

    public String name() {
        return name;
    }

    @Override
    public String toString() {
        return name == null || name.isBlank() ? "k" + id : name;
    }
}
