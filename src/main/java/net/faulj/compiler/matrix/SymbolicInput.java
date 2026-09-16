package net.faulj.compiler.matrix;

/**
 * Shape-only input for planning and optimizer tests.
 *
 * <p>A symbolic input has no runtime matrix and therefore cannot be
 * evaluated.</p>
 */
public final class SymbolicInput implements MatrixExpr {
    private final String name;
    private final MatrixShape shape;

    public SymbolicInput(String name, MatrixShape shape) {
        if (shape == null) {
            throw new IllegalArgumentException("Symbolic input shape must not be null");
        }
        this.name = normalizeName(name);
        this.shape = shape;
    }

    public SymbolicInput(MatrixShape shape, String name) {
        this(name, shape);
    }

    public String name() {
        return name;
    }

    public boolean hasName() {
        return name != null;
    }

    @Override
    public MatrixShape shape() {
        return shape;
    }

    private static String normalizeName(String name) {
        if (name == null) {
            return null;
        }
        String trimmed = name.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }
}
