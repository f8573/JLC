package net.faulj.compiler.matrix;

import net.faulj.matrix.Matrix;

/**
 * Runtime matrix input to an expression graph.
 *
 * <p>The matrix reference is intentionally not copied. M1's contract is that
 * an input matrix must not be mutated concurrently with graph evaluation.</p>
 */
public final class Input implements MatrixExpr {
    private final Matrix matrix;
    private final String name;
    private final MatrixShape shape;

    public Input(Matrix matrix) {
        this(null, matrix);
    }

    public Input(String name, Matrix matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Input matrix must not be null");
        }
        this.matrix = matrix;
        this.name = normalizeName(name);
        this.shape = MatrixShape.from(matrix);
    }

    public Input(Matrix matrix, String name) {
        this(name, matrix);
    }

    public Matrix matrix() {
        return matrix;
    }

    public Matrix value() {
        return matrix;
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
