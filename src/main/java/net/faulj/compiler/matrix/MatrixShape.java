package net.faulj.compiler.matrix;

import net.faulj.matrix.Matrix;

/**
 * Immutable matrix dimensions used by the expression IR and planner.
 */
public record MatrixShape(int rows, int columns) {
    public MatrixShape {
        if (rows < 0 || columns < 0) {
            throw new IllegalArgumentException(
                "Matrix shape dimensions must be non-negative: " + rows + "x" + columns);
        }
    }

    /**
     * Create a shape from an existing JLC matrix.
     */
    public static MatrixShape from(Matrix matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        return new MatrixShape(matrix.getRowCount(), matrix.getColumnCount());
    }

    /**
     * Alias useful when reading matrix code that uses {@code cols}.
     */
    public int cols() {
        return columns;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public int getColumnCount() {
        return columns;
    }

    public int getRowCount() {
        return rows;
    }

    @Override
    public String toString() {
        return rows + "x" + columns;
    }
}
