package net.faulj.matrix;

/**
 * Lightweight row-major view into an existing matrix buffer.
 * <p>
 * Uses an explicit base offset and leading dimension (row stride)
 * to describe submatrices without copying.
 */
public final class MatrixView {
    private final double[] data;
    private final int offset;
    private final int rows;
    private final int cols;
    private final int ld;

    public MatrixView(double[] data, int offset, int rows, int cols, int ld) {
        if (data == null) {
            throw new IllegalArgumentException("Data must not be null");
        }
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Matrix dimensions must be non-negative");
        }
        if (ld < cols) {
            throw new IllegalArgumentException("Leading dimension must be >= number of columns");
        }
        if (offset < 0) {
            throw new IllegalArgumentException("Offset must be non-negative");
        }
        if (rows > 0 && cols > 0) {
            int last = offset + (rows - 1) * ld + (cols - 1);
            if (last >= data.length) {
                throw new IllegalArgumentException("View exceeds backing array length");
            }
        } else if (offset > data.length) {
            throw new IllegalArgumentException("Offset exceeds backing array length");
        }
        this.data = data;
        this.offset = offset;
        this.rows = rows;
        this.cols = cols;
        this.ld = ld;
    }

    /**
     * Create a contiguous row-major view over the full array.
     */
    public static MatrixView wrap(double[] data, int rows, int cols) {
        return new MatrixView(data, 0, rows, cols, cols);
    }

    /**
     * Create a row-major view over a Matrix (no copy).
     */
    public static MatrixView of(Matrix matrix, int rowStart, int colStart, int rows, int cols) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (rowStart < 0 || colStart < 0 || rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Invalid view dimensions");
        }
        if (rowStart + rows > matrix.getRowCount() || colStart + cols > matrix.getColumnCount()) {
            throw new IllegalArgumentException("View exceeds matrix bounds");
        }
        int ld = matrix.getColumnCount();
        int offset = rowStart * ld + colStart;
        return new MatrixView(matrix.getRawData(), offset, rows, cols, ld);
    }

    /**
     * Create a subview of this view.
     */
    public MatrixView view(int rowStart, int colStart, int rows, int cols) {
        if (rowStart < 0 || colStart < 0 || rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Invalid view dimensions");
        }
        if (rowStart + rows > this.rows || colStart + cols > this.cols) {
            throw new IllegalArgumentException("View exceeds matrix bounds");
        }
        int newOffset = offset + rowStart * ld + colStart;
        return new MatrixView(data, newOffset, rows, cols, ld);
    }

    public double[] data() {
        return data;
    }

    public int offset() {
        return offset;
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public int ld() {
        return ld;
    }
}
