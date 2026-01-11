package net.faulj.matrix;

import net.faulj.vector.Vector;

public class Matrix {
    private Vector[] data;
    private final int columns;
    private int exchanges;

    public Matrix(Vector[] data) {
        this.data = data;
        this.columns = data.length;
    }

    public Vector[] getData() {
        return data;
    }

    public void setData(Vector[] data) {
        this.data = data;
    }

    public double get(int row, int column) {
        return data[column].get(row);
    }

    public void set(int row, int column, double value) {
        data[column].set(row, value);
    }

    public int getRowCount() {
        return data.length > 0 ? data[0].dimension() : 0;
    }

    public int getColumnCount() {
        return columns;
    }

    public void exchangeRows(int row1, int row2) {
        if (row1 < 0 || row2 < 0 || row1 >= getRowCount() || row2 >= getRowCount()) {
            throw new IllegalArgumentException("Invalid row indices");
        }
        for (int col = 0; col < columns; col++) {
            double temp = get(row1, col);
            set(row1, col, get(row2, col));
            set(row2, col, temp);
        }
    }

    public void addMultipleOfRow(int sourceRow, int targetRow, double multiplier) {
        if (sourceRow < 0 || targetRow < 0 || sourceRow >= getRowCount() || targetRow >= getRowCount()) {
            throw new IllegalArgumentException("Invalid row indices");
        }
        for (int col = 0; col < columns; col++) {
            set(targetRow, col, get(targetRow, col) + multiplier * get(sourceRow, col));
        }
    }

    public void multiplyRow(int row, double multiplier) {
        if (row < 0 || row >= getRowCount()) {
            throw new IllegalArgumentException("Invalid row index");
        }
        for (int col = 0; col < columns; col++) {
            set(row, col, get(row, col) * multiplier);
        }
    }

    private Vector[] transposeVectors() {
        int rows = getRowCount();
        Vector[] transposed = new Vector[rows];
        for (int i = 0; i < rows; i++) {
            double[] rowData = new double[columns];
            for (int j = 0; j < columns; j++) {
                rowData[j] = get(i, j);
            }
            transposed[i] = new Vector(rowData);
        }
        return transposed;
    }

    public Matrix transpose() {
        return new Matrix(transposeVectors());
    }

    public Matrix copy() {
        Vector[] copiedData = new Vector[data.length];
        for (int i = 0; i < data.length; i++) {
            copiedData[i] = data[i].copy();
        }
        return new Matrix(copiedData);
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        for(Vector v : data) {
            s.append(v.toString()).append("\n");
        }
        return s.toString();
    }

    public void toRowEchelonForm() {
        exchanges = 0;
        int pivotRow = 0;

        for (int col = 0; col < getColumnCount(); col++) {
            // Find pivot
            int nonZeroRow = -1;
            for (int row = pivotRow; row < getRowCount(); row++) {
                if (Math.abs(get(row, col)) > 1e-10) {
                    nonZeroRow = row;
                    break;
                }
            }

            // If no pivot found in this column, move to next
            if (nonZeroRow == -1) {
                continue;
            }

            // Exchange rows if necessary
            if (nonZeroRow != pivotRow) {
                exchangeRows(pivotRow, nonZeroRow);
                exchanges++;
            }

            // Zero entries below pivot
            for (int row = pivotRow + 1; row < getRowCount(); row++) {
                if (Math.abs(get(row, col)) > 1e-10) {
                    double multiplier = -get(row, col) / get(pivotRow, col);
                    addMultipleOfRow(pivotRow, row, multiplier);
                }
            }

            pivotRow++;
            if (pivotRow >= getRowCount()) {
                break;
            }
        }
    }

    /**
     * Transforms the matrix into its Reduced Row Echelon Form (RREF)
     * where:
     * 1. Each pivot is 1
     * 2. Pivots are the only non-zero entries in their columns
     * 3. All entries above and below pivots are zero
     */
    public void toReducedRowEchelonForm() {
        int pivotRow = 0;
        int pivotCol = 0;

        // We'll process columns from left to right
        while (pivotCol < getColumnCount() && pivotRow < getRowCount()) {
            // Find pivot
            int nonZeroRow = -1;
            for (int row = pivotRow; row < getRowCount(); row++) {
                if (Math.abs(get(row, pivotCol)) > 1e-10) {
                    nonZeroRow = row;
                    break;
                }
            }

            // If no pivot found in this column, move to next column
            if (nonZeroRow == -1) {
                pivotCol++;
                continue;
            }

            // Exchange rows if necessary
            if (nonZeroRow != pivotRow) {
                exchangeRows(pivotRow, nonZeroRow);
            }

            // Make pivot element 1
            double pivotValue = get(pivotRow, pivotCol);
            if (Math.abs(pivotValue) > 1e-10) {
                multiplyRow(pivotRow, 1.0 / pivotValue);
            }

            // Zero entries above pivot
            for (int row = 0; row < pivotRow; row++) {
                if (Math.abs(get(row, pivotCol)) > 1e-10) {
                    double multiplier = -get(row, pivotCol);
                    addMultipleOfRow(pivotRow, row, multiplier);
                }
            }

            // Zero entries below pivot
            for (int row = pivotRow + 1; row < getRowCount(); row++) {
                if (Math.abs(get(row, pivotCol)) > 1e-10) {
                    double multiplier = -get(row, pivotCol);
                    addMultipleOfRow(pivotRow, row, multiplier);
                }
            }

            pivotRow++;
            pivotCol++;
        }
    }

    /**
     * Solves the linear equation Ax = b by forming the augmented matrix [A|b],
     * reducing it to RREF, and reading the solution from the augmented column.
     * If the system is inconsistent or underdetermined (non-unique), an
     * ArithmeticException is thrown.
     *
     * @param b the right-hand side vector
     * @return solution vector x
     */
    public Vector solve(Vector b) {
        if (b == null) {
            throw new IllegalArgumentException("Right-hand side vector b must not be null");
        }
        if (b.dimension() != getRowCount()) {
            throw new IllegalArgumentException("Dimension mismatch: b has length " + b.dimension() +
                    " but matrix has " + getRowCount() + " rows");
        }

        // Form augmented matrix [A | b]
        Matrix augmented = this.AppendVector(b, "RIGHT");

        // Reduce to RREF
        augmented.toReducedRowEchelonForm();

        int rows = augmented.getRowCount();
        int cols = augmented.getColumnCount();
        int originalCols = this.getColumnCount();
        int augmentedColIndex = cols - 1;

        // Check for inconsistency: a row with all zeros in A but non-zero in augmented column
        for (int r = 0; r < rows; r++) {
            boolean allZero = true;
            for (int c = 0; c < originalCols; c++) {
                if (Math.abs(augmented.get(r, c)) > 1e-10) {
                    allZero = false;
                    break;
                }
            }
            if (allZero && Math.abs(augmented.get(r, augmentedColIndex)) > 1e-10) {
                throw new ArithmeticException("No solution exists (inconsistent system)");
            }
        }

        // Count pivot columns (leading ones) to detect uniqueness
        int pivotCount = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < originalCols; c++) {
                if (Math.abs(augmented.get(r, c) - 1.0) < 1e-10) {
                    // ensure it's the only non-zero in its column (RREF should guarantee this)
                    boolean isPivot = true;
                    for (int r2 = 0; r2 < rows; r2++) {
                        if (r2 != r && Math.abs(augmented.get(r2, c)) > 1e-10) {
                            isPivot = false;
                            break;
                        }
                    }
                    if (isPivot) {
                        pivotCount++;
                        break;
                    }
                } else if (Math.abs(augmented.get(r, c)) > 1e-10) {
                    break;
                }
            }
        }

        if (pivotCount < originalCols) {
            throw new ArithmeticException("Infinite solutions exist (underdetermined system)");
        }

        // Extract solution from last column (the augmented column)
        return augmented.getData()[augmentedColIndex].copy();
    }

    /**
     * Appends a vector to this matrix either on the LEFT or RIGHT and returns a new Matrix.
     * Position must be either "LEFT" or "RIGHT" (case-insensitive).
     * The vector's dimension must match the matrix row count.
     */
    public Matrix AppendVector(Vector v, String position) {
        if (v == null) {
            throw new IllegalArgumentException("Vector to append must not be null");
        }
        int rows = getRowCount();
        if (v.dimension() != rows) {
            throw new IllegalArgumentException("Vector dimension " + v.dimension() + " does not match matrix row count " + rows);
        }
        String pos = position == null ? "RIGHT" : position.toUpperCase();
        if (!pos.equals("LEFT") && !pos.equals("RIGHT")) {
            throw new IllegalArgumentException("Position must be either \"LEFT\" or \"RIGHT\"");
        }

        Vector[] newData = new Vector[columns + 1];
        if (pos.equals("RIGHT")) {
            for (int i = 0; i < columns; i++) {
                newData[i] = data[i].copy();
            }
            newData[columns] = v.copy();
        } else {
            // LEFT
            newData[0] = v.copy();
            for (int i = 0; i < columns; i++) {
                newData[i + 1] = data[i].copy();
            }
        }
        return new Matrix(newData);
    }

    /**
     * Appends another matrix to this matrix in the specified direction and returns a new Matrix.
     * Position must be one of "LEFT", "RIGHT", "UP", "DOWN" (case-insensitive).
     * - LEFT/RIGHT: matrices are concatenated horizontally (column-wise) and must have the same row count.
     * - UP/DOWN: matrices are concatenated vertically (row-wise) and must have the same column count.
     */
    public Matrix AppendMatrix(Matrix other, String position) {
        if (other == null) {
            throw new IllegalArgumentException("Matrix to append must not be null");
        }
        String pos = position == null ? "RIGHT" : position.toUpperCase();
        if (pos.equals("LEFT") || pos.equals("RIGHT")) {
            // Horizontal concatenation: row counts must match
            if (other.getRowCount() != getRowCount()) {
                throw new IllegalArgumentException("Row count mismatch for horizontal append");
            }
            int newCols = columns + other.getColumnCount();
            Vector[] newData = new Vector[newCols];
            if (pos.equals("LEFT")) {
                // copy other's columns first
                for (int i = 0; i < other.getColumnCount(); i++) {
                    newData[i] = other.getData()[i].copy();
                }
                for (int i = 0; i < columns; i++) {
                    newData[other.getColumnCount() + i] = data[i].copy();
                }
            } else {
                // RIGHT
                for (int i = 0; i < columns; i++) {
                    newData[i] = data[i].copy();
                }
                for (int i = 0; i < other.getColumnCount(); i++) {
                    newData[columns + i] = other.getData()[i].copy();
                }
            }
            return new Matrix(newData);
        } else if (pos.equals("UP") || pos.equals("DOWN")) {
            // Vertical concatenation: column counts must match
            if (other.getColumnCount() != getColumnCount()) {
                throw new IllegalArgumentException("Column count mismatch for vertical append");
            }
            int newRows = getRowCount() + other.getRowCount();
            Vector[] newData = new Vector[columns];
            for (int col = 0; col < columns; col++) {
                double[] combined = new double[newRows];
                if (pos.equals("UP")) {
                    // place other on top
                    for (int r = 0; r < other.getRowCount(); r++) {
                        combined[r] = other.get(r, col);
                    }
                    for (int r = 0; r < getRowCount(); r++) {
                        combined[other.getRowCount() + r] = get(r, col);
                    }
                } else {
                    // DOWN: place other below
                    for (int r = 0; r < getRowCount(); r++) {
                        combined[r] = get(r, col);
                    }
                    for (int r = 0; r < other.getRowCount(); r++) {
                        combined[getRowCount() + r] = other.get(r, col);
                    }
                }
                newData[col] = new Vector(combined);
            }
            return new Matrix(newData);
        } else {
            throw new IllegalArgumentException("Position must be one of LEFT, RIGHT, UP, DOWN");
        }
    }

    public double diagonalProduct() {
        if (getRowCount() != getColumnCount()) {
            throw new ArithmeticException("Row-column count mismatch");
        }
        double result = 1;
        for (int i = 0; i < columns; i++) {
            result *= get(i,i);
        }

        return result;
    }

    /**
     * Returns the inverse of this matrix.
     * Terminates early using isInvertible(); if not invertible, throws an ArithmeticException.
     * The inverse is computed by forming [A | I], reducing to RREF, and returning the right block.
     * @return inverse matrix A^{-1}
     */
    public Matrix inverse() {
        // Ensure square and invertible
        if (!isInvertible()) {
            throw new ArithmeticException("Matrix is not invertible");
        }
        int n = getRowCount();
        // Append identity to the right
        Matrix augmented = this.AppendMatrix(Matrix.Identity(n), "RIGHT");
        // Reduce to RREF
        augmented.toReducedRowEchelonForm();
        // Extract the right block (columns n..2n-1)
        int originalCols = this.getColumnCount();
        Vector[] invCols = new Vector[originalCols];
        for (int c = 0; c < originalCols; c++) {
            invCols[c] = augmented.getData()[originalCols + c].copy();
        }
        return new Matrix(invCols);
    }

    /**
     * Factory method to create an n x n identity matrix.
     * Note: method named Identity(int n) because a Java method must have a name.
     */
    public static Matrix Identity(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("Size n must be positive");
        }
        Vector[] cols = new Vector[n];
        for (int c = 0; c < n; c++) {
            double[] col = new double[n];
            for (int r = 0; r < n; r++) {
                col[r] = (r == c) ? 1.0 : 0.0;
            }
            cols[c] = new Vector(col);
        }
        return new Matrix(cols);
    }

    public double determinant() {
        if (getRowCount() != getColumnCount()) {
            throw new ArithmeticException("Row-column count mismatch");
        }
        Matrix m = this.copy();
        m.toRowEchelonForm();
        return m.diagonalProduct() * Math.pow(-1,m.exchanges);
    }

    public boolean isInvertible() {
        return determinant() != 0;
    }

    /**
     * Multiplies this matrix (m x p) by another matrix (p x n) and returns the product (m x n).
     * Uses the column-oriented storage: each column of the product is A * (column j of other).
     * @param other right-hand-side matrix
     * @return product matrix
     */
    public Matrix multiply(Matrix other) {
        if (other == null) {
            throw new IllegalArgumentException("Other matrix must not be null");
        }
        int m = getRowCount();           // rows of this
        int p = getColumnCount();        // columns of this
        int p2 = other.getRowCount();    // rows of other
        int n = other.getColumnCount();  // columns of other

        if (p != p2) {
            throw new IllegalArgumentException("Inner dimensions must agree for multiplication: " + p + " != " + p2);
        }

        Vector[] resultCols = new Vector[n];
        // For each column j of 'other', compute column j of the product
        for (int j = 0; j < n; j++) {
            double[] col = new double[m];
            for (int k = 0; k < p; k++) {
                double alpha = other.get(k, j);
                if (Math.abs(alpha) <= 1e-15) {
                    continue;
                }
                Vector aCol = this.data[k];
                for (int i = 0; i < m; i++) {
                    col[i] += alpha * aCol.get(i);
                }
            }
            resultCols[j] = new Vector(col);
        }
        return new Matrix(resultCols);
    }

}