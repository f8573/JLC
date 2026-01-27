package net.faulj.matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;

import net.faulj.compute.BlockedMultiply;
import net.faulj.vector.Vector;
import net.faulj.spaces.SubspaceBasis;

/**
 * Dense matrix implementation with optional complex components.
 * Provides common linear algebra operations and utility helpers.
 */
public class Matrix {
    private final int rows;
    private final int columns;
    private double[] data;
    private double[] imag;
    private int exchanges;
    private final ArrayList<Vector> pivotColumns;
    private double tol = 1e-10;
    private transient Vector[] columnViews;

    /**
     * Construct a matrix from column vectors.
     *
     * @param data column vectors
     */
    public Matrix(Vector[] data) {
        if (data == null || data.length == 0) {
            this.rows = 0;
            this.columns = 0;
            this.data = new double[0];
            this.pivotColumns = new ArrayList<>();
            return;
        }
        int rowCount = data[0].dimension();
        this.rows = rowCount;
        this.columns = data.length;
        this.data = new double[rows * columns];
        boolean anyImag = false;
        for (Vector v : data) {
            if (v.dimension() != rowCount) {
                throw new IllegalArgumentException("Column dimension mismatch");
            }
            if (v.hasImag()) {
                anyImag = true;
            }
        }
        if (anyImag) {
            this.imag = new double[rows * columns];
        }
        for (int col = 0; col < columns; col++) {
            Vector v = data[col];
            int idx = col;
            double[] colData = v.getData();
            double[] colImag = anyImag ? (v.hasImag() ? v.getImagData() : null) : null;
            for (int row = 0; row < rows; row++) {
                this.data[idx] = colData[row];
                if (this.imag != null && colImag != null) {
                    this.imag[idx] = colImag[row];
                }
                idx += columns;
            }
        }
        pivotColumns = new ArrayList<>();
    }

    /**
     * Construct a real matrix from a 2D array.
     *
     * @param data row-major data
     */
    public Matrix(double[][] data) {
        if (data == null || data.length == 0 || data[0].length == 0) {
            throw new IllegalArgumentException("Data must be non-empty");
        }
        this.rows = data.length;
        this.columns = data[0].length;
        this.data = new double[rows * columns];
        for (int row = 0; row < rows; row++) {
            if (data[row].length != columns) {
                throw new IllegalArgumentException("All rows must have the same length");
            }
            System.arraycopy(data[row], 0, this.data, row * columns, columns);
        }
        pivotColumns = new ArrayList<>();
    }

    /**
     * Construct a complex matrix from real and imaginary arrays.
     *
     * @param real real data
     * @param imag imaginary data
     */
    public Matrix(double[][] real, double[][] imag) {
        if (real == null || real.length == 0 || real[0].length == 0) {
            throw new IllegalArgumentException("Data must be non-empty");
        }
        this.rows = real.length;
        this.columns = real[0].length;
        if (imag != null && (imag.length != rows || imag[0].length != columns)) {
            throw new IllegalArgumentException("Imaginary data dimensions must match real data dimensions");
        }
        this.data = new double[rows * columns];
        if (imag != null) {
            this.imag = new double[rows * columns];
        }
        for (int row = 0; row < rows; row++) {
            if (real[row].length != columns) {
                throw new IllegalArgumentException("All rows must have the same length");
            }
            System.arraycopy(real[row], 0, this.data, row * columns, columns);
            if (imag != null) {
                System.arraycopy(imag[row], 0, this.imag, row * columns, columns);
            }
        }
        pivotColumns = new ArrayList<>();
    }

    /**
     * Construct an empty real matrix of the given size.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public Matrix(int rows, int cols) {
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Matrix dimensions must be non-negative");
        }
        this.rows = rows;
        this.columns = cols;
        this.data = new double[rows * cols];
        this.pivotColumns = new ArrayList<>();
    }

    private Matrix(int rows, int cols, double[] data, double[] imag, boolean wrap) {
        this.rows = rows;
        this.columns = cols;
        this.data = data;
        this.imag = imag;
        this.pivotColumns = new ArrayList<>();
    }

    /**
     * Wrap a raw data array as a matrix without copying.
     *
     * @param data raw data array
     * @param rows number of rows
     * @param cols number of columns
     * @return wrapped matrix
     */
    public static Matrix wrap(double[] data, int rows, int cols) {
        return wrap(data, null, rows, cols);
    }

    /**
     * Wrap raw real/imag data arrays as a matrix without copying.
     *
     * @param data real data array
     * @param imag imaginary data array
     * @param rows number of rows
     * @param cols number of columns
     * @return wrapped matrix
     */
    public static Matrix wrap(double[] data, double[] imag, int rows, int cols) {
        if (data == null) {
            throw new IllegalArgumentException("Data must not be null");
        }
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Matrix dimensions must be non-negative");
        }
        if (data.length != rows * cols) {
            throw new IllegalArgumentException("Data length does not match dimensions");
        }
        if (imag != null && imag.length != data.length) {
            throw new IllegalArgumentException("Imaginary data length does not match dimensions");
        }
        return new Matrix(rows, cols, data, imag, true);
    }

    /**
     * Set a column from a vector.
     *
     * @param colIndex column index
     * @param column vector data
     */
    public void setColumn(int colIndex, Vector column) {
        if (colIndex < 0 || colIndex >= columns) {
            throw new IllegalArgumentException("Invalid column index");
        }
        if (column.dimension() != getRowCount()) {
            throw new IllegalArgumentException("Column dimension mismatch");
        }
        double[] colData = column.getData();
        double[] colImag = column.hasImag() ? column.getImagData() : null;
        int idx = colIndex;
        if (colImag != null) {
            ensureImagData();
        }
        for (int row = 0; row < rows; row++) {
            data[idx] = colData[row];
            if (imag != null) {
                imag[idx] = colImag == null ? 0.0 : colImag[row];
            }
            idx += columns;
        }
    }

    /**
     * Set a column from a raw array.
     *
     * @param colIndex column index
     * @param column column data
     */
    public void setColumn(int colIndex, double[] column) {
        if (colIndex < 0 || colIndex >= columns) {
            throw new IllegalArgumentException("Invalid column index");
        }
        if (column.length != getRowCount()) {
            throw new IllegalArgumentException("Column dimension mismatch");
        }
        int idx = colIndex;
        for (int row = 0; row < rows; row++) {
            data[idx] = column[row];
            if (imag != null) {
                imag[idx] = 0.0;
            }
            idx += columns;
        }
    }

    /**
     * Get a column as a new array.
     *
     * @param colIndex column index
     * @return column data
     */
    public double[] getColumn(int colIndex) {
        if (colIndex < 0 || colIndex >= columns) {
            throw new IllegalArgumentException("Invalid column index");
        }
        double[] col = new double[rows];
        int idx = colIndex;
        for (int row = 0; row < rows; row++) {
            col[row] = data[idx];
            idx += columns;
        }
        return col;
    }

    /**
     * Get a row as a new array.
     *
     * @param rowIndex row index
     * @return row data
     */
    public double[] getRow(int rowIndex) {
        if (rowIndex < 0 || rowIndex >= rows) {
            throw new IllegalArgumentException("Invalid row index");
        }
        double[] row = new double[columns];
        System.arraycopy(data, rowIndex * columns, row, 0, columns);
        return row;
    }

    /**
     * Get column vector views over this matrix.
     *
     * @return array of column vectors
     */
    public Vector[] getData() {
        if (columns == 0) {
            return new Vector[0];
        }
        if (columnViews == null || columnViews.length != columns) {
            Vector[] views = new Vector[columns];
            for (int col = 0; col < columns; col++) {
                views[col] = new Vector(this, col);
            }
            columnViews = views;
        }
        return columnViews;
    }

    /**
     * Set all columns from a vector array.
     *
     * @param data column vectors
     */
    public void setData(Vector[] data) {
        if (data == null) {
            throw new IllegalArgumentException("Data must not be null");
        }
        if (data.length != columns) {
            throw new IllegalArgumentException("Column count mismatch");
        }
        for (int col = 0; col < columns; col++) {
            setColumn(col, data[col]);
        }
    }

    /**
     * Get a real value at a position.
     *
     * @param row row index
     * @param column column index
     * @return real value
     */
    public double get(int row, int column) {
        return data[row * columns + column];
    }

    /**
     * Get an imaginary value at a position.
     *
     * @param row row index
     * @param column column index
     * @return imaginary value (0 if none)
     */
    public double getImag(int row, int column) {
        if (imag == null) {
            return 0.0;
        }
        return imag[row * columns + column];
    }

    /**
     * Set a real value at a position.
     *
     * @param row row index
     * @param column column index
     * @param value real value
     */
    public void set(int row, int column, double value) {
        data[row * columns + column] = value;
    }

    /**
     * Set an imaginary value at a position.
     *
     * @param row row index
     * @param column column index
     * @param value imaginary value
     */
    public void setImag(int row, int column, double value) {
        if (imag == null) {
            if (value == 0.0) {
                return;
            }
            ensureImagData();
        }
        imag[row * columns + column] = value;
    }

    /**
     * Set both real and imaginary values at a position.
     *
     * @param row row index
     * @param column column index
     * @param real real value
     * @param imaginary imaginary value
     */
    public void setComplex(int row, int column, double real, double imaginary) {
        data[row * columns + column] = real;
        setImag(row, column, imaginary);
    }

    /**
     * Get number of rows.
     *
     * @return row count
     */
    public int getRowCount() {
        return rows;
    }

    /**
     * Get number of columns.
     *
     * @return column count
     */
    public int getColumnCount() {
        return columns;
    }

    /**
     * Get pivot columns used in row-reduction routines.
     *
     * @return pivot columns list
     */
    public ArrayList<Vector> getPivotColumns() {
        return pivotColumns;
    }

    /**
     * Get the raw real data backing array.
     *
     * @return raw real data
     */
    public double[] getRawData() {
        return data;
    }

    /**
     * Get the raw imaginary backing array, if present.
     *
     * @return raw imaginary data or null
     */
    public double[] getRawImagData() {
        return imag;
    }

    /**
     * Ensure the imaginary array exists and return it.
     *
     * @return imaginary data array
     */
    public double[] ensureImagData() {
        if (imag == null) {
            imag = new double[data.length];
        }
        return imag;
    }

    /**
     * Replace the imaginary data array.
     *
     * @param imag imaginary data array
     */
    public void setImagData(double[] imag) {
        if (imag != null && imag.length != data.length) {
            throw new IllegalArgumentException("Imaginary data length does not match dimensions");
        }
        this.imag = imag;
    }

    /**
     * Check if imaginary data storage exists.
     *
     * @return true if complex data is present
     */
    public boolean hasImagData() {
        return imag != null;
    }

    /**
     * Check if the matrix contains only real values.
     *
     * @return true if real
     */
    public boolean isReal() {
        if (imag == null) {
            return true;
        }
        for (double v : imag) {
            if (v != 0.0) {
                return false;
            }
        }
        return true;
    }

    private void ensureReal(String operation) {
        if (!isReal()) {
            throw new UnsupportedOperationException(operation + " requires a real-valued matrix");
        }
    }

    /**
     * Swap two rows in place.
     *
     * @param row1 first row index
     * @param row2 second row index
     */
    public void exchangeRows(int row1, int row2) {
        if (row1 < 0 || row2 < 0 || row1 >= rows || row2 >= rows) {
            throw new IllegalArgumentException("Invalid row indices");
        }
        if (row1 == row2) {
            return;
        }
        int offset1 = row1 * columns;
        int offset2 = row2 * columns;
        for (int col = 0; col < columns; col++) {
            double temp = data[offset1 + col];
            data[offset1 + col] = data[offset2 + col];
            data[offset2 + col] = temp;
            if (imag != null) {
                double tempI = imag[offset1 + col];
                imag[offset1 + col] = imag[offset2 + col];
                imag[offset2 + col] = tempI;
            }
        }
    }

    /**
     * Add a multiple of one row to another.
     *
     * @param sourceRow source row index
     * @param targetRow target row index
     * @param multiplier scalar multiplier
     */
    public void addMultipleOfRow(int sourceRow, int targetRow, double multiplier) {
        if (sourceRow < 0 || targetRow < 0 || sourceRow >= rows || targetRow >= rows) {
            throw new IllegalArgumentException("Invalid row indices");
        }
        int sourceOffset = sourceRow * columns;
        int targetOffset = targetRow * columns;
        for (int col = 0; col < columns; col++) {
            data[targetOffset + col] += multiplier * data[sourceOffset + col];
            if (imag != null) {
                imag[targetOffset + col] += multiplier * imag[sourceOffset + col];
            }
        }
    }

    /**
     * Add another matrix to this matrix.
     *
     * @param other matrix to add
     * @return sum matrix
     */
    public Matrix add(Matrix other) {
        if (columns != other.getColumnCount() || rows != other.getRowCount()) {
            throw new IllegalArgumentException("Matrices must have equal dimensions");
        }
        Matrix result = new Matrix(rows, columns);
        double[] out = result.data;
        double[] a = this.data;
        double[] b = other.data;
        double[] ai = this.imag;
        double[] bi = other.imag;
        double[] oi = null;
        if (ai != null || bi != null) {
            oi = result.ensureImagData();
        }
        for (int i = 0; i < out.length; i++) {
            out[i] = a[i] + b[i];
            if (oi != null) {
                oi[i] = (ai == null ? 0.0 : ai[i]) + (bi == null ? 0.0 : bi[i]);
            }
        }
        return result;
    }

    /**
     * Subtract another matrix from this matrix.
     *
     * @param other matrix to subtract
     * @return difference matrix
     */
    public Matrix subtract(Matrix other) {
        if (columns != other.getColumnCount() || rows != other.getRowCount()) {
            throw new IllegalArgumentException("Matrices must have equal dimensions");
        }
        Matrix result = new Matrix(rows, columns);
        double[] out = result.data;
        double[] a = this.data;
        double[] b = other.data;
        double[] ai = this.imag;
        double[] bi = other.imag;
        double[] oi = null;
        if (ai != null || bi != null) {
            oi = result.ensureImagData();
        }
        for (int i = 0; i < out.length; i++) {
            out[i] = a[i] - b[i];
            if (oi != null) {
                oi[i] = (ai == null ? 0.0 : ai[i]) - (bi == null ? 0.0 : bi[i]);
            }
        }
        return result;
    }

    /**
     * Multiply a row by a scalar.
     *
     * @param row row index
     * @param multiplier scalar multiplier
     */
    public void multiplyRow(int row, double multiplier) {
        if (row < 0 || row >= rows) {
            throw new IllegalArgumentException("Invalid row index");
        }
        int offset = row * columns;
        for (int col = 0; col < columns; col++) {
            data[offset + col] *= multiplier;
            if (imag != null) {
                imag[offset + col] *= multiplier;
            }
        }
    }

    /**
     * Transpose the matrix.
     *
     * @return transposed matrix
     */
    public Matrix transpose() {
        Matrix result = new Matrix(columns, rows);
        double[] out = result.data;
        double[] outImag = null;
        if (imag != null) {
            outImag = result.ensureImagData();
        }
        for (int row = 0; row < rows; row++) {
            int rowOffset = row * columns;
            for (int col = 0; col < columns; col++) {
                int outIndex = col * rows + row;
                out[outIndex] = data[rowOffset + col];
                if (outImag != null) {
                    outImag[outIndex] = imag[rowOffset + col];
                }
            }
        }
        return result;
    }

    /**
     * Create a deep copy of this matrix.
     *
     * @return copied matrix
     */
    public Matrix copy() {
        Matrix result = new Matrix(rows, columns);
        System.arraycopy(data, 0, result.data, 0, data.length);
        if (imag != null) {
            result.imag = Arrays.copyOf(imag, imag.length);
        }
        return result;
    }

    /**
     * Render the matrix as a string.
     *
     * @return string representation
     */
    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        for (int row = 0; row < rows; row++) {
            if (imag == null) {
                s.append(Arrays.toString(getRow(row))).append("\n");
                continue;
            }
            double[] rowReal = getRow(row);
            double[] rowImag = new double[columns];
            int offset = row * columns;
            for (int col = 0; col < columns; col++) {
                rowImag[col] = imag[offset + col];
            }
            StringBuilder line = new StringBuilder("[");
            for (int col = 0; col < columns; col++) {
                if (col > 0) {
                    line.append(", ");
                }
                double re = rowReal[col];
                double im = rowImag[col];
                if (im == 0) {
                    line.append(re);
                } else if (im > 0) {
                    line.append(re).append(" + ").append(im).append("i");
                } else {
                    line.append(re).append(" - ").append(-im).append("i");
                }
            }
            line.append("]");
            s.append(line).append("\n");
        }
        return s.toString();
    }

    /**
     * Transform this matrix to row echelon form in place.
     */
    public void toRowEchelonForm() {
        ensureReal("Row echelon form");
        exchanges = 0;
        pivotColumns.clear();
        int pivotRow = 0;

        for (int col = 0; col < columns && pivotRow < rows; col++) {
            int pivotIndex = -1;
            double maxAbs = tol;
            for (int row = pivotRow; row < rows; row++) {
                double val = Math.abs(get(row, col));
                if (val > maxAbs) {
                    maxAbs = val;
                    pivotIndex = row;
                }
            }

            if (pivotIndex == -1) {
                continue;
            }

            if (pivotIndex != pivotRow) {
                exchangeRows(pivotRow, pivotIndex);
                exchanges++;
            }

            double pivot = get(pivotRow, col);
            if (Math.abs(pivot) <= tol) {
                continue;
            }

            for (int row = pivotRow + 1; row < rows; row++) {
                double val = get(row, col);
                if (Math.abs(val) > tol) {
                    double multiplier = -val / pivot;
                    addMultipleOfRow(pivotRow, row, multiplier);
                }
            }

            pivotRow++;
        }
    }

    /**
     * Transform this matrix to reduced row echelon form in place.
     */
    public void toReducedRowEchelonForm() {
        ensureReal("Reduced row echelon form");
        pivotColumns.clear();
        Matrix original = this.copy();
        int pivotRow = 0;

        for (int col = 0; col < columns && pivotRow < rows; col++) {
            int pivotIndex = -1;
            double maxAbs = tol;
            for (int row = pivotRow; row < rows; row++) {
                double val = Math.abs(get(row, col));
                if (val > maxAbs) {
                    maxAbs = val;
                    pivotIndex = row;
                }
            }

            if (pivotIndex == -1) {
                continue;
            }

            if (pivotIndex != pivotRow) {
                exchangeRows(pivotRow, pivotIndex);
            }

            double pivot = get(pivotRow, col);
            if (Math.abs(pivot) <= tol) {
                continue;
            }

            multiplyRow(pivotRow, 1.0 / pivot);

            for (int row = 0; row < rows; row++) {
                if (row == pivotRow) {
                    continue;
                }
                double val = get(row, col);
                if (Math.abs(val) > tol) {
                    addMultipleOfRow(pivotRow, row, -val);
                }
            }

            pivotColumns.add(original.getData()[col].copy());
            pivotRow++;
        }
    }

    /**
     * Solve a linear system $Ax=b$ using row reduction.
     *
     * @param b right-hand side vector
     * @return solution vector
     */
    public Vector solve(Vector b) {
        ensureReal("Solve");
        if (b == null) {
            throw new IllegalArgumentException("Right-hand side vector b must not be null");
        }
        if (b.dimension() != rows) {
            throw new IllegalArgumentException("Dimension mismatch: b has length " + b.dimension() +
                    " but matrix has " + rows + " rows");
        }

        Matrix augmented = this.AppendVector(b, "RIGHT");
        augmented.toReducedRowEchelonForm();

        int originalCols = this.columns;
        int augmentedColIndex = augmented.getColumnCount() - 1;

        for (int r = 0; r < rows; r++) {
            boolean allZero = true;
            for (int c = 0; c < originalCols; c++) {
                if (Math.abs(augmented.get(r, c)) > tol) {
                    allZero = false;
                    break;
                }
            }
            if (allZero && Math.abs(augmented.get(r, augmentedColIndex)) > tol) {
                throw new ArithmeticException("No solution exists (inconsistent system)");
            }
        }

        int pivotCount = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < originalCols; c++) {
                if (Math.abs(augmented.get(r, c) - 1.0) < tol) {
                    boolean isPivot = true;
                    for (int r2 = 0; r2 < rows; r2++) {
                        if (r2 != r && Math.abs(augmented.get(r2, c)) > tol) {
                            isPivot = false;
                            break;
                        }
                    }
                    if (isPivot) {
                        pivotCount++;
                        break;
                    }
                } else if (Math.abs(augmented.get(r, c)) > tol) {
                    break;
                }
            }
        }

        if (pivotCount < originalCols) {
            throw new ArithmeticException("Infinite solutions exist (underdetermined system)");
        }

        return augmented.getData()[augmentedColIndex].copy();
    }

    /**
     * Append a vector to this matrix as a row or column.
     *
     * @param v vector to append
     * @param position "row" or "column"
     * @return new matrix
     */
    public Matrix AppendVector(Vector v, String position) {
        if (v == null) {
            throw new IllegalArgumentException("Vector to append must not be null");
        }
        if (v.dimension() != rows) {
            throw new IllegalArgumentException("Vector dimension " + v.dimension() + " does not match matrix row count " + rows);
        }
        String pos = position == null ? "RIGHT" : position.toUpperCase();
        if (!pos.equals("LEFT") && !pos.equals("RIGHT")) {
            throw new IllegalArgumentException("Position must be either \"LEFT\" or \"RIGHT\"");
        }

        Matrix result = new Matrix(rows, columns + 1);
        double[] out = result.data;
        double[] outImag = null;
        double[] vImag = v.hasImag() ? v.getImagData() : null;
        if (imag != null || vImag != null) {
            outImag = result.ensureImagData();
        }
        if (pos.equals("RIGHT")) {
            for (int row = 0; row < rows; row++) {
                int srcOffset = row * columns;
                int dstOffset = row * (columns + 1);
                System.arraycopy(data, srcOffset, out, dstOffset, columns);
                out[dstOffset + columns] = v.get(row);
                if (outImag != null) {
                    if (imag != null) {
                        System.arraycopy(imag, srcOffset, outImag, dstOffset, columns);
                    }
                    outImag[dstOffset + columns] = vImag == null ? 0.0 : vImag[row];
                }
            }
        } else {
            for (int row = 0; row < rows; row++) {
                int srcOffset = row * columns;
                int dstOffset = row * (columns + 1);
                out[dstOffset] = v.get(row);
                System.arraycopy(data, srcOffset, out, dstOffset + 1, columns);
                if (outImag != null) {
                    outImag[dstOffset] = vImag == null ? 0.0 : vImag[row];
                    if (imag != null) {
                        System.arraycopy(imag, srcOffset, outImag, dstOffset + 1, columns);
                    }
                }
            }
        }
        return result;
    }

    /**
     * Append another matrix to this matrix.
     *
     * @param other matrix to append
     * @param position "row" or "column"
     * @return new matrix
     */
    public Matrix AppendMatrix(Matrix other, String position) {
        if (other == null) {
            throw new IllegalArgumentException("Matrix to append must not be null");
        }
        String pos = position == null ? "RIGHT" : position.toUpperCase();
        if (pos.equals("LEFT") || pos.equals("RIGHT")) {
            if (other.getRowCount() != rows) {
                throw new IllegalArgumentException("Row count mismatch for horizontal append");
            }
            int newCols = columns + other.getColumnCount();
            Matrix result = new Matrix(rows, newCols);
            double[] out = result.data;
            double[] left = other.data;
            double[] right = this.data;
            double[] outImag = null;
            if (imag != null || other.imag != null) {
                outImag = result.ensureImagData();
            }
            for (int row = 0; row < rows; row++) {
                int dstOffset = row * newCols;
                if (pos.equals("LEFT")) {
                    System.arraycopy(left, row * other.columns, out, dstOffset, other.columns);
                    System.arraycopy(right, row * columns, out, dstOffset + other.columns, columns);
                    if (outImag != null) {
                        if (other.imag != null) {
                            System.arraycopy(other.imag, row * other.columns, outImag, dstOffset, other.columns);
                        }
                        if (imag != null) {
                            System.arraycopy(imag, row * columns, outImag, dstOffset + other.columns, columns);
                        }
                    }
                } else {
                    System.arraycopy(right, row * columns, out, dstOffset, columns);
                    System.arraycopy(left, row * other.columns, out, dstOffset + columns, other.columns);
                    if (outImag != null) {
                        if (imag != null) {
                            System.arraycopy(imag, row * columns, outImag, dstOffset, columns);
                        }
                        if (other.imag != null) {
                            System.arraycopy(other.imag, row * other.columns, outImag, dstOffset + columns, other.columns);
                        }
                    }
                }
            }
            return result;
        } else if (pos.equals("UP") || pos.equals("DOWN")) {
            if (other.getColumnCount() != columns) {
                throw new IllegalArgumentException("Column count mismatch for vertical append");
            }
            int newRows = rows + other.getRowCount();
            Matrix result = new Matrix(newRows, columns);
            double[] out = result.data;
            double[] outImag = null;
            if (imag != null || other.imag != null) {
                outImag = result.ensureImagData();
            }
            if (pos.equals("UP")) {
                System.arraycopy(other.data, 0, out, 0, other.data.length);
                System.arraycopy(this.data, 0, out, other.data.length, this.data.length);
                if (outImag != null) {
                    if (other.imag != null) {
                        System.arraycopy(other.imag, 0, outImag, 0, other.imag.length);
                    }
                    if (imag != null) {
                        System.arraycopy(imag, 0, outImag, other.data.length, imag.length);
                    }
                }
            } else {
                System.arraycopy(this.data, 0, out, 0, this.data.length);
                System.arraycopy(other.data, 0, out, this.data.length, other.data.length);
                if (outImag != null) {
                    if (imag != null) {
                        System.arraycopy(imag, 0, outImag, 0, imag.length);
                    }
                    if (other.imag != null) {
                        System.arraycopy(other.imag, 0, outImag, this.data.length, other.imag.length);
                    }
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("Position must be one of LEFT, RIGHT, UP, DOWN");
        }
    }

    /**
     * Compute the product of diagonal entries.
     *
     * @return product of diagonal
     */
    public double diagonalProduct() {
        ensureReal("Diagonal product");
        if (rows != columns) {
            throw new ArithmeticException("Row-column count mismatch");
        }
        double result = 1;
        for (int i = 0; i < columns; i++) {
            result *= data[i * columns + i];
        }
        return result;
    }

    /**
     * Compute the matrix trace.
     *
     * @return trace
     */
    public double trace() {
        ensureReal("Trace");
        if (rows != columns) {
            throw new ArithmeticException("Row-column count mismatch");
        }
        double result = 0;
        for (int i = 0; i < columns; i++) {
            result += data[i * columns + i];
        }
        return result;
    }

    /**
     * Compute the inverse using LU-based inversion.
     *
     * @return inverse matrix
     */
    public Matrix inverse() {
        ensureReal("Inverse");
        if (!isInvertible()) {
            throw new ArithmeticException("Matrix is not invertible");
        }
        int n = rows;
        Matrix augmented = this.AppendMatrix(Matrix.Identity(n), "RIGHT");
        augmented.toReducedRowEchelonForm();
        int originalCols = this.columns;
        Vector[] invCols = new Vector[originalCols];
        for (int c = 0; c < originalCols; c++) {
            invCols[c] = augmented.getData()[originalCols + c].copy();
        }
        return new Matrix(invCols);
    }

    public static Matrix Identity(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("Size n must be positive");
        }
        Matrix m = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            m.data[i * n + i] = 1.0;
        }
        return m;
    }

    /**
     * Compute the determinant.
     *
     * @return determinant
     */
    public double determinant() {
        ensureReal("Determinant");
        if (rows != columns) {
            throw new ArithmeticException("Row-column count mismatch");
        }
        Matrix m = this.copy();
        int n = m.rows;
        int exchanges = 0;

        for (int col = 0; col < n; col++) {
            int maxRow = col;
            double maxAbs = Math.abs(m.get(col, col));
            for (int r = col + 1; r < n; r++) {
                double a = Math.abs(m.get(r, col));
                if (a > maxAbs) {
                    maxAbs = a;
                    maxRow = r;
                }
            }

            if (maxAbs <= tol) {
                return 0.0;
            }

            if (maxRow != col) {
                m.exchangeRows(col, maxRow);
                exchanges++;
            }

            double pivot = m.get(col, col);
            for (int r = col + 1; r < n; r++) {
                double factor = m.get(r, col) / pivot;
                int rowOffset = r * n;
                int pivotOffset = col * n;
                for (int c = col; c < n; c++) {
                    m.data[rowOffset + c] -= factor * m.data[pivotOffset + c];
                }
            }
        }

        double det = 1.0;
        for (int i = 0; i < n; i++) det *= m.get(i, i);
        if ((exchanges & 1) == 1) det = -det;
        return det;
    }

    /**
     * Check whether this matrix is invertible.
     *
     * @return true if invertible
     */
    public boolean isInvertible() {
        return determinant() != 0;
    }

    /**
     * Multiply by another matrix.
     *
     * @param other matrix to multiply
     * @return product matrix
     */
    public Matrix multiply(Matrix other) {
        return BlockedMultiply.multiply(this, other);
    }

    /**
     * Compute a basis for the row space.
     *
     * @return row space basis
     */
    public Set<Vector> rowSpaceBasis() {
        return SubspaceBasis.rowSpaceBasis(this);
    }

    /**
     * Compute a basis for the column space.
     *
     * @return column space basis
     */
    public Set<Vector> columnSpaceBasis() {
        return SubspaceBasis.columnSpaceBasis(this);
    }

    /**
     * Compute a basis for the null space.
     *
     * @return null space basis
     */
    public Set<Vector> nullSpaceBasis() {
        return SubspaceBasis.nullSpaceBasis(this);
    }

    /**
     * Compute the QR decomposition.
     *
     * @return array containing Q and R
     */
    public Matrix[] QR() {
        net.faulj.decomposition.result.QRResult res = net.faulj.decomposition.qr.HouseholderQR.decompose(this);
        return new Matrix[]{res.getQ(), res.getR()};
    }

    /**
     * Compute the thin QR decomposition.
     *
     * @return array containing Q and R
     */
    public Matrix[] thinQR() {
        net.faulj.decomposition.result.QRResult res = net.faulj.decomposition.qr.HouseholderQR.decomposeThin(this);
        return new Matrix[]{res.getQ(), res.getR()};
    }

    public net.faulj.decomposition.result.HessenbergResult Hessenberg() {
        return net.faulj.decomposition.hessenberg.HessenbergReduction.decompose(this);
    }

    /**
     * Check if this matrix is square.
     *
     * @return true if square
     */
    public boolean isSquare() {
        return rows == columns;
    }

    public static Matrix diag(int num, Matrix matrix) {
        if (!matrix.isSquare()) {
            throw new IllegalArgumentException("Both matrices must be square to make a new diagonal");
        }

        if (num == 0) {
            return matrix;
        }

        Matrix I = Matrix.Identity(num);

        int i = I.getRowCount();
        int m = matrix.getRowCount();
        I = I.AppendMatrix(Matrix.zero(i, m), "RIGHT");
        matrix = matrix.AppendMatrix(Matrix.zero(m, i), "LEFT");
        return I.AppendMatrix(matrix, "DOWN");
    }

    public static Matrix zero(int rows, int columns) {
        return new Matrix(rows, columns);
    }

    /**
     * Multiply this matrix by a scalar.
     *
     * @param d scalar value
     * @return scaled matrix
     */
    public Matrix multiplyScalar(double d) {
        Matrix result = new Matrix(rows, columns);
        double[] out = result.data;
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i] * d;
        }
        if (imag != null) {
            double[] outImag = result.ensureImagData();
            for (int i = 0; i < imag.length; i++) {
                outImag[i] = imag[i] * d;
            }
        }
        return result;
    }

    /**
     * Compute the Frobenius norm.
     *
     * @return Frobenius norm
     */
    public double frobeniusNorm() {
        return MatrixNorms.frobeniusNorm(this);
    }

    /**
     * Compute the induced 1-norm.
     *
     * @return 1-norm
     */
    public double norm1() {
        return MatrixNorms.norm1(this);
    }

    /**
     * Compute the induced infinity norm.
     *
     * @return infinity norm
     */
    public double normInf() {
        return MatrixNorms.normInf(this);
    }

    /**
     * Round values below a tolerance to zero.
     *
     * @param tolerance rounding tolerance
     * @return rounded matrix
     */
    public Matrix round(double tolerance) {
        Matrix result = new Matrix(rows, columns);
        double[] out = result.data;
        for (int i = 0; i < data.length; i++) {
            double val = data[i];
            out[i] = val < tolerance ? 0.0 : val;
        }
        if (imag != null) {
            double[] outImag = result.ensureImagData();
            for (int i = 0; i < imag.length; i++) {
                double val = imag[i];
                outImag[i] = val < tolerance ? 0.0 : val;
            }
        }
        return result;
    }

    /**
     * Extract a submatrix.
     *
     * @param fromRow starting row (inclusive)
     * @param toRow ending row (exclusive)
     * @param fromCol starting column (inclusive)
     * @param toCol ending column (exclusive)
     * @return submatrix
     */
    public Matrix crop(int fromRow, int toRow, int fromCol, int toCol) {
        if (fromRow < 0 || toRow >= rows || fromCol < 0 || toCol >= columns
                || fromRow > toRow || fromCol > toCol) {
            throw new IllegalArgumentException("Invalid crop dimensions");
        }

        int newRows = toRow - fromRow + 1;
        int newCols = toCol - fromCol + 1;
        Matrix result = new Matrix(newRows, newCols);
        double[] out = result.data;
        double[] outImag = null;
        if (imag != null) {
            outImag = result.ensureImagData();
        }

        for (int row = 0; row < newRows; row++) {
            int srcOffset = (fromRow + row) * columns + fromCol;
            int dstOffset = row * newCols;
            System.arraycopy(data, srcOffset, out, dstOffset, newCols);
            if (outImag != null) {
                System.arraycopy(imag, srcOffset, outImag, dstOffset, newCols);
            }
        }

        return result;
    }

    /**
     * Compute the minor matrix removing row i and column j.
     *
     * @param i row index to remove
     * @param j column index to remove
     * @return minor matrix
     */
    public Matrix minor(int i, int j) {
        if (rows != columns) {
            throw new IllegalArgumentException("Matrix must be square to compute a minor");
        }
        Matrix result = new Matrix(rows - 1, columns - 1);
        double[] out = result.data;
        double[] outImag = null;
        if (imag != null) {
            outImag = result.ensureImagData();
        }
        int outCols = result.columns;
        int outRow = 0;
        for (int row = 0; row < rows; row++) {
            if (row == i) {
                continue;
            }
            int outCol = 0;
            int rowOffset = row * columns;
            int outOffset = outRow * outCols;
            for (int col = 0; col < columns; col++) {
                if (col == j) {
                    continue;
                }
                int srcIndex = rowOffset + col;
                out[outOffset + outCol] = data[srcIndex];
                if (outImag != null) {
                    outImag[outOffset + outCol] = imag[srcIndex];
                }
                outCol++;
            }
            outRow++;
        }
        return result;
    }

    public static Matrix randomMatrix(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        double[] out = m.data;
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.random();
        }
        return m;
    }
}
