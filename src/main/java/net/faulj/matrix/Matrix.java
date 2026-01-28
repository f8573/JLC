package net.faulj.matrix;

import java.util.Arrays;
import java.util.Set;

import net.faulj.compute.BlockedMultiply;
import net.faulj.core.Tolerance;
import net.faulj.vector.Vector;
import net.faulj.spaces.SubspaceBasis;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Dense matrix implementation with optional complex components.
 * Provides common linear algebra operations and utility helpers.
 */
public class Matrix {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int TRANSPOSE_BLOCK = 64;
    private final int rows;
    private final int columns;
    private final double[] data;
    private double[] imag;

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
    }

    private Matrix(int rows, int cols, double[] data, double[] imag, boolean wrap) {
        this.rows = rows;
        this.columns = cols;
        this.data = data;
        this.imag = imag;
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
        Vector[] views = new Vector[columns];
        for (int col = 0; col < columns; col++) {
            views[col] = new Vector(this, col);
        }
        return views;
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
        return imag == null;
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
        kernelAddRow(sourceRow, targetRow, multiplier);
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
        int length = out.length;
        int i = 0;
        int loopBound = SPECIES.loopBound(length);
        for (; i < loopBound; i += SPECIES.length()) {
            DoubleVector va = DoubleVector.fromArray(SPECIES, a, i);
            DoubleVector vb = DoubleVector.fromArray(SPECIES, b, i);
            va.add(vb).intoArray(out, i);
        }
        for (; i < length; i++) {
            out[i] = a[i] + b[i];
        }
        if (ai != null || bi != null) {
            double[] oi = result.ensureImagData();
            if (ai != null && bi != null) {
                i = 0;
                loopBound = SPECIES.loopBound(length);
                for (; i < loopBound; i += SPECIES.length()) {
                    DoubleVector vai = DoubleVector.fromArray(SPECIES, ai, i);
                    DoubleVector vbi = DoubleVector.fromArray(SPECIES, bi, i);
                    vai.add(vbi).intoArray(oi, i);
                }
                for (; i < length; i++) {
                    oi[i] = ai[i] + bi[i];
                }
            } else if (ai != null) {
                System.arraycopy(ai, 0, oi, 0, length);
            } else {
                System.arraycopy(bi, 0, oi, 0, length);
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
        int length = out.length;
        int i = 0;
        int loopBound = SPECIES.loopBound(length);
        for (; i < loopBound; i += SPECIES.length()) {
            DoubleVector va = DoubleVector.fromArray(SPECIES, a, i);
            DoubleVector vb = DoubleVector.fromArray(SPECIES, b, i);
            va.sub(vb).intoArray(out, i);
        }
        for (; i < length; i++) {
            out[i] = a[i] - b[i];
        }
        if (ai != null || bi != null) {
            double[] oi = result.ensureImagData();
            if (ai != null && bi != null) {
                i = 0;
                loopBound = SPECIES.loopBound(length);
                for (; i < loopBound; i += SPECIES.length()) {
                    DoubleVector vai = DoubleVector.fromArray(SPECIES, ai, i);
                    DoubleVector vbi = DoubleVector.fromArray(SPECIES, bi, i);
                    vai.sub(vbi).intoArray(oi, i);
                }
                for (; i < length; i++) {
                    oi[i] = ai[i] - bi[i];
                }
            } else if (ai != null) {
                System.arraycopy(ai, 0, oi, 0, length);
            } else {
                i = 0;
                loopBound = SPECIES.loopBound(length);
                for (; i < loopBound; i += SPECIES.length()) {
                    DoubleVector vbi = DoubleVector.fromArray(SPECIES, bi, i);
                    vbi.neg().intoArray(oi, i);
                }
                for (; i < length; i++) {
                    oi[i] = -bi[i];
                }
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
        kernelScaleRow(row, multiplier);
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
        int blockSize = TRANSPOSE_BLOCK;
        for (int i = 0; i < rows; i += blockSize) {
            int iMax = Math.min(i + blockSize, rows);
            for (int j = 0; j < columns; j += blockSize) {
                int jMax = Math.min(j + blockSize, columns);
                for (int row = i; row < iMax; row++) {
                    int rowOffset = row * columns;
                    for (int col = j; col < jMax; col++) {
                        int outIndex = col * rows + row;
                        int srcIndex = rowOffset + col;
                        out[outIndex] = data[srcIndex];
                        if (outImag != null) {
                            outImag[outIndex] = imag[srcIndex];
                        }
                    }
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
        MatrixUtils.toReducedRowEchelonForm(augmented);
        double tol = Tolerance.get();

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
        if (rows != columns) {
            throw new ArithmeticException("Row-column count mismatch");
        }
        int n = rows;
        double tol = Tolerance.get();
        if (n <= 4) {
            return inverseSmall(n, tol);
        }
        if (!isInvertible()) {
            throw new ArithmeticException("Matrix is not invertible");
        }
        Matrix augmented = this.AppendMatrix(Matrix.Identity(n), "RIGHT");
        MatrixUtils.toReducedRowEchelonForm(augmented);
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
        int n = rows;
        if (n == 1) {
            return data[0];
        }
        if (n == 2) {
            return data[0] * data[3] - data[1] * data[2];
        }
        if (n == 3) {
            return det3x3(
                    data[0], data[1], data[2],
                    data[3], data[4], data[5],
                    data[6], data[7], data[8]);
        }
        if (n == 4) {
            double a00 = data[0];
            double a01 = data[1];
            double a02 = data[2];
            double a03 = data[3];
            double a10 = data[4];
            double a11 = data[5];
            double a12 = data[6];
            double a13 = data[7];
            double a20 = data[8];
            double a21 = data[9];
            double a22 = data[10];
            double a23 = data[11];
            double a30 = data[12];
            double a31 = data[13];
            double a32 = data[14];
            double a33 = data[15];
            double c00 = det3x3(a11, a12, a13, a21, a22, a23, a31, a32, a33);
            double c01 = -det3x3(a10, a12, a13, a20, a22, a23, a30, a32, a33);
            double c02 = det3x3(a10, a11, a13, a20, a21, a23, a30, a31, a33);
            double c03 = -det3x3(a10, a11, a12, a20, a21, a22, a30, a31, a32);
            return a00 * c00 + a01 * c01 + a02 * c02 + a03 * c03;
        }
        double tol = Tolerance.get();
        Matrix m = this.copy();
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
        int length = out.length;
        int i = 0;
        int loopBound = SPECIES.loopBound(length);
        DoubleVector factor = DoubleVector.broadcast(SPECIES, d);
        for (; i < loopBound; i += SPECIES.length()) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, data, i);
            v.mul(factor).intoArray(out, i);
        }
        for (; i < length; i++) {
            out[i] = data[i] * d;
        }
        if (imag != null) {
            double[] outImag = result.ensureImagData();
            i = 0;
            loopBound = SPECIES.loopBound(length);
            for (; i < loopBound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, imag, i);
                v.mul(factor).intoArray(outImag, i);
            }
            for (; i < length; i++) {
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
        int length = data.length;
        int i = 0;
        int loopBound = SPECIES.loopBound(length);
        DoubleVector sum = DoubleVector.zero(SPECIES);
        for (; i < loopBound; i += SPECIES.length()) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, data, i);
            sum = v.lanewise(VectorOperators.FMA, v, sum);
        }
        double total = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            total += data[i] * data[i];
        }
        if (imag != null) {
            i = 0;
            sum = DoubleVector.zero(SPECIES);
            for (; i < loopBound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, imag, i);
                sum = v.lanewise(VectorOperators.FMA, v, sum);
            }
            total += sum.reduceLanes(VectorOperators.ADD);
            for (; i < length; i++) {
                total += imag[i] * imag[i];
            }
        }
        return Math.sqrt(total);
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

    private void kernelScaleRow(int row, double factor) {
        if (factor == 1.0) {
            return;
        }
        int offset = row * columns;
        int length = columns;
        int i = 0;
        int loopBound = SPECIES.loopBound(length);
        DoubleVector fv = DoubleVector.broadcast(SPECIES, factor);
        for (; i < loopBound; i += SPECIES.length()) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, data, offset + i);
            v.mul(fv).intoArray(data, offset + i);
        }
        for (; i < length; i++) {
            data[offset + i] *= factor;
        }
        if (imag != null) {
            i = 0;
            for (; i < loopBound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, imag, offset + i);
                v.mul(fv).intoArray(imag, offset + i);
            }
            for (; i < length; i++) {
                imag[offset + i] *= factor;
            }
        }
    }

    private void kernelAddRow(int sourceRow, int targetRow, double factor) {
        if (factor == 0.0) {
            return;
        }
        int sourceOffset = sourceRow * columns;
        int targetOffset = targetRow * columns;
        int length = columns;
        int i = 0;
        int loopBound = SPECIES.loopBound(length);
        DoubleVector fv = DoubleVector.broadcast(SPECIES, factor);
        for (; i < loopBound; i += SPECIES.length()) {
            DoubleVector src = DoubleVector.fromArray(SPECIES, data, sourceOffset + i);
            DoubleVector dst = DoubleVector.fromArray(SPECIES, data, targetOffset + i);
            src.mul(fv).add(dst).intoArray(data, targetOffset + i);
        }
        for (; i < length; i++) {
            data[targetOffset + i] += factor * data[sourceOffset + i];
        }
        if (imag != null) {
            i = 0;
            for (; i < loopBound; i += SPECIES.length()) {
                DoubleVector src = DoubleVector.fromArray(SPECIES, imag, sourceOffset + i);
                DoubleVector dst = DoubleVector.fromArray(SPECIES, imag, targetOffset + i);
                src.mul(fv).add(dst).intoArray(imag, targetOffset + i);
            }
            for (; i < length; i++) {
                imag[targetOffset + i] += factor * imag[sourceOffset + i];
            }
        }
    }

    private static double det3x3(double a00, double a01, double a02,
                                 double a10, double a11, double a12,
                                 double a20, double a21, double a22) {
        return a00 * (a11 * a22 - a12 * a21)
                - a01 * (a10 * a22 - a12 * a20)
                + a02 * (a10 * a21 - a11 * a20);
    }

    private Matrix inverseSmall(int n, double tol) {
        if (n == 1) {
            double a = data[0];
            if (Math.abs(a) <= tol) {
                throw new ArithmeticException("Matrix is not invertible");
            }
            Matrix result = new Matrix(1, 1);
            result.data[0] = 1.0 / a;
            return result;
        }
        if (n == 2) {
            double a = data[0];
            double b = data[1];
            double c = data[2];
            double d = data[3];
            double det = a * d - b * c;
            if (Math.abs(det) <= tol) {
                throw new ArithmeticException("Matrix is not invertible");
            }
            double invDet = 1.0 / det;
            Matrix result = new Matrix(2, 2);
            result.data[0] = d * invDet;
            result.data[1] = -b * invDet;
            result.data[2] = -c * invDet;
            result.data[3] = a * invDet;
            return result;
        }
        if (n == 3) {
            double a00 = data[0];
            double a01 = data[1];
            double a02 = data[2];
            double a10 = data[3];
            double a11 = data[4];
            double a12 = data[5];
            double a20 = data[6];
            double a21 = data[7];
            double a22 = data[8];
            double det = det3x3(a00, a01, a02, a10, a11, a12, a20, a21, a22);
            if (Math.abs(det) <= tol) {
                throw new ArithmeticException("Matrix is not invertible");
            }
            double c00 = (a11 * a22 - a12 * a21);
            double c01 = -(a10 * a22 - a12 * a20);
            double c02 = (a10 * a21 - a11 * a20);
            double c10 = -(a01 * a22 - a02 * a21);
            double c11 = (a00 * a22 - a02 * a20);
            double c12 = -(a00 * a21 - a01 * a20);
            double c20 = (a01 * a12 - a02 * a11);
            double c21 = -(a00 * a12 - a02 * a10);
            double c22 = (a00 * a11 - a01 * a10);
            double invDet = 1.0 / det;
            Matrix result = new Matrix(3, 3);
            result.data[0] = c00 * invDet;
            result.data[1] = c10 * invDet;
            result.data[2] = c20 * invDet;
            result.data[3] = c01 * invDet;
            result.data[4] = c11 * invDet;
            result.data[5] = c21 * invDet;
            result.data[6] = c02 * invDet;
            result.data[7] = c12 * invDet;
            result.data[8] = c22 * invDet;
            return result;
        }
        double a00 = data[0];
        double a01 = data[1];
        double a02 = data[2];
        double a03 = data[3];
        double a10 = data[4];
        double a11 = data[5];
        double a12 = data[6];
        double a13 = data[7];
        double a20 = data[8];
        double a21 = data[9];
        double a22 = data[10];
        double a23 = data[11];
        double a30 = data[12];
        double a31 = data[13];
        double a32 = data[14];
        double a33 = data[15];

        double c00 = det3x3(a11, a12, a13, a21, a22, a23, a31, a32, a33);
        double c01 = -det3x3(a10, a12, a13, a20, a22, a23, a30, a32, a33);
        double c02 = det3x3(a10, a11, a13, a20, a21, a23, a30, a31, a33);
        double c03 = -det3x3(a10, a11, a12, a20, a21, a22, a30, a31, a32);
        double c10 = -det3x3(a01, a02, a03, a21, a22, a23, a31, a32, a33);
        double c11 = det3x3(a00, a02, a03, a20, a22, a23, a30, a32, a33);
        double c12 = -det3x3(a00, a01, a03, a20, a21, a23, a30, a31, a33);
        double c13 = det3x3(a00, a01, a02, a20, a21, a22, a30, a31, a32);
        double c20 = det3x3(a01, a02, a03, a11, a12, a13, a31, a32, a33);
        double c21 = -det3x3(a00, a02, a03, a10, a12, a13, a30, a32, a33);
        double c22 = det3x3(a00, a01, a03, a10, a11, a13, a30, a31, a33);
        double c23 = -det3x3(a00, a01, a02, a10, a11, a12, a30, a31, a32);
        double c30 = -det3x3(a01, a02, a03, a11, a12, a13, a21, a22, a23);
        double c31 = det3x3(a00, a02, a03, a10, a12, a13, a20, a22, a23);
        double c32 = -det3x3(a00, a01, a03, a10, a11, a13, a20, a21, a23);
        double c33 = det3x3(a00, a01, a02, a10, a11, a12, a20, a21, a22);

        double det = a00 * c00 + a01 * c01 + a02 * c02 + a03 * c03;
        if (Math.abs(det) <= tol) {
            throw new ArithmeticException("Matrix is not invertible");
        }
        double invDet = 1.0 / det;
        Matrix result = new Matrix(4, 4);
        result.data[0] = c00 * invDet;
        result.data[1] = c10 * invDet;
        result.data[2] = c20 * invDet;
        result.data[3] = c30 * invDet;
        result.data[4] = c01 * invDet;
        result.data[5] = c11 * invDet;
        result.data[6] = c21 * invDet;
        result.data[7] = c31 * invDet;
        result.data[8] = c02 * invDet;
        result.data[9] = c12 * invDet;
        result.data[10] = c22 * invDet;
        result.data[11] = c32 * invDet;
        result.data[12] = c03 * invDet;
        result.data[13] = c13 * invDet;
        result.data[14] = c23 * invDet;
        result.data[15] = c33 * invDet;
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
