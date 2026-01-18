package net.faulj.matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;
import net.faulj.spaces.SubspaceBasis;

public class Matrix {
    private Vector[] data;
    private final int columns;
    private int exchanges;
    private ArrayList<Vector> pivotColumns;
    private double tol = 1e-10;

    public Matrix(Vector[] data) {
        this.data = data;
        this.columns = data.length;
        pivotColumns = new ArrayList<>();
    }

    public Matrix(double[][] data) {
        Vector[] vectors = new Vector[data[0].length];
        for (int j = 0; j < data[0].length; j++) {
            double[] col = new double[data.length];
            for (int i = 0; i < data.length; i++) {
                col[i] = data[i][j];
            }
            vectors[j] = new Vector(col);
        }
        this.data = vectors;
        this.columns = vectors.length;
        pivotColumns = new ArrayList<>();
    }

    public Matrix(int i, int j) {
        Matrix m = Matrix.zero(i,j);
        this.data = m.data;
        this.columns = m.columns;
        this.pivotColumns = m.pivotColumns;
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

    public ArrayList<Vector> getPivotColumns() {
        return pivotColumns;
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

    public Matrix add(Matrix other) {
        if (columns != other.getColumnCount() || getRowCount() != other.getRowCount()) {
            throw new IllegalArgumentException("Matrices must have equal dimensions");
        }
        Matrix m = this.copy();
        Vector[] mData = m.getData();
        Vector[] oData = other.getData();
        for (int i = 0; i < m.getColumnCount(); i++) {
            mData[i] = mData[i].add(oData[i]);
        }

        return m;
    }

    public Matrix subtract(Matrix other) {
        if (columns != other.getColumnCount() || getRowCount() != other.getRowCount()) {
            throw new IllegalArgumentException("Matrices must have equal dimensions");
        }
        Matrix m = this.copy();
        Vector[] mData = m.getData();
        Vector[] oData = other.getData();
        for (int i = 0; i < m.getColumnCount(); i++) {
            mData[i] = mData[i].subtract(oData[i]);
        }

        return m;
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
                if (Math.abs(get(row, col)) > tol) {
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
                if (Math.abs(get(row, col)) > tol) {
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
        pivotColumns.clear();

        // We'll process columns from left to right
        while (pivotCol < getColumnCount() && pivotRow < getRowCount()) {
            // Find pivot
            int nonZeroRow = -1;
            for (int row = pivotRow; row < getRowCount(); row++) {
                if (Math.abs(get(row, pivotCol)) > tol) {
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
            pivotColumns.add(data[pivotCol]);
            if (Math.abs(pivotValue) > tol) {
                multiplyRow(pivotRow, 1.0 / pivotValue);
            }

            // Zero entries above pivot
            for (int row = 0; row < pivotRow; row++) {
                if (Math.abs(get(row, pivotCol)) > tol) {
                    double multiplier = -get(row, pivotCol);
                    addMultipleOfRow(pivotRow, row, multiplier);
                }
            }

            // Zero entries below pivot
            for (int row = pivotRow + 1; row < getRowCount(); row++) {
                if (Math.abs(get(row, pivotCol)) > tol) {
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
                if (Math.abs(augmented.get(r, c)) > tol) {
                    allZero = false;
                    break;
                }
            }
            if (allZero && Math.abs(augmented.get(r, augmentedColIndex)) > tol) {
                throw new ArithmeticException("No solution exists (inconsistent system)");
            }
        }

        // Count pivot columns (leading ones) to detect uniqueness
        int pivotCount = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < originalCols; c++) {
                if (Math.abs(augmented.get(r, c) - 1.0) < tol) {
                    // ensure it's the only non-zero in its column (RREF should guarantee this)
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

    public double trace() {
        if (getRowCount() != getColumnCount()) {
            throw new ArithmeticException("Row-column count mismatch");
        }
        double result = 0;
        for (int i = 0; i < columns; i++) {
            result += get(i,i);
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
        // Use partial-pivot Gaussian elimination on a copy for numerical stability
        Matrix m = this.copy();
        int n = m.getRowCount();
        int exchanges = 0;

        for (int col = 0; col < n; col++) {
            // find pivot with maximum absolute value in this column (partial pivoting)
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
                return 0.0; // singular (within tolerance)
            }

            if (maxRow != col) {
                m.exchangeRows(col, maxRow);
                exchanges++;
            }

            // eliminate below
            double pivot = m.get(col, col);
            for (int r = col + 1; r < n; r++) {
                double factor = m.get(r, col) / pivot;
                for (int c = col; c < n; c++) {
                    m.set(r, c, m.get(r, c) - factor * m.get(col, c));
                }
            }
        }

        double det = 1.0;
        for (int i = 0; i < n; i++) det *= m.get(i, i);
        if ((exchanges & 1) == 1) det = -det;
        return det;
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

    //rref nonzero rows
    public Set<Vector> rowSpaceBasis() {
        return SubspaceBasis.rowSpaceBasis(this);
    }

    //columns with pivots
    public Set<Vector> columnSpaceBasis() {
        return SubspaceBasis.columnSpaceBasis(this);
    }

    public Set<Vector> nullSpaceBasis() {
        return SubspaceBasis.nullSpaceBasis(this);
    }

    public Matrix[] QR() {
        net.faulj.decomposition.result.QRResult res = net.faulj.decomposition.qr.HouseholderQR.decompose(this);
        return new Matrix[]{res.getQ(), res.getR()};
    }

    public Matrix[] implicitQR() {
        return net.faulj.decomposition.qr.ImplicitQR.decompose(this);
    }

    

    public Matrix[] Hessenberg() {
        return net.faulj.decomposition.hessenberg.HessenbergReduction.decompose(this);
    }

    public boolean isSquare() {
        return getRowCount() == getColumnCount();
    }

    public static Matrix diag(int num, Matrix matrix) {
        //put zeroes on the right of I
        //put zeroes on the left of matrix
        //put the matrix under the I
        if (!matrix.isSquare()) {
            throw new IllegalArgumentException("Both matrices must be square to make a new diagonal");
        }

        if (num == 0) {
            return matrix;
        }

        Matrix I = Matrix.Identity(num);

        int i = I.getRowCount();
        int m = matrix.getRowCount();
        I = I.AppendMatrix(Matrix.zero(i,m), "RIGHT");
        matrix = matrix.AppendMatrix(Matrix.zero(m,i), "LEFT");
        return I.AppendMatrix(matrix, "DOWN");
    }

    public static Matrix zero(int rows, int columns) {
        Vector[] data = new Vector[columns];
        for(int i = 0; i < columns; i++) {
            data[i] = VectorUtils.zero(rows);
        }
        return new Matrix(data);
    }

    public Matrix multiplyScalar(double d) {
        Matrix m = this.copy();
        Vector[] mData = m.getData();
        for(int i = 0; i < m.getColumnCount(); i++) {
            mData[i] = mData[i].multiplyScalar(d);
        }
        return m;
    }

    /**
     * Computes the Frobenius norm: sqrt(sum of squares of all entries).
     */
    public double frobeniusNorm() {
        return MatrixNorms.frobeniusNorm(this);
    }

    /**
     * Computes the 1-norm (maximum absolute column sum).
     */
    public double norm1() {
        return MatrixNorms.norm1(this);
    }

    /**
     * Computes the infinity norm (maximum absolute row sum).
     */
    public double normInf() {
        return MatrixNorms.normInf(this);
    }

    public Matrix round(double tolerance) {
        Matrix m = this.copy();
        Vector[] mData = m.getData();
        for (int i = 0; i < m.getColumnCount(); i++) {
            Vector v = mData[i];
            double[] vData = v.getData();
            for (int j = 0; j < v.dimension(); j++) {
                if (vData[j] < tolerance) {
                    vData[j] = 0;
                }
            }
        }
        return m;
    }

    public Matrix crop(int fromRow, int toRow, int fromCol, int toCol) {
        if (fromRow < 0 || toRow >= getRowCount() || fromCol < 0 || toCol >= getColumnCount()
                || fromRow > toRow || fromCol > toCol) {
            throw new IllegalArgumentException("Invalid crop dimensions");
        }

        int newRows = toRow - fromRow + 1;
        int newCols = toCol - fromCol + 1;
        Vector[] newData = new Vector[newCols];

        for (int col = 0; col < newCols; col++) {
            double[] colData = new double[newRows];
            for (int row = 0; row < newRows; row++) {
                colData[row] = get(fromRow + row, fromCol + col);
            }
            newData[col] = new Vector(colData);
        }

        return new Matrix(newData);
    }

    public Matrix minor(int i, int j) {
        //basically, add all the columns to the arraylist, then transpose, then add rows to the other arraylist, then return
        Matrix m = this.copy();
        List<Vector> cols = new ArrayList<>(Arrays.asList(m.getData()));
        cols.remove(j);
        Matrix m1 = new Matrix(cols.toArray(new Vector[0]));
        m1 = m1.transpose();
        List<Vector> rows = new ArrayList<>(Arrays.asList(m1.getData()));
        rows.remove(i);
        return new Matrix(rows.toArray(new Vector[0])).transpose();
    }

    public static Matrix randomMatrix(int rows, int cols) {
        Vector[] data = new Vector[cols];
        for (int i = 0; i < cols; i++) {
            data[i] = VectorUtils.random(rows);
        }
        return new Matrix(data);
    }
    
    
}