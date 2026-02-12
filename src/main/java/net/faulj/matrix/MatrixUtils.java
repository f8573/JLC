package net.faulj.matrix;

import java.util.Arrays;

import net.faulj.core.Tolerance;

/**
 * Utility methods for matrix operations and validation.
 * 
 * <p>Note: For comprehensive accuracy validation with adaptive thresholds,
 * consider using {@link MatrixAccuracyValidator} instead of the legacy methods here.
 */
public class MatrixUtils {

    /** Machine epsilon for double precision */
    private static final double EPS = 2.220446049250313e-16;

    /**
     * Compute relative Frobenius norm error: ||A - Ahat||_F / ||A||_F
     * 
     * @param A Original matrix
     * @param Ahat Reconstructed matrix
     * @return Relative residual (0.0 for zero matrices)
     */
    public static double relativeError(Matrix A, Matrix Ahat) {
        double normA = A.frobeniusNorm();
        if (normA < EPS) return 0.0;
        return A.subtract(Ahat).frobeniusNorm() / normA;
    }
    
    /**
     * Compute orthogonality error: ||Q^T*Q - I||_F
     * 
     * @param Q Matrix to test for orthogonality
     * @return Frobenius norm of deviation from identity
     */
    public static double orthogonalityError(Matrix Q) {
        int n = Q.getColumnCount();
        Matrix QtQ = Q.transpose().multiply(Q);
        Matrix I = Matrix.Identity(n);
        return QtQ.subtract(I).frobeniusNorm();
    }

    /**
     * Transform the matrix to row echelon form in place.
     *
     * @param matrix matrix to transform
     * @return row-reduction metadata (pivot columns, exchanges)
     */
    public static RowReductionResult toRowEchelonForm(Matrix matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!matrix.isReal()) {
            throw new UnsupportedOperationException("Row echelon form requires a real-valued matrix");
        }
        double tol = Tolerance.get();
        int rows = matrix.getRowCount();
        int columns = matrix.getColumnCount();
        double[] data = matrix.getRawData();
        int exchanges = 0;
        int pivotRow = 0;
        int[] pivotCols = new int[Math.min(rows, columns)];
        int pivotCount = 0;

        for (int col = 0; col < columns && pivotRow < rows; col++) {
            int pivotIndex = -1;
            double maxAbs = tol;
            int scanIndex = pivotRow * columns + col;
            for (int row = pivotRow; row < rows; row++) {
                double val = Math.abs(data[scanIndex]);
                if (val > maxAbs) {
                    maxAbs = val;
                    pivotIndex = row;
                }
                scanIndex += columns;
            }

            if (pivotIndex == -1) {
                continue;
            }

            if (pivotIndex != pivotRow) {
                matrix.exchangeRows(pivotRow, pivotIndex);
                exchanges++;
            }

            double pivot = data[pivotRow * columns + col];
            if (Math.abs(pivot) <= tol) {
                continue;
            }

            for (int row = pivotRow + 1; row < rows; row++) {
                double val = data[row * columns + col];
                if (Math.abs(val) > tol) {
                    double multiplier = -val / pivot;
                    matrix.addMultipleOfRow(pivotRow, row, multiplier);
                }
            }

            pivotCols[pivotCount++] = col;
            pivotRow++;
        }
        return new RowReductionResult(exchanges, Arrays.copyOf(pivotCols, pivotCount));
    }

    /**
     * Transform the matrix to reduced row echelon form in place.
     *
     * @param matrix matrix to transform
     * @return row-reduction metadata (pivot columns, exchanges)
     */
    public static RowReductionResult toReducedRowEchelonForm(Matrix matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!matrix.isReal()) {
            throw new UnsupportedOperationException("Reduced row echelon form requires a real-valued matrix");
        }
        double tol = Tolerance.get();
        int rows = matrix.getRowCount();
        int columns = matrix.getColumnCount();
        double[] data = matrix.getRawData();
        int exchanges = 0;
        int pivotRow = 0;
        int[] pivotCols = new int[Math.min(rows, columns)];
        int pivotCount = 0;

        for (int col = 0; col < columns && pivotRow < rows; col++) {
            int pivotIndex = -1;
            double maxAbs = tol;
            int scanIndex = pivotRow * columns + col;
            for (int row = pivotRow; row < rows; row++) {
                double val = Math.abs(data[scanIndex]);
                if (val > maxAbs) {
                    maxAbs = val;
                    pivotIndex = row;
                }
                scanIndex += columns;
            }

            if (pivotIndex == -1) {
                continue;
            }

            if (pivotIndex != pivotRow) {
                matrix.exchangeRows(pivotRow, pivotIndex);
                exchanges++;
            }

            double pivot = data[pivotRow * columns + col];
            if (Math.abs(pivot) <= tol) {
                continue;
            }

            matrix.multiplyRow(pivotRow, 1.0 / pivot);

            for (int row = 0; row < rows; row++) {
                if (row == pivotRow) {
                    continue;
                }
                double val = data[row * columns + col];
                if (Math.abs(val) > tol) {
                    matrix.addMultipleOfRow(pivotRow, row, -val);
                }
            }

            pivotCols[pivotCount++] = col;
            pivotRow++;
        }
        return new RowReductionResult(exchanges, Arrays.copyOf(pivotCols, pivotCount));
    }

    /**
     * Produce a one-line concise summary of a matrix for logging/diagnostics.
     *
     * The summary includes shape, Frobenius norm, orthogonality error (or 'n/a'),
     * and a small top-left sample of entries up to the provided limits.
     */
    public static String matrixSummary(Matrix m, int maxRows, int maxCols) {
        if (m == null) return "null";
        int rows = m.getRowCount();
        int cols = m.getColumnCount();
        double frob = m.frobeniusNorm();
        String ortho = m.isSquare() ? String.format("%.6e", orthogonalityError(m)) : "n/a";

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%dx%d", rows, cols));
        sb.append(", frob=").append(String.format("%.6e", frob));
        sb.append(", ortho=").append(ortho);

        int r = Math.min(rows, Math.max(0, maxRows));
        int c = Math.min(cols, Math.max(0, maxCols));
        if (r > 0 && c > 0) {
            sb.append(", sample=");
            sb.append('[');
            for (int i = 0; i < r; i++) {
                if (i > 0) sb.append(',');
                sb.append('[');
                for (int j = 0; j < c; j++) {
                    if (j > 0) sb.append(',');
                    double re = m.get(i, j);
                    if (!m.hasImagData()) {
                        sb.append(String.format("%.6e", re));
                    } else {
                        double im = m.getImag(i, j);
                        if (im == 0.0) sb.append(String.format("%.6e", re));
                        else if (im > 0) sb.append(String.format("%.6e+%.6ei", re, im));
                        else sb.append(String.format("%.6e%.6ei", re, im));
                    }
                }
                sb.append(']');
            }
            sb.append(']');
        }

        return sb.toString();
    }

    public static final class RowReductionResult {
        private final int exchanges;
        private final int[] pivotColumns;

        public RowReductionResult(int exchanges, int[] pivotColumns) {
            this.exchanges = exchanges;
            this.pivotColumns = pivotColumns == null ? new int[0] : pivotColumns;
        }

        public int getExchanges() {
            return exchanges;
        }

        public int[] getPivotColumns() {
            return Arrays.copyOf(pivotColumns, pivotColumns.length);
        }
    }
}
