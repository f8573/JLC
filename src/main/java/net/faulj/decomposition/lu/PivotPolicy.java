package net.faulj.decomposition.lu;

import net.faulj.matrix.Matrix;

/**
 * Pivoting policy interface for LU decomposition.
 */
public interface PivotPolicy {
    
    /**
     * Selects the pivot row for column k.
     * @param A the matrix being factored
     * @param k current column
     * @param startRow first candidate row
     * @return row index of selected pivot
     */
    int selectPivotRow(Matrix A, int k, int startRow);
    
    /**
     * Partial pivoting: select largest magnitude in column.
     */
    PivotPolicy PARTIAL = (A, k, startRow) -> {
        int n = A.getRowCount();
        int maxRow = startRow;
        double maxAbs = Math.abs(A.get(startRow, k));
        for (int r = startRow + 1; r < n; r++) {
            double abs = Math.abs(A.get(r, k));
            if (abs > maxAbs) {
                maxAbs = abs;
                maxRow = r;
            }
        }
        return maxRow;
    };
    
    /**
     * No pivoting (use natural order).
     */
    PivotPolicy NONE = (A, k, startRow) -> startRow;
}
