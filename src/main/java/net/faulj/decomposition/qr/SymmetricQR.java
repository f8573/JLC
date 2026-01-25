package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

/**
 * QR decomposition specialized for symmetric input matrices.
 * Validates symmetry and falls back to Householder QR.
 */
public class SymmetricQR {
    private static final double SYM_TOL = 1e-12;

    public static QRResult decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isReal()) {
            throw new UnsupportedOperationException("Symmetric QR requires a real-valued matrix");
        }
        if (!isSymmetric(A)) {
            throw new IllegalArgumentException("Matrix is not symmetric within tolerance");
        }
        return HouseholderQR.decompose(A);
    }

    private static boolean isSymmetric(Matrix A) {
        int n = A.getRowCount();
        if (n != A.getColumnCount()) {
            return false;
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(A.get(i, j) - A.get(j, i)) > SYM_TOL) {
                    return false;
                }
            }
        }
        return true;
    }
}
