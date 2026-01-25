package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Rank-one update/downdate QR. This implementation recomputes QR for correctness.
 */
public class UpdatingQR {
    public static QRResult rankOneUpdate(Matrix A, Vector u, Vector v) {
        Matrix updated = applyRankOne(A, u, v, 1.0);
        return HouseholderQR.decompose(updated);
    }

    public static QRResult rankOneDowndate(Matrix A, Vector u, Vector v) {
        Matrix updated = applyRankOne(A, u, v, -1.0);
        return HouseholderQR.decompose(updated);
    }

    private static Matrix applyRankOne(Matrix A, Vector u, Vector v, double sign) {
        if (A == null || u == null || v == null) {
            throw new IllegalArgumentException("Matrix and vectors must not be null");
        }
        if (!A.isReal() || !u.isReal() || !v.isReal()) {
            throw new UnsupportedOperationException("Rank-one updates require real-valued inputs");
        }
        int m = A.getRowCount();
        int n = A.getColumnCount();
        if (u.dimension() != m) {
            throw new IllegalArgumentException("Vector u dimension mismatch");
        }
        if (v.dimension() != n) {
            throw new IllegalArgumentException("Vector v dimension mismatch");
        }
        Matrix result = A.copy();
        for (int i = 0; i < m; i++) {
            double ui = u.get(i);
            for (int j = 0; j < n; j++) {
                double val = result.get(i, j) + sign * ui * v.get(j);
                result.set(i, j, val);
            }
        }
        return result;
    }
}
