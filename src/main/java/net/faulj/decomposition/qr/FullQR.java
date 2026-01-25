package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

/**
 * Full QR decomposition wrapper: returns full-sized Q (m x m) and R (m x n).
 */
public class FullQR {
    public static QRResult decompose(Matrix A) {
        return HouseholderQR.decompose(A);
    }
}
