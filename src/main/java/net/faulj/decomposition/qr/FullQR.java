package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

/**
 * Full QR decomposition wrapper: returns full-sized Q (m x m) and R (m x n).
 */
public class FullQR {
    /**
     * Compute a full QR decomposition.
     *
     * @param A matrix to decompose
     * @return QR result containing full Q and R
     */
    public static QRResult decompose(Matrix A) {
        return HouseholderQR.decompose(A);
    }
}
