package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

/**
 * Thin (economy) QR decomposition wrapper: returns Q (m x k) and R (k x n),
 * where k = min(m, n).
 */
public class ThinQR {
    /**
     * Compute an economy-size QR decomposition.
     *
     * @param A matrix to decompose
     * @return QR result containing thin Q and R
     */
    public static QRResult decompose(Matrix A) {
        return HouseholderQR.decomposeThin(A);
    }
}
