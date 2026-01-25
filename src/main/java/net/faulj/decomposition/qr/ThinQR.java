package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

/**
 * Thin (economy) QR decomposition wrapper: returns Q (m x k) and R (k x n),
 * where k = min(m, n).
 */
public class ThinQR {
    public static QRResult decompose(Matrix A) {
        return HouseholderQR.decomposeThin(A);
    }
}
