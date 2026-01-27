package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

/**
 * Communication-Avoiding QR (CAQR).
 * Implemented using the tall-skinny QR tree structure to reduce data movement.
 */
public class CommunicationAvoidingQR {
    /**
     * Compute a QR decomposition using a communication-avoiding strategy.
     *
     * @param A matrix to decompose
     * @return QR result containing Q and R
     */
    public static QRResult decompose(Matrix A) {
        return TallSkinnyQR.decompose(A);
    }
}
