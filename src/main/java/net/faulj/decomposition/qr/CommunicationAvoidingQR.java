package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

/**
 * Communication-Avoiding QR (CAQR).
 * Implemented using the tall-skinny QR tree structure to reduce data movement.
 */
public class CommunicationAvoidingQR {
    public static QRResult decompose(Matrix A) {
        return TallSkinnyQR.decompose(A);
    }
}
