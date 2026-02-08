package net.faulj.decomposition.qr;

import net.faulj.decomposition.qr.caqr.CommunicationAvoidingQR;
import net.faulj.decomposition.qr.caqr.QRConfig;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

/**
 * Factory to select QR strategy at runtime. Default is HOUSEHOLDER.
 * Strategy can be set via system property `la.qr.strategy` = {HOUSEHOLDER, CAQR}.
 */
public final class QRFactory {
    public enum QRStrategy { HOUSEHOLDER, CAQR }

    private QRFactory() {}

    public static QRStrategy currentStrategy() {
        String s = System.getProperty("la.qr.strategy", "HOUSEHOLDER");
        try {
            return QRStrategy.valueOf(s.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            return QRStrategy.HOUSEHOLDER;
        }
    }

    public static QRResult decompose(Matrix A, boolean thin) {
        QRStrategy strat = currentStrategy();
        if (strat == QRStrategy.CAQR) {
            // Call CAQR implementation (may be a stub until implemented)
            try {
                return CommunicationAvoidingQR.factor(A, thin, QRConfig.defaultConfig());
            } catch (UnsupportedOperationException ex) {
                // Fall back to Householder if CAQR not implemented
                return HouseholderQR.decomposeHouseholder(A, thin);
            }
        } else {
            return HouseholderQR.decomposeHouseholder(A, thin);
        }
    }
}
