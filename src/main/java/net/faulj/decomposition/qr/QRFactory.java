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

    // Guard to prevent re-entrant CAQR -> QRFactory recursion when
    // CAQR implementation calls back into QRFactory for local factorizations.
    private static final ThreadLocal<Boolean> inCAQR = ThreadLocal.withInitial(() -> Boolean.FALSE);

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
        if (strat == QRStrategy.CAQR && !inCAQR.get()) {
            // mark we are in CAQR to avoid recursive re-entry
            inCAQR.set(Boolean.TRUE);
            try {
                return CommunicationAvoidingQR.factor(A, thin, QRConfig.defaultConfig());
            } catch (UnsupportedOperationException ex) {
                return HouseholderQR.decomposeHouseholder(A, thin);
            } finally {
                inCAQR.set(Boolean.FALSE);
            }
        } else {
            return HouseholderQR.decomposeHouseholder(A, thin);
        }
    }
}
