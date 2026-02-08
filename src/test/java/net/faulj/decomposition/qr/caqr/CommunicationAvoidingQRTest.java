package net.faulj.decomposition.qr.caqr;

import org.junit.Test;
import static org.junit.Assert.*;

public class CommunicationAvoidingQRTest {

    @Test
    public void smokeFactorPanel() {
        int m = 128;
        int b = 16;
        double[] A = new double[m * b];
        // fill A with small deterministic pattern
        for (int j = 0; j < b; j++) {
            for (int i = 0; i < m; i++) {
                A[j * m + i] = (i + 1) + 0.01 * (j + 1);
            }
        }
        QRConfig cfg = new QRConfig(b, 4, 64, 4, 0.6);
        CAQRPanelResult res = CommunicationAvoidingQR.factorPanel(A, 0, m, b, m, cfg);
        assertNotNull(res);
        assertNotNull(res.rTop);
        assertEquals(b * b, res.rTop.length);
    }
}
