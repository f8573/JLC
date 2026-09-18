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
        for (int row = 1; row < b; row++) {
            for (int col = 0; col < row; col++) {
                assertEquals(0.0, res.rTop[row * b + col], 0.0);
            }
        }
    }

    @Test
    public void factorPanelWyUpdateMatchesPanelQTranspose() {
        int m = 10;
        int b = 4;
        int trailingCols = 3;
        double[] panelColMajor = new double[m * b];
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < b; col++) {
                double value = Math.sin((row + 1) * 0.37 + (col + 1) * 0.19);
                panelColMajor[col * m + row] = value;
            }
        }

        QRConfig cfg = new QRConfig(b, 2, 64, 2, 0.6);
        CAQRPanelResult panel = CommunicationAvoidingQR.factorPanel(panelColMajor, 0, m, b, m, cfg);

        double[] trailing = new double[m * trailingCols];
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < trailingCols; col++) {
                double value = Math.cos((row + 1) * 0.23 - (col + 1) * 0.41);
                trailing[row * trailingCols + col] = value;
            }
        }

        double[] expected = compactWyUpdate(panel.workspace, trailing.clone(), m, trailingCols, b);
        CommunicationAvoidingQR.applyWYUpdate(trailing, 0, m, trailingCols, trailingCols, panel.workspace);

        for (int row = 0; row < m; row++) {
            for (int col = 0; col < trailingCols; col++) {
                assertEquals(expected[row * trailingCols + col], trailing[row * trailingCols + col], 1e-9);
            }
        }
    }

    private static double[] compactWyUpdate(WorkspaceManager workspace, double[] matrix, int rows, int cols, int panelWidth) {
        double[] y = new double[rows * panelWidth];
        double[] t = new double[panelWidth * panelWidth];
        java.nio.DoubleBuffer view = workspace.doubleBuffer().duplicate();
        view.position(workspace.yCombinedOffset);
        view.get(y);
        view.position(workspace.tCombinedOffset);
        view.get(t);

        double[] z = new double[panelWidth * cols];
        for (int k = 0; k < panelWidth; k++) {
            for (int col = 0; col < cols; col++) {
                double sum = 0.0;
                for (int row = 0; row < rows; row++) {
                    sum += y[row * panelWidth + k] * matrix[row * cols + col];
                }
                z[k * cols + col] = sum;
            }
        }

        double[] w = new double[panelWidth * cols];
        for (int row = 0; row < panelWidth; row++) {
            for (int col = 0; col < cols; col++) {
                double sum = 0.0;
                for (int k = 0; k < panelWidth; k++) {
                    sum += t[k * panelWidth + row] * z[k * cols + col];
                }
                w[row * cols + col] = sum;
            }
        }

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                double correction = 0.0;
                for (int k = 0; k < panelWidth; k++) {
                    correction += y[row * panelWidth + k] * w[k * cols + col];
                }
                matrix[row * cols + col] -= correction;
            }
        }
        return matrix;
    }
}
