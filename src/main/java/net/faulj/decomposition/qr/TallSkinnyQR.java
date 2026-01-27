package net.faulj.decomposition.qr;

import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Tall-skinny QR (TSQR) for m >> n matrices.
 * Uses a two-level tree: local thin QR per block, then QR of stacked R's.
 */
public class TallSkinnyQR {
    private static final int DEFAULT_BLOCK_SIZE = 64;

    /**
     * Compute a tall-skinny QR decomposition using a two-level TSQR tree.
     *
     * @param A matrix to decompose
     * @return QR result containing Q and R
     */
    public static QRResult decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isReal()) {
            throw new UnsupportedOperationException("Tall-skinny QR requires a real-valued matrix");
        }
        int m = A.getRowCount();
        int n = A.getColumnCount();
        if (m == 0 || n == 0) {
            return new QRResult(A, Matrix.Identity(m), A.copy());
        }
        if (m <= n) {
            return HouseholderQR.decomposeThin(A);
        }

        int blockSize = Math.max(n, DEFAULT_BLOCK_SIZE);
        List<Matrix> qBlocks = new ArrayList<>();
        List<Matrix> rBlocks = new ArrayList<>();

        int row = 0;
        while (row < m) {
            int end = Math.min(m, row + blockSize) - 1;
            Matrix block = A.crop(row, end, 0, n - 1);
            QRResult qrBlock = HouseholderQR.decomposeThin(block);
            qBlocks.add(qrBlock.getQ());
            rBlocks.add(qrBlock.getR());
            row = end + 1;
        }

        int stackedRows = 0;
        for (Matrix r : rBlocks) {
            stackedRows += r.getRowCount();
        }
        Matrix rStack = new Matrix(stackedRows, n);
        int offset = 0;
        for (Matrix r : rBlocks) {
            for (int i = 0; i < r.getRowCount(); i++) {
                for (int j = 0; j < n; j++) {
                    rStack.set(offset + i, j, r.get(i, j));
                }
            }
            offset += r.getRowCount();
        }

        QRResult qr2 = HouseholderQR.decomposeThin(rStack);
        Matrix q2 = qr2.getQ();
        Matrix rFinal = qr2.getR();

        List<Matrix> qFinalBlocks = new ArrayList<>();
        offset = 0;
        int q2Cols = q2.getColumnCount();
        for (int i = 0; i < qBlocks.size(); i++) {
            Matrix qBlock = qBlocks.get(i);
            int kRows = qBlock.getColumnCount();
            Matrix q2Block = q2.crop(offset, offset + kRows - 1, 0, q2Cols - 1);
            Matrix qFinal = qBlock.multiply(q2Block);
            qFinalBlocks.add(qFinal);
            offset += kRows;
        }

        Matrix Q = null;
        for (Matrix block : qFinalBlocks) {
            if (Q == null) {
                Q = block;
            } else {
                Q = Q.AppendMatrix(block, "DOWN");
            }
        }
        if (Q == null) {
            Q = Matrix.Identity(m);
        }

        return new QRResult(A, Q, rFinal);
    }
}
