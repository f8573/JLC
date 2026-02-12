package net.faulj.svd;

import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.decomposition.result.SVDResult;
import net.faulj.matrix.Matrix;

/**
 * Computes the SVD via Golub-Kahan bidiagonalization followed by implicit QR iteration
 * on the bidiagonal matrix.
 */
public class GolubKahanSVD {
    /**
     * Create a Golub-Kahan SVD solver.
     */
    public GolubKahanSVD() {
    }

    /**
     * Computes the full SVD of A.
     *
     * @param A input matrix
     * @return full SVD result
     */
    public SVDResult decompose(Matrix A) {
        return decomposeInternal(A, false);
    }

    /**
     * Computes the thin SVD of A.
     *
     * @param A input matrix
     * @return thin SVD result
     */
    public SVDResult decomposeThin(Matrix A) {
        return decomposeInternal(A, true);
    }

    /**
     * Internal decomposition routine.
     *
     * @param A input matrix
     * @param thin whether to return the economy-size factors
     * @return SVD result
     */
    private SVDResult decomposeInternal(Matrix A, boolean thin) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        Bidiagonalization bidiagonalization = new Bidiagonalization();
        BidiagonalizationResult bidiag = bidiagonalization.decompose(A);

        BidiagonalQR.BidiagonalSVDResult bidiagSvd = BidiagonalQR.decompose(bidiag.getB());

        Matrix U = bidiag.getU().multiply(bidiagSvd.getU());
        Matrix V = bidiag.getV().multiply(bidiagSvd.getV());
        double[] singularValues = bidiagSvd.getSingularValues();

        int r = Math.min(A.getRowCount(), A.getColumnCount());
        int[] order = sortIndicesDescending(singularValues);
        singularValues = reorderValues(singularValues, order);
        U = reorderColumns(U, order, r);
        V = reorderColumns(V, order, r);

        if (thin) {
            int m = A.getRowCount();
            int n = A.getColumnCount();
            U = U.getColumnCount() == r ? U : U.crop(0, m - 1, 0, r - 1);
            V = V.getColumnCount() == r ? V : V.crop(0, n - 1, 0, r - 1);
            singularValues = java.util.Arrays.copyOf(singularValues, r);
        }

        // If assembled U is not orthogonal, re-orthonormalize via QR and a small SVD
        try {
            double uOrtho = net.faulj.matrix.MatrixUtils.orthogonalityError(U);
            if (uOrtho > 1e-10) {
                int rcols = Math.min(U.getColumnCount(), singularValues.length);
                Matrix[] qr = U.thinQR();
                Matrix Q = qr[0];
                Matrix R = qr[1].crop(0, rcols - 1, 0, rcols - 1);
                Matrix SigmaR = new Matrix(rcols, rcols);
                for (int i = 0; i < rcols; i++) SigmaR.set(i, i, singularValues[i]);
                Matrix S = R.multiply(SigmaR);
                SVDResult small = new GolubKahanSVD().decomposeThin(S);
                Matrix Usmall = small.getU();
                double[] sigmaNew = small.getSingularValues();
                Matrix Vsmall = small.getV();
                Matrix Ufinal = Q.multiply(Usmall);
                Matrix Vleft = V.getColumnCount() == rcols ? V : V.crop(0, V.getRowCount() - 1, 0, rcols - 1);
                Matrix Vfinal = Vleft.multiply(Vsmall);
                double[] finalSig = java.util.Arrays.copyOf(sigmaNew, sigmaNew.length);
                return new SVDResult(A, Ufinal, finalSig, Vfinal);
            }
        } catch (Exception e) {
            // fallback to returning the original computed result
        }

        return new SVDResult(A, U, singularValues, V);
    }

    /**
     * Sort indices by descending values.
     *
     * @param values input values
     * @return index order
     */
    private static int[] sortIndicesDescending(double[] values) {
        Integer[] order = new Integer[values.length];
        for (int i = 0; i < values.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(values[b], values[a]));
        int[] result = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = order[i];
        }
        return result;
    }

    /**
     * Reorder values based on an index order.
     *
     * @param values input values
     * @param order index order
     * @return reordered values
     */
    private static double[] reorderValues(double[] values, int[] order) {
        double[] reordered = new double[values.length];
        for (int i = 0; i < order.length; i++) {
            reordered[i] = values[order[i]];
        }
        return reordered;
    }

    /**
     * Reorder matrix columns according to an index order.
     *
     * @param M matrix to reorder
     * @param order column order
     * @param reorderCount number of columns to reorder
     * @return reordered matrix
     */
    private static Matrix reorderColumns(Matrix M, int[] order, int reorderCount) {
        int cols = M.getColumnCount();
        net.faulj.vector.Vector[] data = new net.faulj.vector.Vector[cols];
        for (int i = 0; i < reorderCount; i++) {
            data[i] = M.getData()[order[i]].copy();
        }
        for (int i = reorderCount; i < cols; i++) {
            data[i] = M.getData()[i].copy();
        }
        return new Matrix(data);
    }
}
