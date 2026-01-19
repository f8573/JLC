package net.faulj.eigen.qr;

import net.faulj.matrix.Matrix;

/**
 * Implements the Bulge Chasing step of the Implicit QR algorithm.
 * <p>
 * This class applies the implicit similarity transformations defined by the shifts.
 * It effectively calculates <i>Q<sup>T</sup>HQ</i> where Q is determined by the
 * polynomial <i>p(H) = (H-s<sub>1</sub>I)...(H-s<sub>k</sub>I)</i>.
 * </p>
 *
 * <h2>Process:</h2>
 * <ol>
 * <li><b>Bulge Generation:</b> A "bulge" is introduced at the top-left corner
 * by transforming the first column according to the shifts.</li>
 * <li><b>Bulge Chasing:</b> The bulge is "chased" down the diagonal by
 * Householder reflectors that restore the Hessenberg structure column-by-column.</li>
 * <li><b>Removal:</b> The bulge eventually falls off the bottom-right corner.</li>
 * </ol>
 *
 * <h2>Structure Preservation:</h2>
 * <p>
 * The operation preserves the Upper Hessenberg form everywhere except strictly
 * within the small window of the bulge.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @see ImplicitQRFrancis
 */
public class BulgeChasing {

    /**
     * Performs a bulge chasing sweep using the provided shifts.
     *
     * @param H The Hessenberg matrix.
     * @param Q The accumulation matrix.
     * @param l The start index of the active submatrix.
     * @param m The end index of the active submatrix.
     * @param shifts The array of shifts to apply.
     */
    public static void performSweep(Matrix H, Matrix Q, int l, int m, double[] shifts) {
        // This is a simplified "Double Shift" implementation (2 shifts at a time)
        // extended to loop over all provided shifts.

        int n = H.getRowCount();

        // Process shifts in pairs
        for (int s = 0; s < shifts.length - 1; s += 2) {
            double s1 = shifts[s];
            double s2 = shifts[s+1];

            // 1. Compute first column of (H - s1*I)(H - s2*I)
            // We only need the first 3 elements: x, y, z
            double h00 = H.get(l, l);
            double h10 = H.get(l + 1, l);
            double h01 = H.get(l, l + 1);
            double h11 = H.get(l + 1, l + 1);
            double h21 = (l + 2 <= m) ? H.get(l + 2, l + 1) : 0;

            double sum = s1 + s2;
            double prod = s1 * s2; // If complex conjugate, this is real

            // v = (H - s1)(H - s2) e1
            // x = h00^2 + h01*h10 - sum*h00 + prod
            double x = h00 * h00 + h01 * h10 - sum * h00 + prod;
            // y = h10 * (h00 + h11 - sum)
            double y = h10 * (h00 + h11 - sum);
            // z = h10 * h21
            double z = h10 * h21;

            // 2. Chase the bulge
            for (int k = l; k < m - 1; k++) {
                // Determine Householder vector v to annihilate y and z
                double norm = Math.sqrt(x*x + y*y + z*z);
                if (norm == 0) break;

                double alpha = (x > 0) ? -norm : norm;
                double f = Math.sqrt(2 * (norm * norm - x * alpha));

                double v0 = (x - alpha) / f;
                double v1 = y / f;
                double v2 = z / f;

                // Apply Householder P = I - 2vv^T to rows k..k+2
                // H[k..k+2, :] = H - 2v(v^T H)
                applyReflectorLeft(H, k, v0, v1, v2);
                applyReflectorRight(H, k, v0, v1, v2);
                applyReflectorRight(Q, k, v0, v1, v2);

                // Prepare for next step
                // New x, y, z are the elements pushed down the diagonal
                if (k < m - 2) {
                    x = H.get(k + 1, k);
                    y = H.get(k + 2, k);
                    z = H.get(k + 3, k);
                }
            }
        }
    }

    private static void applyReflectorLeft(Matrix A, int row, double v0, double v1, double v2) {
        int n = A.getColumnCount();
        for (int j = 0; j < n; j++) {
            // dot = v^T * column j
            double dot = v0 * A.get(row, j) + v1 * A.get(row + 1, j) + v2 * A.get(row + 2, j);
            if (dot != 0) {
                // col = col - 2 * dot * v
                A.set(row, j, A.get(row, j) - 2 * v0 * dot);
                A.set(row + 1, j, A.get(row + 1, j) - 2 * v1 * dot);
                A.set(row + 2, j, A.get(row + 2, j) - 2 * v2 * dot);
            }
        }
    }

    private static void applyReflectorRight(Matrix A, int col, double v0, double v1, double v2) {
        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            // dot = row i * v
            double dot = A.get(i, col) * v0 + A.get(i, col + 1) * v1 + A.get(i, col + 2) * v2;
            if (dot != 0) {
                // row = row - 2 * dot * v^T
                A.set(i, col, A.get(i, col) - 2 * v0 * dot);
                A.set(i, col + 1, A.get(i, col + 1) - 2 * v1 * dot);
                A.set(i, col + 2, A.get(i, col + 2) - 2 * v2 * dot);
            }
        }
    }
}