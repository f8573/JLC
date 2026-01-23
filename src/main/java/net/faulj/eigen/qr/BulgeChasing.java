package net.faulj.eigen.qr;

import net.faulj.matrix.Matrix;

/**
 * Implements the Bulge Chasing step of the Implicit QR algorithm.
 * <p>
 * This simplified implementation includes bounds checking and safety limits
 * to prevent array access errors and stack overflow issues.
 * </p>
 *
 * <h2>Process:</h2>
 * <ol>
 * <li>Generate a bulge at the top-left corner using the shifts.</li>
 * <li>Chase the bulge down the diagonal using Householder reflectors.</li>
 * <li>Maintain Hessenberg structure throughout the process.</li>
 * </ol>
 *
 * @author JLC Development Team
 * @version 1.0
 */
public class BulgeChasing {

    private static final double EPSILON = 1e-12;

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
        if (shifts == null || shifts.length < 2) {
            return; // Need at least 2 shifts
        }

        int n = H.getRowCount();
        
        // Safety check: ensure active block is large enough
        if (m - l < 2 || l < 0 || m >= n) {
            return;
        }

        // Process shifts in pairs
        for (int s = 0; s < shifts.length - 1; s += 2) {
            double s1 = shifts[s];
            double s2 = shifts[s + 1];

            // Safety check before accessing elements
            if (l + 2 > m) {
                break; // Not enough space for 3x3 reflector
            }

            // 1. Compute first column of (H - s1*I)(H - s2*I)
            double h00 = H.get(l, l);
            double h10 = H.get(l + 1, l);
            double h01 = H.get(l, l + 1);
            double h11 = H.get(l + 1, l + 1);
            double h21 = (l + 2 <= m) ? H.get(l + 2, l + 1) : 0;

            double sum = s1 + s2;
            double prod = s1 * s2;

            // First column of (H - s1*I)(H - s2*I)
            double x = h00 * h00 + h01 * h10 - sum * h00 + prod;
            double y = h10 * (h00 + h11 - sum);
            double z = h10 * h21;

            // 2. Chase the bulge
            for (int k = l; k <= m - 2; k++) {
                // Determine Householder vector v to annihilate y and z
                double norm = Math.sqrt(x * x + y * y + z * z);
                if (norm < EPSILON) break;

                double alpha = (x > 0) ? -norm : norm;
                double denom = Math.sqrt(2 * (norm * norm - x * alpha));
                if (Math.abs(denom) < EPSILON) break;

                double v0 = (x - alpha) / denom;
                double v1 = y / denom;
                double v2 = z / denom;

                // Bounds check
                if (k + 2 >= n) break;

                // Apply Householder reflector
                applyReflectorLeft(H, k, m, v0, v1, v2);
                applyReflectorRight(H, k, m, v0, v1, v2);
                if (Q != null) {
                    applyReflectorRight(Q, k, n - 1, v0, v1, v2);
                }

                // Prepare for next step
                if (k < m - 2) {
                    x = H.get(k + 1, k);
                    y = H.get(k + 2, k);
                    z = (k + 3 <= m) ? H.get(k + 3, k) : 0;
                } else {
                    break;
                }
            }
        }
    }

    /**
     * Applies Householder reflector from the left with column range limiting.
     */
    private static void applyReflectorLeft(Matrix A, int row, int colEnd, double v0, double v1, double v2) {
        int n = A.getColumnCount();
        int maxCol = Math.min(n, colEnd + 3); // Only update relevant columns
        
        for (int j = 0; j < maxCol; j++) {
            double dot = v0 * A.get(row, j) + v1 * A.get(row + 1, j) + v2 * A.get(row + 2, j);
            if (Math.abs(dot) > EPSILON) {
                A.set(row, j, A.get(row, j) - 2 * v0 * dot);
                A.set(row + 1, j, A.get(row + 1, j) - 2 * v1 * dot);
                A.set(row + 2, j, A.get(row + 2, j) - 2 * v2 * dot);
            }
        }
    }

    /**
     * Applies Householder reflector from the right with row range limiting.
     */
    private static void applyReflectorRight(Matrix A, int col, int rowEnd, double v0, double v1, double v2) {
        int maxRow = Math.min(A.getRowCount(), rowEnd + 1);
        
        for (int i = 0; i < maxRow; i++) {
            double dot = A.get(i, col) * v0 + A.get(i, col + 1) * v1 + A.get(i, col + 2) * v2;
            if (Math.abs(dot) > EPSILON) {
                A.set(i, col, A.get(i, col) - 2 * v0 * dot);
                A.set(i, col + 1, A.get(i, col + 1) - 2 * v1 * dot);
                A.set(i, col + 2, A.get(i, col + 2) - 2 * v2 * dot);
            }
        }
    }
}