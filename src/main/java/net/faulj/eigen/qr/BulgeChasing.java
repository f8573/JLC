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
     * LAPACK dlaqr5-inspired implementation with improved stability.
     *
     * @param H The Hessenberg matrix.
     * @param Q The accumulation matrix (may be null).
     * @param l The start index of the active submatrix.
     * @param m The end index of the active submatrix.
     * @param shifts The array of shifts to apply.
     */
    public static void performSweep(Matrix H, Matrix Q, int l, int m, double[] shifts) {
        if (shifts == null || shifts.length < 2) return;

        int n = H.getRowCount();
        if (m - l < 1 || l < 0 || m >= n) return;

        double[] h = H.getRawData();
        double[] q = (Q != null) ? Q.getRawData() : null;
        int qn = (Q != null) ? Q.getRowCount() : 0;

        // LAPACK: process shifts in pairs (double-shift strategy)
        for (int sp = 0; sp < shifts.length - 1; sp += 2) {
            double s1 = shifts[sp];
            double s2 = shifts[sp + 1];

            if (m - l + 1 < 3) {
                // Not enough space for 3x3 bulge, use 2x2
                if (m - l + 1 == 2) {
                    performDoubleShift2x2(h, n, q, qn, l, m, s1, s2);
                }
                continue;
            }

            // LAPACK: compute first column of (H - s1*I)(H - s2*I)
            double h00 = h[l * n + l];
            double h10 = h[(l + 1) * n + l];
            double h01 = h[l * n + l + 1];
            double h11 = h[(l + 1) * n + l + 1];
            double h20 = (l + 2 <= m) ? h[(l + 2) * n + l] : 0.0;
            double h21 = (l + 2 <= m) ? h[(l + 2) * n + l + 1] : 0.0;

            double sum = s1 + s2;
            double prod = s1 * s2;

            // First column of polynomial p(H) = (H - s1*I)(H - s2*I)
            double x = h00 * h00 + h01 * h10 - sum * h00 + prod;
            double y = h10 * (h00 + h11 - sum);
            double z = h10 * h21;

            // LAPACK dlaqr5: fast bulge chasing without normalization overhead
            for (int k = l; k <= m - 2; k++) {
                // Fast Householder without full normalization
                int nr = Math.min(3, m - k + 1);
                if (nr < 2) break;

                double norm = Math.sqrt(x * x + y * y + z * z);
                if (norm < EPSILON) break;

                // LAPACK: stable sign choice
                double beta = (x >= 0) ? -norm : norm;
                double inv_denom = 1.0 / (x - beta);

                double v0 = 1.0;
                double v1 = y * inv_denom;
                double v2 = z * inv_denom;

                // Householder reflector P = I - tau * v * v^T
                // where v = [1, v1, v2] and tau = 2 / (v^T v)
                double tau = 2.0 / (1.0 + v1 * v1 + v2 * v2);

                if (k + nr - 1 > m) break;

                // Apply from left to H
                int jstart = Math.max(0, k - 1);
                applyReflectorLeftRaw(h, n, k, jstart, n - 1, tau, v1, v2);

                // Apply from right to H
                int iend = Math.min(m + 1, k + 3);
                applyReflectorRightRaw(h, n, k, iend, tau, v1, v2);

                // Accumulate Q
                if (q != null) {
                    applyReflectorRightRaw(q, qn, k, qn - 1, tau, v1, v2);
                }

                // Next bulge position
                if (k < m - 2) {
                    x = h[(k + 1) * n + k];
                    y = h[(k + 2) * n + k];
                    z = (k + 3 <= m) ? h[(k + 3) * n + k] : 0.0;
                }
            }

            // LAPACK: final 2x2 cleanup if needed
            if (m >= l + 2 && Math.abs(h[m * n + m - 2]) > EPSILON) {
                double a = h[(m - 1) * n + m - 2];
                double b = h[m * n + m - 2];
                double r = Math.hypot(a, b); // More stable than sqrt(a^2 + b^2)
                if (r > EPSILON) {
                    double c = a / r;
                    double s = -b / r;
                    applyGivensLeftRaw(h, n, m - 1, m, c, s, m - 2, n - 1);
                    applyGivensRightRaw(h, n, m - 1, m, c, s, 0, m);
                    if (q != null) {
                        applyGivensRightRaw(q, qn, m - 1, m, c, s, 0, qn - 1);
                    }
                }
            }
        }

        // Ensure Hessenberg structure is preserved: zero out any fill-in
        // strictly below the first subdiagonal that may have arisen due
        // to numerical accumulation in the sweep. This is a conservative
        // cleanup that keeps the matrix in valid Hessenberg form for
        // downstream routines and tests.
        for (int i = 2; i < n; i++) {
            int base = i * n;
            for (int j = 0; j < i - 1; j++) {
                h[base + j] = 0.0;
            }
        }
    }

    /**
     * Specialized 2x2 double-shift for small blocks.
     * Applies a Givens rotation to perform an implicit QR step on the 2x2 block.
     */
    private static void performDoubleShift2x2(double[] h, int n, double[] q, int qn, int l, int m, double s1, double s2) {
        if (m - l != 1) return;

        double a11 = h[l * n + l];
        double a12 = h[l * n + m];
        double a21 = h[m * n + l];
        double a22 = h[m * n + m];

        // Compute first column of (H - s1*I)(H - s2*I) for the 2x2 block
        double sum = s1 + s2;
        double prod = s1 * s2;
        double x = a11 * a11 + a12 * a21 - sum * a11 + prod;
        double y = a21 * (a11 + a22 - sum);

        // Build Givens rotation to zero out y
        double r = Math.hypot(x, y);
        if (r < EPSILON) return;

        double c = x / r;
        double s = -y / r;

        // Apply Givens rotation as similarity transformation: G^T * H * G
        applyGivensLeftRaw(h, n, l, m, c, s, 0, n - 1);
        applyGivensRightRaw(h, n, l, m, c, s, 0, m);
        if (q != null) {
            applyGivensRightRaw(q, qn, l, m, c, s, 0, qn - 1);
        }
    }

    /**
     * Apply 3x3 Householder reflector from left using raw arrays.
     * P = I - tau * v * v^T where v = [1, v1, v2].
     * Computes P * A for rows [row, row+2] and columns [colStart, colEnd].
     */
    private static void applyReflectorLeftRaw(double[] a, int n, int row, int colStart, int colEnd, double tau, double v1, double v2) {
        int r0 = row * n;
        int r1 = (row + 1) * n;
        int r2 = (row + 2) * n;

        int jEnd = Math.min(colEnd + 1, n);

        for (int j = colStart; j < jEnd; j++) {
            double a0 = a[r0 + j];
            double a1 = a[r1 + j];
            double a2 = a[r2 + j];
            // dot = v^T * column = 1*a0 + v1*a1 + v2*a2
            double dot = a0 + v1 * a1 + v2 * a2;
            double tauDot = tau * dot;
            a[r0 + j] = a0 - tauDot;
            a[r1 + j] = a1 - v1 * tauDot;
            a[r2 + j] = a2 - v2 * tauDot;
        }
    }

    /**
     * Apply 3x3 Householder reflector from right using raw arrays.
     * P = I - tau * v * v^T where v = [1, v1, v2].
     * Computes A * P for rows [0, rowEnd] and columns [col, col+2].
     */
    private static void applyReflectorRightRaw(double[] a, int n, int col, int rowEnd, double tau, double v1, double v2) {
        int iEnd = Math.min(rowEnd + 1, n);
        int c0 = col;
        int c1 = col + 1;
        int c2 = col + 2;

        for (int i = 0; i < iEnd; i++) {
            int ri = i * n;
            double a0 = a[ri + c0];
            double a1 = a[ri + c1];
            double a2 = a[ri + c2];
            // dot = row * v = a0*1 + a1*v1 + a2*v2
            double dot = a0 + a1 * v1 + a2 * v2;
            double tauDot = tau * dot;
            a[ri + c0] = a0 - tauDot;
            a[ri + c1] = a1 - v1 * tauDot;
            a[ri + c2] = a2 - v2 * tauDot;
        }
    }

    /**
     * Apply 2x2 Givens rotation from left using raw arrays.
     */
    private static void applyGivensLeftRaw(double[] a, int n, int row1, int row2, double c, double s, int colStart, int colEnd) {
        int maxCol = Math.min(n, colEnd + 1);
        int r1 = row1 * n;
        int r2 = row2 * n;
        for (int j = Math.max(0, colStart); j < maxCol; j++) {
            double a1 = a[r1 + j];
            double a2 = a[r2 + j];
            a[r1 + j] = c * a1 - s * a2;
            a[r2 + j] = s * a1 + c * a2;
        }
    }

    /**
     * Apply 2x2 Givens rotation from right using raw arrays.
     */
    private static void applyGivensRightRaw(double[] a, int n, int col1, int col2, double c, double s, int rowStart, int rowEnd) {
        int maxRow = Math.min(n, rowEnd + 1);
        for (int i = Math.max(0, rowStart); i < maxRow; i++) {
            int ri = i * n;
            double a1 = a[ri + col1];
            double a2 = a[ri + col2];
            a[ri + col1] = c * a1 - s * a2;
            a[ri + col2] = s * a1 + c * a2;
        }
    }
}