package net.faulj.eigen.qr;

import net.faulj.matrix.Matrix;

/**
 * Simplified Multi-Shift strategy for the Implicit QR algorithm.
 * <p>
 * This implementation uses conservative shift counts and direct eigenvalue
 * approximation to avoid recursion and stack overflow issues.
 * </p>
 *
 * <h2>Shift Strategy:</h2>
 * <ul>
 * <li>Uses smaller shift counts (2-6) to maintain stability.</li>
 * <li>Estimates shifts from diagonal elements and 2x2 blocks.</li>
 * <li>No recursive Schur decomposition calls.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 */
public class MultiShiftQR {

    /**
     * Determines the optimal number of shifts based on matrix size.
     * LAPACK dlaqr0: aggressive multi-shift for 512x512.
     *
     * @param n The size of the active submatrix.
     * @return Number of shifts (always even, typically 2-12).
     */
    public static int computeOptimalShiftCount(int n) {
        // LAPACK tuning for 512x512 sweet spot
        if (n < 30) return 2;
        if (n < 60) return 4;
        if (n < 150) return 6;
        if (n < 400) return 8;
        // For 512x512: use 10-12 shifts for fast convergence
        int ns = ((n + 6) / 9) & ~1;
        return Math.min(12, Math.max(10, ns));
    }

    /**
     * Generates shifts from the bottom-right corner of the matrix.
     * Uses direct eigenvalue formulas for 2x2 blocks instead of recursion.
     *
     * @param H The Hessenberg matrix.
     * @param l The start index of the active submatrix.
     * @param m The end index of the active submatrix.
     * @param numShifts The requested number of shifts.
     * @return Array of shift values.
     */
    public static double[] generateShifts(Matrix H, int l, int m, int numShifts) {
        return generateShiftsRaw(H.getRawData(), H.getRowCount(), l, m, numShifts);
    }

    /**
     * Generates shifts using raw array access for maximum performance.
     * Uses Wilkinson shift strategy for 2x2 blocks (LAPACK approach).
     *
     * @param h The raw Hessenberg matrix data (row-major).
     * @param n The matrix dimension.
     * @param l The start index of the active submatrix.
     * @param m The end index of the active submatrix.
     * @param numShifts The requested number of shifts.
     * @return Array of shift values.
     */
    public static double[] generateShiftsRaw(double[] h, int n, int l, int m, int numShifts) {
        // Ensure even number
        int ns = Math.max(2, (numShifts / 2) * 2);
        double[] shifts = new double[ns];

        // LAPACK dlaqr4: extract shifts from trailing ns x ns submatrix
        int shiftCount = 0;
        for (int i = m; i >= l + 1 && shiftCount < ns - 1; i -= 2) {
            // 2x2 block
            double a11 = h[(i - 1) * n + (i - 1)];
            double a12 = h[(i - 1) * n + i];
            double a21 = h[i * n + (i - 1)];
            double a22 = h[i * n + i];

            // LAPACK: compute eigenvalues with stable formula
            double s = Math.abs(a11) + Math.abs(a12) + Math.abs(a21) + Math.abs(a22);
            if (s == 0.0) {
                shifts[shiftCount++] = 0.0;
                shifts[shiftCount++] = 0.0;
            } else {
                a11 /= s; a12 /= s; a21 /= s; a22 /= s;
                double tr = a11 + a22;
                double det = a11 * a22 - a12 * a21;
                double rtdisc = Math.sqrt(Math.abs(tr * tr * 0.25 - det));

                shifts[shiftCount++] = s * (tr * 0.5 + rtdisc);
                shifts[shiftCount++] = s * (tr * 0.5 - rtdisc);
            }
        }

        // Fill remaining with diagonal elements
        while (shiftCount < ns) {
            int idx = Math.max(l, m - shiftCount);
            shifts[shiftCount++] = h[idx * n + idx];
        }

        return shifts;
    }

    /**
     * Generate exceptional shifts when standard shifts cause stagnation.
     * Uses random-like shifts based on iteration count.
     *
     * @param H The Hessenberg matrix.
     * @param l The start index of the active submatrix.
     * @param m The end index of the active submatrix.
     * @param numShifts The requested number of shifts.
     * @param stagnationCount Number of iterations without progress.
     * @return Array of exceptional shift values.
     */
    public static double[] generateExceptionalShifts(Matrix H, int l, int m, int numShifts, int stagnationCount) {
        return generateExceptionalShiftsRaw(H.getRawData(), H.getRowCount(), l, m, numShifts, stagnationCount);
    }

    /**
     * Generate exceptional shifts using raw array access for maximum performance.
     * LAPACK dlaqr4 approach: use ad-hoc perturbations.
     *
     * @param h The raw Hessenberg matrix data (row-major).
     * @param n The matrix dimension.
     * @param l The start index of the active submatrix.
     * @param m The end index of the active submatrix.
     * @param numShifts The requested number of shifts.
     * @param stagnationCount Number of iterations without progress.
     * @return Array of exceptional shift values.
     */
    public static double[] generateExceptionalShiftsRaw(double[] h, int n, int l, int m, int numShifts, int stagnationCount) {
        int ns = Math.max(2, (numShifts / 2) * 2);
        double[] shifts = new double[ns];

        // LAPACK: exceptional shift = 0.75*|H[m,m-1]| + |H[m,m]|
        double h_mm = h[m * n + m];
        double h_m_mm1 = (m > 0) ? Math.abs(h[m * n + (m - 1)]) : 0.0;
        double ss = Math.abs(h_mm) + h_m_mm1;
        double aa = 0.75 * h_m_mm1 + ss;
        double bb = ss;

        // Pair-wise exceptional shifts
        for (int i = 0; i < ns; i += 2) {
            shifts[i] = aa;
            shifts[i + 1] = bb;
            // Vary slightly for consecutive exceptional shifts
            aa = aa * 1.05;
            bb = bb * 0.95;
        }

        return shifts;
    }

    /**
     * Compute the AED window size for a given block size.
     *
     * @param blockSize Size of the active block.
     * @param numShifts Number of shifts being used.
     * @return Recommended AED window size.
     */
    public static int computeAEDWindowSize(int blockSize, int numShifts) {
        // Use window about 3 times the shift count, capped at block size
        int windowSize = Math.min(3 * numShifts, blockSize / 2);
        return Math.max(2, windowSize);
    }
}