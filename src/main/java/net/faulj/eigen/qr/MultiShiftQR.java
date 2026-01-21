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
     * Conservative approach: always use 2 (standard double-shift).
     *
     * @param n The size of the active submatrix.
     * @return Number of shifts (always even, typically 2-6).
     */
    public static int computeOptimalShiftCount(int n) {
        if (n < 30) return 2;
        if (n < 100) return 4;
        if (n < 300) return 6;
        return 6; // Cap at 6 for stability
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
        double[] shifts = new double[numShifts];
        
        // Ensure even number of shifts
        int actualShifts = (numShifts / 2) * 2;
        if (actualShifts < 2) actualShifts = 2;

        // Generate shifts from bottom 2x2 blocks
        int shiftCount = 0;
        for (int i = m; i >= l + 1 && shiftCount < actualShifts - 1; i -= 2) {
            // Get 2x2 block
            double a = H.get(i - 1, i - 1);
            double b = H.get(i - 1, i);
            double c = H.get(i, i - 1);
            double d = H.get(i, i);

            // Compute eigenvalues of 2x2 block
            double tr = a + d;
            double det = a * d - b * c;
            double disc = tr * tr - 4 * det;

            if (disc >= 0) {
                double sqrt = Math.sqrt(disc);
                shifts[shiftCount++] = (tr + sqrt) / 2;
                shifts[shiftCount++] = (tr - sqrt) / 2;
            } else {
                // Complex pair: use real part twice (Francis double-shift handles this)
                shifts[shiftCount++] = tr / 2.0;
                shifts[shiftCount++] = tr / 2.0;
            }
        }

        // If we didn't get enough shifts, use diagonal elements
        while (shiftCount < actualShifts && m - shiftCount >= l) {
            shifts[shiftCount] = H.get(m - shiftCount, m - shiftCount);
            shiftCount++;
        }

        // Ensure we have exactly actualShifts shifts
        if (shiftCount < actualShifts) {
            for (int i = shiftCount; i < actualShifts; i++) {
                shifts[i] = H.get(m, m);
            }
        }

        return shifts;
    }
}