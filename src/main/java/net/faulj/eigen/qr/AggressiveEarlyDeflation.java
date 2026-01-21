package net.faulj.eigen.qr;

import net.faulj.matrix.Matrix;

/**
 * Simplified Aggressive Early Deflation (AED) for large matrices.
 * <p>
 * This implementation avoids recursion and uses direct eigenvalue estimation
 * for deflation windows. It's only applied to large matrices where the benefit
 * outweighs the complexity.
 * </p>
 *
 * <h2>Strategy:</h2>
 * <ul>
 * <li>Only used for matrices larger than 50x50.</li>
 * <li>Uses simple 2x2 eigenvalue formulas instead of full Schur decomposition.</li>
 * <li>Checks deflation criteria directly without recursive QR calls.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 */
public class AggressiveEarlyDeflation {

    private static final int MIN_SIZE_FOR_AED = 50;

    /**
     * Attempts to deflate eigenvalues within the specified window.
     * Returns 0 if matrix is too small or deflation is not beneficial.
     *
     * @param H The global Hessenberg matrix.
     * @param Q The global accumulation matrix.
     * @param m The index of the last row/col of the active submatrix.
     * @param winSize The size of the deflation window.
     * @param tol The tolerance for deflation.
     * @return The number of eigenvalues successfully deflated.
     */
    public static int process(Matrix H, Matrix Q, int m, int winSize, double tol) {
        int n = H.getRowCount();
        
        // Only use AED for large matrices
        if (m + 1 < MIN_SIZE_FOR_AED) {
            return 0;
        }

        int winStart = Math.max(0, m - winSize + 1);
        int actualWinSize = m - winStart + 1;

        if (actualWinSize < 2) {
            return 0;
        }

        // Simple deflation check: scan bottom rows of window
        // Check if last 1x1 or 2x2 blocks are already converged
        int deflatedCount = 0;

        // Check last element
        if (m > 0) {
            double subdiag = Math.abs(H.get(m, m - 1));
            double diagSum = Math.abs(H.get(m - 1, m - 1)) + Math.abs(H.get(m, m));
            if (subdiag <= tol * (diagSum + tol)) {
                H.set(m, m - 1, 0.0);
                deflatedCount = 1;
            }
        }

        // Check if there's a 2x2 block that's converged
        if (deflatedCount == 0 && m > 1) {
            double subdiag1 = Math.abs(H.get(m, m - 1));
            double subdiag2 = Math.abs(H.get(m - 1, m - 2));
            
            // Check if bottom 2x2 forms a complex conjugate pair (converged block)
            if (subdiag1 > tol && subdiag2 <= tol) {
                double a = H.get(m - 1, m - 1);
                double b = H.get(m - 1, m);
                double c = H.get(m, m - 1);
                double d = H.get(m, m);
                double disc = (a + d) * (a + d) - 4 * (a * d - b * c);
                
                if (disc < 0) {
                    // Complex eigenvalues => converged 2x2 block
                    H.set(m - 1, m - 2, 0.0);
                    deflatedCount = 2;
                }
            }
        }

        return deflatedCount;
    }
}