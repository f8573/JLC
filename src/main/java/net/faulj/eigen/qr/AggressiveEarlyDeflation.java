// src/main/java/net/faulj/eigen/qr/AggressiveEarlyDeflation.java
package net.faulj.eigen.qr;

import net.faulj.core.Tolerance;
import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.SchurResult;
import java.util.ArrayList;
import java.util.List;

/**
 * Implements Aggressive Early Deflation (AED) for the Hessenberg QR algorithm.
 * <p>
 * AED is a convergence acceleration technique that identifies and deflates eigenvalues
 * effectively before they manifest as negligible subdiagonal elements in the main
 * QR iteration. It works by analyzing a "window" at the bottom-right of the
 * active Hessenberg matrix.
 * </p>
 *
 * <h2>Algorithm Overview:</h2>
 * <ol>
 * <li><b>Window Selection:</b> A window of size <i>w</i> is chosen at the bottom-right of the active block.</li>
 * <li><b>Local Schur Decomposition:</b> The window is reduced to Real Schur form (T<sub>w</sub>)
 * using a standard QR iteration (like {@link ImplicitQRFrancis}).</li>
 * <li><b>Spike Computation:</b> The coupling vector (spike) connecting the window to the
 * rest of the matrix is transformed into the Schur basis.</li>
 * <li><b>Deflation Check:</b> The algorithm scans the spike from the bottom up. If the
 * spike component corresponding to a diagonal block in T<sub>w</sub> is negligible,
 * that block is considered deflated.</li>
 * <li><b>Similarity Update:</b> The global matrix is updated with the orthogonal
 * transformations derived from the window, and deflated eigenvalues are removed
 * from the active computational domain.</li>
 * </ol>
 *
 * <h2>Advantages:</h2>
 * <ul>
 * <li>Significantly reduces the total number of QR sweeps required.</li>
 * <li>Detects converged eigenvalues earlier than standard deflation.</li>
 * <li>Provides high-quality shifts for subsequent QR sweeps if deflation fails.</li>
 * </ul>
 *
 * <h2>Mathematical Context:</h2>
 * <p>
 * Given an active Hessenberg matrix H partitioned as:
 * </p>
 * <pre>
 * ┌ H₁₁  H₁₂ ┐
 * H = │            │
 * └ H₂₁  H₂₂ ┘
 * </pre>
 * <p>
 * Where H₂₂ is the window. We compute H₂₂ = UTU<sup>T</sup>. The updated matrix becomes:
 * </p>
 * <pre>
 * ┌ H₁₁   H₁₂U ┐
 * H' = │             │
 * └ U<sup>T</sup>H₂₁  T   ┘
 * </pre>
 * <p>
 * The term U<sup>T</sup>H₂₁ is the "spike". Since H is Hessenberg, H₂₁ has only one
 * non-zero element (top-right), making the spike calculation efficient.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see ImplicitQRFrancis
 * @see BlockedHessenbergQR
 */
public class AggressiveEarlyDeflation {

    /**
     * Encapsulates the result of an AED step, including deflation statistics and shift suggestions.
     */
    public static class AEDResult {
        /** The number of eigenvalues successfully deflated (removed from active submatrix). */
        public final int deflatedCount;
        /** Real parts of eigenvalues from the undeflated portion of the window (used as shifts). */
        public final double[] shiftsReal;
        /** Imaginary parts of eigenvalues from the undeflated portion of the window. */
        public final double[] shiftsImag;

        public AEDResult(int deflatedCount, double[] shiftsReal, double[] shiftsImag) {
            this.deflatedCount = deflatedCount;
            this.shiftsReal = shiftsReal;
            this.shiftsImag = shiftsImag;
        }
    }

    /**
     * Performs Aggressive Early Deflation on the active Hessenberg block.
     *
     * <h2>Process:</h2>
     * <ol>
     * <li>Define window W within H[activeStart...activeEnd].</li>
     * <li>Compute Schur decomposition of W: W = STS<sup>T</sup>.</li>
     * <li>Calculate spike v = S<sup>T</sup> * e₁ * beta.</li>
     * <li>Identify deflatable eigenvalues by checking spike magnitude against tolerance.</li>
     * <li>Update H globally: H = diag(I, S, I)<sup>T</sup> * H * diag(I, S, I).</li>
     * <li>Return deflated count and shifts (remaining eigenvalues in window).</li>
     * </ol>
     *
     * @param H The full Hessenberg matrix (modified in-place).
     * @param U The global Schur vector accumulator (can be null).
     * @param activeStart Start index (inclusive) of the active submatrix (top-left).
     * @param activeEnd End index (inclusive) of the active submatrix (bottom-right).
     * @param windowSize The target size of the deflation window (w).
     * @return Result containing number of deflated eigenvalues and shifts for the next sweep.
     */
    public static AEDResult process(Matrix H, Matrix U, int activeStart, int activeEnd, int windowSize) {
        int n = H.getRowCount();
        int m = activeEnd - activeStart + 1;

        // 1. Determine actual window size
        int w = Math.min(windowSize, m);
        int k = activeEnd - w + 1; // Window starts at k

        // If window covers the whole block, standard QR is sufficient (or handled here).
        // If w is too small, just return 0.
        if (w <= 1) {
            return new AEDResult(0, new double[0], new double[0]);
        }

        // 2. Extract the window W
        Matrix W = H.crop(k, activeEnd, k, activeEnd);

        // 3. Compute local Schur decomposition of W
        // We use ImplicitQRFrancis to get the Schur form and the transformation matrix S (stored in result.U)
        // W_new = S^T * W_old * S
        SchurResult localSchur = ImplicitQRFrancis.process(W, null);
        Matrix Tw = localSchur.getT(); // Local Schur form
        Matrix S = localSchur.getU();  // Local Similarity transformation

        // 4. Spike Analysis
        // The coupling of the window to the rest of the matrix is via H(k, k-1).
        // The "spike" vector in the local basis is v = S^T * e_1 * H(k, k-1).
        // Since S is orthogonal, v components are just the first row of S scaled by beta.
        double beta = H.get(k, k - 1);
        double[] spike = new double[w];
        for (int i = 0; i < w; i++) {
            // S.get(0, i) is the first element of the i-th column, which is S^T(i, 0).
            spike[i] = S.get(0, i) * beta;
        }

        // 5. Check for Deflation (Scan from bottom up)
        int deflatableCount = 0;
        int i = w - 1;

        // Threshold: epsilon * (norm(H) + |lambda|)
        // Approximate norm(H) with Frobenius of the window or global tolerance
        double tol = Tolerance.get();
        double wNorm = Tw.frobeniusNorm(); // Heuristic for local scale

        while (i >= 0) {
            boolean is2x2 = (i > 0 && Math.abs(Tw.get(i, i - 1)) > Tolerance.get() * (Math.abs(Tw.get(i-1, i-1)) + Math.abs(Tw.get(i, i))));

            if (is2x2) {
                // Check 2x2 block ending at i (indices i-1, i)
                double s1 = Math.abs(spike[i]);
                double s2 = Math.abs(spike[i - 1]);
                // Criterion: max(s1, s2) or norm(spike_block) is small
                double spikeNorm = Math.hypot(s1, s2);

                // Eigenvalues of the 2x2 block
                double diagAvg = (Math.abs(Tw.get(i-1, i-1)) + Math.abs(Tw.get(i, i))) / 2.0;

                if (spikeNorm <= tol * Math.max(wNorm, diagAvg)) {
                    deflatableCount += 2;
                    i -= 2;
                } else {
                    break; // Blocked by undeflatable eigenvalue
                }
            } else {
                // Check 1x1 block at i
                double s = Math.abs(spike[i]);
                double lambda = Math.abs(Tw.get(i, i));

                if (s <= tol * Math.max(wNorm, lambda)) {
                    deflatableCount++;
                    i--;
                } else {
                    break;
                }
            }
        }

        // 6. Apply Transformation S to Global Matrix H and U
        // Update H:
        // Rows k..activeEnd (from left): H = S^T * H
        // Cols k..activeEnd (from right): H = H * S

        // 1. Right update off-diagonal (Top-Right strip)
        if (k > 0) {
            Matrix topStrip = H.crop(0, k - 1, k, activeEnd);
            Matrix newTopStrip = topStrip.multiply(S);
            insertMatrix(H, newTopStrip, 0, k);
        }

        // 2. Paste diagonal block (Tw)
        insertMatrix(H, Tw, k, k);

        // 3. Left update off-diagonal (Right side of window)
        if (activeEnd < n - 1) {
            Matrix rightStrip = H.crop(k, activeEnd, activeEnd + 1, n - 1);
            // S^T * rightStrip
            Matrix newRightStrip = S.transpose().multiply(rightStrip);
            insertMatrix(H, newRightStrip, k, activeEnd + 1);
        }

        // 4. Update the subdiagonal coupling (The Spike)
        for (int j = 0; j < w; j++) {
            H.set(k + j, k - 1, spike[j]);
        }

        // Update U (Accumulate transformations)
        if (U != null) {
            // U = U * (I_k + S + I)
            // U(:, k:activeEnd) = U(:, k:activeEnd) * S
            Matrix uStrip = U.crop(0, n - 1, k, activeEnd);
            Matrix newUStrip = uStrip.multiply(S);
            insertMatrix(U, newUStrip, 0, k);
        }

        // 7. Perform Deflation
        // Set the coupling entries for deflated eigenvalues to zero.
        for (int j = 0; j < deflatableCount; j++) {
            H.set(activeEnd - j, k - 1, 0.0);
        }

        // 8. Collect Shifts
        // The eigenvalues of the UNDEFLATED top part of the window are the shifts for the next sweep.
        int undeflatedW = w - deflatableCount;
        List<Double> shiftsR = new ArrayList<>();
        List<Double> shiftsI = new ArrayList<>();

        // Extract eigenvalues from the top part of Tw
        int idx = 0;
        while (idx < undeflatedW) {
            boolean isBlock = (idx < undeflatedW - 1) &&
                    Math.abs(Tw.get(idx + 1, idx)) > Tolerance.get() * (Math.abs(Tw.get(idx, idx)) + Math.abs(Tw.get(idx+1, idx+1)));

            if (isBlock) {
                // 2x2 block
                double[] real = new double[w];
                double[] imag = new double[w];
                compute2x2(Tw, idx, idx + 1, real, imag);
                shiftsR.add(real[idx]);
                shiftsI.add(imag[idx]);
                shiftsR.add(real[idx + 1]);
                shiftsI.add(imag[idx + 1]);
                idx += 2;
            } else {
                shiftsR.add(Tw.get(idx, idx));
                shiftsI.add(0.0);
                idx++;
            }
        }

        return new AEDResult(deflatableCount,
                shiftsR.stream().mapToDouble(Double::doubleValue).toArray(),
                shiftsI.stream().mapToDouble(Double::doubleValue).toArray());
    }

    private static void insertMatrix(Matrix Target, Matrix Source, int row, int col) {
        for (int r = 0; r < Source.getRowCount(); r++) {
            for (int c = 0; c < Source.getColumnCount(); c++) {
                Target.set(row + r, col + c, Source.get(r, c));
            }
        }
    }

    private static void compute2x2(Matrix T, int i, int j, double[] real, double[] imag) {
        double a = T.get(i, i);
        double b = T.get(i, j);
        double c = T.get(j, i);
        double d = T.get(j, j);

        double trace = a + d;
        double det = a * d - b * c;
        double discriminant = trace * trace - 4 * det;

        if (discriminant >= 0) {
            double root = Math.sqrt(discriminant);
            real[i] = (trace + root) / 2.0;
            real[j] = (trace - root) / 2.0;
            imag[i] = 0;
            imag[j] = 0;
        } else {
            double root = Math.sqrt(-discriminant);
            real[i] = trace / 2.0;
            real[j] = trace / 2.0;
            imag[i] = root / 2.0;
            imag[j] = -root / 2.0;
        }
    }
}