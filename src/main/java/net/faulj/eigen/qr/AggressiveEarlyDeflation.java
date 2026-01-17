// src/main/java/net/faulj/eigen/qr/AggressiveEarlyDeflation.java
package net.faulj.eigen.qr;

import net.faulj.core.Tolerance;
import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.SchurResult;
import java.util.ArrayList;
import java.util.List;

public class AggressiveEarlyDeflation {

    /**
     * Result structure containing deflation statistics and shifts for the next sweep.
     */
    public static class AEDResult {
        public final int deflatedCount;
        public final double[] shiftsReal;
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
     * @param H The full Hessenberg matrix.
     * @param U The global Schur vector accumulator (can be null).
     * @param activeStart Start index (inclusive) of the active submatrix.
     * @param activeEnd End index (inclusive) of the active submatrix.
     * @param windowSize Target size of the deflation window.
     * @return Result containing number of deflated eigenvalues and shifts.
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
            // Actually, ImplicitQRFrancis returns U where U columns are eigenvectors.
            // S^T * e_1 corresponds to the first row of S if S is the matrix applied as U^T A U.
            // In ImplicitQRFrancis: T = Z^T H Z. So S (returned as U) is Z.
            // We need the first component of the transformed basis vectors.
            // v_hat = Z^T * e_1 * beta.
            // The i-th component is (column i of Z) dot e_1 * beta = Z(0, i) * beta.
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
        // H(k:activeEnd, k:activeEnd) is replaced by Tw (if we simply paste it),
        // BUT we must also update off-diagonal blocks H(0:k-1, k:activeEnd) and H(k:activeEnd, activeEnd+1:n)
        // using S.

        // Update H:
        // Rows k..activeEnd (from left): H = S^T * H
        // Cols k..activeEnd (from right): H = H * S

        // Optimization: The window W block (Tw) is already computed.
        // We can just paste Tw into H, but we MUST update the off-diagonal strips.

        // Apply S from right to columns k..activeEnd (affecting all rows 0..n)
        // H * S
        // We perform this manually or using matrix multiplication on strips.
        // Since S is w x w, this is efficient.

        // Strategy:
        // 1. Update H(0:k-1, k:activeEnd) = H(0:k-1, k:activeEnd) * S
        // 2. Paste Tw into H(k:activeEnd, k:activeEnd)
        // 3. Update H(k:activeEnd, activeEnd+1:n) = S^T * H(k:activeEnd, activeEnd+1:n)
        // 4. Update coupling spike H(k:activeEnd, k-1) = spike vector (computed above)

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
        // H(k+i, k-1) becomes spike[i]
        // Theoretically, if we reordered, the spike components for deflated eigenvalues would be 0.
        // Since we scanned from the bottom, spike[activeEnd - k - j] corresponds to the bottom rows.
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
        // The scan established that spike components for indices [w-deflatableCount ... w-1] are negligible.
        // In the global matrix, these are rows [activeEnd-deflatableCount+1 ... activeEnd].
        // The coupling is at column (k-1).

        // Note: The spike vector is placed in column k-1.
        // We explicitely zero out the specific spike entries that passed the test.
        for (int j = 0; j < deflatableCount; j++) {
            H.set(activeEnd - j, k - 1, 0.0);
        }

        // 8. Collect Shifts
        // The eigenvalues of the UNDEFLATED top part of the window are the shifts for the next sweep.
        // Tw is strictly upper Hessenberg (quasi-triangular).
        // The top part is Tw(0 : w-deflatableCount-1, ...).
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
                // Use the helper from ImplicitQRFrancis or re-implement 2x2 eigen solver
                // For shifts, we just need the values.
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

    // Duplicated from ImplicitQRFrancis for independence or refactor to utility
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