package net.faulj.eigen.qr;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Implements the Bulge Chasing mechanism for the Implicit Multi-Shift QR Algorithm.
 * <p>
 * This class handles the creation and "chasing" of a bulge created by implicitly
 * applying a polynomial of shifts to a Hessenberg matrix. This is the core engine
 * for Multi-Shift QR and Aggressive Early Deflation.
 * </p>
 */
public class BulgeChasing {

    private static final double EPSILON = 1e-14;

    /**
     * Performs a single multi-shift QR sweep on the specified submatrix of H.
     * <p>
     * 1. Computes the first column of p(H) where p is the polynomial defined by the shifts.
     * 2. Introduces a bulge at the top of the active block.
     * 3. Chases the bulge down to the bottom of the active block (or off the matrix).
     * </p>
     *
     * @param H          The Hessenberg matrix (modified in-place).
     * @param Q          The orthogonal accumulator (modified in-place, can be null).
     * @param startRow   The starting row/col index of the active submatrix.
     * @param endRow     The ending row/col index (inclusive) of the active submatrix.
     * @param realShifts Array of real parts of the shifts.
     * @param imagShifts Array of imaginary parts of the shifts.
     */
    public static void process(Matrix H, Matrix Q, int startRow, int endRow, double[] realShifts, double[] imagShifts) {
        int m = realShifts.length; // Degree of the shift polynomial (bulge size)
        int n = H.getRowCount();

        // 1. Compute the initial vector x = p(H)e_1 restricted to the active window.
        // The vector will have non-zero elements in the first m+1 positions relative to startRow.
        double[] x = computeInitialVector(H, startRow, endRow, realShifts, imagShifts);

        // 2. Introduce the bulge (Step 0 of chasing)
        // Construct Householder reflector P0 to annihilate x[1...m]
        // Apply P0 to H from left (rows) and right (cols), and to Q from right.
        int bulgeLimit = Math.min(endRow, startRow + m);
        applyHouseholderStep(H, Q, x, startRow, startRow - 1, bulgeLimit, n);

        // 3. Chase the bulge down the matrix
        // The bulge initially occupies rows startRow+1 to startRow+m+1 (roughly).
        // We iterate to annihilate sub-diagonal elements created by the previous step.
        // k is the column index we are clearing (the column *left* of the pivot).
        for (int k = startRow; k <= endRow - 2; k++) {
            // Determine the size of the bulge at this position.
            // The non-zero entries in column k are at indices k+1 (diagonal sub) ... k+m+1.
            // We want to keep k+1 and annihilate k+2 ... k+m+1.

            int firstNonZero = k + 1;
            int lastNonZero = Math.min(k + m + 1, endRow);

            // If we are at the bottom and there's nothing to annihilate, stop.
            if (lastNonZero <= firstNonZero) break;

            // Extract the vector to be reduced: H[k+1...lastNonZero, k]
            double[] v = new double[lastNonZero - firstNonZero + 1];
            for (int i = 0; i < v.length; i++) {
                v[i] = H.get(firstNonZero + i, k);
            }

            // Apply Householder step to annihilate v[1...] (which corresponds to H[k+2...])
            applyHouseholderStep(H, Q, v, firstNonZero, k, lastNonZero, n);

            // Optimization: If the bulge falls off the matrix, the loop condition or lastNonZero handles it.
        }
    }

    /**
     * Computes the vector x = (H - s_1 I)...(H - s_m I) e_1.
     * Handles complex conjugate pairs in shifts to ensure the result is real.
     * <p>
     * Only computes the relevant top elements, as the vector grows by 1 nonzero per degree.
     * </p>
     */
    private static double[] computeInitialVector(Matrix H, int start, int end, double[] re, double[] im) {
        int m = re.length;
        int limit = Math.min(end, start + m);
        int size = limit - start + 1;

        // Initial vector e_1 (relative to start)
        // We only track the non-zero part.
        double[] v = new double[size];
        v[0] = 1.0;

        // Current valid length of the vector (number of non-zeros)
        int currentLen = 1;

        for (int i = 0; i < m; i++) {
            // Check for complex conjugate pair
            boolean isComplexPair = (i < m - 1) && (im[i] != 0) && (Math.abs(im[i] + im[i+1]) < EPSILON) && (Math.abs(re[i] - re[i+1]) < EPSILON);

            if (isComplexPair) {
                // Apply quadratic factor: (H - s)(H - s_conj) = H^2 - 2*Re(s)*H + |s|^2 I
                // Op: v <- H*(H*v) - 2*Re*H*v + ModSq*v
                double twoRe = 2 * re[i];
                double modSq = re[i]*re[i] + im[i]*im[i];

                // We need to apply H twice.
                // First H*v
                // The non-zero part grows by 1.
                int nextLen = Math.min(size, currentLen + 1);
                double[] Hv = multiplyHessenbergSection(H, start, v, currentLen, nextLen);

                // Second H*(Hv)
                int finalLen = Math.min(size, nextLen + 1);
                double[] HHv = multiplyHessenbergSection(H, start, Hv, nextLen, finalLen);

                // Combine: HHv - twoRe * Hv + modSq * v
                for (int j = 0; j < finalLen; j++) {
                    double valHv = (j < nextLen) ? Hv[j] : 0;
                    double valV = (j < currentLen) ? v[j] : 0;
                    v[j] = HHv[j] - twoRe * valHv + modSq * valV;
                }
                currentLen = finalLen;
                i++; // Skip next shift
            } else {
                // Apply linear factor: (H - s I)
                // Op: v <- H*v - s*v
                double s = re[i]; // Ignore imaginary part if not paired (should be 0 for real matrices or handled elsewhere)

                int nextLen = Math.min(size, currentLen + 1);
                double[] Hv = multiplyHessenbergSection(H, start, v, currentLen, nextLen);

                for (int j = 0; j < nextLen; j++) {
                    double valV = (j < currentLen) ? v[j] : 0;
                    v[j] = Hv[j] - s * valV;
                }
                currentLen = nextLen;
            }
        }
        return v;
    }

    /**
     * Multiplies the Hessenberg submatrix H[start...start+outLen, start...start+inLen] by v.
     * Assumes H is Hessenberg (lower triangle is zero below first subdiagonal).
     */
    private static double[] multiplyHessenbergSection(Matrix H, int startAbs, double[] v, int inLen, int outLen) {
        double[] result = new double[outLen];

        // y = H * v
        // y_i = sum(H_ij * v_j)
        // Since H is Hessenberg, H_ij = 0 if i > j + 1
        // So for row i, we sum j from max(0, i-1) to inLen-1

        for (int i = 0; i < outLen; i++) {
            double sum = 0;
            int rowAbs = startAbs + i;

            // Optimization: H is upper Hessenberg.
            // Loop range: j starts from 0? No, j can go up to inLen.
            // But H_ij is only non-zero for j >= i - 1.
            // So j starts at max(0, i-1).
            int jStart = Math.max(0, i - 1);

            for (int j = jStart; j < inLen; j++) {
                int colAbs = startAbs + j;
                sum += H.get(rowAbs, colAbs) * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    /**
     * Constructs a Householder reflector for vector v and applies it to H and Q.
     * The reflector acts on the window defined by pivotRow to lastRow.
     * * @param v        The vector to annihilate (v[0] is pivot, v[1...] are zeroed).
     * @param pivotRow The absolute row index corresponding to v[0].
     * @param colRef   The column index defining the left edge of the update (used for H update logic).
     * For bulge introduction, this might be start-1. For chasing, it's the column we create zeros in.
     * @param lastRow  The absolute row index of the last element of v.
     */
    private static void applyHouseholderStep(Matrix H, Matrix Q, double[] v, int pivotRow, int colRef, int lastRow, int n) {
        int len = lastRow - pivotRow + 1;
        if (len <= 1) return;

        // 1. Construct Householder vector u
        // v = [x0, x1, ... ]^T
        // u = v + sign(x0)||v|| e1
        double x0 = v[0];
        double normSq = 0;
        for (double val : v) normSq += val * val;

        if (normSq < EPSILON * EPSILON) return; // Vector is effectively zero

        double norm = Math.sqrt(normSq);
        double alpha = (x0 >= 0) ? -norm : norm;

        double u0 = x0 - alpha;
        // normalization factor beta = 2 / (u^T u)
        // u^T u = (x0 - alpha)^2 + x1^2 + ...
        //       = x0^2 - 2 x0 alpha + alpha^2 + (sum xi^2)
        //       = x0^2 + (sum xi^2) + alpha^2 - 2 x0 alpha
        //       = normSq + normSq - 2 x0 alpha
        //       = 2 (normSq - x0 alpha)

        // Wait, simpler: use the constructed u vector explicitly
        double[] u = new double[len];
        u[0] = 1.0; // We usually normalize u so u[0] = 1 for storage, but here we calculate explicit beta

        // Let's stick to the v - alpha*e1 formulation without implicit 1
        // u_raw = [x0 - alpha, x1, x2, ...]
        double u_normSq = 0;
        u_normSq += (x0 - alpha) * (x0 - alpha);
        for(int i=1; i<len; i++) {
            u_normSq += v[i] * v[i];
        }

        if (u_normSq < EPSILON) return;

        double beta = 2.0 / u_normSq;

        // Fill u vector
        u[0] = x0 - alpha;
        for(int i=1; i<len; i++) u[i] = v[i];

        // 2. Apply (I - beta u u^T) to H from Left (Rows pivotRow...lastRow)
        // A -> P A = A - beta u (u^T A)
        // We only update columns from colRef + 1 to n (Hessenberg structure optimization)
        // Actually, we must update all columns to the right that have non-zeros.
        // Since this is a similarity on H, full width is relevant?
        // Standard H update: cols k to n.
        // But for "Chase", we might be in the middle of the matrix.
        // Update columns: from (colRef + 1) to n. (Since colRef is where we created zeros, elements left of it are 0 or untouched)

        // Optimization: Col loop
        // Compute w = u^T A
        // A = A - beta u w

        int startCol = Math.max(0, colRef + 1); // Typically we start updating at the column containing the bulge + 1
        // But if we are introducing, we update from startRow.
        // Safe bet: Update from 0 to n? No, expensive.
        // Hessenberg: Zeros below subdiagonal.
        // Left mul acts on rows. Affected rows are pivotRow..lastRow.
        // Any column j where A[row, j] is non-zero needs update.
        // Since H is Hessenberg, lower rows have zeros on left.
        // pivotRow is roughly k+1.
        // So columns 0 to k-1 are zero in these rows?
        // Yes, H[i, j] = 0 for i > j+1.
        // If i >= pivotRow > colRef+1, then for j < colRef, H[i,j] is 0.
        // So we can start columns at (pivotRow - 1) or roughly colRef.

        for (int j = Math.max(0, colRef); j < n; j++) {
            double dot = 0;
            for (int i = 0; i < len; i++) {
                dot += u[i] * H.get(pivotRow + i, j);
            }
            dot *= beta;
            for (int i = 0; i < len; i++) {
                H.set(pivotRow + i, j, H.get(pivotRow + i, j) - dot * u[i]);
            }
        }

        // 3. Apply (I - beta u u^T) to H from Right (Cols pivotRow...lastRow)
        // A -> A P = A - beta (A u) u^T
        // Right mul acts on columns. Affected cols are pivotRow..lastRow.
        // Update rows 0 to lastRow+1 (Hessenberg structure).
        // For rows > lastRow+1, elements are 0.

        int rowLimit = Math.min(n, lastRow + 2);

        for (int i = 0; i < rowLimit; i++) {
            double dot = 0;
            for (int j = 0; j < len; j++) {
                dot += H.get(i, pivotRow + j) * u[j];
            }
            dot *= beta;
            for (int j = 0; j < len; j++) {
                H.set(i, pivotRow + j, H.get(i, pivotRow + j) - dot * u[j]);
            }
        }

        // 4. Apply to Q from Right (Accumulate)
        // Q -> Q P
        if (Q != null) {
            for (int i = 0; i < n; i++) {
                double dot = 0;
                for (int j = 0; j < len; j++) {
                    dot += Q.get(i, pivotRow + j) * u[j];
                }
                dot *= beta;
                for (int j = 0; j < len; j++) {
                    Q.set(i, pivotRow + j, Q.get(i, pivotRow + j) - dot * u[j]);
                }
            }
        }

        // Force the created zeros if this was a chasing step
        // We eliminated v[1]...v[len-1] at column colRef (which corresponds to H[pivotRow+1...lastRow, colRef])
        // pivotRow corresponds to the diagonal/subdiagonal element we kept.
        // So H[pivotRow+1, colRef] ... H[lastRow, colRef] should be 0.
        // Note: applyHouseholderStep takes colRef purely for the loop optimization above,
        // but explicit zeroing is good for numerical stability.

        // If colRef corresponds to the column *left* of pivotRow (standard chase), zero it.
        if (colRef == pivotRow - 1) {
            H.set(pivotRow, colRef, x0 - alpha); // The modified subdiagonal value
            for (int i = 1; i < len; i++) {
                H.set(pivotRow + i, colRef, 0.0);
            }
        }
    }
}