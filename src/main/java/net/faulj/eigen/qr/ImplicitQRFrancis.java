package net.faulj.eigen.qr;

import net.faulj.core.Tolerance;
import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.SchurResult;

public class ImplicitQRFrancis {

    private static final int MAX_ITERATIONS_PER_EIGENVALUE = 30;

    /**
     * Performs the Implicit QR algorithm with Francis double-shift on a Hessenberg matrix.
     *
     * @param H The upper Hessenberg matrix (will be modified to T).
     * @param U The transformation matrix (will be updated U = U * Z).
     * @return SchurResult containing final T, U, and eigenvalues.
     */
    public static SchurResult process(Matrix H, Matrix U) {
        int n = H.getRowCount();
        Matrix T = H.copy(); // Work on a copy if you want to preserve H, or assume H is owned.
        Matrix Z = (U != null) ? U.copy() : Matrix.Identity(n);

        int n_minus_1 = n - 1;
        int m = n_minus_1; // m points to the last row/col of the active submatrix
        int iter = 0;

        double[] realEigenvalues = new double[n];
        double[] imagEigenvalues = new double[n];

        // Main Loop: Reduce the matrix size m as blocks converge
        while (m >= 0) {
            // 1. Look for a single small subdiagonal element to split the matrix (Deflation)
            int l = 0;
            boolean converged = false;

            // Find the active submatrix range [l, m]
            for (int i = m; i > 0; i--) {
                // Deflation criterion: |h_{i, i-1}| <= epsilon * (|h_{i-1, i-1}| + |h_{i, i}|)
                double h_i_i_minus_1 = Math.abs(T.get(i, i - 1));
                double diagSum = Math.abs(T.get(i - 1, i - 1)) + Math.abs(T.get(i, i));

                // Safety for zero diagonal
                if (diagSum == 0) diagSum = 1.0;

                if (h_i_i_minus_1 <= Tolerance.get() * diagSum) {
                    T.set(i, i - 1, 0.0); // Clean deflation
                    l = i;
                    // Check if we found a split at the very bottom
                    if (l == m) {
                        // 1x1 block converged
                        realEigenvalues[m] = T.get(m, m);
                        imagEigenvalues[m] = 0.0;
                        m--;
                        iter = 0;
                        converged = true;
                    } else if (l == m - 1) {
                        // 2x2 block converged
                        compute2x2Eigenvalues(T, m - 1, m, realEigenvalues, imagEigenvalues);
                        m -= 2;
                        iter = 0;
                        converged = true;
                    }
                    break;
                }

                // If we go all the way up to 0, the active block is 0..m
                if (i == 1) l = 0;
            }

            if (converged) continue;

            // Check for excessive iterations
            if (iter > MAX_ITERATIONS_PER_EIGENVALUE * (m - l + 1)) {
                throw new ArithmeticException("Implicit QR failed to converge after too many iterations.");
                // Alternatively: use a fallback shift or random shift here.
            }
            iter++;

            // 2. Francis Double Shift Step on active submatrix T[l..m, l..m]
            double h_mm = T.get(m, m);
            double h_mm1 = T.get(m, m - 1); // h_{m, m-1}
            double h_m1m = T.get(m - 1, m); // h_{m-1, m}
            double h_m1m1 = T.get(m - 1, m - 1); // h_{m-1, m-1}

            // Wilkinson shift invariants (trace and determinant of the bottom 2x2)
            double s = h_m1m1 + h_mm;
            double t = h_m1m1 * h_mm - h_mm1 * h_m1m;

            // 3. Bulge Introduction (Double shift: x^2 - sx + t)
            // We compute the first column of (H^2 - sH + tI) e_1
            // Only need the first 3 elements (because H is Hessenberg)
            double h_ll = T.get(l, l);
            double h_l1l = T.get(l + 1, l);
            double h_l1l1 = T.get(l + 1, l + 1);

            // x1 = h_ll^2 + h_l1l * h_ll1 - s * h_ll + t
            // Note: T.get(l, l+1) is h_ll1
            double x = h_ll * h_ll + T.get(l, l + 1) * h_l1l - s * h_ll + t;
            double y = h_l1l * (h_ll + h_l1l1 - s);
            double z = h_l1l * T.get(l + 2, l + 1);

            // 4. Bulge Chasing
            for (int k = l; k <= m - 2; k++) {
                // Compute Householder reflector P to annihilate y and z (using x as pivot)
                // v = [x, y, z]^T -> P maps v to [alpha, 0, 0]^T

                // Simple Householder construction
                double norm = Math.sqrt(x * x + y * y + z * z);
                if (norm == 0) break; // Should not happen if Hessenberg is unreduced

                // Sign choice to avoid cancellation
                double alpha = (x > 0) ? -norm : norm;
                double div = 1.0 / (x - alpha);

                // Householder vector v (normalized implicitly during application usually,
                // but let's be explicit for clarity or use a helper)
                double v1 = x - alpha;
                double v2 = y;
                double v3 = z;

                // Normalization factor beta = 2 / (v^T v)
                double vTv = v1*v1 + v2*v2 + v3*v3;
                double beta = 2.0 / vTv;

                // Apply P to T from left (rows k..k+2)
                // P = I - beta * v * v^T
                // T = T - beta * v * (v^T * T)
                int maxCol = n; // Apply to all columns to right
                for (int col = k; col < maxCol; col++) {
                    double dot = v1 * T.get(k, col) + v2 * T.get(k + 1, col) + v3 * T.get(k + 2, col);
                    dot *= beta;
                    T.set(k, col, T.get(k, col) - dot * v1);
                    T.set(k + 1, col, T.get(k + 1, col) - dot * v2);
                    T.set(k + 2, col, T.get(k + 2, col) - dot * v3);
                }

                // Apply P to T from right (cols k..min(k+3, n))
                // Note: Hessenberg structure limits non-zeros, but we act on full block typically in standard formulation
                // or just relevant columns. For full similarity, apply to all rows.
                int minRow = 0;
                int maxRow = Math.min(k + 4, n); // Optimization: only need to go down slightly below diagonal?
                // Actually for T to maintain form, we apply to rows 0..n

                for (int row = 0; row < n; row++) {
                    double dot = v1 * T.get(row, k) + v2 * T.get(row, k + 1) + v3 * T.get(row, k + 2);
                    dot *= beta;
                    T.set(row, k, T.get(row, k) - dot * v1);
                    T.set(row, k + 1, T.get(row, k + 1) - dot * v2);
                    T.set(row, k + 2, T.get(row, k + 2) - dot * v3);
                }

                // Accumulate Z: Z = Z * P
                // Apply P to Z from right
                for (int row = 0; row < n; row++) {
                    double dot = v1 * Z.get(row, k) + v2 * Z.get(row, k + 1) + v3 * Z.get(row, k + 2);
                    dot *= beta;
                    Z.set(row, k, Z.get(row, k) - dot * v1);
                    Z.set(row, k + 1, Z.get(row, k + 1) - dot * v2);
                    Z.set(row, k + 2, Z.get(row, k + 2) - dot * v3);
                }

                // Setup x, y, z for next step (chasing)
                // Next pivot vector is column k of transformed T, specifically elements k+1, k+2, k+3
                x = T.get(k + 1, k);
                y = T.get(k + 2, k);
                if (k < m - 2) {
                    z = T.get(k + 3, k);
                } else {
                    z = 0.0;
                }
            }

            // Handle the last step (size 2 Householder) for the bottom corner if needed?
            // The loop goes to m-2, handling the 3x3 block ending at m.
            // Standard Francis implementation usually handles the "falling off the edge"
            // naturally or with a specific 2x2 handler at the end.

            // Re-clean subdiagonal zeros created by chase (optional but good for stability)
            if (l > 0) T.set(l, l-1, 0.0);
            if (m < n-1) T.set(m+1, m, 0.0);
        }

        return new SchurResult(T, Z, realEigenvalues, imagEigenvalues);
    }

    private static void compute2x2Eigenvalues(Matrix T, int i, int j, double[] real, double[] imag) {
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