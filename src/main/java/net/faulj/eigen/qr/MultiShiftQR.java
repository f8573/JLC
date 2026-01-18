package net.faulj.eigen.qr;

import net.faulj.matrix.Matrix;

/**
 * Implements the Multi-Shift QR algorithm for computing the Real Schur Form
 * of an upper Hessenberg matrix.
 * <p>
 * This algorithm generalizes the Francis double-shift method by allowing an arbitrary
 * number of shifts (<i>m</i>) to be applied simultaneously.
 * </p>
 *
 * <h2>Algorithm Strategy:</h2>
 * <p>
 * Instead of applying shifts one by one or in pairs:
 * </p>
 * <ol>
 * <li><b>Shift Selection:</b> Select <i>m</i> simultaneous shifts (e.g., eigenvalues of a bottom <i>m×m</i> block).</li>
 * <li><b>Polynomial Evaluation:</b> Compute the first column of the operator p(H) = (H-μ₁I)...(H-μₘI).</li>
 * <li><b>Bulge Creation:</b> Introduce a "large" bulge of size <i>m+1</i> at the top of the matrix.</li>
 * <li><b>Bulge Chasing:</b> Chase the bulge down the matrix using Householder reflections to restore Hessenberg form.</li>
 * </ol>
 *
 * <h2>Benefits:</h2>
 * <ul>
 * <li><b>Cache Efficiency:</b> Large bulges confine update operations to localized blocks, allowing for BLAS-3 level performance.</li>
 * <li><b>Convergence:</b> Higher order polynomials can improve convergence rates in difficult cases.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see BulgeChasing
 */
public class MultiShiftQR {

    private static final double EPSILON = 1e-12;
    private static final int MAX_ITERATIONS_FACTOR = 100;

    /**
     * Computes the Real Schur form of the given upper Hessenberg matrix.
     * The matrix is modified in-place to become the Schur form.
     *
     * @param H An upper Hessenberg matrix.
     * @return The matrix H in Real Schur form (quasi-upper triangular).
     */
    public static Matrix decompose(Matrix H) {
        Matrix T = H.copy();
        int n = T.getRowCount();
        int maxIterations = MAX_ITERATIONS_FACTOR * n;
        int iterations = 0;

        // Active block range [n0, nn]
        int nn = n - 1;

        while (nn > 0) {
            if (iterations >= maxIterations) {
                throw new RuntimeException("MultiShiftQR did not converge after " + maxIterations + " iterations");
            }

            // 1. Deflation check: find the effective bottom of the active matrix
            if (isNegligible(T, nn, nn - 1)) {
                T.set(nn, nn - 1, 0.0);
                nn--;
                continue;
            }

            // 2. Find the top of the active unreduced Hessenberg block
            int n0 = nn - 1;
            while (n0 > 0) {
                if (isNegligible(T, n0, n0 - 1)) {
                    T.set(n0, n0 - 1, 0.0);
                    break;
                }
                n0--;
            }
            // Active block is now T[n0..nn, n0..nn]

            // 3. Select Shifts
            // For this implementation, we stick to the robust m=2 (Francis double-shift) logic
            // which is a specific instance of multi-shift.
            int m = 2;
            int blockSize = nn - n0 + 1;

            if (blockSize < m) {
                m = blockSize;
            }

            // Get shifts from the trailing m x m block
            double[] realShifts = new double[m];
            double[] imagShifts = new double[m];
            computeShifts(T, nn, m, realShifts, imagShifts);

            // 4. Bulge Chasing Sweep
            // Perform an implicit QR step using the computed shifts.
            bulgeChase(T, n0, nn, realShifts, imagShifts);

            iterations++;
        }

        return T;
    }

    private static boolean isNegligible(Matrix T, int row, int col) {
        double val = Math.abs(T.get(row, col));
        double neighborSum = Math.abs(T.get(row, row)) + Math.abs(T.get(col, col));
        if (neighborSum == 0) neighborSum = 1.0;
        return val <= EPSILON * neighborSum;
    }

    /**
     * Computes m shifts from the bottom-right m x m submatrix of the active block.
     * Stores results in the provided arrays.
     */
    private static void computeShifts(Matrix T, int nn, int m, double[] realShifts, double[] imagShifts) {
        if (m == 2) {
            double a = T.get(nn - 1, nn - 1);
            double b = T.get(nn - 1, nn);
            double c = T.get(nn, nn - 1);
            double d = T.get(nn, nn);

            double trace = a + d;
            double det = a * d - b * c;
            double discriminant = trace * trace - 4 * det;

            if (discriminant >= 0) {
                double sqrt = Math.sqrt(discriminant);
                realShifts[0] = (trace + sqrt) / 2.0;
                realShifts[1] = (trace - sqrt) / 2.0;
                imagShifts[0] = 0;
                imagShifts[1] = 0;
            } else {
                realShifts[0] = trace / 2.0;
                realShifts[1] = trace / 2.0;
                double sqrt = Math.sqrt(-discriminant);
                imagShifts[0] = sqrt / 2.0;
                imagShifts[1] = -sqrt / 2.0;
            }
        } else {
            for (int i = 0; i < m; i++) {
                realShifts[i] = T.get(nn, nn);
                imagShifts[i] = 0;
            }
        }
    }

    /**
     * Performs a single multi-shift implicit QR sweep on the active block T[n0..nn, n0..nn].
     */
    private static void bulgeChase(Matrix T, int n0, int nn, double[] realShifts, double[] imagShifts) {
        int m = realShifts.length;
        int n = T.getRowCount();

        // 1. Construct the first column of the shift polynomial p(H) applied to e1.
        double[] x = new double[m + 1];
        x[0] = 1.0;

        for (int k = 0; k < m; k++) {
            if (imagShifts[k] == 0) {
                applyRealShiftToVector(T, n0, x, realShifts[k], k);
            } else {
                double re = realShifts[k];
                double im = imagShifts[k];
                double s = 2 * re;
                double p = re * re + im * im;

                applyQuadraticShiftToVector(T, n0, x, s, p, k);
                k++;
            }
        }

        // 2. Introduce and Chase the bulge
        for (int i = n0; i < nn - 1; i++) {
            int bulgeSize = (i == n0) ? m : calculateBulgeSize(i, nn, m);
            double[] v = new double[bulgeSize + 1];

            if (i == n0) {
                System.arraycopy(x, 0, v, 0, Math.min(x.length, v.length));
            } else {
                for (int k = 0; k <= bulgeSize; k++) {
                    int row = i + 1 + k;
                    if (row < n) {
                        v[k] = T.get(row, i);
                    }
                }
            }

            double sigma = 0;
            for (int k = 1; k < v.length; k++) sigma += v[k] * v[k];

            if (sigma == 0 && i > n0) continue;

            double mu = Math.sqrt(v[0]*v[0] + sigma);
            if (v[0] <= 0) v[0] -= mu; else v[0] = -sigma / (v[0] + mu);

            double v0Sq = v[0] * v[0];
            double beta = 0;
            if (v0Sq + sigma > 1e-20) {
                beta = 2.0 * v0Sq / (v0Sq + sigma);
                double inv = 1.0 / v[0];
                for (int k = 0; k < v.length; k++) v[k] *= inv;
            } else {
                continue;
            }

            // Apply Householder P to T
            int startRow = (i == n0) ? n0 : i + 1;
            int blockSizeRef = v.length;

            // Apply from Left
            for (int c = 0; c < n; c++) {
                double sum = 0;
                for (int k = 0; k < blockSizeRef; k++) {
                    if (startRow + k < n)
                        sum += v[k] * T.get(startRow + k, c);
                }
                sum *= beta;
                for (int k = 0; k < blockSizeRef; k++) {
                    if (startRow + k < n)
                        T.set(startRow + k, c, T.get(startRow + k, c) - v[k] * sum);
                }
            }

            // Apply from Right
            for (int r = 0; r < n; r++) {
                double sum = 0;
                for (int k = 0; k < blockSizeRef; k++) {
                    if (startRow + k < n)
                        sum += v[k] * T.get(r, startRow + k);
                }
                sum *= beta;
                for (int k = 0; k < blockSizeRef; k++) {
                    if (startRow + k < n)
                        T.set(r, startRow + k, T.get(r, startRow + k) - v[k] * sum);
                }
            }
        }
    }

    private static void applyRealShiftToVector(Matrix T, int n0, double[] x, double shift, int currentHeight) {
        int n = T.getRowCount();
        double[] newX = new double[x.length];

        for (int i = 0; i <= currentHeight + 1; i++) {
            if (i >= x.length) break;
            double sum = 0;

            for (int j = 0; j <= currentHeight; j++) {
                if (j >= x.length) break;
                int r = n0 + i;
                int c = n0 + j;
                if (r < n && c < n) {
                    sum += T.get(r, c) * x[j];
                }
            }
            newX[i] = sum - shift * x[i];
        }
        System.arraycopy(newX, 0, x, 0, x.length);
    }

    private static void applyQuadraticShiftToVector(Matrix T, int n0, double[] x, double s, double p, int currentHeight) {
        // y = H*x
        double[] y = new double[x.length];
        multiplyHessenbergVector(T, n0, x, y, currentHeight);

        // z = H*y = H^2*x
        double[] z = new double[x.length];
        multiplyHessenbergVector(T, n0, y, z, currentHeight + 1);

        // combine: z - s*y + p*x
        for(int i=0; i<x.length; i++) {
            x[i] = z[i] - s*y[i] + p*x[i];
        }
    }

    private static void multiplyHessenbergVector(Matrix T, int n0, double[] input, double[] output, int inputHeight) {
        int n = T.getRowCount();
        for (int i = 0; i <= inputHeight + 1; i++) {
            if (i >= output.length) break;
            double sum = 0;
            for (int j = 0; j <= inputHeight; j++) {
                if (j >= input.length) break;
                int r = n0 + i;
                int c = n0 + j;
                if (r < n && c < n) {
                    sum += T.get(r, c) * input[j];
                }
            }
            output[i] = sum;
        }
    }

    private static int calculateBulgeSize(int i, int nn, int m) {
        return Math.min(m, nn - 1 - i);
    }
}