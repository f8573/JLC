package net.faulj.eigen.qr;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils; // Assuming existence based on Matrix imports

import java.util.ArrayList;
import java.util.List;

/**
 * Implements the Multi-Shift QR algorithm for computing the Real Schur Form
 * of an upper Hessenberg matrix.
 * * This implementation employs an implicit bulge-chasing strategy.
 * For m shifts, a bulge of size roughly (m+1) is created and chased down the matrix.
 * This allows for better cache efficiency (BLAS-3) compared to single shifts
 * and faster convergence than simple QR.
 */
public class MultiShiftQR {

    private static final double EPSILON = 1e-12;
    private static final int MAX_ITERATIONS_FACTOR = 100;

    /**
     * Computes the Real Schur form of the given upper Hessenberg matrix.
     * The matrix is modified in-place to become the Schur form.
     * * @param H An upper Hessenberg matrix.
     * @return The matrix H in Real Schur form (quasi-upper triangular).
     */
    public static Matrix decompose(Matrix H) {
        // Work on a copy if you don't want to modify the original,
        // but typically decomposition updates the matrix.
        // Based on void return types in other decompositions, we might update in place,
        // but here we return the result. Let's work on a copy.
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
            // Check subdiagonal entry T[nn, nn-1]
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
            // We typically use m=2 (Francis) for small blocks, or larger even m for large blocks.
            // For this implementation, we stick to the robust m=2 (Francis double-shift) logic
            // which is a specific instance of multi-shift. To extend to m > 2,
            // one would simply extract more eigenvalues from a larger trailing block.
            int m = 2;
            int blockSize = nn - n0 + 1;

            // If block is very small, just solving it or using simple shifts is sufficient.
            if (blockSize < m) {
                // Should not happen if m=2 and we are in the loop (nn > 0 implies size >= 2)
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
        if (neighborSum == 0) neighborSum = 1.0; // Avoid division by zero/nonsense check
        return val <= EPSILON * neighborSum;
    }

    /**
     * Computes m shifts from the bottom-right m x m submatrix of the active block.
     * Stores results in the provided arrays.
     */
    private static void computeShifts(Matrix T, int nn, int m, double[] realShifts, double[] imagShifts) {
        // For m=2, this is the eigenvalues of the bottom-right 2x2 block.
        // T[nn-1, nn-1]  T[nn-1, nn]
        // T[nn,   nn-1]  T[nn,   nn]

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
            // Placeholder for generic m > 2:
            // Would extract the mxm block and recursively call a simplified QR
            // or an eigenvalue solver to get approximate eigenvalues.
            // Fallback to Wilkinson shift (bottom element) for all shifts if not implemented.
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
        // We only need the first m+1 rows because H is Hessenberg.
        // Vector v represents the direction to map to alpha*e1.

        // Start with e1 scaled. We simulate the multiplication (H - sI).
        // Since we are looking at the submatrix starting at n0, we work in local coordinates 0..m

        // We need a temporary buffer for the vector "x" which will determine the first reflector.
        // The degree of the polynomial is m. The vector will have non-zeros in first m+1 positions.
        double[] x = new double[m + 1];
        x[0] = 1.0;
        // implicit e1, others are 0.

        // Apply shifts (H - s_k I) * x
        // We iterate through shifts. Complex shifts come in conjugate pairs,
        // so we can apply them as real quadratic blocks to keep arithmetic real.
        // (H^2 - 2Re(s)H + |s|^2 I)

        for (int k = 0; k < m; k++) {
            // Determine if this is a single real shift or start of a conjugate pair
            if (imagShifts[k] == 0) {
                // Real shift: x <- (H - s I) * x
                applyRealShiftToVector(T, n0, x, realShifts[k], k);
            } else {
                // Complex conjugate pair.
                // Apply (H - s)(H - conj(s)) = H^2 - 2*real*H + (real^2+imag^2)I
                // This consumes two shifts (k and k+1).
                double re = realShifts[k];
                double im = imagShifts[k];
                double s = 2 * re;
                double p = re * re + im * im;

                applyQuadraticShiftToVector(T, n0, x, s, p, k);
                k++; // Skip next as it is the conjugate
            }
        }

        // 2. Introduce the bulge
        // Compute Householder reflector P0 to zero out x[1..m]
        // Apply P0 to T from left (rows n0..n0+m) and right (cols n0..n0+m + bandwidth)

        // We chase the bulge from row n0 down to nn - 1.
        // The loop index i is the column index we are clearing.

        for (int i = n0; i < nn - 1; i++) {
            // For the first step (i == n0), the vector to annihilate is 'x' computed above.
            // For subsequent steps, the vector comes from the current column i of T.

            int bulgeSize = (i == n0) ? m : calculateBulgeSize(i, nn, m);
            // The vector v to construct reflector from.
            double[] v = new double[bulgeSize + 1];

            if (i == n0) {
                // Copy from x, ensuring we respect bounds (though x is sized m+1)
                System.arraycopy(x, 0, v, 0, Math.min(x.length, v.length));
            } else {
                // Extract T[i+1..i+bulgeSize, i]
                // The pivot is at T[i+1, i] because we are restoring Hessenberg at col i.
                for (int k = 0; k <= bulgeSize; k++) {
                    int row = i + 1 + k;
                    if (row < n) {
                        v[k] = T.get(row, i);
                    }
                }
            }

            // Compute Householder vector u and beta
            // P = I - beta * u * u'
            // u is stored in v for convenience
            double sigma = 0;
            for (int k = 1; k < v.length; k++) sigma += v[k] * v[k];

            if (sigma == 0 && i > n0) continue; // Already zero (except first step where x could be anything)

            double mu = Math.sqrt(v[0]*v[0] + sigma);
            if (v[0] <= 0) v[0] -= mu; else v[0] = -sigma / (v[0] + mu);

            // Normalize u (v)
            double v0Sq = v[0] * v[0];
            double beta = 0;
            if (v0Sq + sigma > 1e-20) {
                beta = 2.0 * v0Sq / (v0Sq + sigma);
                // divide v by v[0] so that v[0] = 1 (standard convention)
                double inv = 1.0 / v[0];
                for (int k = 0; k < v.length; k++) v[k] *= inv;
            } else {
                continue; // Identity
            }

            // Apply Householder P to T
            // P affects rows/cols from startRow to startRow + bulgeSize
            // where startRow = (i == n0) ? n0 : i + 1;
            int startRow = (i == n0) ? n0 : i + 1;
            int blockSizeRef = v.length;

            // Apply from Left: T[start..start+sz, :] -= beta * u * (u' * submatrix)
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

            // Apply from Right: T[:, start..start+sz] -= beta * (submatrix * u) * u'
            // Need to be careful: transformations are symmetric only if symmetric matrix.
            // Here Q'HQ.
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

            // After the first step, we check if the bulge has fallen off the matrix
            // Optimization: restrict column/row loops to relevant bandwidth
        }
    }

    // Helper: x <- (H - alpha*I) * x
    // x is effectively in indices n0..n0+current_height
    // Since H is Hessenberg, multiplying by H shifts data down by 1.
    private static void applyRealShiftToVector(Matrix T, int n0, double[] x, double shift, int currentHeight) {
        int n = T.getRowCount();
        // New x will have height + 1 non-zeros
        double[] newX = new double[x.length];

        // Perform matrix-vector multiply limited to the non-zero structure
        // y = T[n0:..., n0:...] * x

        for (int i = 0; i <= currentHeight + 1; i++) {
            if (i >= x.length) break;
            double sum = 0;
            // row n0 + i
            // dot product with x (which corresponds to cols n0..n0+currentHeight)
            // T is Hessenberg: row R has non-zeros up to col R+1.
            // So col C has non-zeros starting at row C-1.

            for (int j = 0; j <= currentHeight; j++) {
                if (j >= x.length) break;
                // Get T element
                int r = n0 + i;
                int c = n0 + j;
                if (r < n && c < n) {
                    sum += T.get(r, c) * x[j];
                }
            }
            newX[i] = sum - shift * x[i]; // (H - sI)x
        }
        System.arraycopy(newX, 0, x, 0, x.length);
    }

    private static void applyQuadraticShiftToVector(Matrix T, int n0, double[] x, double s, double p, int currentHeight) {
        // x <- (H^2 - sH + pI)x
        // This effectively performs two real shift steps in one logic, or explicitly computes the quadratic.
        // For simplicity, we can just chain two "applyRealShift" but that requires complex arithmetic logic?
        // No, we use the property: (H - mu)(H - conj(mu)) = H^2 - sH + pI.
        // This is purely real.
        // Doing it directly avoids complex numbers.

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
        // How much space is left?
        // We are at col i. The bulge usually extends m rows below diagonal.
        // But we must stop at nn.
        return Math.min(m, nn - 1 - i);
    }
}