package net.faulj.eigen.qr;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.eigen.schur.SchurEigenExtractor;
import net.faulj.scalar.Complex;

import java.util.List;

/**
 * Implements the Francis Implicit QR Algorithm with adaptive size-based strategy.
 * <p>
 * This class provides an intelligent approach to computing the Real Schur Decomposition
 * that adapts to matrix size, avoiding stack overflow for small matrices (5x5, etc.)
 * while using advanced techniques for larger matrices.
 * </p>
 *
 * <h2>Size-Based Strategy:</h2>
 * <ul>
 * <li><b>n &lt; 10:</b> Falls back to ExplicitQRIteration (simpler, no recursion).</li>
 * <li><b>10 &le; n &lt; 30:</b> Uses simple double-shift Francis.</li>
 * <li><b>n &ge; 30:</b> Uses multi-shift with aggressive deflation.</li>
 * </ul>
 *
 * <h2>Algorithm:</h2>
 * <ol>
 * <li>Reduce to Hessenberg form using {@link BlockedHessenbergQR}.</li>
 * <li>Apply size-appropriate QR iteration strategy.</li>
 * <li>Deflate as subdiagonal elements become negligible.</li>
 * </ol>
 *
 * @author JLC Development Team
 * @version 1.0
 * @see ExplicitQRIteration
 * @see BulgeChasing
 */
public class ImplicitQRFrancis {

    private static final double EPSILON = 1e-12;
    private static final int MAX_ITERATIONS_PER_EIGENVALUE = 30;
    private static final int SMALL_MATRIX_THRESHOLD = 10;
    private static final int MEDIUM_MATRIX_THRESHOLD = 30;

    /**
     * Computes the Real Schur Decomposition of the given matrix.
     *
     * @param A The square matrix to decompose.
     * @return The SchurResult containing T (Schur form), U (Vectors), and eigenvalues.
     */
    public static SchurResult decompose(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Matrix must be square");
        }

        int n = A.getRowCount();

        // For very small matrices, use explicit QR (no fancy shifts, no recursion)
        if (n < SMALL_MATRIX_THRESHOLD) {
            Matrix[] result = ExplicitQRIteration.decompose(A);
            List<Complex> eigenvalues = ExplicitQRIteration.getEigenvalues(A);
            return new SchurResult(result[0], result[1], eigenvalues.toArray(new Complex[0]));
        }

        // Step 1: Reduce to Hessenberg
        HessenbergResult hessResult = BlockedHessenbergQR.decompose(A);
        Matrix H = hessResult.getH();
        Matrix U = hessResult.getQ();

        // Step 2: Main iteration loop with adaptive strategy
        int m = n - 1; // Active block end
        int totalIter = 0;

        while (m > 0 && totalIter < MAX_ITERATIONS_PER_EIGENVALUE * n) {
            // Find active block by checking for deflation
            int l = findActiveBlockStart(H, m);
            int blockSize = m - l + 1;

            if (blockSize == 1) {
                // Single eigenvalue converged
                m--;
                totalIter = 0;
                continue;
            }

            if (blockSize == 2) {
                // Check if 2x2 block is converged (complex conjugate pair)
                if (is2x2BlockConverged(H, l, m)) {
                    m -= 2;
                    totalIter = 0;
                    continue;
                }
            }

            // Perform one QR step on active block
            if (blockSize < MEDIUM_MATRIX_THRESHOLD) {
                // Small to medium: use simple double-shift Francis
                performDoubleShiftStep(H, U, l, m);
            } else {
                // Large: use multi-shift strategy with optional deflation
                int numShifts = MultiShiftQR.computeOptimalShiftCount(blockSize);
                double[] shifts = MultiShiftQR.generateShifts(H, l, m, numShifts);
                BulgeChasing.performSweep(H, U, l, m, shifts);
            }

            totalIter++;
        }

        // Extract eigenvalues
        SchurEigenExtractor extractor = new SchurEigenExtractor(H);
        return new SchurResult(H, U, extractor.getEigenvalues());
    }

    /**
     * Finds the start of the active block by scanning for negligible subdiagonal entries.
     */
    private static int findActiveBlockStart(Matrix H, int m) {
        int l = m;
        while (l > 0) {
            double subdiag = Math.abs(H.get(l, l - 1));
            double diagSum = Math.abs(H.get(l - 1, l - 1)) + Math.abs(H.get(l, l));
            if (subdiag <= EPSILON * (diagSum + EPSILON)) {
                H.set(l, l - 1, 0.0);
                break;
            }
            l--;
        }
        return l;
    }

    /**
     * Checks if a 2x2 block represents a converged complex conjugate pair.
     */
    private static boolean is2x2BlockConverged(Matrix H, int i, int j) {
        double subdiag = Math.abs(H.get(j, j - 1));
        if (subdiag < EPSILON) {
            return true; // Already split
        }
        
        // Check if eigenvalues are complex (then it's a converged 2x2 Schur block)
        double a = H.get(i, i);
        double b = H.get(i, i + 1);
        double c = H.get(i + 1, i);
        double d = H.get(i + 1, i + 1);
        double disc = (a + d) * (a + d) - 4 * (a * d - b * c);
        
        return disc < 0; // Complex eigenvalues => converged 2x2 block
    }

    /**
     * Performs a simple double-shift Francis QR step (2 shifts).
     */
    private static void performDoubleShiftStep(Matrix H, Matrix U, int l, int m) {
        // Compute Wilkinson shift from bottom 2x2 block
        double a = H.get(m - 1, m - 1);
        double b = H.get(m - 1, m);
        double c = H.get(m, m - 1);
        double d = H.get(m, m);

        double tr = a + d;
        double det = a * d - b * c;
        double disc = tr * tr - 4 * det;

        double shift1, shift2;
        if (disc >= 0) {
            double sqrt = Math.sqrt(disc);
            shift1 = (tr + sqrt) / 2;
            shift2 = (tr - sqrt) / 2;
        } else {
            // Complex conjugate pair: use real and imaginary parts
            shift1 = tr / 2.0;
            shift2 = tr / 2.0; // Simplified: just use trace
        }

        double[] shifts = {shift1, shift2};
        BulgeChasing.performSweep(H, U, l, m, shifts);
    }
}