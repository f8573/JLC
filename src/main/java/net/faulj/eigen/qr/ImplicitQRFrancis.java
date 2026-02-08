package net.faulj.eigen.qr;

import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.schur.SchurEigenExtractor;
import net.faulj.matrix.Matrix;

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
 * <h2>EJML-inspired optimizations:</h2>
 * <ul>
 * <li>Raw array access for performance</li>
 * <li>Exceptional shift handling when convergence stalls</li>
 * <li>Relative tolerance deflation detection</li>
 * <li>2x2 block scaling for overflow protection</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @see ExplicitQRIteration
 * @see BulgeChasing
 */
public class ImplicitQRFrancis {

    private static final double EPSILON = 1e-12;
    private static final double MACHINE_EPS = 2.220446049250313e-16;
    private static final int MAX_ITERATIONS_PER_EIGENVALUE = 30;
    private static final int SMALL_MATRIX_THRESHOLD = 9;
    private static final int MEDIUM_MATRIX_THRESHOLD = 30;
    private static final int LARGE_MATRIX_THRESHOLD = 60; // Lower threshold for faster AED
    // LAPACK dlaqr0: exceptional shift every 6 iterations
    private static final int EXCEPTIONAL_THRESHOLD = 5; // Faster exceptional shifts
    private static final int AED_FREQUENCY = 6; // More frequent AED for 512x512

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

        // Use the robust Hessenberg -> implicit Francis QR path for all sizes.
        // Falling back to the explicit QR iteration for small matrices produced
        // less accurate results (notably for ill-conditioned or nearly singular
        // matrices). Always reduce to Hessenberg and run the implicit algorithm
        // which includes more careful deflation and shift strategies.
        HessenbergResult hessResult = BlockedHessenbergQR.decompose(A);
        return decomposeFromHessenbergInternal(hessResult, A, true);
    }

    /**
     * Computes the Real Schur Decomposition from a pre-computed Hessenberg form.
     * This is useful when you want to benchmark just the QR iteration phase,
     * or when you've already computed the Hessenberg form for other purposes.
     *
     * @param hessResult The pre-computed Hessenberg result (H, Q).
     * @return The SchurResult containing T (Schur form), U (Vectors), and eigenvalues.
     */
    public static SchurResult decomposeFromHessenberg(HessenbergResult hessResult) {
        return decomposeFromHessenbergInternal(hessResult, hessResult.getOriginal(), true);
    }

    /**
     * Computes just the Schur form T without accumulating the orthogonal matrix U.
     * This is faster when only eigenvalues are needed.
     *
     * @param hessResult The pre-computed Hessenberg result.
     * @return The SchurResult with T and eigenvalues (U will be identity).
     */
    public static SchurResult decomposeSchurFormOnly(HessenbergResult hessResult) {
        return decomposeFromHessenbergInternal(hessResult, hessResult.getOriginal(), false);
    }

    /**
     * Internal implementation of QR iteration on Hessenberg matrix.
     *
     * @param hessResult The Hessenberg decomposition result.
     * @param originalA The original matrix.
     * @param accumulateU Whether to accumulate the orthogonal matrix U.
     */
    private static SchurResult decomposeFromHessenbergInternal(HessenbergResult hessResult, Matrix originalA, boolean accumulateU) {
        Matrix H = hessResult.getH();
        Matrix U = accumulateU ? hessResult.getQ() : Matrix.Identity(H.getRowCount());
        int n = H.getRowCount();

        // Get raw arrays for faster deflation checks
        double[] h = H.getRawData();

        // Step 2: Main iteration loop with adaptive strategy
        int m = n - 1; // Active block end
        int globalIter = 0;
        int exceptionalCount = 0;
        int aedCounter = 0;
        int maxIter = MAX_ITERATIONS_PER_EIGENVALUE * n;
        boolean usesAED = (n >= LARGE_MATRIX_THRESHOLD);
        int totalIterations = 0; // Diagnostic

        while (m > 0 && globalIter < maxIter) {
            totalIterations++;
            // LAPACK strategy: apply AED more aggressively for large matrices
            if (usesAED && aedCounter >= AED_FREQUENCY && m >= MEDIUM_MATRIX_THRESHOLD) {
                int ns = MultiShiftQR.computeOptimalShiftCount(m + 1);
                int winSize = Math.min(Math.max(ns + 1, (m + 1) / 3), m + 1);
                if (winSize >= 4) {
                    AggressiveEarlyDeflation.AEDResult aedResult =
                        AggressiveEarlyDeflation.process(H, accumulateU ? U : null, 0, m, winSize, EPSILON, globalIter);
                    if (aedResult.deflatedCount > 0) {
                        m = aedResult.newActiveEnd;
                        globalIter = 0;
                        exceptionalCount = 0;
                    }
                }
                aedCounter = 0;
            }

            // LAPACK-style deflation check: more aggressive threshold
            while (m > 0) {
                double subdiag = Math.abs(h[m * n + m - 1]);
                double diagAbove = Math.abs(h[(m - 1) * n + m - 1]);
                double diagBelow = Math.abs(h[m * n + m]);
                // LAPACK: uses ulp * (|H[i-1,i-1]| + |H[i,i]|) where ulp ≈ eps
                double threshold = MACHINE_EPS * (diagAbove + diagBelow);

                if (subdiag <= Math.max(threshold, EPSILON)) {
                    h[m * n + m - 1] = 0.0;
                    m--;
                    globalIter = 0;
                    exceptionalCount = 0;
                } else {
                    break;
                }
            }

            if (m <= 0) break;

            // Find active block by checking for deflation
            int l = findActiveBlockStartRaw(h, n, m);
            int blockSize = m - l + 1;

            if (blockSize == 1) {
                m--;
                globalIter = 0;
                exceptionalCount = 0;
                continue;
            }

            if (blockSize == 2) {
                if (is2x2BlockConvergedRaw(h, n, l, m)) {
                    m -= 2;
                    globalIter = 0;
                    exceptionalCount = 0;
                    continue;
                }
                // 2x2 block with real eigenvalues: explicitly split via Schur rotation.
                // The implicit QR step becomes a no-op when shifts equal the eigenvalues
                // (Cayley-Hamilton), so we must directly diagonalize the 2x2 block.
                if (split2x2BlockIfReal(h, n, l, m, accumulateU ? U.getRawData() : null,
                        accumulateU ? U.getRowCount() : 0)) {
                    m -= 2;
                    globalIter = 0;
                    exceptionalCount = 0;
                    continue;
                }
            }

            // Check for exceptional shift need
            boolean useExceptionalShift = (exceptionalCount >= EXCEPTIONAL_THRESHOLD);

            // Compute shifts
            int numShifts = (blockSize < MEDIUM_MATRIX_THRESHOLD) ? 2 : MultiShiftQR.computeOptimalShiftCount(blockSize);
            double[] shifts;
            if (useExceptionalShift) {
                shifts = generateExceptionalShiftsRaw(h, n, l, m, numShifts);
                exceptionalCount = 0;
            } else if (numShifts == 2) {
                shifts = computeDoubleShiftRaw(h, n, l, m);
            } else {
                shifts = MultiShiftQR.generateShiftsRaw(h, n, l, m, numShifts);
            }

            // Perform bulge chasing sweep using Matrix-based method
            BulgeChasing.performSweep(H, accumulateU ? U : null, l, m, shifts);

            globalIter++;
            exceptionalCount++;
            aedCounter++;
        }

        // Final cleanup
        cleanupSchurFormRaw(h, n);

        // Removed debug output for speed

        // Extract eigenvalues
        SchurEigenExtractor extractor = new SchurEigenExtractor(H, U);
        return new SchurResult(originalA, H, U, extractor.getEigenvalues(), extractor.getEigenvectors());
    }

    /**
     * Compute double-shift values from bottom 2x2 block using raw arrays.
     * Uses Wilkinson shift for better convergence (LAPACK approach).
     */
    private static double[] computeDoubleShiftRaw(double[] H, int n, int l, int m) {
        double a11 = H[(m - 1) * n + m - 1];
        double a12 = H[(m - 1) * n + m];
        double a21 = H[m * n + m - 1];
        double a22 = H[m * n + m];

        // LAPACK: scale for numerical stability
        double s = Math.abs(a11) + Math.abs(a12) + Math.abs(a21) + Math.abs(a22);
        if (s == 0.0) {
            return new double[]{0.0, 0.0};
        }

        a11 /= s; a12 /= s; a21 /= s; a22 /= s;
        double tr = a11 + a22;
        double det = a11 * a22 - a12 * a21;
        double disc = tr * tr * 0.25 - det;

        double rtdisc = Math.sqrt(Math.abs(disc));
        double shift1 = s * (tr * 0.5 + rtdisc);
        double shift2 = s * (tr * 0.5 - rtdisc);

        return new double[]{shift1, shift2};
    }

    /**
     * Final cleanup of Schur form - zero small elements.
     * Uses LAPACK-style threshold based on matrix norm.
     */
    private static void cleanupSchurFormRaw(double[] h, int n) {
        // Compute Frobenius norm of H for a robust threshold
        double normH = 0.0;
        for (int i = 0; i < n * n; i++) {
            normH += h[i] * h[i];
        }
        normH = Math.sqrt(normH);
        double thresh = MACHINE_EPS * normH;

        // Zero elements well below the first subdiagonal
        for (int i = 2; i < n; i++) {
            int base = i * n;
            for (int j = 0; j < i - 1; j++) {
                h[base + j] = 0.0;
            }
        }

        // Clean up subdiagonal elements that should be zero
        // Use both local (LAPACK) and global (norm-based) thresholds
        for (int i = 1; i < n; i++) {
            double subdiag = Math.abs(h[i * n + i - 1]);
            double diagSum = Math.abs(h[(i - 1) * n + i - 1]) + Math.abs(h[i * n + i]);
            double localThresh = MACHINE_EPS * (diagSum + EPSILON);
            if (subdiag <= Math.max(localThresh, thresh)) {
                h[i * n + i - 1] = 0.0;
            }
        }
    }

    /**
     * Finds the start of the active block using raw array access.
     * LAPACK dlahqr approach: conservative but accurate deflation criterion.
     */
    private static int findActiveBlockStartRaw(double[] H, int n, int m) {
        int l = m;
        // LAPACK: scan from bottom up, look for negligible subdiagonal
        while (l > 0) {
            double h_ll1 = H[l * n + l - 1];
            if (h_ll1 == 0.0) break;

            double tst = Math.abs(H[(l - 1) * n + l - 1]) + Math.abs(H[l * n + l]);
            if (tst == 0.0) {
                // Check neighbors if diagonal sum is zero
                if (l - 2 >= 0) tst += Math.abs(H[(l - 1) * n + l - 2]);
                if (l + 1 < n) tst += Math.abs(H[(l + 1) * n + l]);
            }
            // LAPACK criterion: |H[l,l-1]| <= ulp * tst
            if (Math.abs(h_ll1) <= MACHINE_EPS * tst) {
                H[l * n + l - 1] = 0.0;
                break;
            }
            l--;
        }
        return l;
    }

    /**
     * Checks if a 2x2 block represents a converged complex conjugate pair using raw array access.
     */
    private static boolean is2x2BlockConvergedRaw(double[] H, int n, int i, int j) {
        double subdiag = Math.abs(H[j * n + j - 1]);
        if (subdiag < EPSILON) {
            return true; // Already split
        }

        // EJML: scale 2x2 block for numerical stability
        double a = H[i * n + i];
        double b = H[i * n + i + 1];
        double c = H[(i + 1) * n + i];
        double d = H[(i + 1) * n + i + 1];

        // Scale by max element for overflow protection
        double maxElem = Math.max(Math.max(Math.abs(a), Math.abs(b)),
                                   Math.max(Math.abs(c), Math.abs(d)));
        if (maxElem > 0) {
            a /= maxElem;
            b /= maxElem;
            c /= maxElem;
            d /= maxElem;
        }

        double disc = (a + d) * (a + d) - 4 * (a * d - b * c);

        return disc < 0; // Complex eigenvalues => converged 2x2 block
    }

    /**
     * Explicitly split a 2x2 block with real eigenvalues via a Schur rotation.
     * Computes a Givens rotation G such that G^T * H[l:m,l:m] * G is upper triangular,
     * then applies it as a similarity transformation to the full matrix.
     *
     * @return true if the block was successfully split (real eigenvalues), false if complex
     */
    private static boolean split2x2BlockIfReal(double[] h, int n, int l, int m,
                                                double[] q, int qn) {
        double a = h[l * n + l];
        double b = h[l * n + m];
        double c = h[m * n + l];
        double d = h[m * n + m];

        double tr = a + d;
        double det = a * d - b * c;
        double disc = tr * tr - 4 * det;

        if (disc < 0) return false; // Complex eigenvalues, don't split

        // Real eigenvalues: compute them
        double sqrtDisc = Math.sqrt(disc);
        double lambda1 = (tr + sqrtDisc) / 2.0;
        double lambda2 = (tr - sqrtDisc) / 2.0;

        // Compute Givens rotation to zero out c (subdiagonal)
        // We want G^T * [a-lambda; c] = [*; 0], choosing the eigenvalue closest to d
        // for better numerical behavior (Wilkinson shift strategy)
        double target = (Math.abs(d - lambda1) < Math.abs(d - lambda2)) ? lambda1 : lambda2;
        double x = a - target;
        double y = c;
        double r = Math.hypot(x, y);
        if (r < EPSILON) {
            // Subdiagonal is already negligible
            h[m * n + l] = 0.0;
            return true;
        }

        double cs = x / r;
        double sn = -y / r;

        // Apply Givens rotation: G^T * H * G (similarity transformation)
        // Left: G^T * H (affects rows l and m)
        for (int j = 0; j < n; j++) {
            double h1 = h[l * n + j];
            double h2 = h[m * n + j];
            h[l * n + j] = cs * h1 - sn * h2;
            h[m * n + j] = sn * h1 + cs * h2;
        }
        // Right: H * G (affects columns l and m)
        for (int i = 0; i < n; i++) {
            double h1 = h[i * n + l];
            double h2 = h[i * n + m];
            h[i * n + l] = cs * h1 - sn * h2;
            h[i * n + m] = sn * h1 + cs * h2;
        }
        // Accumulate Q
        if (q != null) {
            for (int i = 0; i < qn; i++) {
                double q1 = q[i * qn + l];
                double q2 = q[i * qn + m];
                q[i * qn + l] = cs * q1 - sn * q2;
                q[i * qn + m] = sn * q1 + cs * q2;
            }
        }

        // Ensure exact zero on subdiagonal
        h[m * n + l] = 0.0;
        return true;
    }

    /**
     * Generate exceptional shifts for multi-shift QR using raw array access.
     */
    private static double[] generateExceptionalShiftsRaw(double[] H, int n, int l, int m, int numShifts) {
        double[] shifts = new double[numShifts];
        double cornerVal = Math.abs(H[m * n + m]);
        double scale = Math.max(cornerVal, 1.0);

        // Use deterministic but varied shifts to break symmetry
        for (int i = 0; i < numShifts; i++) {
            // LAPACK-style: use ad-hoc formulas based on matrix elements
            double perturbation = (i + 1) * 0.9473 + 0.1307;
            shifts[i] = scale * (((i % 2) == 0) ? perturbation : -perturbation);
        }
        return shifts;
    }
}
