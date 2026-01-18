package net.faulj.eigen.qr;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.givens.GivensRotation;
import net.faulj.scalar.Complex;

import java.util.ArrayList;
import java.util.List;

/**
 * Implements the Explicit QR Algorithm for eigenvalue computation.
 * <p>
 * This class provides a direct implementation of the QR iteration A<sub>k+1</sub> = R<sub>k</sub>Q<sub>k</sub>
 * with explicit shifts. It is primarily used for educational purposes or for matrices where
 * implicit methods (like Francis double-shift) are not applicable or desired.
 * </p>
 *
 * <h2>Algorithm:</h2>
 * <ol>
 * <li><b>Hessenberg Reduction:</b> Reduce A to upper Hessenberg form H.</li>
 * <li><b>Iteration:</b> For each step k:
 * <ul>
 * <li>Determine a shift μ (Wilkinson shift strategy).</li>
 * <li>Compute QR factorization: H - μI = Q R.</li>
 * <li>Update H: H<sub>new</sub> = R Q + μI = Q<sup>T</sup> H Q.</li>
 * </ul>
 * </li>
 * <li><b>Deflation:</b> When a subdiagonal element becomes negligible, split the problem
 * into smaller sub-problems.</li>
 * </ol>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li><b>Shift Strategy:</b> Uses the Wilkinson shift (eigenvalue of bottom 2x2 block closer to corner)
 * to ensure cubic convergence for symmetric matrices and quadratic generally.</li>
 * <li><b>Explicit vs Implicit:</b> This class forms H - μI explicitly, which can be
 * less numerically stable than implicit bulge chasing for large shifts, but sufficient for
 * general use.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see ImplicitQRFrancis
 */
public class ExplicitQRIteration {
    private static final int MAX_ITERATIONS = 1000;
    private static final double EPSILON = 1e-12;

    /**
     * Computes the Real Schur Form T and unitary matrix Q such that A = Q * T * Q^T.
     *
     * @param A Square real matrix.
     * @return Matrix array {T, Q} where T is the quasi-upper triangular Schur form
     * and Q is the accumulated orthogonal matrix.
     */
    public static Matrix[] decompose(Matrix A) {
        if (!A.isSquare()) {
            throw new ArithmeticException("Matrix must be square");
        }

        // 1. Reduce to Hessenberg Form
        Matrix[] hessDecomp = HessenbergReduction.decompose(A);
        Matrix H = hessDecomp[0];
        Matrix Q = hessDecomp[1]; // Accumulated Q from Hessenberg reduction

        int n = H.getRowCount();
        int iter = 0;
        int nMinus1 = n - 1;

        // 2. Iterate
        // We work on active submatrices H[p:q, p:q]
        // Deflation happens when subdiagonal entry H[i, i-1] is negligible.
        int q = n; // Active block end (exclusive)

        while (q > 0 && iter < MAX_ITERATIONS * n) {
            // Find the active block size.
            // Scan up from q-1 to find the first negligible subdiagonal entry.
            int p = q - 1;
            while (p > 0) {
                double hVal = Math.abs(H.get(p, p - 1));
                double neighborSum = Math.abs(H.get(p - 1, p - 1)) + Math.abs(H.get(p, p));
                if (hVal < EPSILON * (neighborSum + EPSILON)) {
                    H.set(p, p - 1, 0.0);
                    break;
                }
                p--;
            }

            // Now the active block is H[p:q, p:q]
            int blockSize = q - p;

            if (blockSize == 1) {
                // 1x1 block is already diagonal
                q--;
                iter = 0;
            } else if (blockSize == 2) {
                // 2x2 block: check if it splits (real eigs) or stays (complex eigs)
                if (Math.abs(H.get(q - 1, q - 2)) < EPSILON) {
                    q -= 2; // Split into 1x1s effectively
                } else {
                    // Check if eigenvalues are real
                    double a = H.get(q - 2, q - 2);
                    double b = H.get(q - 2, q - 1);
                    double c = H.get(q - 1, q - 2);
                    double d = H.get(q - 1, q - 1);
                    double disc = (a + d) * (a + d) - 4 * (a * d - b * c);

                    if (disc >= 0) {
                        // Real eigenvalues, continue iterating to zero the subdiagonal
                        // Use Wilkinson shift to converge this 2x2 block
                        performStep(H, Q, p, q);
                        iter++;
                    } else {
                        // Complex eigenvalues. This is a converged 2x2 block in Real Schur Form.
                        q -= 2;
                        iter = 0;
                    }
                }
            } else {
                // blockSize > 2, perform iteration
                performStep(H, Q, p, q);
                iter++;
            }
        }

        return new Matrix[]{H, Q};
    }

    /**
     * Performs a single Explicit QR step with Wilkinson shift on the active block H[p:q, p:q].
     */
    private static void performStep(Matrix H, Matrix Q, int p, int q) {
        int n = H.getRowCount();

        // 1. Calculate Wilkinson Shift based on trailing 2x2 of the active block
        int end = q - 1;
        double a = H.get(end - 1, end - 1);
        double b = H.get(end - 1, end);
        double c = H.get(end, end - 1);
        double d = H.get(end, end);

        double tr = a + d;
        double det = a * d - b * c;
        double disc = tr * tr - 4 * det;

        double shift;
        if (disc >= 0) {
            double sqrt = Math.sqrt(disc);
            double l1 = (tr + sqrt) / 2;
            double l2 = (tr - sqrt) / 2;
            shift = (Math.abs(l1 - d) < Math.abs(l2 - d)) ? l1 : l2;
        } else {
            // Use real part of the eigenvalue (Rayleigh quotient approximation)
            shift = tr / 2.0;
        }

        // 2. Explicit Shifted QR Step: H - mu*I = Q_step * R
        // We compute Q_step using Givens rotations to zero the subdiagonal of (H - mu*I).

        List<GivensRotation> rotations = new ArrayList<>();

        for (int i = p; i < q; i++) {
            H.set(i, i, H.get(i, i) - shift);
        }

        // Compute Q_step = G_0 * G_1 * ... * G_{m-2}
        for (int i = p; i < q - 1; i++) {
            double x = H.get(i, i);
            double y = H.get(i + 1, i);

            GivensRotation rot = GivensRotation.compute(x, y);
            rotations.add(rot);

            // Apply G^T from left to H (affecting rows i and i+1)
            rot.applyLeft(H, i, i + 1, i, q - 1);
        }

        // H is now R (upper triangular in the active block).

        // 3. Complete the similarity transform: H_new = R * Q_step + shift*I
        // Apply rotations from the right.
        for (int i = 0; i < rotations.size(); i++) {
            GivensRotation rot = rotations.get(i);
            int colIdx = p + i;
            rot.applyRight(H, colIdx, colIdx + 1, 0, n - 1);
        }

        // Restore shift
        for (int i = p; i < q; i++) {
            H.set(i, i, H.get(i, i) + shift);
        }

        // 4. Accumulate Q
        if (Q != null) {
            for (int i = 0; i < rotations.size(); i++) {
                GivensRotation rot = rotations.get(i);
                int colIdx = p + i;
                rot.applyRight(Q, colIdx, colIdx + 1, 0, Q.getRowCount() - 1);
            }
        }
    }

    /**
     * Calculates the eigenvalues of the matrix A by reducing it to Real Schur form
     * and extracting diagonal blocks.
     *
     * @param A Square real matrix.
     * @return List of complex eigenvalues.
     */
    public static List<Complex> getEigenvalues(Matrix A) {
        Matrix[] schur = decompose(A.copy());
        Matrix T = schur[0];
        int n = T.getRowCount();
        List<Complex> eigenValues = new ArrayList<>();

        int i = 0;
        while (i < n) {
            if (i == n - 1) {
                // Last 1x1 block
                eigenValues.add(Complex.valueOf(T.get(i, i)));
                i++;
            } else {
                // Check if 1x1 or 2x2
                if (Math.abs(T.get(i + 1, i)) < EPSILON) {
                    // 1x1 block
                    eigenValues.add(Complex.valueOf(T.get(i, i)));
                    i++;
                } else {
                    // 2x2 block
                    double a = T.get(i, i);
                    double b = T.get(i, i + 1);
                    double c = T.get(i + 1, i);
                    double d = T.get(i + 1, i + 1);

                    double tr = a + d;
                    double det = a * d - b * c;
                    double disc = tr * tr - 4 * det;

                    if (disc >= 0) {
                        double sqrt = Math.sqrt(disc);
                        eigenValues.add(Complex.valueOf((tr + sqrt) / 2));
                        eigenValues.add(Complex.valueOf((tr - sqrt) / 2));
                    } else {
                        double real = tr / 2;
                        double imag = Math.sqrt(-disc) / 2;
                        eigenValues.add(Complex.valueOf(real, imag));
                        eigenValues.add(Complex.valueOf(real, -imag));
                    }
                    i += 2;
                }
            }
        }
        return eigenValues;
    }
}