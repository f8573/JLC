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
 * This implementation:
 * 1. Reduces the input matrix A to Hessenberg form H.
 * 2. Iteratively applies Q*H*Q^T updates (explicit shift) using Givens rotations
 * to drive H to Real Schur Form (quasi-triangular).
 * 3. Uses the Wilkinson shift strategy for convergence.
 * </p>
 */
public class ExplicitQRIteration {
    private static final int MAX_ITERATIONS = 1000;
    private static final double EPSILON = 1e-12;

    /**
     * Computes the Real Schur Form T and unitary matrix Q such that A = Q * T * Q^T.
     *
     * @param A Square matrix
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
                // In Real Schur form, we accept 2x2 blocks on diagonal.
                // We can perform a check if it's already in a standard form, but typically we just deflate.
                // Check if we can split it with a real shift?
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

        // Eigenvalues of [[a, b], [c, d]]
        // lambda = (tr +/- sqrt(tr^2 - 4*det))/2
        // shift is the eigenvalue closer to d.
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
            // Complex eigenvalues.
            // For Explicit QR on Real matrices, we can't use a complex shift directly
            // without complex arithmetic (which would make H complex).
            // Strategy: Use the real part of the eigenvalue (Rayleigh quotient approximation)
            // or zero shift if unstable.
            shift = tr / 2.0;
        }

        // 2. Explicit Shifted QR Step: H - mu*I = Q_step * R
        // We compute Q_step using Givens rotations to zero the subdiagonal of (H - mu*I).
        // Since H is Hessenberg, we only need to zero H[i+1, i].

        List<GivensRotation> rotations = new ArrayList<>();

        // Apply shift to diagonal implicitly during rotation computation
        // We don't want to destroy H, so we simulate the factorization
        // H_shifted = H.copy(); for (int i=p; i<q; i++) H_shifted.set(i,i, H.get(i,i) - shift);
        // But for speed, we can modify H temporarily or just track the column logic.
        // Explicitly forming H - shift*I is fine for clarity here.

        for (int i = p; i < q; i++) {
            H.set(i, i, H.get(i, i) - shift);
        }

        // Compute Q_step = G_0 * G_1 * ... * G_{m-2}
        // such that Q_step^T * (H - shift*I) = R (Upper Triangular)
        for (int i = p; i < q - 1; i++) {
            double x = H.get(i, i);
            double y = H.get(i + 1, i);

            GivensRotation rot = GivensRotation.compute(x, y);
            rotations.add(rot);

            // Apply G^T from left to H (affecting rows i and i+1)
            // Note: GivensRotation.applyLeft applies G^T * A
            rot.applyLeft(H, i, i + 1, i, q - 1);
            // Optimization: col range starts at i (since cols < i are zero below diag)
        }

        // H is now R (upper triangular in the active block).

        // 3. Complete the similarity transform: H_new = R * Q_step + shift*I
        // H_new = R * (G_0 * ... * G_{m-2}) + shift*I
        // Apply rotations from the right.

        for (int i = 0; i < rotations.size(); i++) {
            GivensRotation rot = rotations.get(i);
            int colIdx = p + i;
            // applyRight applies A * G (which is A * Q_step if we use same rotations)
            // Careful: Q_step = G_0 ... G_k.
            // We want R * G_0 * G_1 ...
            // GivensRotation stores (c, s).
            // applyRight uses the transpose of the rotation matrix used in applyLeft.
            // If applyLeft used G^T, applyRight uses (G^T)^T = G.
            // We need R * Q_step? No, similarity is Q_step^T * A_old * Q_step.
            // We computed R = Q_step^T * (A_old - shift*I).
            // Next is A_new = R * Q_step + shift*I.
            // So we multiply R by G_0, then G_1... from the right.
            // rot.applyRight applies the rotation G.

            // Range: rows can be from 0 to min(colIdx + 2, q).
            // Since H is Hessenberg (now triangular R), we need to update rows 0..colIdx+1
            rot.applyRight(H, colIdx, colIdx + 1, 0, Math.min(colIdx + 2, q - 1));
        }

        // Restore shift
        for (int i = p; i < q; i++) {
            H.set(i, i, H.get(i, i) + shift);
        }

        // 4. Accumulate Q
        // Q_total = Q_total * Q_step
        if (Q != null) {
            for (int i = 0; i < rotations.size(); i++) {
                GivensRotation rot = rotations.get(i);
                int colIdx = p + i;
                rot.applyRight(Q, colIdx, colIdx + 1, 0, Q.getRowCount() - 1);
            }
        }
    }

    /**
     * Calculates the eigenvalues of the matrix A.
     * @param A Square matrix
     * @return List of complex eigenvalues
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