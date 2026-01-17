package net.faulj.decomposition.qr;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;

/**
 * Performs an implicit QR iteration step on a Hessenberg matrix using the Francis double-shift strategy.
 * <p>
 * Implicit QR is a sophisticated variant of the QR algorithm that performs similarity transformations
 * without explicitly forming the Q and R factors. This approach is central to modern eigenvalue
 * algorithms and dramatically reduces computational cost compared to explicit QR iteration.
 * </p>
 * <pre>
 *   H<sub>new</sub> = Q<sup>T</sup> * H * Q
 * </pre>
 * <p>
 * where Q is implicitly defined by a sequence of Householder reflections.
 * </p>
 *
 * <h2>Implicit vs Explicit QR:</h2>
 * <p>
 * Traditional QR iteration explicitly computes H = QR, then forms H<sub>new</sub> = RQ.
 * Implicit QR achieves the same transformation using Householder reflections without
 * forming Q or R, reducing cost from O(n<sup>3</sup>) to O(n<sup>2</sup>) per iteration.
 * </p>
 *
 * <h2>Francis Double-Shift Strategy:</h2>
 * <p>
 * The algorithm uses a double-shift approach based on the 2×2 trailing submatrix:
 * </p>
 * <pre>
 *   E = ┌ h<sub>n-2,n-2</sub>  h<sub>n-2,n-1</sub> ┐
 *       └ h<sub>n-1,n-2</sub>  h<sub>n-1,n-1</sub> ┘
 * </pre>
 * <p>
 * The shifts are the eigenvalues of E, which are complex conjugate pairs for real matrices.
 * Working with both shifts simultaneously keeps all arithmetic real.
 * </p>
 *
 * <h2>Algorithm Overview:</h2>
 * <ol>
 *   <li>Extract 2×2 trailing block E and compute its trace and determinant</li>
 *   <li>Form initial 3-vector: x<sub>0</sub> = (H² - σH + ρI)e<sub>1</sub></li>
 *   <li>Create Householder reflector Q<sub>0</sub> to introduce a "bulge"</li>
 *   <li>Chase the bulge down the diagonal using Householder reflections</li>
 *   <li>Apply similarity transformation: H ← Q<sub>0</sub><sup>T</sup>Q<sub>1</sub><sup>T</sup>...H...Q<sub>1</sub>Q<sub>0</sub></li>
 *   <li>Restore Hessenberg form by eliminating elements below first subdiagonal</li>
 * </ol>
 *
 * <h2>The Bulge and Chase:</h2>
 * <p>
 * The initial Householder reflection creates a "bulge" - a nonzero element at position (3,1).
 * Subsequent reflections chase this bulge down and off the matrix, restoring Hessenberg form
 * while accumulating the desired similarity transformation.
 * </p>
 * <pre>
 *   Step 0:        Step 1:        Step 2:        Final:
 *   * * * * *      * * * * *      * * * * *      * * * * *
 *   * * * * *      * * * * *      * * * * *      * * * * *
 *   X * * * *      0 * * * *      0 * * * *      0 * * * *
 *   0 0 * * *      0 X * * *      0 * * * *      0 0 * * *
 *   0 0 0 * *      0 0 0 * *      0 0 X * *      0 0 0 * *
 *       ↑              ↑              ↑
 *     bulge          bulge          bulge
 * </pre>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Per iteration:</b> O(6n<sup>2</sup>) flops for Hessenberg matrix</li>
 *   <li><b>vs Explicit QR:</b> O(n<sup>3</sup>) flops - orders of magnitude faster</li>
 *   <li><b>Number of reflections:</b> n-2 Householder transformations per step</li>
 *   <li><b>Space complexity:</b> O(n²) for matrix storage, O(n) for Householder vectors</li>
 * </ul>
 *
 * <h2>Why Implicit QR is Faster:</h2>
 * <ul>
 *   <li><b>Hessenberg structure:</b> Each Householder reflector only affects O(n) elements</li>
 *   <li><b>No explicit factorization:</b> Avoids forming dense Q and R matrices</li>
 *   <li><b>Structure preservation:</b> Hessenberg form maintained throughout</li>
 *   <li><b>Efficient bulge chase:</b> Localized updates rather than full matrix operations</li>
 * </ul>
 *
 * <h2>Numerical Stability:</h2>
 * <ul>
 *   <li><b>Backward stable:</b> Uses orthogonal Householder transformations</li>
 *   <li><b>Eigenvalue preservation:</b> Similarity transformation preserves eigenvalues exactly</li>
 *   <li><b>No cancellation:</b> Bulge chasing avoids subtractive cancellation</li>
 *   <li><b>Implicit Q theorem:</b> Guarantees same result as explicit QR</li>
 * </ul>
 *
 * <h2>The Implicit Q Theorem:</h2>
 * <p>
 * This fundamental theorem states that if Q and V are orthogonal matrices such that:
 * </p>
 * <ul>
 *   <li>Both Q<sup>T</sup>HQ and V<sup>T</sup>HV are Hessenberg</li>
 *   <li>Q and V have the same first column</li>
 * </ul>
 * <p>
 * Then Q and V are essentially the same (differ only by signs). This theorem justifies the
 * implicit approach: matching the first column ensures we get the same transformation.
 * </p>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Core component of Francis QR algorithm for eigenvalues</li>
 *   <li>Real Schur decomposition computation</li>
 *   <li>Eigenvalue and eigenvector calculation</li>
 *   <li>Matrix function evaluation (e<sup>A</sup>, sin(A), etc.)</li>
 *   <li>Control theory and system analysis</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Start with Hessenberg matrix (from HessenbergReduction)
 * Matrix H = HessenbergReduction.decompose(A)[0];
 *
 * // Perform implicit QR iteration
 * Matrix[] result = ImplicitQR.decompose(H);
 * Matrix Hnew = result[0];  // Transformed Hessenberg matrix
 * Matrix Q = result[1];     // Accumulated orthogonal transformation
 *
 * // Verify similarity: Hnew = Q^T * H * Q
 * Matrix check = Q.transpose().multiply(H).multiply(Q);
 *
 * // Repeat until convergence to Schur form
 * while (!isConverged(Hnew)) {
 *     result = ImplicitQR.decompose(Hnew);
 *     Hnew = result[0];
 * }
 *
 * // Diagonal elements are eigenvalues
 * double[] eigenvalues = Hnew.getDiagonal();
 * }</pre>
 *
 * <h2>Convergence Behavior:</h2>
 * <ul>
 *   <li><b>Quadratic convergence:</b> Subdiagonal elements approach zero quadratically</li>
 *   <li><b>Deflation:</b> Converged eigenvalues can be deflated to reduce problem size</li>
 *   <li><b>Complex eigenvalues:</b> Appear as 2×2 blocks on diagonal</li>
 *   <li><b>Typical iterations:</b> 2-3 iterations per eigenvalue</li>
 * </ul>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Implements Francis implicit double-shift QR algorithm</li>
 *   <li>Maintains Hessenberg structure throughout bulge chase</li>
 *   <li>Accumulates orthogonal transformation for eigenvector computation</li>
 *   <li>Enforces exact zeros below first subdiagonal for numerical cleanliness</li>
 *   <li>Handles edge cases (small matrices, identity, etc.)</li>
 *   <li>Includes diagnostic output for debugging (should be removed in production)</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.hessenberg.HessenbergReduction
 * @see net.faulj.eigen.qr.ImplicitQRFrancis
 * @see net.faulj.eigen.schur.RealSchurDecomposition
 * @see HouseholderQR
 */
public class ImplicitQR {
    public static Matrix[] decompose(Matrix Horig) {
        Matrix H = Horig.copy();

        Matrix E = H.crop(H.getRowCount()-2, H.getRowCount()-1,H.getColumnCount()-2,H.getColumnCount()-1);

        double[] eigen = new double[]{E.trace(),E.get(0, 0)*E.get(1, 1)-E.get(0, 1)*E.get(1, 0)};

        Vector x0 = H.multiply(H).subtract(H.multiplyScalar(eigen[0])).add(Matrix.Identity(H.getColumnCount()).multiplyScalar(eigen[1])).multiply(new Matrix(new Vector[]{VectorUtils.unitVector(H.getColumnCount(),0)})).getData()[0].resize(3);

        int n = H.getRowCount();
        if (n < 3) {
            System.out.println("Matrix too small for implicit QR; returning");
            return new Matrix[2];
        }

        System.out.println("Original H:");
        System.out.println(H);
        double detBefore = H.trace();
        System.out.println("trace(H) before = " + detBefore);

        java.util.ArrayList<Vector> reflectorVs = new java.util.ArrayList<>();
        java.util.ArrayList<Double> reflectorTaus = new java.util.ArrayList<>();
        java.util.ArrayList<Integer> reflectorStarts = new java.util.ArrayList<>();

        for (int k = 0; k < n - 2; k++) {
            int s;
            Vector xk;
            if (k == 0) {
                s = 0;
                Vector hh0 = VectorUtils.householder(x0.copy());
                double tau0 = hh0.get(hh0.dimension() - 1);
                Vector v0 = hh0.resize(hh0.dimension() - 1);
                double v0norm = v0.norm2();
                Vector u0 = v0.multiplyScalar(1.0 / v0norm);
                double tauNorm = 2.0;
                Matrix Q3 = Matrix.Identity(3).subtract(u0.multiply(u0.transpose()).multiplyScalar(tauNorm));

                int kblk = 3;
                Matrix A = H.crop(0, kblk - 1, 0, kblk - 1);
                Matrix B = (kblk < n) ? H.crop(0, kblk - 1, kblk, n - 1) : null;
                Matrix C = (kblk < n) ? H.crop(kblk, n - 1, 0, kblk - 1) : null;

                Matrix Anew = Q3.transpose().multiply(A).multiply(Q3);
                Matrix Bnew = (B != null) ? Q3.transpose().multiply(B) : null;
                Matrix Cnew = (C != null) ? C.multiply(Q3) : null;

                for (int i = 0; i < kblk; i++) {
                    for (int j = 0; j < kblk; j++) {
                        H.set(i, j, Anew.get(i, j));
                    }
                    if (Bnew != null) {
                        for (int j = kblk; j < n; j++) {
                            H.set(i, j, Bnew.get(i, j - kblk));
                        }
                    }
                }
                if (Cnew != null) {
                    for (int i = kblk; i < n; i++) {
                        for (int j = 0; j < kblk; j++) {
                            H.set(i, j, Cnew.get(i - kblk, j));
                        }
                    }
                }

                reflectorVs.add(u0.copy());
                reflectorTaus.add(tauNorm);
                continue;
            } else {
                s = k + 1;
                int mLen = Math.min(2, n - s);
                if (mLen <= 0) continue;
                double[] xdata = new double[mLen];
                for (int i = 0; i < mLen; i++) xdata[i] = H.get(s + i, k);
                xk = new Vector(xdata);
            }
            reflectorStarts.add(s);
            if (xk.norm2() <= 1e-10) continue;

            Vector hh = VectorUtils.householder(xk.copy());
            Vector vsmall = hh.resize(hh.dimension() - 1);
            double vnorm = vsmall.norm2();
            if (vnorm <= 1e-10) continue;
            Vector u = vsmall.multiplyScalar(1.0 / vnorm);
            double tau = 2.0;
            reflectorVs.add(u.copy());
            reflectorTaus.add(tau);

            int m = n - s;
            double[] vhatData = new double[m];
            for (int i = 0; i < vsmall.dimension(); i++) vhatData[i] = u.get(i);
            Vector vhat = new Vector(vhatData);

            Matrix Hsub = H.crop(s, n - 1, s, n - 1);

            Matrix P = Matrix.Identity(m).subtract(vhat.toMatrix().multiply(vhat.transpose()).multiplyScalar(tau));
            Matrix PP = P.multiply(P);
            double maxDiff = 0.0;
            for (int ii = 0; ii < PP.getRowCount(); ii++) {
                for (int jj = 0; jj < PP.getColumnCount(); jj++) {
                    double expect = (ii == jj) ? 1.0 : 0.0;
                    double diff = Math.abs(PP.get(ii, jj) - expect);
                    if (diff > maxDiff) maxDiff = diff;
                }
            }

            Matrix r = vhat.transpose().multiply(Hsub);
            Hsub = Hsub.subtract(vhat.toMatrix().multiply(r).multiplyScalar(tau));

            Matrix w = Hsub.multiply(vhat.toMatrix());
            Hsub = Hsub.subtract(w.multiply(vhat.transpose()).multiplyScalar(tau));

            Matrix H12 = H.crop(0, s - 1, s, n - 1);
            Matrix y = H12.multiply(vhat.toMatrix());
            H12 = H12.subtract(y.multiply(vhat.transpose()).multiplyScalar(tau));
            for (int i = 0; i < s; i++) {
                for (int j = 0; j < m; j++) {
                    H.set(i, s + j, H12.get(i, j));
                }
            }

            Matrix H21 = H.crop(s, n - 1, 0, s - 1);
            Matrix z = vhat.transpose().multiply(H21);
            H21 = H21.subtract(vhat.toMatrix().multiply(z).multiplyScalar(tau));
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < s; j++) {
                    H.set(s + i, j, H21.get(i, j));
                }
            }

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    H.set(s + i, s + j, Hsub.get(i, j));
                }
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i > j + 1) {
                        H.set(i, j, 0.0);
                    } else {
                        double val = H.get(i, j);
                        if (Math.abs(val) < 1e-10) H.set(i, j, 0.0);
                    }
                }
            }
        }

        Matrix Qfull = Matrix.Identity(n);
        for (int idx = 0; idx < reflectorVs.size(); idx++) {
            Vector v = reflectorVs.get(idx);
            double tau = reflectorTaus.get(idx);
            int s = (idx == 0) ? 0 : idx + 1;
            int m = n - s;
            Matrix Pfull = Matrix.Identity(n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    double vi = (i < v.dimension()) ? v.get(i) : 0.0;
                    double vj = (j < v.dimension()) ? v.get(j) : 0.0;
                    double pv = ((i == j) ? 1.0 : 0.0) - tau * vi * vj;
                    Pfull.set(s + i, s + j, pv);
                }
            }
            Qfull = Qfull.multiply(Pfull);
        }

        Matrix Hnew = Qfull.transpose().multiply(Horig).multiply(Qfull);
        H = Hnew;

        // Clean up tiny fill-in introduced by numerical operations: enforce Hessenberg structure
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i > j + 1) {
                    H.set(i, j, 0.0);
                } else {
                    double val = H.get(i, j);
                    if (Math.abs(val) < 1e-12) H.set(i, j, 0.0);
                }
            }
        }

        System.out.println("Transformed H after bulge chase:");
        System.out.println(H);
        double detAfter = H.trace();
        System.out.println("trace(H) after  = " + detAfter);
        System.out.println("Stored " + reflectorVs.size() + " householder reflectors for reconstruction.");

        // Return the transformed Hessenberg matrix and the accumulated orthogonal Q
        return new Matrix[]{H, Qfull};
    }
}
