package net.faulj.eigen.qr;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
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

        int n = A.getRowCount();
        boolean symmetric = isSymmetric(A, EPSILON);
        Matrix T = A.copy();
        Matrix Q = Matrix.Identity(n);
        int maxIterations = MAX_ITERATIONS * n;

        for (int iter = 0; iter < maxIterations; iter++) {
            if (isConverged(T, EPSILON, symmetric)) {
                break;
            }

            double shift = computeWilkinsonShift(T);
            Matrix shifted = T.copy();
            for (int i = 0; i < n; i++) {
                shifted.set(i, i, shifted.get(i, i) - shift);
            }

            QRResult qr = HouseholderQR.decompose(shifted);
            Matrix qStep = qr.getQ();
            Matrix rStep = qr.getR();

            T = rStep.multiply(qStep);
            for (int i = 0; i < n; i++) {
                T.set(i, i, T.get(i, i) + shift);
            }

            Q = Q.multiply(qStep);

            // Deflate tiny subdiagonal elements to stabilize convergence.
            for (int i = 1; i < n; i++) {
                if (Math.abs(T.get(i, i - 1)) < EPSILON) {
                    T.set(i, i - 1, 0.0);
                }
            }
        }

        return new Matrix[]{T, Q};
    }

    private static boolean isConverged(Matrix T, double tol, boolean symmetric) {
        int n = T.getRowCount();
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                if (Math.abs(T.get(i, j)) > tol) {
                    return false;
                }
            }
        }
        if (symmetric) {
            for (int i = 1; i < n; i++) {
                if (Math.abs(T.get(i, i - 1)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isSymmetric(Matrix A, double tol) {
        int n = A.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(A.get(i, j) - A.get(j, i)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    private static double computeWilkinsonShift(Matrix T) {
        int n = T.getRowCount();
        if (n < 2) {
            return T.get(0, 0);
        }

        int i = n - 2;
        int j = n - 1;
        double a = T.get(i, i);
        double b = T.get(i, j);
        double c = T.get(j, i);
        double d = T.get(j, j);

        double tr = a + d;
        double det = a * d - b * c;
        double disc = tr * tr - 4 * det;

        if (disc >= 0) {
            double sqrt = Math.sqrt(disc);
            double l1 = (tr + sqrt) / 2.0;
            double l2 = (tr - sqrt) / 2.0;
            return (Math.abs(l1 - d) < Math.abs(l2 - d)) ? l1 : l2;
        }

        // Complex pair: use the real part to keep the iteration in real arithmetic.
        return tr / 2.0;
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
