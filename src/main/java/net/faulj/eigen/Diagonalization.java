package net.faulj.eigen;

import net.faulj.core.Tolerance;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.eigen.schur.SchurEigenExtractor;
import net.faulj.matrix.Matrix;
import net.faulj.scalar.Complex;
import net.faulj.vector.Vector;

import java.util.Arrays;

/**
 * Handles the diagonalization of square matrices via eigendecomposition.
 */
public class Diagonalization {
    private final Matrix A;
    private final Matrix D;
    private final Matrix P;
    private final Complex[] eigenvalues;

    /**
     * Create a diagonalization container.
     *
     * @param A original matrix
     * @param D diagonal matrix of eigenvalues
     * @param P eigenvector matrix
     * @param eigenvalues eigenvalues as complex numbers
     */
    public Diagonalization(Matrix A, Matrix D, Matrix P, Complex[] eigenvalues) {
        this.A = A;
        this.D = D;
        this.P = P;
        this.eigenvalues = eigenvalues;
    }

    /**
     * @return original matrix
     */
    public Matrix getA() {
        return A;
    }

    /**
     * @return diagonal eigenvalue matrix
     */
    public Matrix getD() {
        return D;
    }

    /**
     * @return eigenvector matrix
     */
    public Matrix getP() {
        return P;
    }

    /**
     * @return eigenvalues as complex numbers
     */
    public Complex[] getEigenvalues() {
        return eigenvalues;
    }

    /**
     * Compute the diagonalization A = P D P^{-1} via Schur decomposition.
     *
     * @param A matrix to diagonalize
     * @return diagonalization result
     */
    public static Diagonalization decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Diagonalization requires a square matrix");
        }
        SchurResult schur = RealSchurDecomposition.decompose(A);
        SchurEigenExtractor extractor = new SchurEigenExtractor(schur.getT(), schur.getU());
        Matrix pMatrix = extractor.getEigenvectors();
        Complex[] eigenvalues = extractor.getEigenvalues();
        CanonicalEigenpairs canonical = canonicalizeEigenpairs(pMatrix, eigenvalues);
        Matrix D = diagonalMatrix(canonical.eigenvalues);
        return new Diagonalization(A, D, canonical.pMatrix, canonical.eigenvalues);
    }

    private static CanonicalEigenpairs canonicalizeEigenpairs(Matrix pMatrix, Complex[] eigenvalues) {
        int n = eigenvalues.length;
        double tol = Tolerance.get();
        double[][] colsReal = new double[n][n];
        double[][] colsImag = new double[n][n];
        boolean[] hasImag = new boolean[n];

        extractColumns(pMatrix, colsReal, colsImag, hasImag, tol);
        normalizePhases(colsReal, colsImag, hasImag, tol);

        Integer[] order = buildEigenvalueOrder(eigenvalues);
        CanonicalMatrixResult canonical = buildCanonicalMatrix(order, colsReal, colsImag, hasImag, eigenvalues);
        return new CanonicalEigenpairs(canonical.pMatrix, canonical.eigenvalues);
    }

    private static void extractColumns(Matrix pMatrix, double[][] colsReal, double[][] colsImag,
                                       boolean[] hasImag, double tol) {
        int n = colsReal.length;
        for (int c = 0; c < n; c++) {
            boolean anyImag = false;
            for (int r = 0; r < n; r++) {
                colsReal[c][r] = pMatrix.get(r, c);
                double im = pMatrix.getImag(r, c);
                colsImag[c][r] = im;
                if (Math.abs(im) > tol) anyImag = true;
            }
            hasImag[c] = anyImag;
        }
    }

    private static void normalizePhases(double[][] colsReal, double[][] colsImag, boolean[] hasImag, double tol) {
        int n = colsReal.length;
        for (int c = 0; c < n; c++) {
            double[] real = colsReal[c];
            double[] imag = hasImag[c] ? colsImag[c] : null;
            PhaseScale scale = findPhaseScale(real, imag, tol);
            if (scale != null) {
                applyPhase(real, imag, scale);
            }
            if (imag != null) {
                hasImag[c] = hasNonZeroImag(imag, tol);
            }
        }
    }

    private static PhaseScale findPhaseScale(double[] real, double[] imag, double tol) {
        for (int r = 0; r < real.length; r++) {
            double a = real[r];
            double b = imag == null ? 0.0 : imag[r];
            if (Math.hypot(a, b) > tol) {
                double mag = Math.hypot(a, b);
                if (mag <= 0.0) return null;
                return new PhaseScale(a / mag, -b / mag);
            }
        }
        return null;
    }

    private static void applyPhase(double[] real, double[] imag, PhaseScale scale) {
        for (int r = 0; r < real.length; r++) {
            double a = real[r];
            double b = imag == null ? 0.0 : imag[r];
            double newR = a * scale.scaleR - b * scale.scaleI;
            double newI = a * scale.scaleI + b * scale.scaleR;
            real[r] = newR;
            if (imag != null) imag[r] = newI;
        }
    }

    private static boolean hasNonZeroImag(double[] imag, double tol) {
        for (double v : imag) {
            if (Math.abs(v) > tol) return true;
        }
        return false;
    }

    private static final class PhaseScale {
        private final double scaleR;
        private final double scaleI;

        private PhaseScale(double scaleR, double scaleI) {
            this.scaleR = scaleR;
            this.scaleI = scaleI;
        }
    }

    private static Integer[] buildEigenvalueOrder(Complex[] eigenvalues) {
        int n = eigenvalues.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            Complex ea = eigenvalues[a];
            Complex eb = eigenvalues[b];
            int cmp = Double.compare(ea.real, eb.real);
            if (cmp != 0) return cmp;
            return Double.compare(ea.imag, eb.imag);
        });
        return order;
    }

    private static CanonicalMatrixResult buildCanonicalMatrix(Integer[] order, double[][] colsReal,
                                                              double[][] colsImag, boolean[] hasImag,
                                                              Complex[] eigenvalues) {
        int n = order.length;
        Vector[] columns = new Vector[n];
        Complex[] ordered = new Complex[n];
        for (int i = 0; i < n; i++) {
            int src = order[i];
            ordered[i] = eigenvalues[src];
            double[] real = Arrays.copyOf(colsReal[src], n);
            if (hasImag[src]) {
                double[] imag = Arrays.copyOf(colsImag[src], n);
                columns[i] = new Vector(real, imag);
            } else {
                columns[i] = new Vector(real);
            }
        }
        return new CanonicalMatrixResult(new Matrix(columns), ordered);
    }

    private static final class CanonicalMatrixResult {
        private final Matrix pMatrix;
        private final Complex[] eigenvalues;

        private CanonicalMatrixResult(Matrix pMatrix, Complex[] eigenvalues) {
            this.pMatrix = pMatrix;
            this.eigenvalues = eigenvalues;
        }
    }

    private static final class CanonicalEigenpairs {
        private final Matrix pMatrix;
        private final Complex[] eigenvalues;

        private CanonicalEigenpairs(Matrix pMatrix, Complex[] eigenvalues) {
            this.pMatrix = pMatrix;
            this.eigenvalues = eigenvalues;
        }
    }

    /**
     * Build a complex diagonal matrix from eigenvalues.
     *
     * @param eigenvalues eigenvalues
     * @return diagonal matrix
     */
    private static Matrix diagonalMatrix(Complex[] eigenvalues) {
        int n = eigenvalues.length;
        Matrix D = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            Complex lambda = eigenvalues[i];
            D.setComplex(i, i, lambda.real, lambda.imag);
        }
        return D;
    }
}
