package net.faulj.eigen.schur;

import net.faulj.core.Tolerance;
import net.faulj.matrix.Matrix;
import net.faulj.scalar.Complex;
import net.faulj.vector.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for computing eigenvectors from a Real Schur Decomposition.
 */
public class SchurEigenExtractor {
    private final Matrix schur;
    private final Matrix schurVectors;
    private Complex[] eigenvalues;
    private Matrix eigenvectors;

    public SchurEigenExtractor(Matrix S) {
        this(S, null);
    }

    public SchurEigenExtractor(Matrix S, Matrix U) {
        schur = S;
        schurVectors = U;
        initialize();
    }

    private void initialize() {
        eigenvalues = extractEigenvalues(schur);
        if (schurVectors != null) {
            eigenvectors = extractEigenvectors(schur, schurVectors, eigenvalues);
        }
    }

    public Complex[] getEigenvalues() {
        return eigenvalues;
    }

    public Matrix getEigenvectors() {
        return eigenvectors;
    }

    private static Complex[] extractEigenvalues(Matrix T) {
        int n = T.getRowCount();
        List<Complex> values = new ArrayList<>();
        double tol = Tolerance.get();

        int i = 0;
        while (i < n) {
            if (i == n - 1 || Math.abs(T.get(i + 1, i)) <= tol) {
                values.add(Complex.valueOf(T.get(i, i)));
                i++;
            } else {
                double a = T.get(i, i);
                double b = T.get(i, i + 1);
                double c = T.get(i + 1, i);
                double d = T.get(i + 1, i + 1);
                double tr = a + d;
                double det = a * d - b * c;
                double disc = tr * tr - 4 * det;
                if (disc >= 0) {
                    double sqrt = Math.sqrt(disc);
                    values.add(Complex.valueOf((tr + sqrt) / 2.0));
                    values.add(Complex.valueOf((tr - sqrt) / 2.0));
                } else {
                    double real = tr / 2.0;
                    double imag = Math.sqrt(-disc) / 2.0;
                    values.add(Complex.valueOf(real, imag));
                    values.add(Complex.valueOf(real, -imag));
                }
                i += 2;
            }
        }
        return values.toArray(new Complex[0]);
    }

    private static Matrix extractEigenvectors(Matrix T, Matrix U, Complex[] values) {
        int n = T.getRowCount();
        Vector[] columns = new Vector[n];
        double tol = Tolerance.get();

        int i = 0;
        while (i < n) {
            if (i < n - 1 && Math.abs(T.get(i + 1, i)) > tol) {
                Complex lambda = values[i];
                Vector y = solveComplexTriangularEigenvector(T, lambda, i, tol);
                Matrix yMatrix = new Matrix(new Vector[]{y});
                Matrix xMatrix = U.multiply(yMatrix);
                Vector x = xMatrix.getData()[0];
                columns[i] = x;
                columns[i + 1] = x.conjugate();
                i += 2;
                continue;
            }

            Complex lambda = values[i];
            Vector y = solveRealTriangularEigenvector(T, lambda.real, i, tol);
            Matrix yMatrix = new Matrix(new Vector[]{y});
            Matrix xMatrix = U.multiply(yMatrix);
            columns[i] = xMatrix.getData()[0];
            i++;
        }
        return new Matrix(columns);
    }

    private static Vector solveRealTriangularEigenvector(Matrix T, double lambda, int index, double tol) {
        int n = T.getRowCount();
        double[] y = new double[n];
        y[index] = 1.0;

        for (int row = index - 1; row >= 0; row--) {
            double sum = 0.0;
            for (int col = row + 1; col <= index; col++) {
                sum += T.get(row, col) * y[col];
            }
            double denom = T.get(row, row) - lambda;
            if (Math.abs(denom) < tol) {
                denom = denom >= 0.0 ? tol : -tol;
            }
            y[row] = -sum / denom;
        }

        return new Vector(y);
    }

    private static Vector solveComplexTriangularEigenvector(Matrix T, Complex lambda, int index, double tol) {
        int n = T.getRowCount();
        double alpha = lambda.real;
        double beta = lambda.imag;
        double[] p = new double[n];
        double[] q = new double[n];

        double a = T.get(index, index);
        double b = T.get(index, index + 1);
        double c = T.get(index + 1, index);
        double d = T.get(index + 1, index + 1);

        Complex v1;
        Complex v2;
        if (Math.abs(b) >= Math.abs(c)) {
            if (Math.abs(b) <= tol) {
                v1 = Complex.ONE;
                v2 = Complex.ZERO;
            } else {
                v1 = Complex.ONE;
                Complex aMinusLambda = Complex.valueOf(a).subtract(lambda);
                v2 = aMinusLambda.multiply(-1).divide(Complex.valueOf(b));
            }
        } else {
            if (Math.abs(c) <= tol) {
                v1 = Complex.ZERO;
                v2 = Complex.ONE;
            } else {
                v2 = Complex.ONE;
                Complex dMinusLambda = Complex.valueOf(d).subtract(lambda);
                v1 = dMinusLambda.multiply(-1).divide(Complex.valueOf(c));
            }
        }

        p[index] = v1.real;
        q[index] = v1.imag;
        p[index + 1] = v2.real;
        q[index + 1] = v2.imag;

        int row = index - 1;
        while (row >= 0) {
            if (row > 0 && Math.abs(T.get(row, row - 1)) > tol) {
                int top = row - 1;
                int bottom = row;
                double sumTopReal = 0.0;
                double sumTopImag = 0.0;
                double sumBottomReal = 0.0;
                double sumBottomImag = 0.0;
                for (int col = bottom + 1; col <= index + 1; col++) {
                    double tTop = T.get(top, col);
                    double tBottom = T.get(bottom, col);
                    if (tTop != 0.0) {
                        sumTopReal += tTop * p[col];
                        sumTopImag += tTop * q[col];
                    }
                    if (tBottom != 0.0) {
                        sumBottomReal += tBottom * p[col];
                        sumBottomImag += tBottom * q[col];
                    }
                }

                Complex sumTop = Complex.valueOf(sumTopReal, sumTopImag);
                Complex sumBottom = Complex.valueOf(sumBottomReal, sumBottomImag);
                Complex a11 = Complex.valueOf(T.get(top, top)).subtract(lambda);
                Complex a22 = Complex.valueOf(T.get(bottom, bottom)).subtract(lambda);
                double a12 = T.get(top, bottom);
                double a21 = T.get(bottom, top);
                Complex det = a11.multiply(a22).subtract(Complex.valueOf(a12 * a21));
                if (det.abs() < tol) {
                    det = det.add(Complex.valueOf(tol));
                }

                Complex vTop = a22.multiply(sumTop).multiply(-1).add(Complex.valueOf(a12).multiply(sumBottom)).divide(det);
                Complex vBottom = Complex.valueOf(a21).multiply(sumTop).subtract(a11.multiply(sumBottom)).divide(det);

                p[top] = vTop.real;
                q[top] = vTop.imag;
                p[bottom] = vBottom.real;
                q[bottom] = vBottom.imag;
                row -= 2;
                continue;
            }

            double sumP = 0.0;
            double sumQ = 0.0;
            for (int col = row + 1; col <= index + 1; col++) {
                double t = T.get(row, col);
                if (t != 0.0) {
                    sumP += t * p[col];
                    sumQ += t * q[col];
                }
            }
            double aDiag = T.get(row, row) - alpha;
            double denom = aDiag * aDiag + beta * beta;
            if (denom < tol) {
                denom = tol * tol;
            }
            p[row] = (-aDiag * sumP + beta * sumQ) / denom;
            q[row] = (-beta * sumP - aDiag * sumQ) / denom;
            row--;
        }

        return new Vector(p, q);
    }
}
