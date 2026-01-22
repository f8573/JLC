package net.faulj.matrix;

import net.faulj.core.Tolerance;

public class MatrixUtils {


    public static double normResidual(Matrix A, Matrix Ahat) {
        Matrix E = A.subtract(Ahat);
        double Ef = MatrixNorms.frobeniusNorm(E);
        double Af = MatrixNorms.frobeniusNorm(A);
        double e = Tolerance.get();
        int n = A.getColumnCount();
        return Ef/(e*Af*n);
    }

    public static double backwardErrorComponentwise(Matrix A, Matrix Ahat) {
        double e = Tolerance.get();
        double max = Double.MIN_VALUE;
        Matrix E = A.subtract(Ahat);
        for (int i = 0; i < A.getColumnCount(); i++) {
            for (int j = 0; j < A.getRowCount(); j++) {
                double n = Math.hypot(E.get(j, i), E.getImag(j, i));
                double realSum = A.get(j, i) + Ahat.get(j, i);
                double imagSum = A.getImag(j, i) + Ahat.getImag(j, i);
                double d = Math.hypot(realSum, imagSum) + Math.random() / e;
                if (n/d > max) {
                    max = n/d;
                }
            }
        }
        return max;
    }
}
