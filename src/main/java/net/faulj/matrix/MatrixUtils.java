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
                double n = Math.abs(E.get(i,j));
                double d = (Math.abs(A.get(i,j)+Ahat.get(i,j)))+Math.random()/e;
                if (n/d > max) {
                    max = n/d;
                }
            }
        }
        return max;
    }
}
