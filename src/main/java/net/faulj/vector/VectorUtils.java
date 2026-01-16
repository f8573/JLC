package net.faulj.vector;

public class VectorUtils {
    public static Vector unitVector(int dimension, int number) {
        if (number >= dimension) {
            throw new ArithmeticException("The index of the unit cannot be larger than the dimension of the vector");
        }
        Vector v = zero(dimension);
        v.set(number,1);
        return v;
    }

    public static Vector zero (int size) {
        return new Vector(new double[size]);
    }

    public static Vector random(int size) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
        }
        return new Vector(data);
    }

    public static Vector householder(Vector x) {
        x = x.copy();
        int m = x.dimension();

        double alpha = x.norm2();
        double beta = -1*Math.copySign(alpha, x.get(0));
        Vector v = x;
        v.getData()[0] -= beta;
        double tau = 2.0 / v.dot(v);
        v = v.resize(m+1);
        v.set(m,tau);
        return v;
    }
}
