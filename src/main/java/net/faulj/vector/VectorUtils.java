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
}
