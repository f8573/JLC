package net.faulj.test;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

public class Test {

    public static void main(String[] args) {
        Vector v = new Vector(new double[]{1, 2, 3});
        Vector v2 = new Vector(new double[]{4, 5, 6});
        System.out.println(v);
        System.out.println(v2);
        System.out.println(v.magnitude());
        System.out.println(v.dot(v2));
        Vector v3 = v.normalize();
        System.out.println(v3);
        Vector v4 = v.copy();
        v4.add(v2);
        System.out.println(v4);
        System.out.println(v);
        Matrix m = new Matrix(
                new Vector[]{
                        new Vector(new double[]{1,0,0,0}),
                        new Vector(new double[]{6,0,0,0}),
                        new Vector(new double[]{0,1,0,0}),
                        new Vector(new double[]{0,9,0,0}),
                        new Vector(new double[]{5,3,0,0})
                });
        Matrix m2 = m.copy();
        System.out.println(Matrix.Identity(3));
        System.out.println(m.nullSpaceBasis());
    }
}
