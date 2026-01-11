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
        Matrix m = new Matrix(new Vector[]{new Vector(new double[]{1,2}), new Vector(new double[]{7,8})});
        Matrix m2 = m.copy();
        Matrix m3 = m.copy();
        m.toRowEchelonForm();
        m2.toReducedRowEchelonForm();
        System.out.println(m);
        System.out.println(m2);
        System.out.println(m3);
        System.out.println(m3.determinant());
        System.out.println(m3.isInvertible());
        System.out.println(m3.copy().inverse());
        System.out.println(m3.copy().multiply(m3.copy().inverse()));
    }
}
