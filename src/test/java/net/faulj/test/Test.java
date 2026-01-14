package net.faulj.test;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

public class Test {

    public static void main(String[] args) {
        Matrix m = new Matrix(
                new Vector[]{
                        new Vector(new double[]{12,6,-4}),
                        new Vector(new double[]{-51,167,24}),
                        new Vector(new double[]{4,-68,-41})
                });
        Matrix[] mqr = m.QR();
        Matrix[] hess = m.Hessenberg();
        for(Matrix matrix : mqr) {
            System.out.println(matrix);
        }
//        for(Matrix matrix : hess) {
//            System.out.println(matrix);
//        }
//
//        System.out.println(Matrix.diag(Matrix.Identity(1),m));

    }
}
