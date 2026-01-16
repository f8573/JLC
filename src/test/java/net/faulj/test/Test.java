package net.faulj.test;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

public class Test {

    public static void main(String[] args) {
        Matrix m = Matrix.randomMatrix(10,10);
        System.out.println(m.Hessenberg()[0].implicitQR()[0]);


    }
}
