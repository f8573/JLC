package net.faulj.test;

import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.matrix.Matrix;
import net.faulj.visualizer.MatrixLatexExporter;

import java.io.File;

public class Test {

    public static void main(String[] args) {
        //We wish to test Hessenberg
        Matrix Hessenberg = new Matrix(new double[][]{
                {4, 3, 2, 1},
                {1, 4, 3, 2},
                {2, 1, 4, 3},
                {3, 2, 1, 4}
        });
        Matrix[] hessenbergReduction = HessenbergReduction.decompose(Hessenberg);
        System.out.println("Trace delta: " + (Math.abs(Hessenberg.trace()-hessenbergReduction[0].trace())));
    }
}
