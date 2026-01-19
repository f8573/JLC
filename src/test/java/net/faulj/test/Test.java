package net.faulj.test;

import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.eigen.qr.ImplicitQRFrancis;
import net.faulj.eigen.schur.SchurEigenExtractor;
import net.faulj.matrix.Matrix;
import net.faulj.orthogonal.OrthogonalProjection;
import net.faulj.orthogonal.Orthonormalization;
import net.faulj.vector.Vector;
import net.faulj.visualizer.MatrixLatexExporter;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class Test {

    public static void main(String[] args) {
        Matrix A = Matrix.randomMatrix(5,5);
        System.out.println(A);
        System.out.println(ExplicitQRIteration.decompose(A)[0]);
        System.out.println(ImplicitQRFrancis.decompose(A));
    }
}
