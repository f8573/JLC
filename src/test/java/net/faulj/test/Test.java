package net.faulj.test;

import net.faulj.core.PermutationVector;
import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.*;
import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.eigen.qr.ImplicitQRFrancis;
import net.faulj.eigen.schur.SchurEigenExtractor;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import net.faulj.orthogonal.OrthogonalProjection;
import net.faulj.orthogonal.Orthonormalization;
import net.faulj.polar.PolarDecomposition;
import net.faulj.vector.Vector;
import net.faulj.visualizer.MatrixLatexExporter;

import java.io.File;
import java.text.Bidi;
import java.util.Arrays;
import java.util.List;

public class Test {

    public static void main(String[] args) {
        Matrix A = Matrix.randomMatrix(5,5);
        System.out.println(A);
        //decompositions
        QRResult qrResult = HouseholderQR.decompose(A);
        LUDecomposition luDecomposition = new LUDecomposition();
        LUResult luResult = luDecomposition.decompose(A);
        HessenbergResult hessenbergResult = HessenbergReduction.decompose(A);
        SchurResult schurResult = ImplicitQRFrancis.decompose(A);
        Bidiagonalization bidiagonalization = new Bidiagonalization();
        BidiagonalizationResult bidiagonalizationResult = bidiagonalization.decompose(A);
        PolarResult polarResult = PolarDecomposition.decompose(A);
        //residuals
        System.out.println(qrResult.residualElement());
        System.out.println(luResult.residualElement());
        System.out.println(hessenbergResult.residualElement());
        System.out.println(schurResult.residualElement());
        System.out.println(bidiagonalizationResult.residualElement());
        System.out.println(polarResult.residualElement());
        System.out.println(qrResult.residualNorm());
        System.out.println(luResult.residualNorm());
        System.out.println(hessenbergResult.residualNorm());
        System.out.println(schurResult.residualNorm());
        System.out.println(bidiagonalizationResult.residualNorm());
        System.out.println(polarResult.residualNorm());
        //orthogonality checks
        System.out.println(Arrays.toString(qrResult.verifyOrthogonality(qrResult.getQ())));
        System.out.println(Arrays.toString(hessenbergResult.verifyOrthogonality(hessenbergResult.getQ())));
        System.out.println(Arrays.toString(schurResult.verifyOrthogonality(schurResult.getU())));
        System.out.println(Arrays.toString(bidiagonalizationResult.verifyOrthogonality(bidiagonalizationResult.getU())));
        System.out.println(Arrays.toString(bidiagonalizationResult.verifyOrthogonality(bidiagonalizationResult.getV())));
        System.out.println(Arrays.toString(polarResult.verifyOrthogonality(polarResult.getU())));
    }
}
