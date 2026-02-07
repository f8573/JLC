package net.faulj.core;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.Diagonalization;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.eigen.schur.SchurEigenExtractor;
import net.faulj.matrix.Matrix;
import net.faulj.scalar.Complex;
import net.faulj.vector.Vector;
import org.junit.Test;

import java.util.List;
import java.util.Set;

import static org.junit.Assert.*;

/**
 * Diagnostic test to reproduce eigenspace/decomposition issues for 3x3+ matrices.
 */
public class EigenspaceDiagnosticTest {

    @Test
    public void testEigenspaceFor3x3Matrix() {
        Matrix A = new Matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 8}
        });

        DiagnosticMetrics.MatrixDiagnostics diag = DiagnosticMetrics.analyze(A);

        System.out.println("=== 3x3 Matrix [[1,2,3],[4,5,6],[7,8,8]] ===");
        System.out.println("Square: " + diag.isSquare());
        System.out.println("Real: " + diag.isReal());
        System.out.println("Diagonalizable: " + diag.getDiagonalizable());

        // Print eigenvalues
        Complex[] eigenvalues = diag.getEigenvalues();
        System.out.println("Eigenvalues:");
        if (eigenvalues != null) {
            for (int i = 0; i < eigenvalues.length; i++) {
                System.out.println("  lambda[" + i + "] = " + eigenvalues[i].real + " + " + eigenvalues[i].imag + "i");
            }
        } else {
            System.out.println("  null!");
        }

        // Print multiplicities
        int[] alg = diag.getAlgebraicMultiplicity();
        int[] geo = diag.getGeometricMultiplicity();
        System.out.println("Algebraic multiplicities: ");
        if (alg != null) {
            for (int i = 0; i < alg.length; i++) System.out.println("  alg[" + i + "] = " + alg[i]);
        }
        System.out.println("Geometric multiplicities: ");
        if (geo != null) {
            for (int i = 0; i < geo.length; i++) System.out.println("  geo[" + i + "] = " + geo[i]);
        }

        // Print eigenspace list
        List<DiagnosticMetrics.DiagnosticItem<Set<Vector>>> eigenspaceList = diag.getEigenspaceList();
        System.out.println("EigenspaceList size: " + (eigenspaceList != null ? eigenspaceList.size() : "null"));
        if (eigenspaceList != null) {
            for (int i = 0; i < eigenspaceList.size(); i++) {
                DiagnosticMetrics.DiagnosticItem<Set<Vector>> item = eigenspaceList.get(i);
                System.out.println("  eigenspace[" + i + "]: name=" + item.getName() +
                    " status=" + item.getStatus() + " message=" + item.getMessage());
                if (item.getValue() != null) {
                    System.out.println("    vectors: " + item.getValue().size());
                    for (Vector v : item.getValue()) {
                        StringBuilder sb = new StringBuilder("    [");
                        for (int j = 0; j < v.dimension(); j++) {
                            if (j > 0) sb.append(", ");
                            sb.append(v.get(j));
                            if (v.hasImag()) sb.append("+" + v.getImag(j) + "i");
                        }
                        sb.append("]");
                        System.out.println(sb.toString());
                    }
                } else {
                    System.out.println("    value is null!");
                }
            }
        }

        // Print eigenbasis list
        List<DiagnosticMetrics.DiagnosticItem<Set<Vector>>> eigenbasisList = diag.getEigenbasisList();
        System.out.println("EigenbasisList size: " + (eigenbasisList != null ? eigenbasisList.size() : "null"));
        if (eigenbasisList != null) {
            for (int i = 0; i < eigenbasisList.size(); i++) {
                DiagnosticMetrics.DiagnosticItem<Set<Vector>> item = eigenbasisList.get(i);
                System.out.println("  eigenbasis[" + i + "]: name=" + item.getName() +
                    " status=" + item.getStatus() + " message=" + item.getMessage());
                if (item.getValue() != null) {
                    System.out.println("    vectors: " + item.getValue().size());
                } else {
                    System.out.println("    value is null!");
                }
            }
        }

        // Print eigenvectors matrix
        Matrix eigenvectors = diag.getEigenvectors();
        System.out.println("Eigenvectors matrix: " + (eigenvectors != null ? eigenvectors.getRowCount() + "x" + eigenvectors.getColumnCount() : "null"));
        if (eigenvectors != null) {
            System.out.println("  hasImagData: " + eigenvectors.hasImagData());
        }

        // Print diagonalization status
        DiagnosticMetrics.DiagnosticItem<?> diagItem = diag.getDiagonalization();
        System.out.println("Diagonalization status: " + (diagItem != null ? diagItem.getStatus() : "null"));
        System.out.println("Diagonalization message: " + (diagItem != null ? diagItem.getMessage() : "null"));
        if (diagItem != null && diagItem.getValidation() != null) {
            System.out.println("Diag validation normLevel: " + diagItem.getValidation().normLevel);
            System.out.println("Diag validation elementLevel: " + diagItem.getValidation().elementLevel);
            System.out.println("Diag validation normResidual: " + diagItem.getValidation().normResidual);
            System.out.println("Diag validation elementResidual: " + diagItem.getValidation().elementResidual);
            System.out.println("Diag validation passes: " + diagItem.getValidation().passes);
            System.out.println("Diag validation message: " + diagItem.getValidation().message);
        }

        // Print SVD status
        DiagnosticMetrics.DiagnosticItem<?> svdItem = diag.getSvd();
        System.out.println("SVD status: " + (svdItem != null ? svdItem.getStatus() : "null"));
        if (svdItem != null && svdItem.getValidation() != null) {
            System.out.println("SVD validation normLevel: " + svdItem.getValidation().normLevel);
            System.out.println("SVD validation elementLevel: " + svdItem.getValidation().elementLevel);
            System.out.println("SVD validation normResidual: " + svdItem.getValidation().normResidual);
            System.out.println("SVD validation elementResidual: " + svdItem.getValidation().elementResidual);
        }

        // Assertions
        assertNotNull("Eigenvalues should not be null", eigenvalues);
        assertEquals("Should have 3 eigenvalues", 3, eigenvalues.length);

        if (geo != null) {
            for (int i = 0; i < geo.length; i++) {
                assertTrue("Geometric multiplicity should be positive, got " + geo[i], geo[i] > 0);
            }
        }

        assertNotNull("Diagonalizable should not be null", diag.getDiagonalizable());
    }

    @Test
    public void testSchurDirectly() {
        Matrix A = new Matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 8}
        });

        System.out.println("\n=== Direct Schur Decomposition ===");

        SchurResult schur = RealSchurDecomposition.decompose(A);
        Matrix T = schur.getT();
        Matrix U = schur.getU();

        System.out.println("Schur form T:");
        for (int i = 0; i < T.getRowCount(); i++) {
            StringBuilder sb = new StringBuilder("  [");
            for (int j = 0; j < T.getColumnCount(); j++) {
                if (j > 0) sb.append(", ");
                sb.append(String.format("%.6f", T.get(i, j)));
            }
            sb.append("]");
            System.out.println(sb.toString());
        }

        System.out.println("Schur eigenvalues from SchurResult:");
        Complex[] schurEv = schur.getEigenvalues();
        for (int i = 0; i < schurEv.length; i++) {
            System.out.println("  lambda[" + i + "] = " + schurEv[i].real + " + " + schurEv[i].imag + "i");
        }

        // Now test via SchurEigenExtractor
        SchurEigenExtractor extractor = new SchurEigenExtractor(T, U);
        Complex[] extractedEv = extractor.getEigenvalues();
        System.out.println("SchurEigenExtractor eigenvalues:");
        for (int i = 0; i < extractedEv.length; i++) {
            System.out.println("  lambda[" + i + "] = " + extractedEv[i].real + " + " + extractedEv[i].imag + "i");
        }

        // Print eigenvectors
        Matrix eigenvectors = extractor.getEigenvectors();
        System.out.println("Eigenvectors P:");
        for (int i = 0; i < eigenvectors.getRowCount(); i++) {
            StringBuilder sb = new StringBuilder("  [");
            for (int j = 0; j < eigenvectors.getColumnCount(); j++) {
                if (j > 0) sb.append(", ");
                sb.append(String.format("%.6f", eigenvectors.get(i, j)));
            }
            sb.append("]");
            System.out.println(sb.toString());
        }

        // Test Diagonalization
        System.out.println("\n=== Direct Diagonalization ===");
        Diagonalization diagResult = Diagonalization.decompose(A);
        Complex[] diagEv = diagResult.getEigenvalues();
        System.out.println("Diagonalization eigenvalues:");
        for (int i = 0; i < diagEv.length; i++) {
            System.out.println("  lambda[" + i + "] = " + diagEv[i].real + " + " + diagEv[i].imag + "i");
        }

        Matrix P = diagResult.getP();
        Matrix D = diagResult.getD();

        System.out.println("D matrix:");
        for (int i = 0; i < D.getRowCount(); i++) {
            StringBuilder sb = new StringBuilder("  [");
            for (int j = 0; j < D.getColumnCount(); j++) {
                if (j > 0) sb.append(", ");
                sb.append(String.format("%.6f", D.get(i, j)));
                if (D.hasImagData()) {
                    sb.append("+").append(String.format("%.6fi", D.getImag(i, j)));
                }
            }
            sb.append("]");
            System.out.println(sb.toString());
        }

        System.out.println("P matrix:");
        for (int i = 0; i < P.getRowCount(); i++) {
            StringBuilder sb = new StringBuilder("  [");
            for (int j = 0; j < P.getColumnCount(); j++) {
                if (j > 0) sb.append(", ");
                sb.append(String.format("%.6f", P.get(i, j)));
            }
            sb.append("]");
            System.out.println(sb.toString());
        }

        // Reconstruct
        Matrix pinv = P.inverse();
        Matrix reconstructed = P.multiply(D).multiply(pinv);
        System.out.println("Reconstructed A = PDP^-1:");
        for (int i = 0; i < reconstructed.getRowCount(); i++) {
            StringBuilder sb = new StringBuilder("  [");
            for (int j = 0; j < reconstructed.getColumnCount(); j++) {
                if (j > 0) sb.append(", ");
                sb.append(String.format("%.6f", reconstructed.get(i, j)));
            }
            sb.append("]");
            System.out.println(sb.toString());
        }

        // Verify: Av = lambda*v for each eigenvector
        System.out.println("\nVerification Av = lambda*v:");
        Vector[] pCols = P.getData();
        for (int i = 0; i < diagEv.length; i++) {
            Vector v = pCols[i];
            double lambda = diagEv[i].real;
            // Compute Av
            double[] av = new double[3];
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    av[r] += A.get(r, c) * v.get(c);
                }
            }
            double[] lv = new double[3];
            for (int r = 0; r < 3; r++) lv[r] = lambda * v.get(r);
            double diff = 0;
            for (int r = 0; r < 3; r++) diff += (av[r] - lv[r]) * (av[r] - lv[r]);
            diff = Math.sqrt(diff);
            System.out.println("  lambda=" + lambda + " ||Av - lambda*v|| = " + diff);
        }
    }

    @Test
    public void testEigenspaceFor2x2Matrix() {
        // Compare with a 2x2 matrix that reportedly works
        Matrix A = new Matrix(new double[][]{
            {1, 2},
            {3, 4}
        });

        DiagnosticMetrics.MatrixDiagnostics diag = DiagnosticMetrics.analyze(A);

        System.out.println("\n=== 2x2 Matrix [[1,2],[3,4]] ===");
        System.out.println("Diagonalizable: " + diag.getDiagonalizable());

        Complex[] eigenvalues = diag.getEigenvalues();
        System.out.println("Eigenvalues:");
        if (eigenvalues != null) {
            for (int i = 0; i < eigenvalues.length; i++) {
                System.out.println("  lambda[" + i + "] = " + eigenvalues[i].real + " + " + eigenvalues[i].imag + "i");
            }
        }

        int[] geo = diag.getGeometricMultiplicity();
        System.out.println("Geometric multiplicities: ");
        if (geo != null) {
            for (int i = 0; i < geo.length; i++) System.out.println("  geo[" + i + "] = " + geo[i]);
        }

        List<DiagnosticMetrics.DiagnosticItem<Set<Vector>>> eigenspaceList = diag.getEigenspaceList();
        System.out.println("EigenspaceList size: " + (eigenspaceList != null ? eigenspaceList.size() : "null"));
        if (eigenspaceList != null) {
            for (int i = 0; i < eigenspaceList.size(); i++) {
                DiagnosticMetrics.DiagnosticItem<Set<Vector>> item = eigenspaceList.get(i);
                System.out.println("  eigenspace[" + i + "]: status=" + item.getStatus() +
                    " hasValue=" + (item.getValue() != null) +
                    " message=" + item.getMessage());
            }
        }

        List<DiagnosticMetrics.DiagnosticItem<Set<Vector>>> eigenbasisList = diag.getEigenbasisList();
        System.out.println("EigenbasisList size: " + (eigenbasisList != null ? eigenbasisList.size() : "null"));
        if (eigenbasisList != null) {
            for (int i = 0; i < eigenbasisList.size(); i++) {
                DiagnosticMetrics.DiagnosticItem<Set<Vector>> item = eigenbasisList.get(i);
                System.out.println("  eigenbasis[" + i + "]: status=" + item.getStatus() +
                    " hasValue=" + (item.getValue() != null) +
                    " message=" + item.getMessage());
            }
        }

        DiagnosticMetrics.DiagnosticItem<?> diagItemResult = diag.getDiagonalization();
        if (diagItemResult != null && diagItemResult.getValidation() != null) {
            System.out.println("Diag validation normResidual: " + diagItemResult.getValidation().normResidual);
            System.out.println("Diag validation elementResidual: " + diagItemResult.getValidation().elementResidual);
        }
    }

    @Test
    public void testComplexEigenvalueMatrix() {
        // Rotation matrix has complex eigenvalues — should produce a valid 2x2 Schur block
        Matrix A = new Matrix(new double[][]{
            {0, -1, 0},
            {1,  0, 0},
            {0,  0, 3}
        });

        DiagnosticMetrics.MatrixDiagnostics diag = DiagnosticMetrics.analyze(A);
        System.out.println("\n=== 3x3 with complex eigenvalues ===");

        Complex[] eigenvalues = diag.getEigenvalues();
        assertNotNull("Eigenvalues should not be null", eigenvalues);
        assertEquals(3, eigenvalues.length);
        System.out.println("Eigenvalues:");
        for (int i = 0; i < eigenvalues.length; i++) {
            System.out.println("  lambda[" + i + "] = " + eigenvalues[i].real + " + " + eigenvalues[i].imag + "i");
        }

        // Should have eigenvalues ±i and 3
        boolean foundReal3 = false;
        boolean foundComplex = false;
        for (Complex ev : eigenvalues) {
            if (Math.abs(ev.real - 3.0) < 1e-8 && Math.abs(ev.imag) < 1e-8) foundReal3 = true;
            if (Math.abs(ev.imag) > 0.5) foundComplex = true;
        }
        assertTrue("Should find eigenvalue 3", foundReal3);
        assertTrue("Should find complex eigenvalue pair", foundComplex);
    }

    @Test
    public void testLargerMatrices() {
        // 4x4 matrix
        Matrix A4 = new Matrix(new double[][]{
            {2, 1, 0, 0},
            {1, 3, 1, 0},
            {0, 1, 4, 1},
            {0, 0, 1, 5}
        });

        DiagnosticMetrics.MatrixDiagnostics diag4 = DiagnosticMetrics.analyze(A4);
        System.out.println("\n=== 4x4 symmetric matrix ===");
        System.out.println("Diagonalizable: " + diag4.getDiagonalizable());

        int[] geo4 = diag4.getGeometricMultiplicity();
        assertNotNull("Geometric multiplicities should not be null", geo4);
        for (int g : geo4) {
            assertTrue("Geometric multiplicity should be positive, got " + g, g > 0);
        }

        List<DiagnosticMetrics.DiagnosticItem<Set<Vector>>> espaces = diag4.getEigenspaceList();
        assertNotNull(espaces);
        for (int i = 0; i < espaces.size(); i++) {
            assertNotNull("Eigenspace " + i + " should have vectors",
                espaces.get(i).getValue());
            assertFalse("Eigenspace " + i + " should not be empty",
                espaces.get(i).getValue().isEmpty());
        }

        // 5x5 non-symmetric matrix
        Matrix A5 = new Matrix(new double[][]{
            {5, 1, 0, 0, 0},
            {0, 4, 1, 0, 0},
            {0, 0, 3, 1, 0},
            {0, 0, 0, 2, 1},
            {0, 0, 0, 0, 1}
        });

        DiagnosticMetrics.MatrixDiagnostics diag5 = DiagnosticMetrics.analyze(A5);
        System.out.println("\n=== 5x5 upper triangular matrix ===");
        System.out.println("Diagonalizable: " + diag5.getDiagonalizable());

        Complex[] ev5 = diag5.getEigenvalues();
        assertNotNull(ev5);
        assertEquals(5, ev5.length);

        int[] geo5 = diag5.getGeometricMultiplicity();
        assertNotNull(geo5);
        for (int g : geo5) {
            assertTrue("Geometric multiplicity should be positive, got " + g, g > 0);
        }
    }
}
