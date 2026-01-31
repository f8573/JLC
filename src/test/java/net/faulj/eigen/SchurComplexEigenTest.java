package net.faulj.eigen;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;
import net.faulj.scalar.Complex;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Unit test for Schur decomposition and complex eigenvector extraction
 * using the specific matrix [[0,-1,0],[1,0,0],[0,0,1]].
 */
public class SchurComplexEigenTest {

    private static Matrix fromRowMajor(double[][] a) {
        // Matrix has a constructor that accepts row-major 2D arrays.
        return new Matrix(a);
    }

    @Test
    public void testSchurComplexEigenvectorsNotZero() {
        double[][] a = new double[][]{
                {0.0, -1.0, 0.0},
                {1.0,  0.0, 0.0},
                {0.0,  0.0, 1.0}
        };

        Matrix A = fromRowMajor(a);
        SchurResult schur = RealSchurDecomposition.decompose(A);
        assertNotNull("Schur result should not be null", schur);

        Complex[] eig = schur.getEigenvalues();
        assertNotNull("Eigenvalues should not be null", eig);
        assertEquals(3, eig.length);

        int idxOne = -1;
        int idxPosImag = -1;
        int idxNegImag = -1;
        double tol = 1e-8;
        for (int i = 0; i < eig.length; i++) {
            if (Math.abs(eig[i].real - 1.0) < tol && Math.abs(eig[i].imag) < tol) {
                idxOne = i;
            }
            if (Math.abs(eig[i].real) < tol && Math.abs(Math.abs(eig[i].imag) - 1.0) < 1e-6) {
                if (eig[i].imag > 0) idxPosImag = i;
                else idxNegImag = i;
            }
        }

        assertTrue("Should find eigenvalue 1.0", idxOne >= 0);
        assertTrue("Should find complex conjugate eigenpair", idxPosImag >= 0 && idxNegImag >= 0);

        Matrix V = schur.getEigenvectors();
        assertNotNull("Eigenvector matrix should not be null", V);
        assertEquals(3, V.getRowCount());
        assertEquals(3, V.getColumnCount());

        // Check real eigenvector corresponding to eigenvalue 1 is concentrated in third component
        double v20 = Math.abs(V.get(0, idxOne));
        double v21 = Math.abs(V.get(1, idxOne));
        double v22 = Math.abs(V.get(2, idxOne));
        double norm = Math.sqrt(v20*v20 + v21*v21 + v22*v22);
        assertTrue("Real eigenvector for eigenvalue 1 should have non-zero norm", norm > tol);
        // third component should carry most of the weight (expected ~ [0,0,1])
        assertTrue("Third component should be dominant for eigenvalue 1", Math.abs(v22) > 0.8 * norm);

        // For complex pair, ensure eigenvectors include imaginary parts and are not zero
        assertTrue("Eigenvectors should have imaginary data for complex pair", V.hasImagData());

        // Check positive-imag partner column has non-zero complex vector
        double complexNormSq = 0.0;
        for (int r = 0; r < 3; r++) {
            double re = V.get(r, idxPosImag);
            double im = V.getImag(r, idxPosImag);
            complexNormSq += re*re + im*im;
        }
        double complexNorm = Math.sqrt(complexNormSq);
        assertTrue("Complex eigenvector should not be all zeros", complexNorm > tol);
    }
}
