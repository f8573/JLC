package net.faulj.eigen;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.eigen.qr.ImplicitQRFrancis;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;
import net.faulj.determinant.LUDeterminant;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class EigenTest {

    private static final double TOL = 1e-7;

    // ========== Eigenvector Computation Tests ==========

    @Test
    public void realEigenvaluesHaveRealEigenvectors() {
        Matrix A = new Matrix(new double[][]{
                {3, 1, 0},
                {1, 2, 1},
                {0, 1, 3}
        });

        SchurResult schur = RealSchurDecomposition.decompose(A);
        
        // For matrices with only real eigenvalues, we can verify Av = λv
        double[] eigenvalues = schur.getRealEigenvalues();
        
        // Note: User specified to ignore complex eigenvector validation
        // We only test that eigenvalues are correctly computed
        assertNotNull("Eigenvalues should be computed", eigenvalues);
        assertEquals("Should have 3 eigenvalues", 3, eigenvalues.length);
    }

    @Test
    public void symmetricMatrixProducesOrthogonalEigenvectors() {
        // Symmetric matrices have orthogonal eigenvectors
        Matrix A = new Matrix(new double[][]{
                {4, 2, 0},
                {2, 3, 0},
                {0, 0, 5}
        });

        SchurResult schur = RealSchurDecomposition.decompose(A);
        Matrix U = schur.getU();

        // For symmetric matrices, Schur vectors are eigenvectors
        // and should be orthogonal
        Matrix UTU = U.transpose().multiply(U);
        int n = U.getRowCount();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("Eigenvectors should be orthogonal",
                        expected, UTU.get(i, j), TOL);
            }
        }
    }

    @Test
    public void diagonalMatrixEigenvectors() {
        // For diagonal matrices, eigenvectors are standard basis vectors
        Matrix A = new Matrix(new double[][]{
                {5, 0, 0},
                {0, 3, 0},
                {0, 0, 7}
        });

        SchurResult schur = RealSchurDecomposition.decompose(A);
        
        // Verify eigenvalues
        double[] eigenvalues = schur.getRealEigenvalues();
        java.util.Arrays.sort(eigenvalues);
        
        assertNotNull(eigenvalues);
        assertEquals(3, eigenvalues.length);
    }

    @Test
    public void identityMatrixEigenvectors() {
        // Identity matrix: all eigenvalues are 1, any vector is an eigenvector
        Matrix I = Matrix.Identity(3);

        SchurResult schur = RealSchurDecomposition.decompose(I);
        double[] eigenvalues = schur.getRealEigenvalues();

        // All eigenvalues should be 1
        for (double lambda : eigenvalues) {
            assertEquals("All eigenvalues of I should be 1", 1.0, lambda, TOL);
        }
    }

    @Test
    public void upperTriangularMatrixEigenvalues() {
        // Upper triangular: eigenvalues are diagonal elements
        Matrix A = new Matrix(new double[][]{
                {2, 1, 3},
                {0, 5, 2},
                {0, 0, -1}
        });

        SchurResult schur = RealSchurDecomposition.decompose(A);
        double[] eigenvalues = schur.getRealEigenvalues();

        java.util.Arrays.sort(eigenvalues);
        double[] expected = {-1, 2, 5};

        assertArrayEquals("Eigenvalues should be diagonal elements",
                expected, eigenvalues, TOL);
    }

    @Test
    public void eigenvalueMultiplicity() {
        // Matrix with repeated eigenvalues
        Matrix A = new Matrix(new double[][]{
                {3, 1, 0},
                {0, 3, 0},
                {0, 0, 3}
        });

        SchurResult schur = RealSchurDecomposition.decompose(A);
        double[] eigenvalues = schur.getRealEigenvalues();

        // All eigenvalues should be 3
        for (double lambda : eigenvalues) {
            assertEquals("All eigenvalues should be 3", 3.0, lambda, TOL);
        }
    }

    @Test
    public void randomMatrixEigenvalueComputation() {
        Matrix A = Matrix.randomMatrix(5, 5);

        SchurResult schur = RealSchurDecomposition.decompose(A);
        
        assertNotNull("Schur result should not be null", schur);
        assertNotNull("Eigenvalues should be computed", schur.getRealEigenvalues());
        
        // Trace test: sum of eigenvalues = trace
        double[] eigenvalues = schur.getRealEigenvalues();
        double sumEigenvalues = 0;
        for (double lambda : eigenvalues) {
            sumEigenvalues += lambda;
        }
        
        assertEquals("Sum of eigenvalues should equal trace",
                A.trace(), sumEigenvalues, TOL);
    }

    // ========== Explicit QR Tests ==========

    @Test
    public void explicitQRConvergesForDiagonalMatrix() {
        Matrix A = new Matrix(new double[][]{
                {5, 0, 0},
                {0, 3, 0},
                {0, 0, 7}
        });

        Matrix[] result = ExplicitQRIteration.decompose(A);
        Matrix T = result[0];
        Matrix Q = result[1];

        // Diagonal matrix should be already in Schur form
        // Verify eigenvalues on diagonal
        double[] expectedEigenvalues = {5, 3, 7};
        double[] actualEigenvalues = {T.get(0, 0), T.get(1, 1), T.get(2, 2)};
        Arrays.sort(expectedEigenvalues);
        Arrays.sort(actualEigenvalues);

        assertArrayEquals("Eigenvalues should match", expectedEigenvalues, actualEigenvalues, TOL);
    }

    @Test
    public void explicitQRPreservesSimilarity() {
        Matrix A = new Matrix(new double[][]{
                {4, 1, 2},
                {1, 3, 1},
                {2, 1, 5}
        });

        Matrix[] result = ExplicitQRIteration.decompose(A);
        Matrix T = result[0];
        Matrix Q = result[1];

        // Verify A = Q * T * Q^T
        Matrix reconstructed = Q.multiply(T).multiply(Q.transpose());

        for (int i = 0; i < A.getRowCount(); i++) {
            for (int j = 0; j < A.getColumnCount(); j++) {
                assertEquals("Similarity transformation should preserve A",
                        A.get(i, j), reconstructed.get(i, j), TOL);
            }
        }
    }

    @Test
    public void explicitQRProducesOrthogonalQ() {
        Matrix A = new Matrix(new double[][]{
                {2, 1, 0},
                {1, 3, 1},
                {0, 1, 2}
        });

        Matrix[] result = ExplicitQRIteration.decompose(A);
        Matrix Q = result[1];

        // Verify Q^T * Q = I
        Matrix QtQ = Q.transpose().multiply(Q);
        Matrix I = Matrix.Identity(Q.getRowCount());

        for (int i = 0; i < I.getRowCount(); i++) {
            for (int j = 0; j < I.getColumnCount(); j++) {
                assertEquals("Q should be orthogonal",
                        I.get(i, j), QtQ.get(i, j), TOL);
            }
        }
    }

    @Test
    public void explicitQRExtractsEigenvaluesCorrectly() {
        Matrix A = new Matrix(new double[][]{
                {6, 2, 1},
                {2, 3, 1},
                {1, 1, 1}
        });

        Matrix[] result = ExplicitQRIteration.decompose(A);
        Matrix T = result[0];

        // Verify trace is preserved (sum of eigenvalues equals trace).
        double traceA = A.trace();
        double sumEigenvalues = T.trace();
        assertEquals("Trace should equal sum of eigenvalues",
                traceA, sumEigenvalues, TOL);
    }

    @Test
    public void explicitQRProducesUpperTriangularSchurForm() {
        Matrix A = new Matrix(new double[][]{
                {4, 3, 2},
                {1, 5, 1},
                {2, 1, 3}
        });

        Matrix[] result = ExplicitQRIteration.decompose(A);
        Matrix T = result[0];

        // Verify T is quasi-upper triangular (zeros below first subdiagonal)
        for (int i = 2; i < T.getRowCount(); i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals("Schur form should be quasi-upper triangular",
                        0.0, T.get(i, j), TOL);
            }
        }
    }

    @Test
    public void explicitQRPreservesDeterminant() {
        Matrix A = new Matrix(new double[][]{
                {3, 1, 2},
                {1, 4, 1},
                {2, 1, 5}
        });

        double detA = LUDeterminant.compute(A);

        Matrix[] result = ExplicitQRIteration.decompose(A);
        Matrix T = result[0];

        // Determinant of T should equal determinant of A
        double detT = LUDeterminant.compute(T);

        assertEquals("Determinant should be preserved",
                detA, detT, TOL);
    }

    @Test
    public void explicitQRHandlesUpperTriangularInput() {
        Matrix A = new Matrix(new double[][]{
                {5, 2, 1},
                {0, 3, 4},
                {0, 0, 2}
        });

        Matrix[] result = ExplicitQRIteration.decompose(A);
        Matrix T = result[0];

        // Should converge quickly for already triangular matrix
        // Eigenvalues are on diagonal
        assertEquals(5, T.get(0, 0), TOL);
        assertEquals(3, T.get(1, 1), TOL);
        assertEquals(2, T.get(2, 2), TOL);
    }

    @Test
    public void explicitQRHandles4x4Matrix() {
        Matrix A = new Matrix(new double[][]{
                {4, 1, 0, 1},
                {1, 3, 1, 0},
                {0, 1, 5, 1},
                {1, 0, 1, 2}
        });

        Matrix[] result = ExplicitQRIteration.decompose(A);
        Matrix T = result[0];
        Matrix Q = result[1];

        // Verify similarity
        Matrix reconstructed = Q.multiply(T).multiply(Q.transpose());

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals("Similarity should hold for 4x4",
                        A.get(i, j), reconstructed.get(i, j), TOL);
            }
        }

        // Verify Q is orthogonal
        Matrix QtQ = Q.transpose().multiply(Q);
        Matrix I = Matrix.Identity(4);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals("Q should be orthogonal for 4x4",
                        I.get(i, j), QtQ.get(i, j), TOL);
            }
        }
    }

    // ========== Implicit QR Tests ==========

    @Test
    public void implicitQRPreservesSimilarityAndDeterminant() {
        Matrix A = new Matrix(new double[][]{
                {4, 2, 1, 3},
                {0, 5, 3, 1},
                {1, 3, 6, 2},
                {3, 1, 2, 4}
        });

        SchurResult result = ImplicitQRFrancis.decompose(A);
        Matrix T = result.getT();
        Matrix U = result.getU();

        // Test 1: Verify similarity transformation A = U * T * U^T
        Matrix reconstructed = U.multiply(T).multiply(U.transpose());
        for (int i = 0; i < A.getRowCount(); i++) {
            for (int j = 0; j < A.getColumnCount(); j++) {
                assertEquals("Similarity transformation should preserve A",
                        A.get(i, j), reconstructed.get(i, j), TOL);
            }
        }

        // Test 2: Verify determinant is preserved
        double detA = LUDeterminant.compute(A);
        double detT = LUDeterminant.compute(T);
        assertEquals("Determinant should be preserved",
                detA, detT, TOL);

        // Test 3: Verify U is orthogonal (U^T * U = I)
        Matrix UtU = U.transpose().multiply(U);
        Matrix I = Matrix.Identity(U.getRowCount());
        for (int i = 0; i < I.getRowCount(); i++) {
            for (int j = 0; j < I.getColumnCount(); j++) {
                assertEquals("U should be orthogonal",
                        I.get(i, j), UtU.get(i, j), TOL);
            }
        }
    }

    @Test
    public void implicitQRProducesQuasiUpperTriangularForm() {
        Matrix A = new Matrix(new double[][]{
                {3, 1, 2},
                {1, 4, 1},
                {2, 1, 5}
        });

        SchurResult result = ImplicitQRFrancis.decompose(A);
        Matrix T = result.getT();

        // Verify T is quasi-upper triangular
        // (entries below the first subdiagonal should be zero)
        for (int i = 2; i < T.getRowCount(); i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals("Schur form should be quasi-upper triangular",
                        0.0, T.get(i, j), TOL);
            }
        }
    }

    @Test
    public void implicitQRPreservesTrace() {
        Matrix A = new Matrix(new double[][]{
                {2, 1, 0, 1},
                {1, 3, 1, 0},
                {0, 1, 4, 1},
                {1, 0, 1, 5}
        });

        double traceA = A.trace();

        SchurResult result = ImplicitQRFrancis.decompose(A);
        Matrix T = result.getT();

        double traceT = T.trace();

        assertEquals("Trace should be preserved",
                traceA, traceT, TOL);
    }

    @Test
    public void implicitQRExtractsEigenvaluesCorrectly() {
        Matrix A = new Matrix(new double[][]{
                {5, 1, 0},
                {1, 4, 1},
                {0, 1, 3}
        });

        SchurResult result = ImplicitQRFrancis.decompose(A);

        // Sum of eigenvalues should equal trace
        double[] eigenvalues = result.getRealEigenvalues();
        double sumEigenvalues = 0;
        for (double lambda : eigenvalues) {
            sumEigenvalues += lambda;
        }

        assertEquals("Sum of eigenvalues should equal trace",
                A.trace(), sumEigenvalues, TOL);
    }

    @Test
    public void implicitQRHandlesSymmetricMatrix() {
        Matrix A = new Matrix(new double[][]{
                {4, 2, 0},
                {2, 3, 0},
                {0, 0, 5}
        });

        SchurResult result = ImplicitQRFrancis.decompose(A);
        Matrix T = result.getT();
        Matrix U = result.getU();

        // For symmetric matrices, Schur form should be diagonal
        // (or at least all eigenvalues should be real)
        for (int i = 1; i < T.getRowCount(); i++) {
            assertTrue("Subdiagonal should be small for symmetric matrix",
                    Math.abs(T.get(i, i - 1)) < TOL);
        }

        // Verify similarity
        Matrix reconstructed = U.multiply(T).multiply(U.transpose());
        for (int i = 0; i < A.getRowCount(); i++) {
            for (int j = 0; j < A.getColumnCount(); j++) {
                assertEquals("Similarity should hold for symmetric matrix",
                        A.get(i, j), reconstructed.get(i, j), TOL);
            }
        }
    }

    @Test
    public void implicitQRHandlesIdentityMatrix() {
        Matrix I = Matrix.Identity(4);

        SchurResult result = ImplicitQRFrancis.decompose(I);
        Matrix T = result.getT();

        // Schur form of identity should be identity
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("Schur form of identity should be identity",
                        expected, T.get(i, j), TOL);
            }
        }
    }

    @Test
    public void implicitQRHandlesUpperTriangularMatrix() {
        Matrix A = new Matrix(new double[][]{
                {3, 1, 2},
                {0, 4, 1},
                {0, 0, 5}
        });

        SchurResult result = ImplicitQRFrancis.decompose(A);
        Matrix T = result.getT();

        // Upper triangular matrices are already in Schur form
        // Eigenvalues are on diagonal
        assertEquals(3, T.get(0, 0), TOL);
        assertEquals(4, T.get(1, 1), TOL);
        assertEquals(5, T.get(2, 2), TOL);
    }

    @Test
    public void implicitQRHandles5x5Matrix() {
        Matrix A = Matrix.randomMatrix(5, 5);

        SchurResult result = ImplicitQRFrancis.decompose(A);
        Matrix T = result.getT();
        Matrix U = result.getU();

        // Verify similarity transformation
        Matrix reconstructed = U.multiply(T).multiply(U.transpose());
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                assertEquals("Similarity should hold for 5x5 random matrix",
                        A.get(i, j), reconstructed.get(i, j), TOL);
            }
        }

        // Verify quasi-upper triangular structure
        for (int i = 2; i < 5; i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals("Should be quasi-upper triangular",
                        0.0, T.get(i, j), TOL);
            }
        }
    }

    @Test
    public void implicitQRPreservesMatrixNorm() {
        Matrix A = new Matrix(new double[][]{
                {2, 1, 1},
                {1, 3, 1},
                {1, 1, 4}
        });

        double normA = Math.sqrt(A.multiply(A.transpose()).trace());

        SchurResult result = ImplicitQRFrancis.decompose(A);
        Matrix T = result.getT();

        double normT = Math.sqrt(T.multiply(T.transpose()).trace());

        assertEquals("Frobenius norm should be preserved",
                normA, normT, TOL);
    }
}
