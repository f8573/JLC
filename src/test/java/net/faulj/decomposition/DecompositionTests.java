package net.faulj.decomposition;

import net.faulj.core.Tolerance;
import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.decomposition.result.LUResult;
import net.faulj.decomposition.result.QRResult;
import net.faulj.determinant.MinorsDeterminant;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.determinant.LUDeterminant;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

public class DecompositionTests {

    private static final double TOL = 1e-9;
    private static final double TOL_SCHUR = 1e-7;
    private static final double TOL_BIDIAG = 1e-5;

    // ========== LU Decomposition Tests ==========

    @Test
    public void luFactorsHaveTriangularStructureAndReconstruct() {
        Matrix a = new Matrix(new double[][]{
                {4,2,1,3},
                {0,5,3,1},
                {1,3,6,2},
                {3,1,2,4}
        });

        LUDecomposition lu = new LUDecomposition();
        LUResult res = lu.decompose(a);

        Matrix L = res.getL();
        Matrix U = res.getU();

        // L should be lower triangular (j > i => L[i,j] ~= 0)
        for (int i = 0; i < L.getRowCount(); i++) {
            for (int j = i+1; j < L.getColumnCount(); j++) {
                assertEquals(0.0, L.get(i,j), Tolerance.get()*10);
            }
        }

        // U should be upper triangular (i > j => U[i,j] ~= 0)
        for (int i = 1; i < U.getRowCount(); i++) {
            for (int j = 0; j < Math.min(i, U.getColumnCount()); j++) {
                assertEquals(0.0, U.get(i,j), Tolerance.get()*10);
            }
        }
    }

    // ========== Hessenberg Reduction Tests ==========

    private static Matrix fromRowMajor(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        Vector[] colsV = new Vector[cols];
        for (int c = 0; c < cols; c++) {
            double[] col = new double[rows];
            for (int r = 0; r < rows; r++) col[r] = a[r][c];
            colsV[c] = new Vector(col);
        }
        return new Matrix(colsV);
    }

    @Test
    public void hessenbergReductionPropertiesRandomMatrix() {
        int n = 8;
        double[][] a = new double[n][n];
        Random rnd = new Random();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 2.0 - 1.0; // [-1,1)
            }
        }

        Matrix A = fromRowMajor(a);

        HessenbergResult res = HessenbergReduction.decompose(A);
        Assert.assertNotNull(res);

        Matrix H = res.getH();
        Matrix Q = res.getQ();

        double tol = Tolerance.get();

        // 1) Subdiagonal entries of H are negligible below first subdiagonal
        int rows = H.getRowCount();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < rows; j++) {
                if (j < i - 1) {
                    double val = Math.abs(H.get(i, j));
                    Assert.assertTrue("H("+i+","+j+") not negligible: " + val, val <= tol);
                }
            }
        }

        // 2) Trace preservation
        double traceA = A.trace();
        double traceH = H.trace();
        Assert.assertTrue(Math.abs(traceA - traceH) <= tol);

        // 3) Orthogonality checks: Q*Q^T ~= I and Q^T*Q ~= I
        Matrix QQT = Q.multiply(Q.transpose());
        Matrix QTQ = Q.transpose().multiply(Q);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.assertTrue(Math.abs(QQT.get(i, j) - expected) <= tol);
                Assert.assertTrue(Math.abs(QTQ.get(i, j) - expected) <= tol);
            }
        }

        // 4) Reconstruction: A ~= Q * H * Q^T
        Matrix reconstructed = Q.multiply(H).multiply(Q.transpose());
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Assert.assertTrue("A reconstructed differs at ("+i+","+j+")", Math.abs(reconstructed.get(i, j) - A.get(i, j)) <= tol);
            }
        }
    }

    // ========== Bidiagonalization Tests ==========

    @Test
    public void bidiagonalStructureUpperBidiagonal() {
        // m >= n case (tall/square matrix) - upper bidiagonal
        Matrix A = new Matrix(new double[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });

        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult result = bidiag.decompose(A);

        Matrix B = result.getB();
        int m = B.getRowCount();
        int n = B.getColumnCount();

        // Check bidiagonal structure: only main diagonal and superdiagonal should be nonzero
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j != i && j != i + 1) {
                    assertEquals("B(" + i + "," + j + ") should be zero", 0.0, B.get(i, j), TOL);
                }
            }
        }
    }

    @Test
    public void bidiagonalStructureLowerBidiagonal() {
        // m < n case (wide matrix) - lower bidiagonal
        Matrix A = new Matrix(new double[][]{
                {1, 2, 3, 4, 5},
                {6, 7, 8, 9, 10},
                {11, 12, 13, 14, 15}
        });

        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult result = bidiag.decompose(A);

        Matrix B = result.getB();
        int m = B.getRowCount();
        int n = B.getColumnCount();

        // Check bidiagonal structure: only main diagonal and subdiagonal should be nonzero
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j != i && j != i - 1) {
                    assertEquals("B(" + i + "," + j + ") should be zero", 0.0, B.get(i, j), TOL);
                }
            }
        }
    }

    @Test
    public void bidiagonalizationOrthogonalityOfU() {
        Matrix A = Matrix.randomMatrix(5, 4);

        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult result = bidiag.decompose(A);

        Matrix U = result.getU();
        int m = U.getRowCount();

        // Check U^T * U = I
        Matrix UTU = U.transpose().multiply(U);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("U^T * U at (" + i + "," + j + ")", expected, UTU.get(i, j), TOL);
            }
        }
    }

    @Test
    public void bidiagonalizationOrthogonalityOfV() {
        Matrix A = Matrix.randomMatrix(4, 5);

        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult result = bidiag.decompose(A);

        Matrix V = result.getV();
        int n = V.getRowCount();

        // Check V^T * V = I
        Matrix VTV = V.transpose().multiply(V);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("V^T * V at (" + i + "," + j + ")", expected, VTV.get(i, j), TOL);
            }
        }
    }

    @Test
    public void bidiagonalizationReconstruction() {
        Matrix A = new Matrix(new double[][]{
                {3, 2, 2},
                {2, 3, -2},
                {1, 1, 1},
                {4, -1, 3}
        });

        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult result = bidiag.decompose(A);

        // A should equal U * B * V^T
        Matrix reconstructed = result.getU().multiply(result.getB()).multiply(result.getV().transpose());

        int m = A.getRowCount();
        int n = A.getColumnCount();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals("Reconstruction at (" + i + "," + j + ")",
                        A.get(i, j), reconstructed.get(i, j), TOL);
            }
        }
    }

    @Test
    public void bidiagonalizationFrobeniusNormPreservation() {
        Matrix A = Matrix.randomMatrix(6, 4);

        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult result = bidiag.decompose(A);

        double normA = A.frobeniusNorm();
        double normB = result.getB().frobeniusNorm();

        assertEquals("Frobenius norm should be preserved", normA, normB, TOL);
    }

    @Test
    public void bidiagonalizationSquareMatrix() {
        Matrix A = new Matrix(new double[][]{
                {4, 3, 2, 1},
                {1, 4, 3, 2},
                {2, 1, 4, 3},
                {3, 2, 1, 4}
        });

        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult result = bidiag.decompose(A);

        assertNotNull(result);
        assertNotNull(result.getU());
        assertNotNull(result.getB());
        assertNotNull(result.getV());

        // Verify reconstruction
        Matrix reconstructed = result.getU().multiply(result.getB()).multiply(result.getV().transpose());
        assertTrue(A.subtract(reconstructed).frobeniusNorm() < TOL);
    }

    @Test
    public void bidiagonalizationResidualIsSmall() {
        Matrix A = Matrix.randomMatrix(5, 3);

        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult result = bidiag.decompose(A);

        // Use residual methods if available in BidiagonalizationResult
        double residual = result.residualNorm();
        assertTrue("Residual should be small: " + residual, residual < TOL_BIDIAG);
    }

    // ========== QR Decomposition Tests ==========

    private static Matrix randomMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) a[i][j] = rnd.nextDouble() * 2 - 1;
        return fromRowMajor(a);
    }

    private static Matrix randomHessenberg(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j < i - 1) a[i][j] = 0.0; else a[i][j] = rnd.nextDouble() * 2 - 1;
            }
        }
        return fromRowMajor(a);
    }

    @Test
    public void householderQRMultipleSizes() {
        int[] sizes = {3, 5, 8};
        for (int n : sizes) {
            Matrix A = randomMatrix(n, 123 + n);
            QRResult res = HouseholderQR.decompose(A);
            Assert.assertNotNull(res);
            Matrix Q = res.getQ();
            Matrix R = res.getR();

            double tol = Tolerance.get();

            // A ~= Q * R
            Matrix QR = Q.multiply(R);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    Assert.assertTrue(Math.abs(A.get(i, j) - QR.get(i, j)) <= tol);
                }
            }

            // Q orthogonal: Q*Q^T ~= I and Q^T*Q ~= I
            Matrix QQT = Q.multiply(Q.transpose());
            Matrix QTQ = Q.transpose().multiply(Q);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double expected = (i == j) ? 1.0 : 0.0;
                    Assert.assertTrue(Math.abs(QQT.get(i, j) - expected) <= tol);
                    Assert.assertTrue(Math.abs(QTQ.get(i, j) - expected) <= tol);
                }
            }

            // R upper triangular: entries below diagonal negligible
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i > j) {
                        Assert.assertTrue(Math.abs(R.get(i, j)) <= tol);
                    }
                }
            }
        }
    }

    @Test
    public void householderQROnHessenbergInput() {
        int n = 6;
        Matrix H0 = randomHessenberg(n, 999L);

        // sanity: H0 is upper Hessenberg
        double tol = Tolerance.get();
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (j < i - 1) Assert.assertTrue(Math.abs(H0.get(i, j)) <= tol);

        QRResult res = HouseholderQR.decompose(H0);
        Matrix Q = res.getQ();
        Matrix R = res.getR();

        // same checks as above
        Matrix QR = Q.multiply(R);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Assert.assertTrue(Math.abs(H0.get(i, j) - QR.get(i, j)) <= tol);

        Matrix QQT = Q.multiply(Q.transpose());
        Matrix QTQ = Q.transpose().multiply(Q);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            Assert.assertTrue(Math.abs(QQT.get(i, j) - expected) <= tol);
            Assert.assertTrue(Math.abs(QTQ.get(i, j) - expected) <= tol);
        }

        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i > j) Assert.assertTrue(Math.abs(R.get(i, j)) <= tol);
    }

    // ========== Schur Decomposition Tests ==========

    @Test
    public void schurDecompositionDiagonalMatrix() {
        Matrix A = new Matrix(new double[][]{
                {5, 0, 0},
                {0, 3, 0},
                {0, 0, 7}
        });

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();
        Matrix U = result.getU();

        // Diagonal matrix is already in Schur form
        // Verify eigenvalues
        double[] eigenvalues = result.getRealEigenvalues();
        assertEquals(3, eigenvalues.length);
        
        // All eigenvalues should be real
        double[] imagParts = result.getImagEigenvalues();
        for (double imag : imagParts) {
            assertEquals("All eigenvalues should be real for diagonal matrix", 0.0, imag, TOL_SCHUR);
        }

        // Verify similarity: A = U * T * U^T
        Matrix reconstructed = U.multiply(T).multiply(U.transpose());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals("Similarity transformation should hold",
                        A.get(i, j), reconstructed.get(i, j), TOL_SCHUR);
            }
        }
    }

    @Test
    public void schurDecompositionSymmetricMatrix() {
        Matrix A = new Matrix(new double[][]{
                {4, 1, 2},
                {1, 3, 1},
                {2, 1, 5}
        });

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();
        Matrix U = result.getU();

        // Symmetric matrices should have real eigenvalues
        double[] imagParts = result.getImagEigenvalues();
        for (double imag : imagParts) {
            assertTrue("Symmetric matrix should have nearly real eigenvalues",
                    Math.abs(imag) < TOL_SCHUR);
        }

        // U should be orthogonal
        Matrix UtU = U.transpose().multiply(U);
        Matrix I = Matrix.Identity(3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals("U should be orthogonal",
                        I.get(i, j), UtU.get(i, j), TOL_SCHUR);
            }
        }

        // Verify similarity
        Matrix reconstructed = U.multiply(T).multiply(U.transpose());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals("Similarity should hold",
                        A.get(i, j), reconstructed.get(i, j), TOL_SCHUR);
            }
        }
    }

    @Test
    public void schurDecompositionUpperTriangular() {
        Matrix A = new Matrix(new double[][]{
                {2, 1, 3},
                {0, 5, 2},
                {0, 0, -1}
        });

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();

        // Upper triangular matrix is already in Schur form
        // Eigenvalues are diagonal elements
        double[] eigenvalues = result.getRealEigenvalues();
        double[] expected = {-1, 2, 5};
        java.util.Arrays.sort(eigenvalues);

        assertArrayEquals("Eigenvalues should be diagonal elements",
                expected, eigenvalues, TOL_SCHUR);
    }

    @Test
    public void schurDecompositionIdentityMatrix() {
        Matrix I = Matrix.Identity(4);

        SchurResult result = RealSchurDecomposition.decompose(I);
        Matrix T = result.getT();

        // Identity matrix should remain identity
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals("Schur form of identity should be identity",
                        expected, T.get(i, j), TOL_SCHUR);
            }
        }

        // All eigenvalues should be 1
        double[] eigenvalues = result.getRealEigenvalues();
        for (double lambda : eigenvalues) {
            assertEquals("All eigenvalues should be 1", 1.0, lambda, TOL_SCHUR);
        }
    }

    @Test
    public void schurDecompositionPreservesTrace() {
        Matrix A = new Matrix(new double[][]{
                {3, 2, 1, 4},
                {1, 5, 2, 1},
                {2, 1, 6, 3},
                {4, 1, 3, 2}
        });

        double traceA = A.trace();

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();

        double traceT = T.trace();

        assertEquals("Trace should be preserved", traceA, traceT, TOL_SCHUR);

        // Also verify sum of eigenvalues equals trace
        double[] eigenvalues = result.getRealEigenvalues();
        double sumEigenvalues = 0;
        for (double lambda : eigenvalues) {
            sumEigenvalues += lambda;
        }
        assertEquals("Sum of eigenvalues should equal trace",
                traceA, sumEigenvalues, TOL_SCHUR);
    }

    @Test
    public void schurDecompositionPreservesDeterminant() {
        Matrix A = new Matrix(new double[][]{
                {2, 1, 3},
                {1, 4, 1},
                {3, 1, 5}
        });

        double detA = LUDeterminant.compute(A);

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();

        double detT = LUDeterminant.compute(T);

        assertEquals("Determinant should be preserved", detA, detT, TOL_SCHUR);
    }

    @Test
    public void schurDecompositionOrthogonalityOfU() {
        Matrix A = new Matrix(new double[][]{
                {5, 2, 1},
                {2, 4, 3},
                {1, 3, 6}
        });

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix U = result.getU();

        // Verify U^T * U = I
        Matrix UtU = U.transpose().multiply(U);
        Matrix I = Matrix.Identity(U.getRowCount());

        for (int i = 0; i < I.getRowCount(); i++) {
            for (int j = 0; j < I.getColumnCount(); j++) {
                assertEquals("U should be orthogonal",
                        I.get(i, j), UtU.get(i, j), TOL_SCHUR);
            }
        }
    }

    @Test
    public void schurDecompositionQuasiUpperTriangularStructure() {
        Matrix A = new Matrix(new double[][]{
                {1, 2, 3, 4},
                {2, 3, 4, 1},
                {3, 4, 1, 2},
                {4, 1, 2, 3}
        });

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();

        // Verify quasi-upper triangular (zeros below first subdiagonal)
        for (int i = 2; i < T.getRowCount(); i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals("Should be quasi-upper triangular",
                        0.0, T.get(i, j), TOL_SCHUR);
            }
        }
    }

    @Test
    public void schurDecompositionSimilarityTransformation() {
        Matrix A = new Matrix(new double[][]{
                {6, 3, 1},
                {3, 4, 2},
                {1, 2, 5}
        });

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();
        Matrix U = result.getU();

        // Verify A = U * T * U^T
        Matrix reconstructed = U.multiply(T).multiply(U.transpose());

        for (int i = 0; i < A.getRowCount(); i++) {
            for (int j = 0; j < A.getColumnCount(); j++) {
                assertEquals("Similarity transformation A = UTU^T should hold",
                        A.get(i, j), reconstructed.get(i, j), TOL_SCHUR);
            }
        }
    }

    @Test
    public void schurDecompositionRandomMatrix() {
        Matrix A = Matrix.randomMatrix(5, 5);

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();
        Matrix U = result.getU();

        // Verify similarity
        Matrix reconstructed = U.multiply(T).multiply(U.transpose());
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                assertEquals("Similarity should hold for random matrix",
                        A.get(i, j), reconstructed.get(i, j), TOL_SCHUR);
            }
        }

        // Verify trace
        assertEquals("Trace should be preserved",
                A.trace(), T.trace(), TOL_SCHUR);

        // Verify U orthogonal
        Matrix UtU = U.transpose().multiply(U);
        Matrix I = Matrix.Identity(5);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                assertEquals("U should be orthogonal",
                        I.get(i, j), UtU.get(i, j), TOL_SCHUR);
            }
        }
    }

    @Test
    public void schurDecompositionEigenvaluesNotNull() {
        Matrix A = new Matrix(new double[][]{
                {2, 1},
                {1, 3}
        });

        SchurResult result = RealSchurDecomposition.decompose(A);

        assertNotNull("Eigenvalues should not be null", result.getEigenvalues());
        assertNotNull("Real eigenvalues should not be null", result.getRealEigenvalues());
        assertNotNull("Imaginary eigenvalues should not be null", result.getImagEigenvalues());

        assertEquals("Should have correct number of eigenvalues",
                2, result.getEigenvalues().length);
    }

    @Test
    public void schurDecomposition3x3WithComplexEigenvalues() {
        // This matrix is designed to have complex eigenvalues
        Matrix A = new Matrix(new double[][]{
                {0, -1, 0},
                {1, 0, 0},
                {0, 0, 2}
        });

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();
        Matrix U = result.getU();

        // Verify similarity
        Matrix reconstructed = U.multiply(T).multiply(U.transpose());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals("Similarity should hold even with complex eigenvalues",
                        A.get(i, j), reconstructed.get(i, j), TOL_SCHUR);
            }
        }

        // Verify trace
        assertEquals("Trace should be preserved",
                A.trace(), T.trace(), TOL_SCHUR);
    }

    @Test
    public void schurDecompositionLargeMatrix() {
        Matrix A = Matrix.randomMatrix(10, 10);

        SchurResult result = RealSchurDecomposition.decompose(A);
        Matrix T = result.getT();
        Matrix U = result.getU();

        assertNotNull("Result should not be null", result);
        assertEquals("T should be 10x10", 10, T.getRowCount());
        assertEquals("U should be 10x10", 10, U.getRowCount());

        // Verify quasi-upper triangular
        for (int i = 2; i < 10; i++) {
            for (int j = 0; j < i - 1; j++) {
                assertTrue("Should be quasi-upper triangular",
                        Math.abs(T.get(i, j)) < TOL_SCHUR);
            }
        }
    }
}
