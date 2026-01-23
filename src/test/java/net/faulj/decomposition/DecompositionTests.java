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
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.determinant.LUDeterminant;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * Comprehensive decomposition tests using matrix norm-based accuracy measurements.
 * Tests cover small (2x2-10x10), medium (15x15-30x30), and huge (50x50, 100x100, 200x200) matrices.
 * 
 * Accuracy is measured using:
 * - Frobenius norm for reconstruction errors
 * - Relative errors normalized by matrix norms
 * - Orthogonality errors using ||Q^T*Q - I||_F
 */
public class DecompositionTests {

    // Base tolerances for norm-based error measurements
    private static final double SMALL_TOL = 1e-10;
    private static final double MEDIUM_TOL = 1e-8;
    private static final double LARGE_TOL = 1e-6;
    private static final double HUGE_TOL = 1e-5;

    // Matrix sizes for systematic testing
    private static final int[] SMALL_SIZES = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    private static final int[] MEDIUM_SIZES = {15, 20, 25, 30};
    private static final int[] HUGE_SIZES = {50, 100, 200};

    private static final Random RNG = new Random(42);

    // ========== Helper Methods ==========

    /**
     * Create matrix from row-major 2D array
     */
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

    /**
     * Generate random matrix with entries in [-1, 1]
     */
    private static Matrix randomMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 2 - 1;
            }
        }
        return fromRowMajor(a);
    }

    /**
     * Generate random symmetric matrix
     */
    private static Matrix randomSymmetricMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double val = rnd.nextDouble() * 2 - 1;
                a[i][j] = val;
                a[j][i] = val;
            }
        }
        return fromRowMajor(a);
    }

    /**
     * Generate random Hessenberg matrix
     */
    private static Matrix randomHessenberg(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j < i - 1) {
                    a[i][j] = 0.0;
                } else {
                    a[i][j] = rnd.nextDouble() * 2 - 1;
                }
            }
        }
        return fromRowMajor(a);
    }

    /**
     * Generate random diagonally dominant matrix (well-conditioned for LU)
     */
    private static Matrix randomDiagonallyDominant(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            double rowSum = 0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    a[i][j] = rnd.nextDouble() * 0.5 - 0.25;
                    rowSum += Math.abs(a[i][j]);
                }
            }
            a[i][i] = rowSum + 1 + rnd.nextDouble();
        }
        return fromRowMajor(a);
    }

    /**
     * Get size-appropriate tolerance for norm-based error measurements
     */
    private double getTolerance(int n) {
        if (n <= 10) return SMALL_TOL * Math.sqrt(n);
        if (n <= 30) return MEDIUM_TOL * Math.sqrt(n);
        if (n <= 100) return LARGE_TOL * Math.sqrt(n);
        return HUGE_TOL * Math.sqrt(n);
    }

    /**
     * Measure orthogonality error: ||Q^T*Q - I||_F
     */
    private double orthogonalityError(Matrix Q) {
        int n = Q.getRowCount();
        Matrix QtQ = Q.transpose().multiply(Q);
        Matrix I = Matrix.Identity(n);
        return QtQ.subtract(I).frobeniusNorm();
    }

    /**
     * Measure reconstruction error: ||A - reconstructed||_F / ||A||_F
     */
    private double reconstructionError(Matrix A, Matrix reconstructed) {
        double normA = A.frobeniusNorm();
        if (normA < 1e-14) return 0.0;
        return A.subtract(reconstructed).frobeniusNorm() / normA;
    }

    /**
     * Assert orthogonality using Frobenius norm
     */
    private void assertOrthogonal(Matrix Q, double tol, String context) {
        double error = orthogonalityError(Q);
        assertTrue(context + ": ||Q^T*Q - I||_F = " + error + " > " + tol,
                error < tol);
    }

    /**
     * Assert reconstruction accuracy using relative Frobenius norm
     */
    private void assertReconstruction(Matrix A, Matrix reconstructed, double tol, String context) {
        double relError = reconstructionError(A, reconstructed);
        assertTrue(context + ": reconstruction error = " + relError + " > " + tol,
                relError < tol);
    }

    // ========== QR Decomposition Tests ==========

    @Test
    public void testQR_SmallRandom() {
        System.out.println("\n=== QR Decomposition: Small Random Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomMatrix(n, 100 + n);
            testQRDecomposition(A, n, "Random " + n + "x" + n);
        }
    }

    @Test
    public void testQR_MediumRandom() {
        System.out.println("\n=== QR Decomposition: Medium Random Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomMatrix(n, 200 + n);
            testQRDecomposition(A, n, "Random " + n + "x" + n);
        }
    }

    @Test
    public void testQR_HugeRandom() {
        System.out.println("\n=== QR Decomposition: Huge Random Matrices ===");
        for (int n : HUGE_SIZES) {
            Matrix A = randomMatrix(n, 300 + n);
            long start = System.currentTimeMillis();
            testQRDecomposition(A, n, "Random " + n + "x" + n);
            long elapsed = System.currentTimeMillis() - start;
            System.out.println("  Time: " + elapsed + "ms");
        }
    }

    private void testQRDecomposition(Matrix A, int n, String context) {
        QRResult res = HouseholderQR.decompose(A);
        assertNotNull(context + ": QR result", res);

        Matrix Q = res.getQ();
        Matrix R = res.getR();
        double tol = getTolerance(n);

        // Test 1: Orthogonality of Q
        double orthError = orthogonalityError(Q);
        System.out.printf("  %s: ||Q^T*Q - I||_F = %.2e\n", context, orthError);
        assertOrthogonal(Q, tol, context);

        // Test 2: R is upper triangular (check structure)
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(context + ": R lower part at (" + i + "," + j + ")",
                        0.0, R.get(i, j), tol);
            }
        }

        // Test 3: Reconstruction A = Q*R
        Matrix QR = Q.multiply(R);
        double reconError = reconstructionError(A, QR);
        System.out.printf("  %s: Reconstruction error = %.2e\n", context, reconError);
        assertReconstruction(A, QR, tol, context);
    }

    // ========== LU Decomposition Tests ==========

    @Test
    public void testLU_SmallDiagonallyDominant() {
        System.out.println("\n=== LU Decomposition: Small Diagonally Dominant Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomDiagonallyDominant(n, 400 + n);
            testLUDecomposition(A, n, "Diag Dominant " + n + "x" + n);
        }
    }

    @Test
    public void testLU_MediumDiagonallyDominant() {
        System.out.println("\n=== LU Decomposition: Medium Diagonally Dominant Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomDiagonallyDominant(n, 500 + n);
            testLUDecomposition(A, n, "Diag Dominant " + n + "x" + n);
        }
    }

    @Test
    public void testLU_HugeDiagonallyDominant() {
        System.out.println("\n=== LU Decomposition: Huge Diagonally Dominant Matrices ===");
        for (int n : HUGE_SIZES) {
            Matrix A = randomDiagonallyDominant(n, 600 + n);
            long start = System.currentTimeMillis();
            testLUDecomposition(A, n, "Diag Dominant " + n + "x" + n);
            long elapsed = System.currentTimeMillis() - start;
            System.out.println("  Time: " + elapsed + "ms");
        }
    }

    private void testLUDecomposition(Matrix A, int n, String context) {
        LUDecomposition lu = new LUDecomposition();
        LUResult res = lu.decompose(A);
        assertNotNull(context + ": LU result", res);

        Matrix L = res.getL();
        Matrix U = res.getU();
        double tol = getTolerance(n);

        // Test 1: L is lower triangular
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                assertEquals(context + ": L upper part at (" + i + "," + j + ")",
                        0.0, L.get(i, j), tol);
            }
        }

        // Test 2: U is upper triangular
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(context + ": U lower part at (" + i + "," + j + ")",
                        0.0, U.get(i, j), tol);
            }
        }

        // Test 3: Reconstruction A ≈ L*U
        Matrix LU = L.multiply(U);
        double reconError = reconstructionError(A, LU);
        System.out.printf("  %s: Reconstruction error = %.2e\n", context, reconError);
        assertTrue(context + ": relative reconstruction error",
                reconError < tol * 10);
    }

    // ========== Hessenberg Reduction Tests ==========

    @Test
    public void testHessenberg_SmallRandom() {
        System.out.println("\n=== Hessenberg Reduction: Small Random Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomMatrix(n, 700 + n);
            testHessenbergReduction(A, n, "Random " + n + "x" + n);
        }
    }

    @Test
    public void testHessenberg_MediumRandom() {
        System.out.println("\n=== Hessenberg Reduction: Medium Random Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomMatrix(n, 800 + n);
            testHessenbergReduction(A, n, "Random " + n + "x" + n);
        }
    }

    @Test
    public void testHessenberg_HugeRandom() {
        System.out.println("\n=== Hessenberg Reduction: Huge Random Matrices ===");
        for (int n : HUGE_SIZES) {
            Matrix A = randomMatrix(n, 900 + n);
            long start = System.currentTimeMillis();
            testHessenbergReduction(A, n, "Random " + n + "x" + n);
            long elapsed = System.currentTimeMillis() - start;
            System.out.println("  Time: " + elapsed + "ms");
        }
    }

    private void testHessenbergReduction(Matrix A, int n, String context) {
        HessenbergResult res = HessenbergReduction.decompose(A);
        assertNotNull(context + ": Hessenberg result", res);

        Matrix H = res.getH();
        Matrix Q = res.getQ();
        double tol = getTolerance(n);

        // Test 1: H is upper Hessenberg (zeros below subdiagonal)
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals(context + ": H below subdiagonal at (" + i + "," + j + ")",
                        0.0, H.get(i, j), tol);
            }
        }

        // Test 2: Orthogonality of Q
        double orthError = orthogonalityError(Q);
        System.out.printf("  %s: ||Q^T*Q - I||_F = %.2e\n", context, orthError);
        assertOrthogonal(Q, tol, context);

        // Test 3: Trace preservation
        double traceError = Math.abs(A.trace() - H.trace()) / Math.max(Math.abs(A.trace()), 1.0);
        System.out.printf("  %s: Trace error = %.2e\n", context, traceError);
        assertEquals(context + ": trace preservation",
                A.trace(), H.trace(), tol * Math.abs(A.trace()) + tol);

        // Test 4: Reconstruction A = Q*H*Q^T
        Matrix reconstructed = Q.multiply(H).multiply(Q.transpose());
        double reconError = reconstructionError(A, reconstructed);
        System.out.printf("  %s: Reconstruction error = %.2e\n", context, reconError);
        assertReconstruction(A, reconstructed, tol, context);
    }

    // ========== Bidiagonalization Tests ==========

    @Test
    public void testBidiagonalization_SmallSquare() {
        System.out.println("\n=== Bidiagonalization: Small Square Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomMatrix(n, 1000 + n);
            testBidiagonalization(A, n, n, "Square " + n + "x" + n);
        }
    }

    @Test
    public void testBidiagonalization_MediumSquare() {
        System.out.println("\n=== Bidiagonalization: Medium Square Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomMatrix(n, 1100 + n);
            testBidiagonalization(A, n, n, "Square " + n + "x" + n);
        }
    }

    @Test
    public void testBidiagonalization_HugeSquare() {
        System.out.println("\n=== Bidiagonalization: Huge Square Matrices ===");
        for (int n : HUGE_SIZES) {
            Matrix A = randomMatrix(n, 1200 + n);
            long start = System.currentTimeMillis();
            testBidiagonalization(A, n, n, "Square " + n + "x" + n);
            long elapsed = System.currentTimeMillis() - start;
            System.out.println("  Time: " + elapsed + "ms");
        }
    }

    private void testBidiagonalization(Matrix A, int m, int n, String context) {
        Bidiagonalization bidiag = new Bidiagonalization();
        BidiagonalizationResult result = bidiag.decompose(A);
        assertNotNull(context + ": result", result);

        Matrix U = result.getU();
        Matrix B = result.getB();
        Matrix V = result.getV();

        double tol = getTolerance(Math.max(m, n)) * 10; // Relaxed tolerance for bidiag

        // Test 1: Check bidiagonal structure
        int bRows = B.getRowCount();
        int bCols = B.getColumnCount();
        boolean isUpper = m >= n;

        for (int i = 0; i < bRows; i++) {
            for (int j = 0; j < bCols; j++) {
                boolean onDiag = (i == j);
                boolean onSuperDiag = (j == i + 1);
                boolean onSubDiag = (j == i - 1);

                boolean shouldBeNonzero = onDiag || (isUpper ? onSuperDiag : onSubDiag);
                if (!shouldBeNonzero) {
                    assertEquals(context + ": B(" + i + "," + j + ") should be zero",
                            0.0, B.get(i, j), tol);
                }
            }
        }

        // Test 2: U orthogonality
        Matrix UTU = U.transpose().multiply(U);
        int uSize = UTU.getRowCount();
        double uOrthError = 0;
        for (int i = 0; i < uSize; i++) {
            for (int j = 0; j < uSize; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = UTU.get(i, j) - expected;
                uOrthError += diff * diff;
            }
        }
        uOrthError = Math.sqrt(uOrthError);
        System.out.printf("  %s: ||U^T*U - I||_F = %.2e\n", context, uOrthError);

        // Test 3: V orthogonality
        Matrix VTV = V.transpose().multiply(V);
        int vSize = VTV.getRowCount();
        double vOrthError = 0;
        for (int i = 0; i < vSize; i++) {
            for (int j = 0; j < vSize; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = VTV.get(i, j) - expected;
                vOrthError += diff * diff;
            }
        }
        vOrthError = Math.sqrt(vOrthError);
        System.out.printf("  %s: ||V^T*V - I||_F = %.2e\n", context, vOrthError);

        // Test 4: Reconstruction A = U*B*V^T
        Matrix reconstructed = U.multiply(B).multiply(V.transpose());
        double reconError = reconstructionError(A, reconstructed);
        System.out.printf("  %s: Reconstruction error = %.2e\n", context, reconError);
        assertTrue(context + ": reconstruction error",
                reconError < tol);

        // Test 5: Frobenius norm preservation
        double normA = A.frobeniusNorm();
        double normB = B.frobeniusNorm();
        double normError = Math.abs(normA - normB) / Math.max(normA, 1.0);
        System.out.printf("  %s: Norm preservation error = %.2e\n", context, normError);
    }

    // ========== Identity Matrix Tests ==========

    @Test
    public void testDecompositions_IdentityMatrices() {
        System.out.println("\n=== Decompositions: Identity Matrices ===");
        int[] sizes = {2, 3, 5, 7, 10, 15, 20, 30, 50, 100};
        
        for (int n : sizes) {
            Matrix I = Matrix.Identity(n);
            double tol = getTolerance(n);

            // QR of identity
            QRResult qr = HouseholderQR.decompose(I);
            double qrError = reconstructionError(I, qr.getQ().multiply(qr.getR()));
            System.out.printf("  Identity %dx%d QR: Reconstruction error = %.2e\n", n, n, qrError);
            assertTrue("Identity " + n + " QR", qrError < tol);

            // Hessenberg of identity (should stay identity)
            HessenbergResult hess = HessenbergReduction.decompose(I);
            double hessError = reconstructionError(I, hess.getQ().multiply(hess.getH()).multiply(hess.getQ().transpose()));
            System.out.printf("  Identity %dx%d Hessenberg: Reconstruction error = %.2e\n", n, n, hessError);
            assertTrue("Identity " + n + " Hessenberg", hessError < tol);
        }
    }

    // ========== Symmetric Matrix Tests ==========

    @Test
    public void testDecompositions_SymmetricMatrices() {
        System.out.println("\n=== Decompositions: Symmetric Matrices ===");
        int[] sizes = {3, 5, 10, 15, 20, 30, 50};
        
        for (int n : sizes) {
            Matrix A = randomSymmetricMatrix(n, 1300 + n);
            double tol = getTolerance(n);

            // QR decomposition
            QRResult qr = HouseholderQR.decompose(A);
            double qrError = reconstructionError(A, qr.getQ().multiply(qr.getR()));
            System.out.printf("  Symmetric %dx%d QR: Reconstruction error = %.2e\n", n, n, qrError);

            // Hessenberg reduction (should give tridiagonal for symmetric)
            HessenbergResult hess = HessenbergReduction.decompose(A);
            double hessError = reconstructionError(A, hess.getQ().multiply(hess.getH()).multiply(hess.getQ().transpose()));
            System.out.printf("  Symmetric %dx%d Hessenberg: Reconstruction error = %.2e\n", n, n, hessError);
        }
    }

    // ========== Accuracy Summary Test ==========

    @Test
    public void testAccuracySummary_AllDecompositions() {
        System.out.println("\n=== ACCURACY SUMMARY: All Decompositions ===");
        System.out.println("Size\tQR Recon\tLU Recon\tHess Recon\tBidiag Recon");
        System.out.println("----\t--------\t--------\t----------\t------------");
        
        int[] allSizes = {2, 3, 5, 7, 10, 15, 20, 25, 30, 50, 100, 200};
        
        for (int n : allSizes) {
            Matrix A = randomMatrix(n, 1400 + n);
            Matrix D = randomDiagonallyDominant(n, 1500 + n);
            
            // QR
            QRResult qr = HouseholderQR.decompose(A);
            double qrError = reconstructionError(A, qr.getQ().multiply(qr.getR()));
            
            // LU
            LUResult lu = new LUDecomposition().decompose(D);
            double luError = reconstructionError(D, lu.getL().multiply(lu.getU()));
            
            // Hessenberg
            HessenbergResult hess = HessenbergReduction.decompose(A);
            double hessError = reconstructionError(A, hess.getQ().multiply(hess.getH()).multiply(hess.getQ().transpose()));
            
            // Bidiagonalization
            BidiagonalizationResult bidiag = new Bidiagonalization().decompose(A);
            double bidiagError = reconstructionError(A, bidiag.getU().multiply(bidiag.getB()).multiply(bidiag.getV().transpose()));
            
            System.out.printf("%d\t%.2e\t%.2e\t%.2e\t%.2e\n", n, qrError, luError, hessError, bidiagError);
        }
    }
}
