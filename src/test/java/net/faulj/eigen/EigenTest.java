package net.faulj.eigen;

import net.faulj.decomposition.result.SchurResult;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.eigen.qr.ImplicitQRFrancis;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;
import net.faulj.determinant.LUDeterminant;
import net.faulj.vector.Vector;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Comprehensive eigenvalue and eigenvector tests using matrix norm-based accuracy measurements.
 * Tests cover small (2x2-10x10), medium (15x15-30x30), and huge (50x50, 100x100, 200x200) matrices.
 * 
 * Accuracy is measured using:
 * - Frobenius norm for reconstruction errors
 * - Relative errors normalized by matrix norms
 * - Orthogonality errors using ||Q^T*Q - I||_F
 */
public class EigenTest {

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
     * Generate random diagonal matrix
     */
    private static Matrix randomDiagonalMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            a[i][i] = rnd.nextDouble() * 10 - 5;
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
        if (normA < 1e-14) return 0.0; // Avoid division by zero
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

    // ========== Real Schur Decomposition Tests - Small Matrices ==========

    /**
     * Runs implicit QR on small random matrices.
     */
    @Test
    public void testImplicitQR_SmallRandom() {
        System.out.println("\n=== Implicit QR: Small Random Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomMatrix(n, 800 + n);
            testImplicitQR(A, n, "Random " + n + "x" + n);
        }
    }

    /**
     * Runs implicit QR on small symmetric matrices.
     */
    @Test
    public void testImplicitQR_SmallSymmetric() {
        System.out.println("\n=== Implicit QR: Small Symmetric Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomSymmetricMatrix(n, 900 + n);
            testImplicitQR(A, n, "Symmetric " + n + "x" + n);
        }
    }

    // ========== Implicit QR Tests - Medium Matrices ==========

    /**
     * Runs implicit QR on medium random matrices.
     */
    @Test
    public void testImplicitQR_MediumRandom() {
        System.out.println("\n=== Implicit QR: Medium Random Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomMatrix(n, 1000 + n);
            testImplicitQR(A, n, "Random " + n + "x" + n);
        }
    }

    /**
     * Runs implicit QR on medium symmetric matrices.
     */
    @Test
    public void testImplicitQR_MediumSymmetric() {
        System.out.println("\n=== Implicit QR: Medium Symmetric Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomSymmetricMatrix(n, 1100 + n);
            testImplicitQR(A, n, "Symmetric " + n + "x" + n);
        }
    }

    // ========== Implicit QR Tests - Huge Matrices ==========

    /**
     * Runs implicit QR on large random matrices and logs timing.
     */
    @Test
    public void testImplicitQR_HugeRandom() {
        System.out.println("\n=== Implicit QR: Huge Random Matrices ===");
        for (int n : HUGE_SIZES) {
            Matrix A = randomMatrix(n, 1200 + n);
            long start = System.currentTimeMillis();
            testImplicitQR(A, n, "Random " + n + "x" + n);
            long elapsed = System.currentTimeMillis() - start;
            System.out.println("  Time: " + elapsed + "ms");
        }
    }

    /**
     * Runs implicit QR on large symmetric matrices and logs timing.
     */
    @Test
    public void testImplicitQR_HugeSymmetric() {
        System.out.println("\n=== Implicit QR: Huge Symmetric Matrices ===");
        for (int n : HUGE_SIZES) {
            Matrix A = randomSymmetricMatrix(n, 1300 + n);
            long start = System.currentTimeMillis();
            testImplicitQR(A, n, "Symmetric " + n + "x" + n);
            long elapsed = System.currentTimeMillis() - start;
            System.out.println("  Time: " + elapsed + "ms");
        }
    }

    /**
     * Core implicit QR (Francis) test using matrix norms.
     *
     * @param A input matrix
     * @param n size of the matrix
     * @param context label for assertions
     */
    private void testImplicitQR(Matrix A, int n, String context) {
        SchurResult result = ImplicitQRFrancis.decompose(A);
        assertNotNull(context + ": result", result);

        Matrix T = result.getT();
        Matrix U = result.getU();
        double tol = getTolerance(n);

        // Test 1: Orthogonality of U
        double orthError = orthogonalityError(U);
        System.out.printf("  %s: ||U^T*U - I||_F = %.2e\n", context, orthError);
        assertOrthogonal(U, tol, context);

        // Test 2: Reconstruction
        Matrix reconstructed = U.multiply(T).multiply(U.transpose());
        double reconError = reconstructionError(A, reconstructed);
        System.out.printf("  %s: Reconstruction error = %.2e\n", context, reconError);
        assertReconstruction(A, reconstructed, tol, context);

        // Test 3: Trace preservation
        double traceError = Math.abs(A.trace() - T.trace()) / Math.max(Math.abs(A.trace()), 1.0);
        System.out.printf("  %s: Trace error = %.2e\n", context, traceError);
        assertEquals(context + ": trace preservation",
                A.trace(), T.trace(), tol * Math.abs(A.trace()) + tol);

        // Test 4: Eigenvalue sum
        double[] eigenvalues = result.getRealEigenvalues();
        double sumEigs = 0;
        for (double lambda : eigenvalues) {
            sumEigs += lambda;
        }
        double eigSumError = Math.abs(sumEigs - A.trace()) / Math.max(Math.abs(A.trace()), 1.0);
        System.out.printf("  %s: Eigenvalue sum error = %.2e\n", context, eigSumError);
    }

    // ========== Explicit QR Tests - Small Matrices ==========

    /**
     * Runs explicit QR on small random matrices.
     */
    @Test
    public void testExplicitQR_SmallRandom() {
        System.out.println("\n=== Explicit QR: Small Random Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomMatrix(n, 1400 + n);
            testExplicitQR(A, n, "Random " + n + "x" + n);
        }
    }

    /**
     * Runs explicit QR on small symmetric matrices.
     */
    @Test
    public void testExplicitQR_SmallSymmetric() {
        System.out.println("\n=== Explicit QR: Small Symmetric Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomSymmetricMatrix(n, 1500 + n);
            testExplicitQR(A, n, "Symmetric " + n + "x" + n);
        }
    }

    // ========== Explicit QR Tests - Medium Matrices ==========

    /**
     * Runs explicit QR on medium random matrices.
     */
    @Test
    public void testExplicitQR_MediumRandom() {
        System.out.println("\n=== Explicit QR: Medium Random Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomMatrix(n, 1600 + n);
            testExplicitQR(A, n, "Random " + n + "x" + n);
        }
    }

    /**
     * Core explicit QR test using matrix norms.
     *
     * @param A input matrix
     * @param n size of the matrix
     * @param context label for assertions
     */
    private void testExplicitQR(Matrix A, int n, String context) {
        Matrix[] result = ExplicitQRIteration.decompose(A);
        assertNotNull(context + ": result", result);
        assertEquals(context + ": result length", 2, result.length);

        Matrix T = result[0];
        Matrix Q = result[1];
        double tol = getTolerance(n);

        // Test 1: Orthogonality
        double orthError = orthogonalityError(Q);
        System.out.printf("  %s: ||Q^T*Q - I||_F = %.2e\n", context, orthError);
        assertOrthogonal(Q, tol, context);

        // Test 2: Reconstruction
        Matrix reconstructed = Q.multiply(T).multiply(Q.transpose());
        double reconError = reconstructionError(A, reconstructed);
        System.out.printf("  %s: Reconstruction error = %.2e\n", context, reconError);
        assertReconstruction(A, reconstructed, tol, context);

        // Test 3: Trace preservation
        double traceError = Math.abs(A.trace() - T.trace()) / Math.max(Math.abs(A.trace()), 1.0);
        System.out.printf("  %s: Trace error = %.2e\n", context, traceError);
        assertEquals(context + ": trace preservation",
                A.trace(), T.trace(), tol * Math.abs(A.trace()) + tol);
    }

    // ========== Identity Matrix Tests (All Sizes) ==========

    /**
     * Verifies identity matrices produce unit eigenvalues and orthogonal $U$.
     */
    @Test
    public void testIdentityMatrices_AllSizes() {
        System.out.println("\n=== Identity Matrix Tests ===");
        int[] allSizes = {2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200};
        
        for (int n : allSizes) {
            Matrix I = Matrix.Identity(n);
            SchurResult schur = RealSchurDecomposition.decompose(I);
            
            double tol = getTolerance(n);
            
            // All eigenvalues should be 1
            double[] eigenvalues = schur.getRealEigenvalues();
            for (int i = 0; i < n; i++) {
                assertEquals("Identity " + n + "x" + n + ": eigenvalue " + i,
                        1.0, eigenvalues[i], tol);
            }
            
            // U should be orthogonal
            double orthError = orthogonalityError(schur.getU());
            System.out.printf("  Identity %dx%d: ||U^T*U - I||_F = %.2e\n", n, n, orthError);
            assertTrue("Identity " + n + "x" + n + ": orthogonality",
                    orthError < tol);
        }
    }

    // ========== Diagonal Matrix Tests ==========

    /**
     * Validates diagonal eigenvalues for small sizes.
     */
    @Test
    public void testDiagonalMatrices_SmallSizes() {
        System.out.println("\n=== Diagonal Matrix Tests: Small ===");
        for (int n : SMALL_SIZES) {
            Matrix D = randomDiagonalMatrix(n, 1700 + n);
            testDiagonalMatrix(D, n);
        }
    }

    /**
     * Validates diagonal eigenvalues for medium sizes.
     */
    @Test
    public void testDiagonalMatrices_MediumSizes() {
        System.out.println("\n=== Diagonal Matrix Tests: Medium ===");
        for (int n : MEDIUM_SIZES) {
            Matrix D = randomDiagonalMatrix(n, 1800 + n);
            testDiagonalMatrix(D, n);
        }
    }

    /**
     * Validates diagonal eigenvalues for large sizes.
     */
    @Test
    public void testDiagonalMatrices_HugeSizes() {
        System.out.println("\n=== Diagonal Matrix Tests: Huge ===");
        for (int n : HUGE_SIZES) {
            Matrix D = randomDiagonalMatrix(n, 1900 + n);
            testDiagonalMatrix(D, n);
        }
    }

    /**
     * Compares expected diagonal eigenvalues to computed values.
     *
     * @param D diagonal matrix
     * @param n size of the matrix
     */
    private void testDiagonalMatrix(Matrix D, int n) {
        SchurResult schur = RealSchurDecomposition.decompose(D);
        double tol = getTolerance(n);

        // Extract expected eigenvalues from diagonal
        double[] expectedEigs = new double[n];
        for (int i = 0; i < n; i++) {
            expectedEigs[i] = D.get(i, i);
        }
        Arrays.sort(expectedEigs);

        // Compare with computed eigenvalues
        double[] actualEigs = schur.getRealEigenvalues();
        Arrays.sort(actualEigs);

        // Compute eigenvalue error norm
        double eigError = 0;
        for (int i = 0; i < n; i++) {
            eigError += Math.pow(expectedEigs[i] - actualEigs[i], 2);
        }
        eigError = Math.sqrt(eigError);
        
        System.out.printf("  Diagonal %dx%d: Eigenvalue error norm = %.2e\n", n, n, eigError);
        assertTrue("Diagonal " + n + "x" + n + ": eigenvalue accuracy",
                eigError < tol * n);
    }

    // ========== Determinant Preservation Tests ==========

    /**
     * Checks determinant preservation on small random matrices.
     */
    @Test
    public void testDeterminantPreservation_SmallSizes() {
        System.out.println("\n=== Determinant Preservation: Small ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomMatrix(n, 2000 + n);
            testDeterminantPreservation(A, n, "Random");
        }
    }

    /**
     * Checks determinant preservation on medium random matrices.
     */
    @Test
    public void testDeterminantPreservation_MediumSizes() {
        System.out.println("\n=== Determinant Preservation: Medium ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomMatrix(n, 2100 + n);
            testDeterminantPreservation(A, n, "Random");
        }
    }

    /**
     * Compares determinants between $A$ and its Schur form $T$.
     *
     * @param A input matrix
     * @param n size of the matrix
     * @param type label for assertions
     */
    private void testDeterminantPreservation(Matrix A, int n, String type) {
        double detA = LUDeterminant.compute(A);
        
        SchurResult schur = RealSchurDecomposition.decompose(A);
        Matrix T = schur.getT();
        double detT = LUDeterminant.compute(T);
        
        double tol = getTolerance(n);
        double relError = Math.abs(detA - detT) / Math.max(Math.abs(detA), 1.0);
        
        System.out.printf("  %s %dx%d: Determinant error = %.2e\n", type, n, n, relError);
        assertEquals(type + " " + n + "x" + n + ": determinant preservation",
                detA, detT, Math.abs(detA) * tol + tol);
    }

    // ========== Near-Singular Matrix Tests ==========

    /**
     * Validates behavior on near-singular matrices with small eigenvalues.
     */
    @Test
    public void testNearSingularMatrices() {
        System.out.println("\n=== Near-Singular Matrix Tests ===");
        int[] sizes = {3, 5, 10, 20, 50};
        
        for (int n : sizes) {
            // Create matrix with one very small eigenvalue
            double[][] data = new double[n][n];
            for (int i = 0; i < n; i++) {
                data[i][i] = (i == 0) ? 1e-10 : i + 1;
            }
            Matrix D = fromRowMajor(data);

            // Apply random orthogonal transformation
            Matrix Q = HouseholderQR.decompose(randomMatrix(n, 2200 + n)).getQ();
            Matrix A = Q.multiply(D).multiply(Q.transpose());

            SchurResult schur = RealSchurDecomposition.decompose(A);
            double tol = getTolerance(n) * 100; // Relaxed tolerance

            // Check reconstruction error
            Matrix T = schur.getT();
            Matrix U = schur.getU();
            Matrix reconstructed = U.multiply(T).multiply(U.transpose());
            double reconError = reconstructionError(A, reconstructed);
            
            System.out.printf("  Near-singular %dx%d: Reconstruction error = %.2e\n", 
                    n, n, reconError);
            assertTrue("Near-singular " + n + "x" + n + ": reconstruction",
                    reconError < tol);

            // Eigenvalues should include the small one
            double[] eigs = schur.getRealEigenvalues();
            boolean foundSmall = false;
            for (double eig : eigs) {
                if (Math.abs(eig) < 1e-8) {
                    foundSmall = true;
                    break;
                }
            }
            assertTrue("Near-singular " + n + "x" + n + ": detects small eigenvalue", 
                    foundSmall);
        }
    }

    // ========== Symmetric Positive Definite Matrix Tests ==========

    /**
     * Validates eigenvalues and reconstruction for SPD matrices.
     */
    @Test
    public void testSymmetricPositiveDefinite_AllSizes() {
        System.out.println("\n=== Symmetric Positive Definite Matrices ===");
        int[] sizes = {3, 5, 10, 15, 20, 30, 50, 100};
        
        for (int n : sizes) {
            // Create SPD matrix by adding n*I to random symmetric
            Matrix A = randomSymmetricMatrix(n, 2300 + n);
            Matrix I = Matrix.Identity(n);
            A = A.add(I.multiplyScalar(n));

            SchurResult schur = RealSchurDecomposition.decompose(A);
            double tol = getTolerance(n);

            // All eigenvalues should be positive
            double[] eigs = schur.getRealEigenvalues();
            for (int i = 0; i < n; i++) {
                assertTrue("SPD " + n + "x" + n + ": eigenvalue " + i + " positive",
                        eigs[i] > 0);
            }

            // Reconstruction error
            Matrix T = schur.getT();
            Matrix U = schur.getU();
            Matrix reconstructed = U.multiply(T).multiply(U.transpose());
            double reconError = reconstructionError(A, reconstructed);
            
            System.out.printf("  SPD %dx%d: Reconstruction error = %.2e\n", n, n, reconError);
            assertTrue("SPD " + n + "x" + n + ": reconstruction error",
                    reconError < tol);
        }
    }

    // ========== Accuracy Summary Test ==========

    /**
     * Prints accuracy summaries across a range of sizes.
     */
    @Test
    public void testAccuracySummary_AllSizes() {
        System.out.println("\n=== ACCURACY SUMMARY: Schur Decomposition ===");
        System.out.println("Size\tOrth Error\tRecon Error\tTrace Error");
        System.out.println("----\t----------\t-----------\t-----------");
        
        int[] allSizes = {2, 3, 5, 7, 10, 15, 20, 25, 30, 50, 100, 200};
        
        for (int n : allSizes) {
            Matrix A = randomMatrix(n, 2400 + n);
            SchurResult schur = RealSchurDecomposition.decompose(A);
            
            Matrix T = schur.getT();
            Matrix U = schur.getU();
            
            double orthError = orthogonalityError(U);
            double reconError = reconstructionError(A, U.multiply(T).multiply(U.transpose()));
            double traceError = Math.abs(A.trace() - T.trace()) / Math.max(Math.abs(A.trace()), 1.0);
            
            System.out.printf("%d\t%.2e\t%.2e\t%.2e\n", n, orthError, reconError, traceError);
        }
    }
}
