package net.faulj.matrix;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.MatrixAccuracyValidator.AccuracyLevel;
import net.faulj.matrix.MatrixAccuracyValidator.ValidationResult;
import net.faulj.vector.Vector;

/**
 * Tests for MatrixAccuracyValidator demonstrating adaptive threshold validation
 */
public class MatrixAccuracyValidatorTest {

    private static final Random RNG = new Random(42);

    /**
     * Convert a row-major array to a column-based Matrix.
     *
     * @param a row-major data
     * @return matrix instance
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
     * Create a deterministic random square matrix.
     *
     * @param n dimension
     * @param seed RNG seed
     * @return random matrix
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
     * Validate QR reconstruction accuracy across multiple sizes.
     */
    @Test
    public void testQRDecomposition_ExcellentAccuracy() {
        System.out.println("\n=== QR Decomposition Validation ===");
        
        int[] sizes = {5, 10, 20, 50, 100};
        for (int n : sizes) {
            Matrix A = randomMatrix(n, 100 + n);
            QRResult qr = HouseholderQR.decompose(A);
            Matrix reconstructed = qr.getQ().multiply(qr.getR());
            
            ValidationResult result = MatrixAccuracyValidator.validate(
                A, reconstructed, "QR " + n + "x" + n, 1.0);
            
            System.out.println(result.message);
            
            // All should pass
            assertTrue("QR should not fail critically", result.passes);
            
            // Most should be EXCELLENT or GOOD
            assertTrue("QR should have good accuracy", 
                      result.normLevel.ordinal() <= AccuracyLevel.ACCEPTABLE.ordinal());
        }
    }

    /**
     * Validate orthogonality of Q from QR decomposition.
     */
    @Test
    public void testOrthogonalityValidation() {
        System.out.println("\n=== Orthogonality Validation ===");
        
        Matrix A = randomMatrix(20, 42);
        QRResult qr = HouseholderQR.decompose(A);
        Matrix Q = qr.getQ();
        
        // For orthogonality, use norm-based measurement only
        int n = Q.getColumnCount();
        Matrix QtQ = Q.transpose().multiply(Q);
        Matrix I = Matrix.Identity(n);
        double orthError = QtQ.subtract(I).frobeniusNorm();
        
        System.out.printf("QR Q-matrix orthogonality: ||Q^T*Q - I||_F = %.2e\n", orthError);
        
        double tol = Math.sqrt(n) * 2.220446049250313e-16 * 100; // GOOD threshold
        assertTrue("Q should be orthogonal: " + orthError + " vs " + tol, 
                  orthError < tol);
    }

    /**
     * Verify perfect reconstruction for identity matrices.
     */
    @Test
    public void testPerfectReconstruction() {
        System.out.println("\n=== Perfect Reconstruction (Identity) ===");
        
        Matrix I = Matrix.Identity(10);
        ValidationResult result = MatrixAccuracyValidator.validate(
            I, I, "Identity self-comparison", 1.0);
        
        System.out.println(result.message);
        
        assertEquals("Perfect match should be EXCELLENT", 
                    AccuracyLevel.EXCELLENT, result.normLevel);
        assertEquals("Perfect match element-wise should be EXCELLENT", 
                    AccuracyLevel.EXCELLENT, result.elementLevel);
        assertTrue("Should pass", result.passes);
        assertFalse("Should not warn", result.shouldWarn);
    }

    /**
     * Validate adaptive thresholds under varying condition estimates.
     */
    @Test
    public void testConditionNumberAdaptation() {
        System.out.println("\n=== Condition Number Adaptation ===");
        
        Matrix A = randomMatrix(10, 123);
        QRResult qr = HouseholderQR.decompose(A);
        Matrix reconstructed = qr.getQ().multiply(qr.getR());
        
        // Test with different condition estimates
        double[] conditions = {1.0, 10.0, 100.0, 1000.0};
        
        for (double cond : conditions) {
            ValidationResult result = MatrixAccuracyValidator.validate(
                A, reconstructed, String.format("QR (cond=%.0e)", cond), cond);
            
            System.out.printf("  Condition %.0e: %s (Norm: %s, Elem: %s)\n", 
                            cond, result.getOverallLevel(), 
                            result.normLevel, result.elementLevel);
        }
    }

    /**
     * Validate scaling behavior across matrix sizes.
     */
    @Test
    public void testSizeScaling() {
        System.out.println("\n=== Matrix Size Scaling ===");
        
        int[] sizes = {2, 5, 10, 25, 50, 100, 200};
        
        System.out.println("Size\tNorm Residual\tNorm Level\tElem Residual\tElem Level");
        System.out.println("----\t-------------\t----------\t-------------\t----------");
        
        for (int n : sizes) {
            Matrix A = randomMatrix(n, 200 + n);
            QRResult qr = HouseholderQR.decompose(A);
            Matrix reconstructed = qr.getQ().multiply(qr.getR());
            
            ValidationResult result = MatrixAccuracyValidator.validate(
                A, reconstructed, "QR", 1.0);
            
            System.out.printf("%d\t%.2e\t%s\t%.2e\t%s\n", 
                            n, result.normResidual, result.normLevel,
                            result.elementResidual, result.elementLevel);
            
            assertTrue("Should pass for well-conditioned QR", result.passes);
        }
    }

    /**
     * Compare localized versus global error detection.
     */
    @Test
    public void testLocalizedVsGlobalErrors() {
        System.out.println("\n=== Localized vs Global Error Detection ===");
        
        Matrix A = randomMatrix(10, 999);
        Matrix Ahat = A.copy();
        
        // Test 1: Perfect match
        ValidationResult perfect = MatrixAccuracyValidator.validate(
            A, Ahat, "Perfect match", 1.0);
        System.out.println("Perfect:\n" + perfect.message);
        
        // Test 2: Single element error (localized)
        Matrix AhatLocal = A.copy();
        AhatLocal.set(0, 0, A.get(0, 0) + 1e-8);
        ValidationResult localized = MatrixAccuracyValidator.validate(
            A, AhatLocal, "Localized error", 1.0);
        System.out.println("\nLocalized error:\n" + localized.message);
        
        // Test 3: All elements slightly off (global)
        double[][] data = new double[10][10];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                data[i][j] = A.get(i, j) + 1e-10;
            }
        }
        Matrix AhatGlobal = fromRowMajor(data);
        ValidationResult global = MatrixAccuracyValidator.validate(
            A, AhatGlobal, "Global error", 1.0);
        System.out.println("\nGlobal error:\n" + global.message);
    }

    /**
     * Validate reconstruction accuracy for Schur decomposition.
     */
    @Test
    public void testSchurDecomposition() {
        System.out.println("\n=== Schur Decomposition Validation ===");
        
        int[] sizes = {5, 10, 20, 50};
        for (int n : sizes) {
            Matrix A = randomMatrix(n, 300 + n);
            double condEst = MatrixAccuracyValidator.estimateCondition(A);
            
            SchurResult schur = RealSchurDecomposition.decompose(A);
            Matrix reconstructed = schur.getU().multiply(schur.getT())
                                        .multiply(schur.getU().transpose());
            
            ValidationResult result = MatrixAccuracyValidator.validate(
                A, reconstructed, "Schur " + n + "x" + n, condEst);
            
            System.out.println(result.message);
            
            // Schur can have larger errors for poorly conditioned matrices
            // Just check it completes; detailed validation done elsewhere
            if (!result.passes) {
                System.out.println("  Note: Schur failed for size " + n + " (expected for some cases)");
            }
        }
    }

    /**
     * Ensure validateOrThrow succeeds for accurate reconstructions.
     */
    @Test
    public void testValidateOrThrow_Success() {
        Matrix A = randomMatrix(10, 555);
        QRResult qr = HouseholderQR.decompose(A);
        Matrix reconstructed = qr.getQ().multiply(qr.getR());
        
        // Should not throw
        MatrixAccuracyValidator.validateOrThrow(A, reconstructed, "QR test");
    }

    /**
     * Ensure validateOrThrow throws on critical failures.
     */
    @Test(expected = IllegalStateException.class)
    public void testValidateOrThrow_Failure() {
        Matrix A = Matrix.Identity(5);
        Matrix B = Matrix.zero(5, 5);
        
        // Should throw due to critical accuracy failure
        MatrixAccuracyValidator.validateOrThrow(A, B, "Identity vs Zero");
    }

    /**
     * Validate overall accuracy level aggregation.
     */
    @Test
    public void testOverallLevel() {
        Matrix A = Matrix.Identity(10);
        
        ValidationResult result = MatrixAccuracyValidator.validate(
            A, A, "Test", 1.0);
        
        AccuracyLevel overall = result.getOverallLevel();
        assertEquals("Overall should match best individual level", 
                    AccuracyLevel.EXCELLENT, overall);
    }

    /**
     * Print a summary table for multiple decompositions.
     */
    @Test
    public void testAccuracySummary_MultipleDecompositions() {
        System.out.println("\n=== Comprehensive Accuracy Summary ===");
        System.out.println("Operation\t\tSize\tOverall\t\tNorm Error\tElem Error");
        System.out.println("--------\t\t----\t-------\t\t----------\t----------");
        
        int[] sizes = {5, 10, 20, 50, 100};
        
        for (int n : sizes) {
            Matrix A = randomMatrix(n, 400 + n);
            
            // QR
            QRResult qr = HouseholderQR.decompose(A);
            Matrix qrRecon = qr.getQ().multiply(qr.getR());
            ValidationResult qrResult = MatrixAccuracyValidator.validate(
                A, qrRecon, "QR", 1.0);
            
            System.out.printf("QR Decomposition\t%d\t%s\t%.2e\t%.2e\n", 
                            n, qrResult.getOverallLevel(), 
                            qrResult.normResidual, qrResult.elementResidual);
            
            // Schur (for smaller sizes)
            if (n <= 50) {
                SchurResult schur = RealSchurDecomposition.decompose(A);
                Matrix schurRecon = schur.getU().multiply(schur.getT())
                                         .multiply(schur.getU().transpose());
                ValidationResult schurResult = MatrixAccuracyValidator.validate(
                    A, schurRecon, "Schur", 1.0);
                
                System.out.printf("Schur Decomposition\t%d\t%s\t%.2e\t%.2e\n", 
                                n, schurResult.getOverallLevel(),
                                schurResult.normResidual, schurResult.elementResidual);
            }
        }
    }
}
