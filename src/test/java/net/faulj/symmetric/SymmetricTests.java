package net.faulj.symmetric;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixAccuracyValidator;
import net.faulj.matrix.MatrixAccuracyValidator.ValidationResult;
import net.faulj.vector.Vector;

/**
 * Comprehensive tests for symmetric matrix eigendecomposition and related classes.
 * Tests use random matrices and MatrixAccuracyValidator for precision checking.
 */
public class SymmetricTests {
    
    private static final double TOLERANCE = 1e-10;
    private static final int NUM_RANDOM_TESTS = 10;
    
    // ==================== SpectralDecomposition Tests ====================
    
    @Test
    public void testSpectralDecompositionSmall() {
        // 2x2 symmetric matrix
        Matrix A = new Matrix(new double[][]{
            {4, 1},
            {1, 3}
        });
        
        SpectralDecomposition spectral = SymmetricEigenDecomposition.decompose(A);
        
        // Verify reconstruction: A = Q * Λ * Q^T
        Matrix reconstructed = spectral.reconstruct();
        ValidationResult result = MatrixAccuracyValidator.validate(A, reconstructed, 
            "Spectral decomposition 2x2");
        
        System.out.println(result.message);
        assertTrue("Reconstruction should be accurate", result.passes);
        
        // Verify eigenvalues are real and sorted descending
        double[] eigenvalues = spectral.getEigenvalues();
        assertEquals(2, eigenvalues.length);
        assertTrue("Eigenvalues should be sorted descending", eigenvalues[0] >= eigenvalues[1]);
    }
    
    @Test
    public void testSpectralDecompositionSmallDiagonal() {
        // Well-conditioned diagonal matrix
        Matrix A = new Matrix(new double[][]{
            {5, 0, 0, 0},
            {0, 3, 0, 0},
            {0, 0, 2, 0},
            {0, 0, 0, 1}
        });
        
        SpectralDecomposition spectral = SymmetricEigenDecomposition.decompose(A);
        
        // Verify reconstruction
        Matrix reconstructed = spectral.reconstruct();
        ValidationResult result = MatrixAccuracyValidator.validate(A, reconstructed,
            "Diagonal matrix 4x4");
        
        if (!result.passes) {
            System.out.println(result.message);
        }
        assertTrue("Reconstruction should pass validation", result.passes);
    }
    
    @Test
    public void testQuadraticFormEvaluation() {
        Matrix A = new Matrix(new double[][]{
            {2, 1},
            {1, 2}
        });
        QuadraticForm q = new QuadraticForm(A);
        
        Vector x = new Vector(new double[]{1, 0});
        double value = q.evaluate(x);
        assertEquals("Q([1,0]) = 2", 2.0, value, TOLERANCE);
        
        x = new Vector(new double[]{1, 1});
        value = q.evaluate(x);
        assertEquals("Q([1,1]) = 6", 6.0, value, TOLERANCE);
    }
    
    @Test
    public void testQuadraticFormDefiniteness() {
        // Positive definite
        Matrix A1 = new Matrix(new double[][]{
            {2, 0},
            {0, 3}
        });
        QuadraticForm q1 = new QuadraticForm(A1);
        assertEquals(QuadraticForm.Definiteness.POSITIVE_DEFINITE, q1.classify());
        assertTrue(q1.isPositiveDefinite());
        
        // Negative definite
        Matrix A2 = new Matrix(new double[][]{
            {-2, 0},
            {0, -3}
        });
        QuadraticForm q2 = new QuadraticForm(A2);
        assertEquals(QuadraticForm.Definiteness.NEGATIVE_DEFINITE, q2.classify());
        assertTrue(q2.isNegativeDefinite());
        
        // Indefinite
        Matrix A3 = new Matrix(new double[][]{
            {1, 0},
            {0, -1}
        });
        QuadraticForm q3 = new QuadraticForm(A3);
        assertEquals(QuadraticForm.Definiteness.INDEFINITE, q3.classify());
        assertTrue(q3.isIndefinite());
    }
    
    @Test
    public void testRayleighQuotient() {
        Matrix A = new Matrix(new double[][]{
            {3, 1},
            {1, 2}
        });
        
        Vector x = new Vector(new double[]{1, 0});
        double r = ConstrainedOptimization.rayleighQuotient(A, x);
        assertEquals("R([1,0]) = 3", 3.0, r, TOLERANCE);
        
        x = new Vector(new double[]{0, 1});
        r = ConstrainedOptimization.rayleighQuotient(A, x);
        assertEquals("R([0,1]) = 2", 2.0, r, TOLERANCE);
    }
    
    @Test
    public void testMaximizeMinimize() {
        Matrix A = createRandomSymmetric(5);
        
        ConstrainedOptimization.OptimizationResult maxResult = 
            ConstrainedOptimization.maximize(A);
        ConstrainedOptimization.OptimizationResult minResult = 
            ConstrainedOptimization.minimize(A);
        
        // Maximum should be >= minimum
        assertTrue("Maximum eigenvalue should be >= minimum eigenvalue", 
            maxResult.getValue() >= minResult.getValue());
    }
    
    @Test
    public void testPrincipalAxesTransformation() {
        Matrix A = new Matrix(new double[][]{
            {5, 2},
            {2, 2}
        });
        PrincipalAxes axes = new PrincipalAxes(A);
        
        Vector x = new Vector(new double[]{1, 1});
        Vector y = axes.toPrincipalCoordinates(x);
        Vector xBack = axes.fromPrincipalCoordinates(y);
        
        // Round trip should recover original
        for (int i = 0; i < x.dimension(); i++) {
            assertEquals("Round trip transformation should preserve vector", 
                x.get(i), xBack.get(i), TOLERANCE);
        }
    }
    
    @Test
    public void testProjectionFromVector() {
        Vector v = new Vector(new double[]{3, 4});
        ProjectionMatrix P = ProjectionMatrix.fromVector(v);
        
        // Projecting v onto itself should give v
        Vector projected = P.apply(v);
        for (int i = 0; i < v.dimension(); i++) {
            assertEquals(v.get(i), projected.get(i), TOLERANCE);
        }
        
        // Projection should be idempotent
        assertTrue("Projection should be idempotent", P.isIdempotent(TOLERANCE));
        
        // Projection should be orthogonal (symmetric)
        assertTrue("Projection should be orthogonal", P.isOrthogonal(TOLERANCE));
    }
    
    @Test
    public void testProjectionIdentityZero() {
        int n = 4;
        
        ProjectionMatrix I = ProjectionMatrix.identity(n);
        assertEquals("Identity should have full rank", n, I.getRank());
        
        Vector v = randomVector(n);
        Vector projected = I.apply(v);
        for (int i = 0; i < n; i++) {
            assertEquals("Identity projection should not change vector", 
                v.get(i), projected.get(i), TOLERANCE);
        }
        
        ProjectionMatrix Z = ProjectionMatrix.zero(n);
        assertEquals("Zero projection should have rank 0", 0, Z.getRank());
        
        projected = Z.apply(v);
        for (int i = 0; i < n; i++) {
            assertEquals("Zero projection should give zero vector", 
                0.0, projected.get(i), TOLERANCE);
        }
    }
    
    // ==================== Helper Methods ====================
    
    /**
     * Create a random symmetric matrix with given dimension.
     */
    private static Matrix createRandomSymmetric(int n) {
        Matrix A = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            A.set(i, i, Math.random() * 10 - 5); // Diagonal: -5 to 5
            for (int j = i + 1; j < n; j++) {
                double value = Math.random() * 4 - 2; // Off-diagonal: -2 to 2
                A.set(i, j, value);
                A.set(j, i, value); // Ensure symmetry
            }
        }
        return A;
    }
    
    /**
     * Create a well-conditioned random symmetric matrix with given dimension.
     * Uses diagonal dominance to ensure better conditioning.
     */
    private static Matrix createWellConditionedSymmetric(int n) {
        Matrix A = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            // Diagonal: larger values for better conditioning
            A.set(i, i, 5 + Math.random() * 5); // Diagonal: 5 to 10
            for (int j = i + 1; j < n; j++) {
                double value = Math.random() * 2 - 1; // Off-diagonal: -1 to 1
                A.set(i, j, value);
                A.set(j, i, value); // Ensure symmetry
            }
        }
        return A;
    }
    
    /**
     * Create a random vector with given dimension.
     */
    private static Vector randomVector(int n) {
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = Math.random() * 10 - 5; // -5 to 5
        }
        return new Vector(data);
    }
}
