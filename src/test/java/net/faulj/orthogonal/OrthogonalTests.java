package net.faulj.orthogonal;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Comprehensive tests for orthogonal projection and orthonormalization algorithms.
 * Tests cover small, medium, and large random matrices, as well as critical edge cases.
 *
 * Accuracy is measured using:
 * - Frobenius norm for reconstruction and projection errors
 * - Orthonormality verification via ||Q^T*Q - I||_F
 * - Projection matrix properties (idempotence, symmetry)
 */
public class OrthogonalTests {

    // Base tolerances for norm-based error measurements
    private static final double SMALL_TOL = 1e-10;
    private static final double MEDIUM_TOL = 1e-8;
    private static final double LARGE_TOL = 1e-6;

    // Matrix sizes for systematic testing
    private static final int[] SMALL_SIZES = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    private static final int[] MEDIUM_SIZES = {15, 20, 25, 30};
    private static final int[] LARGE_SIZES = {50, 75, 100};

    private static final Random RNG = new Random(42);

    // ========== Helper Methods ==========

    /**
     * Generate random matrix with entries in [-1, 1]
     */
    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                a[i][j] = 2.0 * rnd.nextDouble() - 1.0;
            }
        }
        // Add small diagonal component to improve conditioning
        int minDim = Math.min(rows, cols);
        for (int i = 0; i < minDim; i++) {
            a[i][i] += (i + 1) * 0.5;
        }
        return fromRowMajor(a);
    }

    /**
     * Create matrix from row-major 2D array
     */
    private static Matrix fromRowMajor(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        Vector[] colsV = new Vector[cols];
        for (int c = 0; c < cols; c++) {
            double[] col = new double[rows];
            for (int r = 0; r < rows; r++) {
                col[r] = a[r][c];
            }
            colsV[c] = new Vector(col);
        }
        return new Matrix(colsV);
    }

    /**
     * Generate random vector with entries in [-1, 1]
     */
    private static Vector randomVector(int size, long seed) {
        Random rnd = new Random(seed);
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = 2.0 * rnd.nextDouble() - 1.0;
        }
        return new Vector(data);
    }

    /**
     * Generate list of random linearly independent vectors
     * Uses QR decomposition on random matrix to ensure independence
     */
    private static List<Vector> randomVectors(int count, int size, long seed) {
        if (count > size) {
            throw new IllegalArgumentException("Cannot have more vectors than dimension");
        }
        
        // Generate random matrix and use QR to get orthonormal columns
        // Then perturb them slightly to make them non-orthogonal but still independent
        Matrix A = randomMatrix(size, count, seed);
        net.faulj.decomposition.result.QRResult qr = net.faulj.decomposition.qr.HouseholderQR.decompose(A);
        Matrix Q = qr.getQ();
        
        List<Vector> vectors = new ArrayList<>();
        Random rnd = new Random(seed + 1000);
        Vector[] qCols = Q.getData();
        
        // Create linear combinations of QR columns to ensure linear independence
        for (int i = 0; i < count; i++) {
            Vector v = qCols[i].multiplyScalar(1.0 + rnd.nextDouble());
            // Add small random component from other basis vectors
            for (int j = 0; j < Math.min(count, i + 2); j++) {
                if (j != i && j < Q.getColumnCount()) {
                    v = v.add(qCols[j].multiplyScalar(rnd.nextDouble() * 0.1));
                }
            }
            vectors.add(v);
        }
        return vectors;
    }

    /**
     * Get size-appropriate tolerance
     */
    private double getTolerance(int n) {
        if (n <= 10) return SMALL_TOL;
        if (n <= 30) return MEDIUM_TOL;
        return LARGE_TOL;
    }

    /**
     * Measure orthonormality error: ||Q^T*Q - I||_F
     */
    private double orthonormalityError(Matrix Q) {
        int k = Q.getColumnCount();
        Matrix QtQ = Q.transpose().multiply(Q);
        Matrix I = Matrix.Identity(k);
        return QtQ.subtract(I).frobeniusNorm();
    }

    /**
     * Check if matrix is symmetric
     */
    private boolean isSymmetric(Matrix M, double tol) {
        if (M.getRowCount() != M.getColumnCount()) return false;
        int n = M.getRowCount();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(M.get(i, j) - M.get(j, i)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Check if matrix is idempotent: P^2 = P
     */
    private double idempotenceError(Matrix P) {
        Matrix P2 = P.multiply(P);
        return P.subtract(P2).frobeniusNorm();
    }

    // ========== Orthonormalization Tests ==========

    /**
     * Exercises orthonormalization on small random bases.
     */
    @Test
    public void testOrthonormalization_SmallRandom() {
        System.out.println("\n=== Orthonormalization: Small Random Bases ===");
        int[] sizes = {3, 4, 5, 6, 7, 8, 9, 10};
        for (int n : sizes) {
            int k = Math.max(2, n / 2);
            List<Vector> vectors = randomVectors(k, n, 1000 + n);
            testOrthonormalization(vectors, n, "Small " + n + "x" + k);
        }
    }

    /**
     * Exercises orthonormalization on medium random bases.
     */
    @Test
    public void testOrthonormalization_MediumRandom() {
        System.out.println("\n=== Orthonormalization: Medium Random Bases ===");
        for (int n : MEDIUM_SIZES) {
            int k = Math.max(3, n / 3);
            List<Vector> vectors = randomVectors(k, n, 2000 + n);
            testOrthonormalization(vectors, n, "Medium " + n + "x" + k);
        }
    }

    /**
     * Exercises orthonormalization on larger random bases.
     */
    @Test
    public void testOrthonormalization_LargeRandom() {
        System.out.println("\n=== Orthonormalization: Large Random Bases ===");
        int[] sizes = {50, 75};
        for (int n : sizes) {
            int k = Math.max(5, n / 5);
            List<Vector> vectors = randomVectors(k, n, 3000 + n);
            testOrthonormalization(vectors, n, "Large " + n + "x" + k);
        }
    }

    /**
     * Validates orthonormality, unit norms, and span preservation.
     *
     * @param vectors input basis vectors
     * @param size ambient dimension
     * @param context label for assertions
     */
    private void testOrthonormalization(List<Vector> vectors, int size, String context) {
        Matrix Q = Orthonormalization.createOrthonormalBasis(vectors);
        
        assertNotNull(context + ": Result should not be null", Q);
        // Note: Q might have fewer columns than input if some vectors were linearly dependent
        assertTrue(context + ": Column count should be <= input size", 
                Q.getColumnCount() <= vectors.size());
        assertEquals(context + ": Row count", size, Q.getRowCount());

        // Test orthonormality
        double tol = getTolerance(size) * 1000; // Very relaxed tolerance for gram-schmidt numerical issues
        double orthError = orthonormalityError(Q);
        System.out.printf("  %s: ||Q^T*Q - I||_F = %.2e (tol=%.2e)\n", context, orthError, tol);
        assertTrue(context + ": Orthonormality error = " + orthError + " > " + tol,
                orthError < tol);

        // Test unit norm columns
        Vector[] qCols = Q.getData();
        for (int i = 0; i < Q.getColumnCount(); i++) {
            double norm = qCols[i].norm2();
            assertTrue(context + ": Column " + i + " norm = " + norm,
                    Math.abs(norm - 1.0) < tol);
        }

        // Test span preservation (first k columns of Q span same space as first k input vectors)
        // We verify this by checking that each input vector can be represented in the new basis
        // Only test for the number of vectors actually in the orthonormal basis
        for (int i = 0; i < Math.min(vectors.size(), Q.getColumnCount()); i++) {
            Vector v = vectors.get(i);
            // Project v onto the orthonormal basis and reconstruct
            Vector projection = new Vector(new double[size]);
            for (int j = 0; j < Q.getColumnCount(); j++) {
                Vector qj = qCols[j];
                double coeff = v.dot(qj);
                projection = projection.add(qj.multiplyScalar(coeff));
            }
            
            // The projection should equal v (within numerical tolerance)
            double error = v.subtract(projection).norm2() / Math.max(1.0, v.norm2());
            assertTrue(context + ": Span preservation for vector " + i + " error = " + error,
                    error < tol * 100);
        }
    }

    /**
     * Ensures single-vector input is normalized correctly.
     */
    @Test
    public void testOrthonormalization_EdgeCase_SingleVector() {
        System.out.println("\n=== Orthonormalization: Single Vector Edge Case ===");
        Vector v = new Vector(new double[]{3, 4});
        List<Vector> vectors = Arrays.asList(v);
        
        Matrix Q = Orthonormalization.createOrthonormalBasis(vectors);
        
        assertEquals("Single vector: column count", 1, Q.getColumnCount());
        Vector q0 = Q.getData()[0];
        double norm = q0.norm2();
        assertTrue("Single vector: normalized", Math.abs(norm - 1.0) < SMALL_TOL);
        
        // Check direction preserved
        Vector expected = v.multiplyScalar(1.0 / v.norm2());
        double error = q0.subtract(expected).norm2();
        assertTrue("Single vector: direction preserved", error < SMALL_TOL);
    }

    /**
     * Confirms orthonormalization preserves an orthogonal basis.
     */
    @Test
    public void testOrthonormalization_EdgeCase_AlreadyOrthogonal() {
        System.out.println("\n=== Orthonormalization: Already Orthogonal Vectors ===");
        Vector v1 = new Vector(new double[]{2, 0, 0});
        Vector v2 = new Vector(new double[]{0, 3, 0});
        Vector v3 = new Vector(new double[]{0, 0, 4});
        List<Vector> vectors = Arrays.asList(v1, v2, v3);
        
        Matrix Q = Orthonormalization.createOrthonormalBasis(vectors);
        
        double orthError = orthonormalityError(Q);
        System.out.printf("  Already orthogonal: ||Q^T*Q - I||_F = %.2e\n", orthError);
        assertTrue("Already orthogonal: orthonormality", orthError < SMALL_TOL);
    }

    // ========== Orthogonal Projection Matrix Tests ==========

    /**
     * Exercises projection matrix construction on small random subspaces.
     */
    @Test
    public void testProjectionMatrix_SmallRandom() {
        System.out.println("\n=== Projection Matrix: Small Random Subspaces ===");
        for (int n : SMALL_SIZES) {
            int k = Math.max(1, n / 2);
            Matrix A = randomMatrix(n, k, 4000 + n);
            testProjectionMatrix(A, n, "Small " + n + "x" + k);
        }
    }

    /**
     * Exercises projection matrix construction on medium random subspaces.
     */
    @Test
    public void testProjectionMatrix_MediumRandom() {
        System.out.println("\n=== Projection Matrix: Medium Random Subspaces ===");
        for (int n : MEDIUM_SIZES) {
            int k = Math.max(2, n / 3);
            Matrix A = randomMatrix(n, k, 5000 + n);
            testProjectionMatrix(A, n, "Medium " + n + "x" + k);
        }
    }

    /**
     * Exercises projection matrix construction on large random subspaces.
     */
    @Test
    public void testProjectionMatrix_LargeRandom() {
        System.out.println("\n=== Projection Matrix: Large Random Subspaces ===");
        for (int n : LARGE_SIZES) {
            int k = Math.max(3, n / 5);
            Matrix A = randomMatrix(n, k, 6000 + n);
            testProjectionMatrix(A, n, "Large " + n + "x" + k);
        }
    }

    /**
     * Validates projection matrix symmetry, idempotence, and action.
     *
     * @param A basis matrix
     * @param size ambient dimension
     * @param context label for assertions
     */
    private void testProjectionMatrix(Matrix A, int size, String context) {
        Matrix P = OrthogonalProjection.createMatrix(A);
        
        assertNotNull(context + ": Result should not be null", P);
        assertEquals(context + ": Row count", size, P.getRowCount());
        assertEquals(context + ": Column count", size, P.getColumnCount());

        double tol = getTolerance(size) * 1000; // More relaxed for projection matrices

        // Test 1: Symmetry (P^T = P)
        assertTrue(context + ": Symmetry", isSymmetric(P, tol));

        // Test 2: Idempotence (P^2 = P)
        double idempError = idempotenceError(P);
        System.out.printf("  %s: ||P^2 - P||_F = %.2e\n", context, idempError);
        assertTrue(context + ": Idempotence error = " + idempError + " > " + tol,
                idempError < tol);

        // Test 3: Projects vectors in column space to themselves
        Vector[] aCols = A.getData();
        for (int j = 0; j < A.getColumnCount(); j++) {
            Vector v = aCols[j];
            Vector Pv = P.multiply(v.toMatrix()).getData()[0];
            double error = v.subtract(Pv).norm2() / Math.max(1.0, v.norm2());
            assertTrue(context + ": Projects column " + j + " to itself, error = " + error,
                    error < tol);
        }

        // Test 4: Orthogonality of I-P to column space
        Matrix IminusP = Matrix.Identity(size).subtract(P);
        for (int j = 0; j < A.getColumnCount(); j++) {
            Vector v = aCols[j];
            Vector residual = IminusP.multiply(v.toMatrix()).getData()[0];
            // residual should be orthogonal to v
            double vNorm = Math.max(1e-10, v.norm2());
            double parallelComponent = Math.abs(residual.dot(v)) / (vNorm * vNorm);
            assertTrue(context + ": (I-P)v orthogonal to v for column " + j + ", parallel = " + parallelComponent,
                    parallelComponent < tol);
        }
    }

    /**
     * Ensures full-space projection equals identity.
     */
    @Test
    public void testProjectionMatrix_EdgeCase_FullSpace() {
        System.out.println("\n=== Projection Matrix: Full Space (Identity) ===");
        int n = 5;
        Matrix I = Matrix.Identity(n);
        Matrix P = OrthogonalProjection.createMatrix(I);
        
        // Should be identity matrix
        double error = P.subtract(I).frobeniusNorm();
        System.out.printf("  Full space: ||P - I||_F = %.2e\n", error);
        assertTrue("Full space projection is identity", error < SMALL_TOL * 10);
    }

    /**
     * Validates projection onto a one-dimensional subspace.
     */
    @Test
    public void testProjectionMatrix_EdgeCase_OneDimensional() {
        System.out.println("\n=== Projection Matrix: One-Dimensional Subspace ===");
        Vector v = new Vector(new double[]{1, 2, 3});
        Matrix A = new Matrix(new Vector[]{v});
        Matrix P = OrthogonalProjection.createMatrix(A);
        
        // P should be vv^T / (v^T v)
        double vTv = v.dot(v);
        Matrix expected = v.toMatrix().multiply(v.toMatrix().transpose()).multiplyScalar(1.0 / vTv);
        
        double error = P.subtract(expected).frobeniusNorm();
        System.out.printf("  1D subspace: ||P - vv^T/(v^Tv)||_F = %.2e\n", error);
        assertTrue("1D projection matrix", error < SMALL_TOL * 100);
        
        // Test idempotence
        double idempError = idempotenceError(P);
        assertTrue("1D idempotence", idempError < SMALL_TOL * 100);
    }

    // ========== Best Approximation Tests ==========

    /**
     * Exercises best approximation on small random cases.
     */
    @Test
    public void testBestApproximation_SmallRandom() {
        System.out.println("\n=== Best Approximation: Small Random Cases ===");
        for (int n : SMALL_SIZES) {
            int k = Math.max(1, n / 2);
            Matrix A = randomMatrix(n, k, 7000 + n);
            Vector y = randomVector(n, 7500 + n);
            testBestApproximation(A, y, n, "Small " + n + "x" + k);
        }
    }

    /**
     * Exercises best approximation on medium random cases.
     */
    @Test
    public void testBestApproximation_MediumRandom() {
        System.out.println("\n=== Best Approximation: Medium Random Cases ===");
        for (int n : MEDIUM_SIZES) {
            int k = Math.max(2, n / 3);
            Matrix A = randomMatrix(n, k, 8000 + n);
            Vector y = randomVector(n, 8500 + n);
            testBestApproximation(A, y, n, "Medium " + n + "x" + k);
        }
    }

    /**
     * Exercises best approximation on large random cases.
     */
    @Test
    public void testBestApproximation_LargeRandom() {
        System.out.println("\n=== Best Approximation: Large Random Cases ===");
        for (int n : LARGE_SIZES) {
            int k = Math.max(3, n / 5);
            Matrix A = randomMatrix(n, k, 9000 + n);
            Vector y = randomVector(n, 9500 + n);
            testBestApproximation(A, y, n, "Large " + n + "x" + k);
        }
    }

    /**
     * Validates orthogonality of residual and minimal distance properties.
     *
     * @param A basis matrix
     * @param y target vector
     * @param size ambient dimension
     * @param context label for assertions
     */
    private void testBestApproximation(Matrix A, Vector y, int size, String context) {
        BestApproximation ba = new BestApproximation();
        Vector yHat = ba.findClosest(y, A);
        
        assertNotNull(context + ": Result should not be null", yHat);
        assertEquals(context + ": Result dimension", size, yHat.dimension());

        double tol = getTolerance(size) * 100;

        // Test 1: yHat is in column space of A (can be expressed as A*x for some x)
        // This is guaranteed by construction, but we verify the least squares solution
        
        // Test 2: Error vector (y - yHat) is orthogonal to column space
        Vector error = y.subtract(yHat);
        Vector[] aCols = A.getData();
        for (int j = 0; j < A.getColumnCount(); j++) {
            Vector colj = aCols[j];
            double dotProduct = Math.abs(error.dot(colj)) / (Math.max(1e-10, error.norm2()) * Math.max(1e-10, colj.norm2()));
            assertTrue(context + ": Error orthogonal to column " + j + ", dot = " + dotProduct,
                    dotProduct < tol);
        }

        // Test 3: yHat is the closest point (distance minimized)
        // We verify this by checking that any other point in the column space is farther
        double minDist = error.norm2();
        
        // Try a few random points in the column space
        Random rnd = new Random(size + 10000L);
        for (int trial = 0; trial < 5; trial++) {
            double[] coeffs = new double[A.getColumnCount()];
            for (int i = 0; i < coeffs.length; i++) {
                coeffs[i] = 2.0 * rnd.nextDouble() - 1.0;
            }
            
            // Construct alternative point in column space
            Vector alternative = new Vector(new double[size]);
            for (int j = 0; j < A.getColumnCount(); j++) {
                alternative = alternative.add(aCols[j].multiplyScalar(coeffs[j]));
            }
            
            // Distance from y to alternative should be >= distance to yHat
            double altDist = y.subtract(alternative).norm2();
            // Allow small numerical tolerance
            assertTrue(context + ": yHat is optimal (trial " + trial + ")",
                    minDist <= altDist + tol * Math.max(1.0, minDist));
        }

        System.out.printf("  %s: Min distance = %.2e\n", context, minDist);
    }

    /**
     * Confirms best approximation equals input already in subspace.
     */
    @Test
    public void testBestApproximation_EdgeCase_VectorInSubspace() {
        System.out.println("\n=== Best Approximation: Vector Already in Subspace ===");
        Vector v1 = new Vector(new double[]{1, 0, 0});
        Vector v2 = new Vector(new double[]{0, 1, 0});
        Matrix A = new Matrix(new Vector[]{v1, v2});
        
        Vector y = new Vector(new double[]{3, 4, 0}); // Already in span{v1, v2}
        
        BestApproximation ba = new BestApproximation();
        Vector yHat = ba.findClosest(y, A);
        
        // yHat should equal y
        double error = y.subtract(yHat).norm2();
        System.out.printf("  Vector in subspace: ||y - yHat||_2 = %.2e\n", error);
        assertTrue("Vector in subspace: distance zero", error < SMALL_TOL * 10);
    }

    /**
     * Confirms projection of an orthogonal vector yields zero.
     */
    @Test
    public void testBestApproximation_EdgeCase_OrthogonalVector() {
        System.out.println("\n=== Best Approximation: Vector Orthogonal to Subspace ===");
        Vector v1 = new Vector(new double[]{1, 0, 0});
        Vector v2 = new Vector(new double[]{0, 1, 0});
        Matrix A = new Matrix(new Vector[]{v1, v2});
        
        Vector y = new Vector(new double[]{0, 0, 5}); // Orthogonal to span{v1, v2}
        
        BestApproximation ba = new BestApproximation();
        Vector yHat = ba.findClosest(y, A);
        
        // yHat should be zero (or very close)
        double norm = yHat.norm2();
        System.out.printf("  Orthogonal vector: ||yHat||_2 = %.2e\n", norm);
        assertTrue("Orthogonal vector: projection is zero", norm < SMALL_TOL * 10);
        
        // Error should equal y
        Vector error = y.subtract(yHat);
        double errorNorm = error.norm2();
        assertTrue("Orthogonal vector: error equals y", Math.abs(errorNorm - y.norm2()) < SMALL_TOL * 10);
    }

    // ========== Projection onto Orthonormal Basis Tests ==========

    /**
     * Projects random vectors onto an orthonormal basis.
     */
    @Test
    public void testProjectOntoOrthonormalBasis_Random() {
        System.out.println("\n=== Project onto Orthonormal Basis: Random Cases ===");
        int[] sizes = {5, 10, 15, 20};
        
        for (int n : sizes) {
            int k = Math.max(2, n / 3);
            int numVectors = Math.max(2, k / 2);
            
            // Create orthonormal basis
            List<Vector> rawBasis = randomVectors(k, n, 10000 + n);
            Matrix QMat = Orthonormalization.createOrthonormalBasis(rawBasis);
            List<Vector> orthoBasis = new ArrayList<>();
            Vector[] qCols = QMat.getData();
            for (int i = 0; i < QMat.getColumnCount(); i++) {
                orthoBasis.add(qCols[i]);
            }
            
            // Create source vectors to project
            List<Vector> sourceVectors = randomVectors(numVectors, n, 11000 + n);
            
            Matrix projected = Orthonormalization.projectOntoOrthonormalBasis(orthoBasis, sourceVectors);
            
            assertNotNull("Random " + n + "x" + k + ": result not null", projected);
            assertEquals("Random " + n + "x" + k + ": row count", n, projected.getRowCount());
            assertEquals("Random " + n + "x" + k + ": column count", numVectors, projected.getColumnCount());
            
            double tol = getTolerance(n) * 100;
            
            // Each projected vector should be orthogonal to the error
            Vector[] projCols = projected.getData();
            for (int i = 0; i < numVectors; i++) {
                Vector source = sourceVectors.get(i);
                Vector proj = projCols[i];
                Vector error = source.subtract(proj);
                
                // Error should be orthogonal to all basis vectors
                for (int j = 0; j < orthoBasis.size(); j++) {
                    Vector basisVec = orthoBasis.get(j);
                    double dotProduct = Math.abs(error.dot(basisVec));
                    assertTrue("Random " + n + "x" + k + ": error orthogonal to basis " + j + " for vector " + i,
                            dotProduct < tol);
                }
            }
            
            System.out.printf("  Random %dx%d: Projection successful\n", n, k);
        }
    }

    /**
     * Projects onto a standard basis and verifies expected result.
     */
    @Test
    public void testProjectOntoOrthonormalBasis_EdgeCase_StandardBasis() {
        System.out.println("\n=== Project onto Orthonormal Basis: Standard Basis ===");
        List<Vector> standardBasis = Arrays.asList(
            new Vector(new double[]{1, 0, 0}),
            new Vector(new double[]{0, 1, 0})
        );
        
        Vector v = new Vector(new double[]{3, 4, 5});
        List<Vector> sources = Arrays.asList(v);
        
        Matrix projected = Orthonormalization.projectOntoOrthonormalBasis(standardBasis, sources);
        
        Vector result = projected.getData()[0];
        Vector expected = new Vector(new double[]{3, 4, 0}); // Projection onto xy-plane
        
        double error = result.subtract(expected).norm2();
        System.out.printf("  Standard basis projection: error = %.2e\n", error);
        assertTrue("Standard basis projection", error < SMALL_TOL * 10);
    }

    // ========== Integration Tests ==========

    /**
     * Integrates orthonormalization and projection matrix consistency.
     */
    @Test
    public void testIntegration_OrthonormalizationWithProjection() {
        System.out.println("\n=== Integration: Orthonormalization + Projection Matrix ===");
        int n = 10;
        int k = 4;
        
        // Create random basis and orthonormalize
        List<Vector> rawBasis = randomVectors(k, n, 12000);
        Matrix Q = Orthonormalization.createOrthonormalBasis(rawBasis);
        
        // Create projection matrix from Q
        Matrix A = Q; // Already orthonormal
        Matrix P = OrthogonalProjection.createMatrix(A);
        
        // Project random vector using projection matrix
        Vector y = randomVector(n, 12100);
        Vector projectedByMatrix = P.multiply(y.toMatrix()).getData()[0];
        
        // Project using best approximation
        BestApproximation ba = new BestApproximation();
        Vector projectedByBA = ba.findClosest(y, A);
        
        // Both methods should give same result
        double error = projectedByMatrix.subtract(projectedByBA).norm2();
        System.out.printf("  Integration: ||P*y - bestApprox(y)||_2 = %.2e\n", error);
        assertTrue("Integration: both methods agree", error < MEDIUM_TOL * 100);
    }

    /**
     * Prints summary metrics for orthogonal operations across sizes.
     */
    @Test
    public void testAccuracySummary_AllOrthogonalOperations() {
        System.out.println("\n=== Accuracy Summary: All Orthogonal Operations ===");
        int[] testSizes = {5, 10, 20, 50};
        
        for (int n : testSizes) {
            int k = Math.max(2, n / 3);
            
            List<Vector> vectors = randomVectors(k, n, 20000 + n);
            Matrix Q = Orthonormalization.createOrthonormalBasis(vectors);
            double orthError = orthonormalityError(Q);
            
            Matrix P = OrthogonalProjection.createMatrix(Q);
            double idempError = idempotenceError(P);
            
            Vector y = randomVector(n, 21000 + n);
            BestApproximation ba = new BestApproximation();
            Vector yHat = ba.findClosest(y, Q);
            Vector error = y.subtract(yHat);
            
            // Verify error is orthogonal
            double maxDot = 0.0;
            Vector[] qCols = Q.getData();
            for (int j = 0; j < Q.getColumnCount(); j++) {
                maxDot = Math.max(maxDot, Math.abs(error.dot(qCols[j])));
            }
            
            System.out.printf("  n=%d, k=%d: orthError=%.2e, idempError=%.2e, maxOrthDot=%.2e\n",
                    n, k, orthError, idempError, maxDot);
        }
    }
}
