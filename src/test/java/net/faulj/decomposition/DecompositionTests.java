package net.faulj.decomposition;

import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.qr.CommunicationAvoidingQR;
import net.faulj.decomposition.qr.FullQR;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.qr.PivotedQR;
import net.faulj.decomposition.qr.StrongRankRevealingQR;
import net.faulj.decomposition.qr.SymmetricQR;
import net.faulj.decomposition.qr.TallSkinnyQR;
import net.faulj.decomposition.qr.ThinQR;
import net.faulj.decomposition.qr.UpdatingQR;
import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.decomposition.result.CholeskyResult;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.decomposition.result.LUResult;
import net.faulj.decomposition.result.PivotedQRResult;
import net.faulj.decomposition.result.QRResult;
import net.faulj.decomposition.result.SVDResult;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import net.faulj.svd.SVDecomposition;
import net.faulj.svd.ThinSVD;
import net.faulj.vector.Vector;
import org.junit.Test;

import java.util.Arrays;
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
     * Generate random rectangular matrix with entries in [-1, 1]
     */
    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
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
     * Generate random symmetric positive definite matrix (suitable for Cholesky)
     * Constructs A = B * B^T + λI where B is random and λ ensures positive definiteness
     */
    private static Matrix randomPositiveDefinite(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] b = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        Matrix B = fromRowMajor(b);
        Matrix BBT = B.multiply(B.transpose());
        
        // Add scaled identity to ensure positive definiteness
        double lambda = n * 0.1;
        Matrix I = Matrix.Identity(n);
        return BBT.add(I.multiplyScalar(lambda));
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

    /**
     * Assert column-orthonormality for thin Q (checks Q^T Q = I).
     */
    private void assertOrthonormalColumns(Matrix Q, double tol, String context) {
        double error = MatrixUtils.orthogonalityError(Q);
        assertTrue(context + ": ||Q^T*Q - I||_F = " + error + " > " + tol,
                error < tol);
    }

    /**
     * Assert upper-trapezoidal structure of R for rectangular QR.
     */
    private void assertUpperTrapezoidal(Matrix R, double tol, String context) {
        int m = R.getRowCount();
        int n = R.getColumnCount();
        int limit = Math.min(m, n);
        for (int i = 1; i < m; i++) {
            int maxCol = Math.min(i - 1, n - 1);
            for (int j = 0; j <= maxCol; j++) {
                if (i < limit || j < limit) {
                    assertEquals(context + ": R lower part at (" + i + "," + j + ")",
                            0.0, R.get(i, j), tol);
                }
            }
        }
        if (m > n) {
            for (int i = n; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    assertEquals(context + ": R extra row at (" + i + "," + j + ")",
                            0.0, R.get(i, j), tol);
                }
            }
        }
    }

    /**
     * Computes a base SVD tolerance scaled by matrix size.
     *
     * @param m rows
     * @param n columns
     * @return base tolerance
     */
    private double svdBaseTolerance(int m, int n) {
        return getTolerance(Math.max(m, n)) * 100;
    }

    /**
     * Extracts a contiguous column slice from a matrix.
     *
     * @param M source matrix
     * @param startCol start column (inclusive)
     * @param endCol end column (exclusive)
     * @return column-sliced matrix
     */
    private static Matrix columnSlice(Matrix M, int startCol, int endCol) {
        if (startCol < 0 || endCol > M.getColumnCount() || startCol >= endCol) {
            throw new IllegalArgumentException("Invalid column slice");
        }
        Vector[] cols = new Vector[endCol - startCol];
        Vector[] data = M.getData();
        for (int i = 0; i < cols.length; i++) {
            cols[i] = data[startCol + i].copy();
        }
        return new Matrix(cols);
    }

    /**
     * Builds a diagonal matrix from the first {@code count} values.
     *
     * @param values diagonal values
     * @param count number of diagonal entries
     * @return diagonal matrix
     */
    private static Matrix diagonalMatrix(double[] values, int count) {
        Matrix d = new Matrix(count, count);
        for (int i = 0; i < count; i++) {
            d.set(i, i, values[i]);
        }
        return d;
    }

    /**
     * Computes the spectral norm via SVD.
     *
     * @param A input matrix
     * @return spectral norm
     */
    private static double spectralNorm(Matrix A) {
        SVDResult svd = new SVDecomposition().decompose(A);
        double[] sigma = svd.getSingularValues();
        return sigma.length == 0 ? 0.0 : Math.abs(sigma[0]);
    }

    /**
     * Estimates singular values via eigenvalues of $A^T A$ or $A A^T$.
     *
     * @param A input matrix
     * @param tol tolerance for eigenvalue checks
     * @param context label for assertions
     * @return singular values sorted descending
     */
    private double[] singularValuesFromEigen(Matrix A, double tol, String context) {
        int m = A.getRowCount();
        int n = A.getColumnCount();
        Matrix sym = (m >= n) ? A.transpose().multiply(A) : A.multiply(A.transpose());
        SchurResult schur = RealSchurDecomposition.decompose(sym);
        double[] real = schur.getRealEigenvalues();
        double[] imag = schur.getImagEigenvalues();
        double[] sigma = new double[real.length];
        for (int i = 0; i < real.length; i++) {
            double imagAbs = Math.abs(imag[i]);
            assertTrue(context + ": eigenvalue imag part = " + imagAbs,
                    imagAbs <= tol * 100);
            double val = real[i];
            if (val < 0 && Math.abs(val) < tol * 100) {
                val = 0.0;
            }
            assertTrue(context + ": negative eigenvalue = " + val,
                    val >= -tol * 100);
            sigma[i] = Math.sqrt(Math.max(0.0, val));
        }
        Arrays.sort(sigma);
        for (int i = 0; i < sigma.length / 2; i++) {
            double tmp = sigma[i];
            sigma[i] = sigma[sigma.length - 1 - i];
            sigma[sigma.length - 1 - i] = tmp;
        }
        return sigma;
    }

    /**
     * Validates that null-space columns in full SVD are consistent.
     *
     * @param A input matrix
     * @param U left singular vectors
     * @param V right singular vectors
     * @param r rank estimate
     * @param tol tolerance for null-space checks
     * @param context label for assertions
     */
    private void validateFullSvdNullspace(Matrix A, Matrix U, Matrix V, int r, double tol, String context) {
        int m = A.getRowCount();
        int n = A.getColumnCount();
        double nullTol = tol * Math.max(1.0, A.frobeniusNorm()) * 10;
        if (m > r) {
            for (int col = r; col < m; col++) {
                Matrix u = columnSlice(U, col, col + 1);
                double norm = A.transpose().multiply(u).frobeniusNorm();
                assertTrue(context + ": U null(A^T) column " + col + " norm = " + norm,
                        norm < nullTol);
            }
        }
        if (n > r) {
            for (int col = r; col < n; col++) {
                Matrix v = columnSlice(V, col, col + 1);
                double norm = A.multiply(v).frobeniusNorm();
                assertTrue(context + ": V null(A) column " + col + " norm = " + norm,
                        norm < nullTol);
            }
        }
    }

    /**
     * Validates SVD factorization, orthogonality, and singular values.
     *
     * @param A original matrix
     * @param result SVD result
     * @param thin whether thin SVD is expected
     * @param context label for assertions
     */
    private void validateSvd(Matrix A, SVDResult result, boolean thin, String context) {
        int m = A.getRowCount();
        int n = A.getColumnCount();
        int r = Math.min(m, n);

        Matrix U = result.getU();
        Matrix V = result.getV();
        double[] sigma = result.getSingularValues();

        double baseTol = svdBaseTolerance(m, n);
        double reconTol = baseTol * 10;
        double orthTol = baseTol * 10;
        double eigTol = baseTol * 50;

        if (thin) {
            assertEquals(context + ": thin U column count", r, U.getColumnCount());
            assertEquals(context + ": thin V column count", r, V.getColumnCount());
        } else {
            assertEquals(context + ": full U column count", m, U.getColumnCount());
            assertEquals(context + ": full V column count", n, V.getColumnCount());
        }
        assertEquals(context + ": singular value count", r, sigma.length);

        double reconError = reconstructionError(A, result.reconstruct());
        assertTrue(context + ": reconstruction error = " + reconError,
                reconError < reconTol);

        double uOrth = MatrixUtils.orthogonalityError(U);
        double vOrth = MatrixUtils.orthogonalityError(V);
        assertTrue(context + ": ||U^T*U - I||_F = " + uOrth,
                uOrth < orthTol);
        assertTrue(context + ": ||V^T*V - I||_F = " + vOrth,
                vOrth < orthTol);

        for (int i = 0; i < sigma.length; i++) {
            assertTrue(context + ": negative sigma[" + i + "] = " + sigma[i],
                    sigma[i] >= -baseTol);
            if (i < sigma.length - 1) {
                assertTrue(context + ": non-descending sigma at " + i,
                        sigma[i] + baseTol >= sigma[i + 1]);
            }
        }

        double[] sigmaEig = singularValuesFromEigen(A, baseTol, context);
        assertEquals(context + ": eigenvalue singular count", r, sigmaEig.length);
        for (int i = 0; i < sigma.length; i++) {
            double diff = Math.abs(sigma[i] - sigmaEig[i]);
            double allowed = eigTol * Math.max(1.0, sigmaEig[i]);
            assertTrue(context + ": sigma mismatch at " + i + " diff=" + diff,
                    diff <= allowed);
        }

        if (!thin) {
            validateFullSvdNullspace(A, U, V, r, baseTol, context);
        }
    }

    // ========== QR Decomposition Tests ==========

    /**
     * Runs QR decomposition on small random square matrices.
     */
    @Test
    public void testQR_SmallRandom() {
        System.out.println("\n=== QR Decomposition: Small Random Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomMatrix(n, 100 + n);
            testQRDecomposition(A, n, "Random " + n + "x" + n);
        }
    }

    /**
     * Runs QR decomposition on medium random square matrices.
     */
    @Test
    public void testQR_MediumRandom() {
        System.out.println("\n=== QR Decomposition: Medium Random Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomMatrix(n, 200 + n);
            testQRDecomposition(A, n, "Random " + n + "x" + n);
        }
    }

    /**
     * Runs QR decomposition on large random square matrices and logs timing.
     */
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

    /**
     * Validates orthogonality, triangularity, and reconstruction for QR.
     *
     * @param A input matrix
     * @param n size of the matrix
     * @param context label for assertions
     */
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

    // ========== Extended QR Variants ==========

    /**
     * Compares full and thin QR on rectangular random matrices.
     */
    @Test
    public void testQR_FullAndThin_RectangularRandom() {
        System.out.println("\n=== Full/Thin QR: Rectangular Random Matrices ===");
        int[][] sizes = {
                {12, 6},
                {25, 10},
                {8, 14},
                {15, 30}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 2600 + dims[0] * 10L + dims[1]);
            testFullQR(A, "Full QR " + dims[0] + "x" + dims[1]);
            testThinQR(A, "Thin QR " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Exercises pivoted QR on random matrices.
     */
    @Test
    public void testQR_Pivoted_Random() {
        System.out.println("\n=== Pivoted QR: Random Matrices ===");
        int[][] sizes = {
                {10, 10},
                {18, 8},
                {8, 16}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 2700 + dims[0] * 10L + dims[1]);
            testPivotedQR(A, "Pivoted QR " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Tests strong rank-revealing QR on a near-singular matrix.
     */
    @Test
    public void testQR_StrongRankRevealing_NearSingular() {
        System.out.println("\n=== Strong RRQR: Near-Singular Matrices ===");
        int n = 12;
        Matrix A = randomMatrix(n, n, 2800 + n);
        for (int i = 0; i < n; i++) {
            double val = A.get(i, n - 2);
            A.set(i, n - 1, val + (RNG.nextDouble() - 0.5) * 1e-8);
        }
        testStrongRRQR(A, "Strong RRQR " + n + "x" + n);
    }

    /**
     * Exercises tall-skinny QR on random tall matrices.
     */
    @Test
    public void testQR_TallSkinny_Random() {
        System.out.println("\n=== Tall-Skinny QR: Random Matrices ===");
        int[][] sizes = {
                {60, 10},
                {80, 5},
                {40, 12}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 2900 + dims[0] * 10L + dims[1]);
            testTallSkinnyQR(A, "TSQR " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Exercises communication-avoiding QR on random tall matrices.
     */
    @Test
    public void testQR_CommunicationAvoiding_Random() {
        System.out.println("\n=== CAQR: Random Matrices ===");
        int[][] sizes = {
                {72, 9},
                {55, 7},
                {48, 16}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 3000 + dims[0] * 10L + dims[1]);
            testCommunicationAvoidingQR(A, "CAQR " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Exercises symmetric QR on random symmetric matrices.
     */
    @Test
    public void testQR_Symmetric_Random() {
        System.out.println("\n=== Symmetric QR: Random Symmetric Matrices ===");
        int[] sizes = {5, 10, 15};
        for (int n : sizes) {
            Matrix A = randomSymmetricMatrix(n, 3100 + n);
            testSymmetricQR(A, "Symmetric QR " + n + "x" + n);
        }
    }

    /**
     * Verifies rank-one update and downdate for QR.
     */
    @Test
    public void testQR_Updating_Downdating_RankOne() {
        System.out.println("\n=== Updating/Downdating QR: Rank-One Updates ===");
        int m = 12;
        int n = 8;
        Matrix A = randomMatrix(m, n, 3200 + m * 10L + n);
        double[] uData = new double[m];
        double[] vData = new double[n];
        for (int i = 0; i < m; i++) uData[i] = RNG.nextDouble() * 2 - 1;
        for (int i = 0; i < n; i++) vData[i] = RNG.nextDouble() * 2 - 1;
        Vector u = new Vector(uData);
        Vector v = new Vector(vData);

        QRResult upd = UpdatingQR.rankOneUpdate(A, u, v);
        Matrix Aplus = A.copy();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Aplus.set(i, j, Aplus.get(i, j) + u.get(i) * v.get(j));
            }
        }
        double tol = getTolerance(Math.max(m, n)) * 10;
        assertOrthonormalColumns(upd.getQ().crop(0, m - 1, 0, upd.getQ().getColumnCount() - 1), tol,
                "Update Q");
        assertReconstruction(Aplus, upd.getQ().multiply(upd.getR()), tol * 10, "Rank-one update");

        QRResult down = UpdatingQR.rankOneDowndate(A, u, v);
        Matrix Aminus = A.copy();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Aminus.set(i, j, Aminus.get(i, j) - u.get(i) * v.get(j));
            }
        }
        assertReconstruction(Aminus, down.getQ().multiply(down.getR()), tol * 10, "Rank-one downdate");
    }

    /**
     * Validates full QR decomposition for rectangular matrices.
     *
     * @param A input matrix
     * @param context label for assertions
     */
    private void testFullQR(Matrix A, String context) {
        QRResult res = FullQR.decompose(A);
        assertNotNull(context + ": result", res);
        Matrix Q = res.getQ();
        Matrix R = res.getR();
        double tol = getTolerance(Math.max(A.getRowCount(), A.getColumnCount()));
        assertOrthogonal(Q, tol, context);
        assertUpperTrapezoidal(R, tol, context);
        Matrix QR = Q.multiply(R);
        assertReconstruction(A, QR, tol * 10, context);
    }

    /**
     * Validates thin QR decomposition for rectangular matrices.
     *
     * @param A input matrix
     * @param context label for assertions
     */
    private void testThinQR(Matrix A, String context) {
        QRResult res = ThinQR.decompose(A);
        assertNotNull(context + ": result", res);
        Matrix Q = res.getQ();
        Matrix R = res.getR();
        int k = Math.min(A.getRowCount(), A.getColumnCount());
        assertEquals(context + ": Q column count", k, Q.getColumnCount());
        assertEquals(context + ": R row count", k, R.getRowCount());
        double tol = getTolerance(Math.max(A.getRowCount(), A.getColumnCount()));
        assertOrthonormalColumns(Q, tol, context);
        Matrix QR = Q.multiply(R);
        assertReconstruction(A, QR, tol * 10, context);
    }

    /**
     * Validates pivoted QR reconstruction on permuted input.
     *
     * @param A input matrix
     * @param context label for assertions
     */
    private void testPivotedQR(Matrix A, String context) {
        PivotedQRResult res = PivotedQR.decompose(A);
        assertNotNull(context + ": result", res);
        Matrix Q = res.getQ();
        Matrix R = res.getR();
        double tol = getTolerance(Math.max(A.getRowCount(), A.getColumnCount()));
        assertOrthogonal(Q, tol, context);
        assertUpperTrapezoidal(R, tol, context);
        Matrix AP = res.permutedA();
        Matrix QR = Q.multiply(R);
        assertReconstruction(AP, QR, tol * 10, context);
    }

    /**
     * Validates strong rank-revealing QR properties and reconstruction.
     *
     * @param A input matrix
     * @param context label for assertions
     */
    private void testStrongRRQR(Matrix A, String context) {
        PivotedQRResult res = StrongRankRevealingQR.decompose(A);
        assertNotNull(context + ": result", res);
        Matrix Q = res.getQ();
        Matrix R = res.getR();
        double tol = getTolerance(Math.max(A.getRowCount(), A.getColumnCount()));
        assertOrthogonal(Q, tol, context);
        assertUpperTrapezoidal(R, tol, context);
        Matrix AP = res.permutedA();
        Matrix QR = Q.multiply(R);
        assertReconstruction(AP, QR, tol * 10, context);

        int kMax = Math.min(R.getRowCount(), R.getColumnCount());
        double maxDiag = 0.0;
        double minDiag = Double.POSITIVE_INFINITY;
        for (int i = 0; i < kMax; i++) {
            double diag = Math.abs(R.get(i, i));
            maxDiag = Math.max(maxDiag, diag);
            minDiag = Math.min(minDiag, diag);
        }
        assertTrue(context + ": rank-revealing diagonal ratio",
                maxDiag == 0.0 || minDiag / maxDiag < 1e-6 || kMax < 2);
    }

    /**
     * Validates tall-skinny QR decomposition.
     *
     * @param A input matrix
     * @param context label for assertions
     */
    private void testTallSkinnyQR(Matrix A, String context) {
        QRResult res = TallSkinnyQR.decompose(A);
        assertNotNull(context + ": result", res);
        Matrix Q = res.getQ();
        Matrix R = res.getR();
        int k = Math.min(A.getRowCount(), A.getColumnCount());
        assertEquals(context + ": Q column count", k, Q.getColumnCount());
        assertEquals(context + ": R row count", k, R.getRowCount());
        double tol = getTolerance(Math.max(A.getRowCount(), A.getColumnCount()));
        assertOrthonormalColumns(Q, tol * 10, context);
        Matrix QR = Q.multiply(R);
        assertReconstruction(A, QR, tol * 50, context);
    }

    /**
     * Validates communication-avoiding QR decomposition.
     *
     * @param A input matrix
     * @param context label for assertions
     */
    private void testCommunicationAvoidingQR(Matrix A, String context) {
        QRResult res = CommunicationAvoidingQR.decompose(A);
        assertNotNull(context + ": result", res);
        Matrix Q = res.getQ();
        Matrix R = res.getR();
        int k = Math.min(A.getRowCount(), A.getColumnCount());
        assertEquals(context + ": Q column count", k, Q.getColumnCount());
        assertEquals(context + ": R row count", k, R.getRowCount());
        double tol = getTolerance(Math.max(A.getRowCount(), A.getColumnCount()));
        assertOrthonormalColumns(Q, tol * 10, context);
        Matrix QR = Q.multiply(R);
        assertReconstruction(A, QR, tol * 50, context);
    }

    /**
     * Validates symmetric QR decomposition.
     *
     * @param A symmetric input matrix
     * @param context label for assertions
     */
    private void testSymmetricQR(Matrix A, String context) {
        QRResult res = SymmetricQR.decompose(A);
        assertNotNull(context + ": result", res);
        Matrix Q = res.getQ();
        Matrix R = res.getR();
        double tol = getTolerance(A.getRowCount());
        assertOrthogonal(Q, tol, context);
        assertUpperTrapezoidal(R, tol, context);
        Matrix QR = Q.multiply(R);
        assertReconstruction(A, QR, tol * 10, context);
    }

    // ========== LU Decomposition Tests ==========

    /**
     * Runs LU decomposition on small diagonally dominant matrices.
     */
    @Test
    public void testLU_SmallDiagonallyDominant() {
        System.out.println("\n=== LU Decomposition: Small Diagonally Dominant Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomDiagonallyDominant(n, 400 + n);
            testLUDecomposition(A, n, "Diag Dominant " + n + "x" + n);
        }
    }

    /**
     * Runs LU decomposition on medium diagonally dominant matrices.
     */
    @Test
    public void testLU_MediumDiagonallyDominant() {
        System.out.println("\n=== LU Decomposition: Medium Diagonally Dominant Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomDiagonallyDominant(n, 500 + n);
            testLUDecomposition(A, n, "Diag Dominant " + n + "x" + n);
        }
    }

    /**
     * Runs LU decomposition on large diagonally dominant matrices.
     */
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

    /**
     * Validates LU factorization structure and reconstruction.
     *
     * @param A input matrix
     * @param n size of the matrix
     * @param context label for assertions
     */
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

    // ========== Cholesky Decomposition Tests ==========

    /**
     * Runs Cholesky decomposition on small SPD matrices.
     */
    @Test
    public void testCholesky_SmallPositiveDefinite() {
        System.out.println("\n=== Cholesky Decomposition: Small Positive Definite Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomPositiveDefinite(n, 4000 + n);
            testCholeskyDecomposition(A, n, "Positive Definite " + n + "x" + n);
        }
    }

    /**
     * Runs Cholesky decomposition on medium SPD matrices.
     */
    @Test
    public void testCholesky_MediumPositiveDefinite() {
        System.out.println("\n=== Cholesky Decomposition: Medium Positive Definite Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomPositiveDefinite(n, 4100 + n);
            testCholeskyDecomposition(A, n, "Positive Definite " + n + "x" + n);
        }
    }

    /**
     * Runs Cholesky decomposition on large SPD matrices.
     */
    @Test
    public void testCholesky_HugePositiveDefinite() {
        System.out.println("\n=== Cholesky Decomposition: Huge Positive Definite Matrices ===");
        for (int n : HUGE_SIZES) {
            Matrix A = randomPositiveDefinite(n, 4200 + n);
            long start = System.currentTimeMillis();
            testCholeskyDecomposition(A, n, "Positive Definite " + n + "x" + n);
            long elapsed = System.currentTimeMillis() - start;
            System.out.println("  Time: " + elapsed + "ms");
        }
    }

    /**
     * Validates Cholesky factorization and reconstruction.
     *
     * @param A input SPD matrix
     * @param n size of the matrix
     * @param context label for assertions
     */
    private void testCholeskyDecomposition(Matrix A, int n, String context) {
        CholeskyDecomposition cholesky = new CholeskyDecomposition();
        CholeskyResult res = cholesky.decompose(A);
        assertNotNull(context + ": Cholesky result", res);

        Matrix L = res.getL();
        double tol = getTolerance(n);

        // Test 1: L is lower triangular
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                assertEquals(context + ": L upper part at (" + i + "," + j + ")",
                        0.0, L.get(i, j), tol);
            }
        }

        // Test 2: L has positive diagonal entries
        for (int i = 0; i < n; i++) {
            double diag = L.get(i, i);
            assertTrue(context + ": L diagonal at " + i + " is positive",
                    diag > 0.0);
        }

        // Test 3: Reconstruction A ≈ L*L^T
        Matrix LLT = L.multiply(L.transpose());
        double reconError = reconstructionError(A, LLT);
        System.out.printf("  %s: Reconstruction error = %.2e\n", context, reconError);
        assertTrue(context + ": relative reconstruction error = " + reconError,
                reconError < tol * 10);

        // Test 4: Verify symmetry of A (should be symmetric for Cholesky)
        Matrix AT = A.transpose();
        double symError = A.subtract(AT).frobeniusNorm() / A.frobeniusNorm();
        assertTrue(context + ": input matrix symmetry = " + symError,
                symError < tol);

        // Test 5: Verify positive definiteness by checking eigenvalues via Sylvester's criterion
        // All leading principal minors should be positive
        for (int k = 1; k <= Math.min(5, n); k++) {
            double det = computeLeadingPrincipalMinor(A, k);
            assertTrue(context + ": Leading principal minor " + k + " = " + det + " > 0",
                    det > -tol * Math.pow(n, k));
        }
    }

    /**
     * Compute the k-th leading principal minor (determinant of top-left k x k submatrix)
     */
    private double computeLeadingPrincipalMinor(Matrix A, int k) {
        if (k == 1) return A.get(0, 0);
        if (k == 2) {
            return A.get(0, 0) * A.get(1, 1) - A.get(0, 1) * A.get(1, 0);
        }
        
        // For larger k, use simple cofactor expansion (not efficient but sufficient for testing)
        double[][] submatrix = new double[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                submatrix[i][j] = A.get(i, j);
            }
        }
        return determinant(submatrix, k);
    }

    /**
     * Compute determinant using cofactor expansion (for small matrices)
     */
    private double determinant(double[][] matrix, int n) {
        if (n == 1) return matrix[0][0];
        if (n == 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        
        double det = 0;
        for (int col = 0; col < n; col++) {
            double[][] minor = new double[n - 1][n - 1];
            for (int i = 1; i < n; i++) {
                int minorCol = 0;
                for (int j = 0; j < n; j++) {
                    if (j != col) {
                        minor[i - 1][minorCol++] = matrix[i][j];
                    }
                }
            }
            det += Math.pow(-1, col) * matrix[0][col] * determinant(minor, n - 1);
        }
        return det;
    }

    /**
     * Confirms Cholesky of identity returns identity.
     */
    @Test
    public void testCholesky_EdgeCase_Identity() {
        System.out.println("\n=== Cholesky Decomposition: Identity Matrix ===");
        int n = 5;
        Matrix I = Matrix.Identity(n);
        
        CholeskyDecomposition cholesky = new CholeskyDecomposition();
        CholeskyResult res = cholesky.decompose(I);
        Matrix L = res.getL();
        
        // L should also be identity
        double error = L.subtract(I).frobeniusNorm();
        System.out.printf("  Identity: ||L - I||_F = %.2e\n", error);
        assertTrue("Identity: L equals I", error < SMALL_TOL);
    }

    /**
     * Confirms Cholesky of diagonal matrices returns sqrt diagonals.
     */
    @Test
    public void testCholesky_EdgeCase_Diagonal() {
        System.out.println("\n=== Cholesky Decomposition: Diagonal Matrix ===");
        int n = 5;
        double[][] diag = new double[n][n];
        for (int i = 0; i < n; i++) {
            diag[i][i] = (i + 1) * 2.0; // Positive diagonal entries
        }
        Matrix A = fromRowMajor(diag);
        
        CholeskyDecomposition cholesky = new CholeskyDecomposition();
        CholeskyResult res = cholesky.decompose(A);
        Matrix L = res.getL();
        
        // L should be diagonal with sqrt of diagonal entries
        for (int i = 0; i < n; i++) {
            double expected = Math.sqrt((i + 1) * 2.0);
            double actual = L.get(i, i);
            assertEquals("Diagonal: L[" + i + "," + i + "]", expected, actual, SMALL_TOL);
            
            // Off-diagonal should be zero
            for (int j = 0; j < i; j++) {
                assertEquals("Diagonal: L[" + i + "," + j + "]", 0.0, L.get(i, j), SMALL_TOL);
            }
        }
        
        // Verify reconstruction
        Matrix LLT = L.multiply(L.transpose());
        double reconError = reconstructionError(A, LLT);
        System.out.printf("  Diagonal: Reconstruction error = %.2e\n", reconError);
        assertTrue("Diagonal: reconstruction", reconError < SMALL_TOL);
    }

    /**
     * Validates Cholesky on a simple $3\times3$ SPD matrix.
     */
    @Test
    public void testCholesky_EdgeCase_SimpleSymmetric() {
        System.out.println("\n=== Cholesky Decomposition: Simple 3x3 Symmetric ===");
        double[][] data = {
            {4.0, 2.0, 1.0},
            {2.0, 5.0, 3.0},
            {1.0, 3.0, 6.0}
        };
        Matrix A = fromRowMajor(data);
        
        CholeskyDecomposition cholesky = new CholeskyDecomposition();
        CholeskyResult res = cholesky.decompose(A);
        Matrix L = res.getL();
        
        // Verify reconstruction
        Matrix LLT = L.multiply(L.transpose());
        double reconError = reconstructionError(A, LLT);
        System.out.printf("  Simple 3x3: Reconstruction error = %.2e\n", reconError);
        assertTrue("Simple 3x3: reconstruction", reconError < SMALL_TOL);
        
        // Verify L is lower triangular with positive diagonal
        for (int i = 0; i < 3; i++) {
            assertTrue("Simple 3x3: positive diagonal", L.get(i, i) > 0);
            for (int j = i + 1; j < 3; j++) {
                assertEquals("Simple 3x3: upper triangular", 0.0, L.get(i, j), SMALL_TOL);
            }
        }
    }

    /**
     * Ensures non-positive-definite inputs fail for Cholesky.
     */
    @Test(expected = ArithmeticException.class)
    public void testCholesky_NegativeCase_NotPositiveDefinite() {
        System.out.println("\n=== Cholesky Decomposition: Not Positive Definite (Expected Failure) ===");
        // Create a symmetric but not positive definite matrix
        double[][] data = {
            {1.0, 2.0},
            {2.0, 1.0} // Eigenvalues: 3 and -1 (one negative)
        };
        Matrix A = fromRowMajor(data);
        
        CholeskyDecomposition cholesky = new CholeskyDecomposition();
        cholesky.decompose(A); // Should throw ArithmeticException
    }

    /**
     * Ensures Cholesky rejects non-square matrices.
     */
    @Test(expected = IllegalArgumentException.class)
    public void testCholesky_NegativeCase_NonSquare() {
        System.out.println("\n=== Cholesky Decomposition: Non-Square Matrix (Expected Failure) ===");
        Matrix A = randomMatrix(5, 3, 9999);
        
        CholeskyDecomposition cholesky = new CholeskyDecomposition();
        cholesky.decompose(A); // Should throw IllegalArgumentException
    }

    // ========== Hessenberg Reduction Tests ==========

    /**
     * Runs Hessenberg reduction on small random matrices.
     */
    @Test
    public void testHessenberg_SmallRandom() {
        System.out.println("\n=== Hessenberg Reduction: Small Random Matrices ===");
        for (int n : SMALL_SIZES) {
            Matrix A = randomMatrix(n, 700 + n);
            testHessenbergReduction(A, n, "Random " + n + "x" + n);
        }
    }

    /**
     * Runs Hessenberg reduction on medium random matrices.
     */
    @Test
    public void testHessenberg_MediumRandom() {
        System.out.println("\n=== Hessenberg Reduction: Medium Random Matrices ===");
        for (int n : MEDIUM_SIZES) {
            Matrix A = randomMatrix(n, 800 + n);
            testHessenbergReduction(A, n, "Random " + n + "x" + n);
        }
    }

    /**
     * Runs Hessenberg reduction on large random matrices.
     */
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

    /**
     * Validates Hessenberg reduction structure and reconstruction.
     *
     * @param A input matrix
     * @param n size of the matrix
     * @param context label for assertions
     */
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

    /**
     * Runs bidiagonalization on square matrices.
     */
    @Test
    public void testBidiagonalization_Square() {
        System.out.println("\n=== Bidiagonalization: Square Matrices ===");
        int[] sizes = {2, 3, 5, 8, 10};
        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 1600 + n);
            validateBidiagonalization(A, "Square " + n + "x" + n);
        }
    }

    /**
     * Runs bidiagonalization on tall rectangular matrices.
     */
    @Test
    public void testBidiagonalization_RectangularTall() {
        System.out.println("\n=== Bidiagonalization: Tall Rectangular Matrices ===");
        int[][] sizes = {
                {6, 3},
                {10, 4},
                {12, 5},
                {15, 7}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 1700 + dims[0] * 10 + dims[1]);
            validateBidiagonalization(A, "Tall " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Runs bidiagonalization on wide rectangular matrices.
     */
    @Test
    public void testBidiagonalization_RectangularWide() {
        System.out.println("\n=== Bidiagonalization: Wide Rectangular Matrices ===");
        int[][] sizes = {
                {3, 6},
                {4, 10},
                {5, 12},
                {7, 15}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 1800 + dims[0] * 10 + dims[1]);
            validateBidiagonalization(A, "Wide " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Validates orthogonality, structure, and reconstruction for bidiagonalization.
     *
     * @param A input matrix
     * @param context label for assertions
     */
    private void validateBidiagonalization(Matrix A, String context) {
        int m = A.getRowCount();
        int n = A.getColumnCount();
        BidiagonalizationResult result = new Bidiagonalization().decompose(A);
        assertNotNull(context + ": result", result);

        Matrix U = result.getU();
        Matrix B = result.getB();
        Matrix V = result.getV();

        double baseTol = getTolerance(Math.max(m, n)) * 50;
        double structureTol = baseTol * 5;
        double reconTol = baseTol * 10;
        double orthTol = baseTol * 10;

        assertEquals(context + ": U row count", m, U.getRowCount());
        assertEquals(context + ": U column count", m, U.getColumnCount());
        assertEquals(context + ": V row count", n, V.getRowCount());
        assertEquals(context + ": V column count", n, V.getColumnCount());
        assertEquals(context + ": B row count", m, B.getRowCount());
        assertEquals(context + ": B column count", n, B.getColumnCount());

        double uOrth = MatrixUtils.orthogonalityError(U);
        double vOrth = MatrixUtils.orthogonalityError(V);
        assertTrue(context + ": ||U^T*U - I||_F = " + uOrth,
                uOrth < orthTol);
        assertTrue(context + ": ||V^T*V - I||_F = " + vOrth,
                vOrth < orthTol);

        boolean isUpper = m >= n;
        for (int i = 0; i < B.getRowCount(); i++) {
            for (int j = 0; j < B.getColumnCount(); j++) {
                boolean onDiag = (i == j);
                boolean onSuper = (j == i + 1);
                boolean onSub = (j == i - 1);
                boolean allowed = onDiag || (isUpper ? onSuper : onSub);
                if (!allowed) {
                    assertEquals(context + ": B(" + i + "," + j + ") should be zero",
                            0.0, B.get(i, j), structureTol);
                }
            }
        }

        double reconError = reconstructionError(A, result.reconstruct());
        assertTrue(context + ": reconstruction error = " + reconError,
                reconError < reconTol);

        double normA = A.frobeniusNorm();
        double normB = B.frobeniusNorm();
        double normError = Math.abs(normA - normB) / Math.max(normA, 1.0);
        assertTrue(context + ": Frobenius norm preservation error = " + normError,
                normError < reconTol);
    }

    // ========== SVD Tests ==========

    /**
     * Runs full SVD on square matrices.
     */
    @Test
    public void testSVD_FullSquare() {
        System.out.println("\n=== SVD: Full Square Matrices ===");
        int[] sizes = {3, 5, 8};
        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 1900 + n);
            SVDResult result = new SVDecomposition().decompose(A);
            validateSvd(A, result, false, "Full square " + n + "x" + n);
        }
    }

    /**
     * Runs full SVD on tall rectangular matrices.
     */
    @Test
    public void testSVD_FullRectangularTall() {
        System.out.println("\n=== SVD: Full Tall Rectangular Matrices ===");
        int[][] sizes = {
                {8, 5},
                {12, 6}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 2000 + dims[0] * 10 + dims[1]);
            SVDResult result = new SVDecomposition().decompose(A);
            validateSvd(A, result, false, "Full tall " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Runs full SVD on wide rectangular matrices.
     */
    @Test
    public void testSVD_FullRectangularWide() {
        System.out.println("\n=== SVD: Full Wide Rectangular Matrices ===");
        int[][] sizes = {
                {5, 8},
                {6, 12}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 2100 + dims[0] * 10 + dims[1]);
            SVDResult result = new SVDecomposition().decompose(A);
            validateSvd(A, result, false, "Full wide " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Runs thin SVD on square matrices.
     */
    @Test
    public void testSVD_ThinSquare() {
        System.out.println("\n=== SVD: Thin Square Matrices ===");
        int[] sizes = {4, 7};
        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 2200 + n);
            SVDResult result = new ThinSVD().decompose(A);
            validateSvd(A, result, true, "Thin square " + n + "x" + n);
        }
    }

    /**
     * Runs thin SVD on tall rectangular matrices.
     */
    @Test
    public void testSVD_ThinRectangularTall() {
        System.out.println("\n=== SVD: Thin Tall Rectangular Matrices ===");
        int[][] sizes = {
                {9, 4},
                {12, 5}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 2300 + dims[0] * 10 + dims[1]);
            SVDResult result = new ThinSVD().decompose(A);
            validateSvd(A, result, true, "Thin tall " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Runs thin SVD on wide rectangular matrices.
     */
    @Test
    public void testSVD_ThinRectangularWide() {
        System.out.println("\n=== SVD: Thin Wide Rectangular Matrices ===");
        int[][] sizes = {
                {4, 9},
                {5, 12}
        };
        for (int[] dims : sizes) {
            Matrix A = randomMatrix(dims[0], dims[1], 2400 + dims[0] * 10 + dims[1]);
            SVDResult result = new ThinSVD().decompose(A);
            validateSvd(A, result, true, "Thin wide " + dims[0] + "x" + dims[1]);
        }
    }

    /**
     * Checks truncated SVD spectral error against next singular value.
     */
    @Test
    public void testSVD_TruncatedSpectralError() {
        System.out.println("\n=== SVD: Truncated Spectral Error ===");
        Matrix A = randomMatrix(9, 6, 2501);
        SVDResult full = new SVDecomposition().decompose(A);
        double[] sigma = full.getSingularValues();
        int r = Math.min(A.getRowCount(), A.getColumnCount());
        assertTrue("Expected at least 2 singular values", r > 2);

        int k = 2;
        Matrix Uk = columnSlice(full.getU(), 0, k);
        Matrix Vk = columnSlice(full.getV(), 0, k);
        Matrix Sk = diagonalMatrix(sigma, k);
        Matrix Ak = Uk.multiply(Sk).multiply(Vk.transpose());

        Matrix residual = A.subtract(Ak);
        double residualNorm2 = spectralNorm(residual);
        double sigmaNext = sigma[k];

        double tol = svdBaseTolerance(A.getRowCount(), A.getColumnCount()) * 50
                * Math.max(1.0, sigmaNext);
        assertTrue("Truncated ||A - A_k||_2 mismatch: " + residualNorm2 + " vs " + sigmaNext,
                Math.abs(residualNorm2 - sigmaNext) <= tol);
    }

    // ========== Identity Matrix Tests ==========

    /**
     * Validates decompositions on identity matrices across sizes.
     */
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

    /**
     * Runs decomposition checks on symmetric matrices.
     */
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

    /**
     * Prints reconstruction accuracy summary across decompositions.
     */
    @Test
    public void testAccuracySummary_AllDecompositions() {
        System.out.println("\n=== ACCURACY SUMMARY: All Decompositions ===");
        System.out.println("Size\tQR Recon\tLU Recon\tChol Recon\tHess Recon\tBidiag Recon");
        System.out.println("----\t--------\t--------\t----------\t----------\t------------");
        
        int[] allSizes = {2, 3, 5, 7, 10, 15, 20, 25, 30, 50, 100, 200};
        
        for (int n : allSizes) {
            Matrix A = randomMatrix(n, 1400 + n);
            Matrix D = randomDiagonallyDominant(n, 1500 + n);
            Matrix P = randomPositiveDefinite(n, 1600 + n);
            
            // QR
            QRResult qr = HouseholderQR.decompose(A);
            double qrError = reconstructionError(A, qr.getQ().multiply(qr.getR()));
            
            // LU
            LUResult lu = new LUDecomposition().decompose(D);
            double luError = reconstructionError(D, lu.getL().multiply(lu.getU()));
            
            // Cholesky
            CholeskyResult chol = new CholeskyDecomposition().decompose(P);
            double cholError = reconstructionError(P, chol.getL().multiply(chol.getL().transpose()));
            
            // Hessenberg
            HessenbergResult hess = HessenbergReduction.decompose(A);
            double hessError = reconstructionError(A, hess.getQ().multiply(hess.getH()).multiply(hess.getQ().transpose()));
            
            // Bidiagonalization
            BidiagonalizationResult bidiag = new Bidiagonalization().decompose(A);
            double bidiagError = reconstructionError(A, bidiag.getU().multiply(bidiag.getB()).multiply(bidiag.getV().transpose()));
            
            System.out.printf("%d\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n", n, qrError, luError, cholError, hessError, bidiagError);
        }
    }
}
