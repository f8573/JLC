package net.faulj.svd;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import net.faulj.decomposition.result.SVDResult;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixAccuracyValidator;
import net.faulj.matrix.MatrixAccuracyValidator.ValidationResult;
import net.faulj.matrix.MatrixUtils;

/**
 * Comprehensive test suite for SVD-related functionality including
 * Pseudoinverse, Rank Estimation, and SVD Algorithms.
 */
public class SVDTests {

    private static final Random RNG = new Random(42);

    /**
     * Generates a random matrix with diagonal conditioning boost.
     *
     * @param m rows
     * @param n columns
     * @param seed RNG seed
     * @return random matrix
     */
    private static Matrix randomMatrix(int m, int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 2 - 1;
            }
            // Add to diagonal for better conditioning
            if (i < n) {
                a[i][i] += Math.min(m, n);
            }
        }
        return fromRowMajor(a);
    }

    /**
     * Builds a matrix from row-major input.
     *
     * @param a row-major values
     * @return matrix with matching entries
     */
    private static Matrix fromRowMajor(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        net.faulj.vector.Vector[] colsV = new net.faulj.vector.Vector[cols];
        for (int c = 0; c < cols; c++) {
            double[] col = new double[rows];
            for (int r = 0; r < rows; r++) col[r] = a[r][c];
            colsV[c] = new net.faulj.vector.Vector(col);
        }
        return new Matrix(colsV);
    }

    /**
     * Asserts accuracy using the matrix validator with a context label.
     *
     * @param expected expected matrix
     * @param actual actual matrix
     * @param context label for assertions
     * @param cond estimated condition number
     */
    private static void assertAccurate(Matrix expected, Matrix actual, String context, double cond) {
        ValidationResult result = MatrixAccuracyValidator.validate(expected, actual, context, cond);
        assertTrue(context + " failed:\n" + result.message, result.passes);
    }

    // ========== Pseudoinverse Tests ==========

    /**
     * Validates pseudoinverse on a full-rank square matrix.
     */
    @Test
    public void testPseudoinverseSquareFullRank() {
        Matrix A = randomMatrix(5, 5, 100);

        Matrix Aplus = new Pseudoinverse().compute(A);
        double cond = MatrixAccuracyValidator.estimateCondition(A);

        assertAccurate(A, A.multiply(Aplus).multiply(A), "A*A+*A (square)", cond);
    }

    /**
     * Validates pseudoinverse on a tall full-rank matrix.
     */
    @Test
    public void testPseudoinverseTallFullRank() {
        Matrix A = randomMatrix(6, 3, 200);

        Matrix Aplus = new Pseudoinverse().compute(A);
        double cond = MatrixAccuracyValidator.estimateCondition(A);

        assertAccurate(A, A.multiply(Aplus).multiply(A), "A*A+*A (tall)", cond);
    }

    /**
     * Validates pseudoinverse on a wide full-rank matrix.
     */
    @Test
    public void testPseudoinverseWideFullRank() {
        Matrix A = randomMatrix(3, 6, 300);

        Matrix Aplus = new Pseudoinverse().compute(A);
        double cond = MatrixAccuracyValidator.estimateCondition(A);

        assertAccurate(A, A.multiply(Aplus).multiply(A), "A*A+*A (wide)", cond);
    }

    /**
     * Validates pseudoinverse on a rank-deficient matrix.
     */
    @Test
    public void testPseudoinverseRankDeficient() {
        Matrix A = new Matrix(new double[][]{
                {1, 2, 3},
                {2, 4, 6},
                {3, 6, 9}
        });

        // Diagnostic: inspect bidiagonalization output before full SVD
        net.faulj.decomposition.bidiagonal.Bidiagonalization bidiag = new net.faulj.decomposition.bidiagonal.Bidiagonalization();
        net.faulj.decomposition.result.BidiagonalizationResult bres = bidiag.decompose(A);
        Matrix B = bres.getB();
        System.out.println("Bidiagonal B: " + MatrixUtils.matrixSummary(B, 3, 3));
        double[] diag = new double[Math.min(B.getRowCount(), B.getColumnCount())];
        for (int i = 0; i < diag.length; i++) diag[i] = B.get(i, i);
        System.out.println("B diag: " + Arrays.toString(diag));
        // Inspect eigenvalues of B^T B which should match squared singular values of B
        Matrix BtB = B.transpose().multiply(B);
        net.faulj.decomposition.result.SchurResult br = net.faulj.eigen.schur.RealSchurDecomposition.decompose(BtB);
        double[] bReal = br.getRealEigenvalues();
        double[] bSigma = new double[bReal.length];
        for (int i = 0; i < bReal.length; i++) bSigma[i] = Math.sqrt(Math.max(0.0, bReal[i]));
        System.out.println("B^T B eigenvalues: " + Arrays.toString(bReal));
        System.out.println("B singular values from eigen: " + Arrays.toString(bSigma));

        // Diagnostic SVD inspection for rank-deficient case
        SVDecomposition svd = new SVDecomposition();
        SVDResult res = svd.decompose(A);
        double[] s = res.getSingularValues();
        System.out.println("Singular values: " + Arrays.toString(s));

        // Additional diagnostics: inspect Sigma, U*Sigma, and reconstructed matrix
        try {
            Matrix Sigma = res.getSigma();
            System.out.println("Sigma matrix: " + MatrixUtils.matrixSummary(Sigma, 3, 3));
            Matrix USigma = res.getU().multiply(Sigma);
            System.out.println("U * Sigma: " + MatrixUtils.matrixSummary(USigma, 3, 3));
            Matrix recon = res.reconstruct();
            System.out.println("Reconstructed: " + MatrixUtils.matrixSummary(recon, 3, 3));
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Effective rank check
        int effRankFromArray = RankEstimation.effectiveRank(s, 1e-12);
        int effRankFromMatrix = RankEstimation.effectiveRank(A);
        System.out.println("Effective rank (from singulars): " + effRankFromArray);
        System.out.println("Effective rank (from matrix): " + effRankFromMatrix);

        // Orthogonality checks for U and V
        Matrix U = res.getU();
        Matrix V = res.getV();
        double uOrth = MatrixUtils.orthogonalityError(U);
        double vOrth = MatrixUtils.orthogonalityError(V);
        System.out.println("U orthogonality error: " + uOrth);
        System.out.println("V orthogonality error: " + vOrth);

        // Reconstruction accuracy
        double reconErr = reconstructionError(A, res.reconstruct());

        // Check that singular values are non-increasing (sorted descending)
        boolean sortedDesc = true;
        for (int i = 0; i < s.length - 1; i++) {
            if (s[i] < s[i + 1]) {
                sortedDesc = false;
                break;
            }
        }
        System.out.println("Singular values sorted descending: " + sortedDesc);

        System.out.println("SVD reconstruction relative error: " + reconErr);

        // Validate SVD factors (orthogonality, reconstruction, singular values)
        validateSvd(A, res, false, "Rank-deficient SVD");

        // Expect rank == 1 for this matrix
        assertEquals("Effective rank (from singulars)", 1, effRankFromArray);
        assertEquals("Effective rank (from matrix)", 1, effRankFromMatrix);

        Matrix Aplus = new Pseudoinverse().compute(A);
        double cond = MatrixAccuracyValidator.estimateCondition(A);

        assertAccurate(A, A.multiply(Aplus).multiply(A), "A*A+*A (rank-deficient)", cond);
    }

    // ========== Rank Estimation Tests ==========

    /**
     * Checks effective rank using default tolerance with dimensions.
     */
    @Test
    public void testEffectiveRankDefaultToleranceWithDimensions() {
        double[] singularValues = {10.0, 1e-4, 1e-16};
        int rank = RankEstimation.effectiveRank(singularValues, 3, 3);
        assertEquals(2, rank);
    }

    /**
     * Checks effective rank under custom tolerance values.
     */
    @Test
    public void testEffectiveRankCustomTolerance() {
        double[] singularValues = {5.0, 1e-3, 1e-6};
        assertEquals(2, RankEstimation.effectiveRank(singularValues, 1e-4));
        assertEquals(1, RankEstimation.effectiveRank(singularValues, 1e-2));
    }

    /**
     * Checks effective rank directly from a matrix.
     */
    @Test
    public void testEffectiveRankFromMatrix() {
        Matrix A = new Matrix(new double[][]{
                {3, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
        });
        int rank = RankEstimation.effectiveRank(A);
        assertEquals(1, rank);
    }

    /**
     * Ensures random matrices are close to full rank.
     */
    @Test
    public void testEffectiveRankFromRandomMatrix() {
        Matrix A = randomMatrix(5, 5, 100);
        int rank = RankEstimation.effectiveRank(A);
        // Random matrix should be full rank
        assertTrue(rank >= 4);
    }

    // ========== SVD Algorithms Tests ==========

    /**
     * Computes relative reconstruction error using Frobenius norm.
     *
     * @param A original matrix
     * @param reconstructed reconstructed matrix
     * @return relative error
     */
    private static double reconstructionError(Matrix A, Matrix reconstructed) {
        double normA = A.frobeniusNorm();
        if (normA < 1e-14) {
            return 0.0;
        }
        return A.subtract(reconstructed).frobeniusNorm() / normA;
    }

    /**
     * Returns base tolerance for a given matrix size.
     *
     * @param m rows
     * @param n columns
     * @return base tolerance
     */
    private static double baseTol(int m, int n) {
        return 1e-8 * Math.max(1.0, Math.max(m, n));
    }

    /**
     * Computes singular values via eigenvalues of $A^T A$ or $A A^T$.
     *
     * @param A input matrix
     * @param tol tolerance for eigenvalue checks
     * @param context label for assertions
     * @return singular values sorted descending
     */
    private static double[] singularValuesFromEigen(Matrix A, double tol, String context) {
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
     * Validates SVD factorization, orthogonality, and singular values.
     *
     * @param A original matrix
     * @param result computed SVD
     * @param thin whether thin SVD is expected
     * @param context label for assertions
     */
    private static void validateSvd(Matrix A, SVDResult result, boolean thin, String context) {
        int m = A.getRowCount();
        int n = A.getColumnCount();
        int r = Math.min(m, n);

        Matrix U = result.getU();
        Matrix V = result.getV();
        double[] sigma = result.getSingularValues();

        double tol = baseTol(m, n);
        double reconTol = tol * 10;
        double orthTol = tol * 10;
        double eigTol = tol * 50;

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
                    sigma[i] >= -tol);
            if (i < sigma.length - 1) {
                assertTrue(context + ": non-descending sigma at " + i,
                        sigma[i] + tol >= sigma[i + 1]);
            }
        }

        double[] sigmaEig = singularValuesFromEigen(A, tol, context);
        assertEquals(context + ": eigenvalue singular count", r, sigmaEig.length);
        for (int i = 0; i < sigma.length; i++) {
            double diff = Math.abs(sigma[i] - sigmaEig[i]);
            double allowed = eigTol * Math.max(1.0, sigmaEig[i]);
            assertTrue(context + ": sigma mismatch at " + i + " diff=" + diff,
                    diff <= allowed);
        }
    }

    /**
     * Tests Golub-Kahan full SVD on a tall matrix.
     */
    @Test
    public void testGolubKahanFullTall() {
        Matrix A = randomMatrix(7, 4, 3101);
        SVDResult result = new GolubKahanSVD().decompose(A);
        validateSvd(A, result, false, "Golub-Kahan full tall");
    }

    /**
     * Tests Golub-Kahan thin SVD on a wide matrix.
     */
    @Test
    public void testGolubKahanThinWide() {
        Matrix A = randomMatrix(4, 7, 3102);
        SVDResult result = new GolubKahanSVD().decomposeThin(A);
        validateSvd(A, result, true, "Golub-Kahan thin wide");
    }

    /**
     * Tests divide-and-conquer full SVD on a wide matrix.
     */
    @Test
    public void testDivideAndConquerFullWide() {
        Matrix A = randomMatrix(5, 8, 3201);
        SVDResult result = new DivideAndConquerSVD().decompose(A);
        validateSvd(A, result, false, "Divide-and-conquer full wide");
    }

    /**
     * Tests divide-and-conquer thin SVD on a tall matrix.
     */
    @Test
    public void testDivideAndConquerThinTall() {
        Matrix A = randomMatrix(8, 5, 3202);
        SVDResult result = new DivideAndConquerSVD().decomposeThin(A);
        validateSvd(A, result, true, "Divide-and-conquer thin tall");
    }
}
