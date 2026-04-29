package net.faulj.decomposition.lu;

import net.faulj.core.PermutationVector;
import net.faulj.core.Tolerance;
import net.faulj.decomposition.result.LUResult;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.NativeFactorizationSupport;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Computes the LU decomposition of a square matrix with partial or full pivoting.
 * <p>
 * The LU decomposition factors a square matrix A (with row permutation P) into the form:
 * </p>
 * <pre>
 *   PA = LU
 * </pre>
 * <p>
 * where:
 * </p>
 * <ul>
 *   <li><b>P</b> is a permutation matrix representing row exchanges</li>
 *   <li><b>L</b> is a unit lower triangular matrix (ones on diagonal)</li>
 *   <li><b>U</b> is an upper triangular matrix</li>
 * </ul>
 *
 * <h2>Gaussian Elimination:</h2>
 * <p>
 * LU decomposition is equivalent to Gaussian elimination with pivoting. The factorization
 * captures the elimination multipliers in L and the reduced form in U.
 * </p>
 *
 * <h2>Algorithm:</h2>
 * <p>
 * The algorithm performs elimination with row pivoting:
 * </p>
 * <ol>
 *   <li>For column k = 0 to n-2:</li>
 *   <li>Select pivot row using the chosen {@link PivotPolicy}</li>
 *   <li>Exchange rows if necessary and update permutation P</li>
 *   <li>Compute elimination multipliers: L[i,k] = U[i,k] / U[k,k]</li>
 *   <li>Update remaining submatrix: U[i,j] -= L[i,k] * U[k,j]</li>
 * </ol>
 *
 * <h2>Computational Complexity:</h2>
 * <ul>
 *   <li><b>Time complexity:</b> O(2n<sup>3</sup>/3) flops for decomposition</li>
 *   <li><b>Solving Ax=b:</b> O(n<sup>2</sup>) additional flops (forward and back substitution)</li>
 *   <li><b>Space complexity:</b> O(n<sup>2</sup>) for L and U (can be stored in-place)</li>
 *   <li><b>Comparison:</b> Twice as fast as Cholesky for non-symmetric matrices</li>
 * </ul>
 *
 * <h2>Pivoting Strategies:</h2>
 * <ul>
 *   <li><b>Partial pivoting (default):</b> Select largest element in current column below diagonal</li>
 *   <li><b>No pivoting:</b> Use natural order (fails for some matrices, less stable)</li>
 *   <li><b>Complete pivoting:</b> Select largest element in entire remaining submatrix (more stable, more expensive)</li>
 * </ul>
 * <p>
 * Partial pivoting provides a good balance between stability and efficiency for most applications.
 * </p>
 *
 * <h2>Numerical Stability:</h2>
 * <ul>
 *   <li><b>With partial pivoting:</b> Backward stable for most practical matrices</li>
 *   <li><b>Growth factor:</b> Measures instability; typically small but can be O(2<sup>n</sup>) worst case</li>
 *   <li><b>Singular detection:</b> Small pivots indicate near-singularity</li>
 *   <li><b>Condition number:</b> Error amplification bounded by κ(A) * machine epsilon</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 *   <li>Solving linear systems Ax = b</li>
 *   <li>Computing determinants: det(A) = det(P) * &prod; U[i,i]</li>
 *   <li>Matrix inversion: A<sup>-1</sup> = U<sup>-1</sup> L<sup>-1</sup> P</li>
 *   <li>Computing matrix rank</li>
 *   <li>Condition number estimation</li>
 *   <li>Backend for many numerical algorithms</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = new Matrix(new double[][] {
 *     {2,  1, -1},
 *     {-3, -1, 2},
 *     {-2, 1,  2}
 * });
 *
 * LUDecomposition lu = new LUDecomposition(PivotPolicy.PARTIAL);
 * LUResult result = lu.decompose(A);
 *
 * Matrix L = result.getL();     // Lower triangular with unit diagonal
 * Matrix U = result.getU();     // Upper triangular
 * PermutationVector P = result.getP();
 *
 * // Solve Ax = b
 * Vector b = new Vector(new double[] {8, -11, -3});
 * Vector x = result.solve(b);   // Forward/backward substitution
 *
 * // Compute determinant
 * double det = result.determinant();
 *
 * // Check singularity
 * if (result.isSingular()) {
 *     System.out.println("Matrix is singular or nearly singular");
 * }
 *
 * // Verify: PA = LU
 * Matrix PA = P.applyTo(A);
 * Matrix reconstructed = L.multiply(U);
 * }</pre>
 *
 * <h2>When to Use LU:</h2>
 * <ul>
 *   <li><b>General square systems:</b> LU is the default choice</li>
 *   <li><b>Multiple right-hand sides:</b> Decompose once, solve many times</li>
 *   <li><b>Determinants and inverses:</b> LU provides these efficiently</li>
 * </ul>
 *
 * <h2>When to Use Alternatives:</h2>
 * <ul>
 *   <li><b>Symmetric positive definite:</b> Use Cholesky (2x faster, more stable)</li>
 *   <li><b>Overdetermined systems:</b> Use QR decomposition for least squares</li>
 *   <li><b>Rank-deficient:</b> Use SVD for reliable rank determination</li>
 *   <li><b>Sparse matrices:</b> Use iterative methods or sparse LU</li>
 * </ul>
 *
 * <h2>Singularity Detection:</h2>
 * <p>
 * The decomposition detects singular or nearly singular matrices by checking for
 * small pivots. The result object provides an {@code isSingular()} method that
 * returns true if any diagonal element of U is effectively zero (below tolerance).
 * </p>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>Supports configurable pivoting strategies</li>
 *   <li>Tracks row permutations efficiently with {@link PermutationVector}</li>
 *   <li>Detects singular matrices during decomposition</li>
 *   <li>Can be used for in-place factorization to save memory</li>
 *   <li>Uses {@link Tolerance} for zero detection</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see PivotPolicy
 * @see net.faulj.decomposition.result.LUResult
 * @see net.faulj.solve.LUSolver
 * @see net.faulj.decomposition.cholesky.CholeskyDecomposition
 */
public class LUDecomposition {
    // Small and medium matrices do better with the simpler unblocked path.
    // Once the trailing update dominates, narrower panels feed GEMM more efficiently.
    private static final int DEFAULT_BLOCK_THRESHOLD = 384;
    private static final int DEFAULT_BLOCK_SIZE = 32;
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    
    private final PivotPolicy pivotPolicy;
    
    /**
     * Create an LU decomposition with partial pivoting.
     */
    public LUDecomposition() {
        this(PivotPolicy.PARTIAL);
    }
    
    /**
     * Create an LU decomposition with a specific pivoting strategy.
     *
     * @param pivotPolicy pivot selection policy
     */
    public LUDecomposition(PivotPolicy pivotPolicy) {
        this.pivotPolicy = pivotPolicy;
    }
    
    /**
     * Computes LU factorization with pivoting.
     * @param A square matrix to factor
     * @return LUResult containing L, U, P, and diagnostics
     */
    public LUResult decompose(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("LU decomposition requires a square matrix");
        }

        int n = A.getRowCount();
        if (pivotPolicy == PivotPolicy.PARTIAL) {
            Matrix vendorPacked = A.copy();
            int[] pivots = new int[n];
            if (NativeFactorizationSupport.tryLu(vendorPacked.getRawData(), n, pivots)) {
                return buildVendorResult(A, vendorPacked.getRawData(), n, pivots);
            }
        }
        if (n >= blockThreshold()) {
            return decomposeBlocked(A, n);
        }

        return decomposeUnblocked(A, n);
    }

    private LUResult decomposeUnblocked(Matrix A, int n) {
        Matrix U = A.copy();
        Matrix L = Matrix.Identity(n);
        double[] ud = U.getRawData();
        double[] ld = L.getRawData();
        PermutationVector P = new PermutationVector(n);
        boolean singular = false;

        for (int k = 0; k < n - 1; k++) {
            int pivotRow = selectPivotRow(U, ud, n, k);
            if (pivotRow != k) {
                swapRows(ud, n, k, pivotRow, 0, n);
                P.exchange(k, pivotRow);
                swapRows(ld, n, k, pivotRow, 0, k);
            }

            int pivotOffset = k * n;
            double pivot = ud[pivotOffset + k];
            if (Tolerance.isZero(pivot)) {
                singular = true;
                continue;
            }

            for (int i = k + 1; i < n; i++) {
                int rowOffset = i * n;
                double factor = ud[rowOffset + k] / pivot;
                ld[rowOffset + k] = factor;
                ud[rowOffset + k] = 0.0;
                for (int j = k + 1; j < n; j++) {
                    ud[rowOffset + j] -= factor * ud[pivotOffset + j];
                }
            }
        }

        if (Tolerance.isZero(ud[(n - 1) * n + (n - 1)])) {
            singular = true;
        }

        return new LUResult(A, L, U, P, singular);
    }

    private static LUResult buildVendorResult(Matrix original, double[] packedLu, int n, int[] pivots) {
        Matrix L = Matrix.Identity(n);
        Matrix U = new Matrix(n, n);
        double[] ld = L.getRawData();
        double[] ud = U.getRawData();
        boolean singular = false;

        for (int row = 0; row < n; row++) {
            int rowOffset = row * n;
            for (int col = 0; col < n; col++) {
                double value = packedLu[rowOffset + col];
                if (row > col) {
                    ld[rowOffset + col] = value;
                    ud[rowOffset + col] = 0.0;
                } else {
                    ud[rowOffset + col] = value;
                    if (row < col) {
                        ld[rowOffset + col] = 0.0;
                    }
                }
            }
            singular |= Tolerance.isZero(ud[rowOffset + row]);
        }

        PermutationVector permutation = new PermutationVector(n);
        for (int i = 0; i < n; i++) {
            int pivotRow = pivots[i];
            if (pivotRow >= 0 && pivotRow < n && pivotRow != i) {
                permutation.exchange(i, pivotRow);
            }
        }
        return new LUResult(original, L, U, permutation, singular);
    }

    private LUResult decomposeBlocked(Matrix A, int n) {
        Matrix U = A.copy();
        Matrix L = Matrix.Identity(n);
        PermutationVector P = new PermutationVector(n);
        double[] ud = U.getRawData();
        double[] ld = L.getRawData();
        boolean singular = false;

        int blockSize = Math.min(blockSize(), n);
        for (int kStart = 0; kStart < n; kStart += blockSize) {
            int kEnd = Math.min(kStart + blockSize, n);

            for (int k = kStart; k < kEnd; k++) {
                int pivotRow = selectPivotRow(U, ud, n, k);
                if (pivotRow != k) {
                    swapRows(ud, n, k, pivotRow, 0, n);
                    P.exchange(k, pivotRow);
                    swapRows(ld, n, k, pivotRow, 0, k);
                }

                int pivotOffset = k * n;
                double pivot = ud[pivotOffset + k];
                if (Tolerance.isZero(pivot)) {
                    singular = true;
                    continue;
                }

                for (int i = k + 1; i < n; i++) {
                    int rowOffset = i * n;
                    double factor = ud[rowOffset + k] / pivot;
                    ld[rowOffset + k] = factor;
                    ud[rowOffset + k] = 0.0;
                    for (int j = k + 1; j < kEnd; j++) {
                        ud[rowOffset + j] -= factor * ud[pivotOffset + j];
                    }
                }
            }

            if (kEnd >= n) {
                continue;
            }

            solveUpperPanel(ld, ud, n, kStart, kEnd);

            int trailing = n - kEnd;
            int panelWidth = kEnd - kStart;
            if (panelWidth > 0 && trailing > 0) {
                Gemm.gemmStrided(
                    ld, kEnd * n + kStart, n,
                    ud, kStart * n + kEnd, n,
                    ud, kEnd * n + kEnd, n,
                    trailing, panelWidth, trailing,
                    -1.0, 1.0, panelWidth
                );
            }
        }

        if (Tolerance.isZero(ud[(n - 1) * n + (n - 1)])) {
            singular = true;
        }

        return new LUResult(A, L, U, P, singular);
    }

    private static void solveUpperPanel(double[] l, double[] u, int n, int kStart, int kEnd) {
        for (int row = kStart + 1; row < kEnd; row++) {
            int rowOffset = row * n;
            for (int prev = kStart; prev < row; prev++) {
                double factor = l[rowOffset + prev];
                if (factor == 0.0) {
                    continue;
                }
                int prevOffset = prev * n;
                subtractScaledRowSegment(u, rowOffset, prevOffset, kEnd, n, factor);
            }
        }
    }

    private int selectPivotRow(Matrix matrix, double[] data, int n, int k) {
        if (pivotPolicy == PivotPolicy.NONE) {
            return k;
        }
        if (pivotPolicy == PivotPolicy.PARTIAL) {
            int maxRow = k;
            double maxAbs = Math.abs(data[k * n + k]);
            for (int row = k + 1; row < n; row++) {
                double abs = Math.abs(data[row * n + k]);
                if (abs > maxAbs) {
                    maxAbs = abs;
                    maxRow = row;
                }
            }
            return maxRow;
        }
        return pivotPolicy.selectPivotRow(matrix, k, k);
    }

    private static void subtractScaledRowSegment(double[] data, int rowOffset, int prevOffset,
                                                 int colStart, int colEnd, double factor) {
        int width = colEnd - colStart;
        if (width <= 0) {
            return;
        }

        int index = 0;
        int vectorBound = SPECIES.loopBound(width);
        DoubleVector factorVector = DoubleVector.broadcast(SPECIES, factor);
        for (; index < vectorBound; index += SPECIES.length()) {
            int target = rowOffset + colStart + index;
            int source = prevOffset + colStart + index;
            DoubleVector rowVector = DoubleVector.fromArray(SPECIES, data, target);
            DoubleVector prevVector = DoubleVector.fromArray(SPECIES, data, source);
            prevVector.mul(factorVector).neg().add(rowVector).intoArray(data, target);
        }
        for (; index < width; index++) {
            int col = colStart + index;
            data[rowOffset + col] = Math.fma(-factor, data[prevOffset + col], data[rowOffset + col]);
        }
    }

    private static void swapRows(double[] data, int ld, int rowA, int rowB, int colStart, int colEnd) {
        if (rowA == rowB || colStart >= colEnd) {
            return;
        }
        int offsetA = rowA * ld;
        int offsetB = rowB * ld;
        for (int col = colStart; col < colEnd; col++) {
            double temp = data[offsetA + col];
            data[offsetA + col] = data[offsetB + col];
            data[offsetB + col] = temp;
        }
    }

    private static int blockThreshold() {
        return integerProperty("net.faulj.decomposition.lu.blockThreshold", DEFAULT_BLOCK_THRESHOLD);
    }

    private static int blockSize() {
        return integerProperty("net.faulj.decomposition.lu.blockSize", DEFAULT_BLOCK_SIZE);
    }

    private static int integerProperty(String key, int fallback) {
        String value = System.getProperty(key);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Math.max(1, Integer.parseInt(value.trim()));
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }
}
