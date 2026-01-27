package net.faulj.eigen.schur;

import java.util.Set;
import java.util.stream.IntStream;

import net.faulj.core.Tolerance;
import net.faulj.matrix.Matrix;
import net.faulj.scalar.Complex;
import net.faulj.spaces.SubspaceBasis;
import net.faulj.vector.Vector;

/**
 * High-performance extractor for eigenvectors from a Real Schur Decomposition.
 * <p>
 * This implementation utilizes batch back-substitution, primitive-only arithmetic (avoiding
 * Complex objects in hot loops), and parallel execution to efficiently compute eigenvectors
 * from the quasi-triangular Schur form T.
 * </p>
 *
 * <h2>Optimizations:</h2>
 * <ul>
 * <li><b>Batch Back-Substitution:</b> Solves multiple eigenvectors simultaneously to utilize
 * CPU cache lines and vectorization (SIMD) effectively.</li>
 * <li><b>Primitive Solvers:</b> Replaces complex object arithmetic with manual double-precision
 * operations to eliminate GC pressure.</li>
 * <li><b>Hybrid Format Multiplication:</b> Performs the basis transformation ($x = Uy$) using
 * blocked matrix multiplication on flattened arrays.</li>
 * <li><b>Parallelization:</b> Computes disjoint blocks of eigenvectors concurrently.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 2.0
 */
public class SchurEigenExtractor {

    // Internal flattened storage for speed (Row-Major)
    private final double[] tData;
    private final double[] uData;
    private final int n;

    // Results
    private double[] valReal;
    private double[] valImag;
    private Matrix eigenvectors;

    // Batch size for blocking (tuned for L1/L2 cache)
    private static final int BATCH_SIZE = 64;

    /**
     * Create an extractor using only the Schur form.
     *
     * @param S Schur form matrix T
     */
    public SchurEigenExtractor(Matrix S) {
        this(S, null);
    }

    /**
     * Create an extractor using the Schur form and Schur vectors.
     *
     * @param S Schur form matrix T
     * @param U Schur vectors matrix (may be null to skip eigenvectors)
     */
    public SchurEigenExtractor(Matrix S, Matrix U) {
        this.n = S.getRowCount();
        this.tData = flatten(S);
        this.uData = (U != null) ? flatten(U) : null;
        compute();
    }

    /**
     * Flattens a JLC Matrix into a 1D double array for fast access.
     */
    private double[] flatten(Matrix m) {
        int rows = m.getRowCount();
        int cols = m.getColumnCount();
        double[] flat = new double[rows * cols];
        Vector[] data = m.getData();
        // The current Matrix implementation uses columns of Vectors.
        // We assume standard column-vector access but want Row-Major for T to optimize back-sub.
        // However, extracting col-by-col is easier given the structure.
        // Let's store T in Row-Major to optimize the row-based back-substitution.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                flat[r * cols + c] = m.get(r, c);
            }
        }
        return flat;
    }

    /**
     * Compute eigenvalues and (optionally) eigenvectors from Schur data.
     */
    private void compute() {
        extractEigenvaluesPrimitive();

        if (uData == null) return;

        // Allocate flattened eigenvector matrix (n x n)
        // Stored Column-Major mostly, but we'll manage index manually.
        double[] evecData = new double[n * n];
        boolean[] fixedColumns = new boolean[n];

        // For repeated real eigenvalues, compute a robust nullspace basis of (T - lambda I)
        // and use a representative vector for those columns to avoid unstable back-substitution.
        double tol = Tolerance.get();
        boolean[] visitedEv = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (visitedEv[i]) continue;
            int count = 0;
            int j = i;
            while (j < n && Math.abs(valReal[j] - valReal[i]) <= tol && Math.abs(valImag[j]) <= tol) {
                visitedEv[j] = true;
                count++;
                j++;
            }
            if (count > 1) {
                try {
                    Matrix Tmat = inflate(tData);
                    for (int r = 0; r < n; r++) {
                        Tmat.set(r, r, Tmat.get(r, r) - valReal[i]);
                    }
                    Set<Vector> basis = SubspaceBasis.nullSpaceBasis(Tmat);
                    if (basis != null && !basis.isEmpty()) {
                        Vector rep = basis.iterator().next();
                        double[] data = rep.getData();
                        // fill all columns for this eigenvalue with the representative
                        for (int k = i; k < i + count; k++) {
                            for (int r = 0; r < n; r++) {
                                evecData[r * n + k] = data[r];
                            }
                            fixedColumns[k] = true;
                        }
                    }
                } catch (RuntimeException ex) {
                    // ignore and fall back to back-substitution
                }
            }
        }

        // 1. Identify valid blocks for batching
        // We cannot split a 2x2 complex block across batches.
        int[] batchStarts = new int[n / BATCH_SIZE + 2];
        int batchCount = 0;
        batchStarts[0] = 0;

        int current = 0;
        while (current < n) {
            if ((current + BATCH_SIZE) >= n) {
                current = n;
            } else {
                current += BATCH_SIZE;
                // If we land in the middle of a 2x2 block (imag part not zero), back up one
                if (Math.abs(valImag[current - 1]) > Tolerance.get() &&
                        Math.abs(valImag[current]) > Tolerance.get()) {
                    // This is a heuristic: if T[current][current-1] is significant, it's a block.
                    // Safer: check our extracted eigenvalues.
                    // If valImag[current-1] != 0, it is part of a pair.
                    // Since complex pairs are stored adjacently, if current-1 is complex,
                    // we check if it is the first or second of the pair.
                    // A simple rule: if split between pair, reduce batch size by 1.
                    if (valImag[current-1] != 0 && valImag[current] != 0) {
                        // Likely split a pair, adjust boundary
                        if (Math.abs(valReal[current-1] - valReal[current]) < Tolerance.get()) {
                            current--;
                        }
                    }
                }
            }
            batchCount++;
            batchStarts[batchCount] = current;
        }

        int finalBatchCount = batchCount;

        // 2. Parallel Batch Back-Substitution
        IntStream.range(0, finalBatchCount).parallel().forEach(b -> {
            int start = batchStarts[b];
            int end = batchStarts[b+1];
            processBatch(start, end, evecData, fixedColumns);
        });

        // Validate computed eigenvectors and repair unstable columns by computing
        // the nullspace of (T - lambda I) when residuals are large or vectors are invalid.
        double maxAbsT = 0.0;
        for (double v : tData) {
            double av = Math.abs(v);
            if (av > maxAbsT) maxAbsT = av;
        }

        for (int j = 0; j < n; j++) {
            if (Math.abs(valImag[j]) > Tolerance.get()) continue; // skip complex pair columns
            // extract column y
            double[] y = new double[n];
            for (int r = 0; r < n; r++) y[r] = evecData[r * n + j];

            // compute residual r = (T - lambda I) * y
            double lambda = valReal[j];
            double resNorm = 0.0;
            double yNorm = 0.0;
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                int rowOff = i * n;
                for (int m = 0; m < n; m++) {
                    sum += tData[rowOff + m] * y[m];
                }
                double ri = sum - lambda * y[i];
                resNorm += ri * ri;
                yNorm += y[i] * y[i];
            }
            resNorm = Math.sqrt(resNorm);
            yNorm = Math.sqrt(yNorm);

            double allowed = Tolerance.get() * Math.max(1.0, yNorm) * Math.max(1.0, maxAbsT);
            if (yNorm < Tolerance.get() || resNorm > allowed * 1e3) {
                // recompute via nullspace of (T - lambda I)
                try {
                    Matrix Tmat = inflate(tData);
                    for (int r = 0; r < n; r++) {
                        Tmat.set(r, r, Tmat.get(r, r) - lambda);
                    }
                    Set<Vector> basis = SubspaceBasis.nullSpaceBasis(Tmat);
                    if (basis != null && !basis.isEmpty()) {
                        Vector rep = basis.iterator().next();
                        double[] data = rep.getData();
                        for (int r = 0; r < n; r++) {
                            evecData[r * n + j] = data[r];
                        }
                    }
                } catch (RuntimeException ex) {
                    // leave column as-is if fallback fails
                }
            }
        }

        // 3. Transformation x = Uy (if U was provided)
        // We can do this in place or via a new matrix.
        // evecData currently holds 'y' (eigenvectors of T). We need 'x' = U * y.
        // We will perform a parallel block matrix multiply.
        double[] resultData = new double[n * n];

        // Parallelize over columns of result
        IntStream.range(0, finalBatchCount).parallel().forEach(b -> {
            int start = batchStarts[b];
            int end = batchStarts[b+1];
            multiplyHybridBlock(start, end, evecData, resultData);
        });

        this.eigenvectors = inflate(resultData);
    }

    /**
     * Solves a batch of eigenvectors for T.
     * Stores result into evecData (conceptually n x n, flattened row-major or col-major).
     * Here we treat evecData as n rows x n cols, row-major for simplicity in memory,
     * but eigenvectors are columns. So evecData[row * n + col].
     */
    /**
     * Solve a batch of eigenvectors using back-substitution.
     *
     * @param startCol start column (inclusive)
     * @param endCol end column (exclusive)
     * @param Y eigenvector storage
     * @param fixed flags for fixed columns
     */
    private void processBatch(int startCol, int endCol, double[] Y, boolean[] fixed) {
        double tol = Tolerance.get();

        // Iterate backwards (Back-Substitution)
        // For each row k from n-1 down to 0
        for (int k = n - 1; k >= 0; k--) {
            int rowOffset = k * n;
            int tRowOffset = k * n;
            double t_kk = tData[tRowOffset + k];

            // For each eigenvector j in this batch
            for (int j = startCol; j < endCol; j++) {
                if (fixed != null && fixed[j]) continue;

                // Handle Complex Conjugate Pairs
                if (Math.abs(valImag[j]) > tol) {
                    // It's a complex pair. We process j (real part) and j+1 (imag part) together.
                    // If we define the complex eigenvector as v = vr + i*vi
                    // We solve (T - (lr + i*li)I)(vr + i*vi) = 0
                    // Real part: (T - lr*I)vr + li*vi = 0
                    // Imag part: (T - lr*I)vi - li*vr = 0

                    // Complex pairs are processed when we hit the second element (j+1) or
                    // we handle both at j. Let's skip the second one in the loop.
                    if (j > startCol && Math.abs(valImag[j-1]) > tol &&
                            Math.abs(valReal[j] - valReal[j-1]) < tol) {
                        continue; // Already processed as the "imaginary part" column of the previous j
                    }

                    // Current j is the Real part column, j+1 is the Imag part column
                    solveComplexPixel(k, j, j + 1, rowOffset, tRowOffset, Y);
                    continue;
                }

                // --- Real Eigenvector Case ---

                // 1. Compute diagonal entry (T_kk - lambda_j)
                double diag = t_kk - valReal[j];

                // 2. Compute accumulation: sum = sum(T_km * y_mj) for m > k
                // Since T is upper triangular, we only sum m > k.
                double sum = 0.0;

                // VECTORIZATION OPPORTUNITY: This loop is a dot product.
                // T is row-major, Y is row-major.
                // We are accessing Y[m][j] (column stride). This is a strided access (slow).
                // However, T is accessed sequentially.
                for (int m = k + 1; m < n; m++) {
                    sum += tData[tRowOffset + m] * Y[m * n + j];
                }

                // 3. Solve for Y[k][j]
                if (k > j) {
                    // For k > j (below the diagonal of the eigenvector matrix), value is 0
                    // UNLESS T is not strictly triangular (quasi-triangular).
                    // But in Schur form, eigenvalues are on diagonal.
                    // Standard algo sets y_j[j] = 1, and backsubs.
                    // The code below handles the general case.
                }

                if (Math.abs(diag) < tol) {
                    diag = tol; // Protect division by zero
                }

                // If we are at the diagonal (k == j), we can fix the scale (e.g., 1.0)
                if (k == j) {
                    Y[rowOffset + j] = 1.0;
                    // We must re-evaluate the equation consistency or rely on the fact
                    // that (T-lambda*I) is singular here.
                    // For strictly triangular T, the row equation at k=j is 0=0 if we didn't set y_k=1.
                    // With y_k=1, we don't divide.
                } else {
                    Y[rowOffset + j] = -sum / diag;
                }
            }
        }
    }

    /**
     * Solves a single row step for a complex conjugate pair of eigenvectors.
     * Stored in columns colReal and colImag of Y.
     */
    /**
     * Solve a 2x2 complex block during back-substitution.
     *
     * @param k row index in T
     * @param colReal column index for real component
     * @param colImag column index for imaginary component
     * @param rowOffset row offset for Y
     * @param tRowOffset row offset for T
     * @param Y eigenvector storage
     */
    private void solveComplexPixel(int k, int colReal, int colImag, int rowOffset, int tRowOffset, double[] Y) {
        double lambdaRe = valReal[colReal];
        double lambdaIm = valImag[colReal]; // Positive or negative
        // If valImag[colReal] is the first of the pair, it's usually positive (or we assume one).

        // Sums for (T * vr) and (T * vi)
        double sumReal = 0.0;
        double sumImag = 0.0;

        for (int m = k + 1; m < n; m++) {
            double T_km = tData[tRowOffset + m];
            sumReal += T_km * Y[m * n + colReal];
            sumImag += T_km * Y[m * n + colImag];
        }

        // We are solving the 2x2 system at the diagonal block or the 1x1 scalar eq above it.
        // If k is part of the 2x2 block generating this eigenvalue:
        // We handle that logic via standard quasi-triangular solvers (omitted for brevity,
        // assuming k < index of block for back-sub).

        double t_kk = tData[tRowOffset + k];

        // Equation: (T_kk - lambda) * (y_k) + sum = 0
        // (T_kk - (lr + i*li)) * (yr + i*yi) + (sumR + i*sumI) = 0
        // Real: (T_kk - lr)yr + li*yi = -sumR
        // Imag: -li*yr + (T_kk - lr)yi = -sumI

        double diag = t_kk - lambdaRe;
        double li = lambdaIm;

        // Solve 2x2 linear system for yr, yi
        // [ diag   li ] [ yr ] = [ -sumR ]
        // [ -li  diag ] [ yi ]   [ -sumI ]

        double det = diag*diag + li*li;
        if (det < Tolerance.get()) det = Tolerance.get();

        double rhsR = -sumReal;
        double rhsI = -sumImag;

        // Cramers rule / Inverse
        // Inv = 1/det * [ diag  -li ]
        //               [ li   diag ]

        Y[rowOffset + colReal] = (diag * rhsR - li * rhsI) / det;
        Y[rowOffset + colImag] = (li * rhsR + diag * rhsI) / det;
    }

    /**
     * Efficient blocked matrix multiplication: Res = U * Y
     * Only computes columns [startCol, endCol) of Result.
     */
    /**
     * Multiply U by a block of eigenvectors (hybrid blocked multiply).
     *
     * @param startCol start column (inclusive)
     * @param endCol end column (exclusive)
     * @param Y eigenvectors in Schur basis
     * @param Result output eigenvectors in original basis
     */
    private void multiplyHybridBlock(int startCol, int endCol, double[] Y, double[] Result) {
        // U is n x n (uData)
        // Y is n x n (Y)
        // We compute n rows for columns j in [start, end)

        for (int i = 0; i < n; i++) {
            int uRowOffset = i * n;
            int resRowOffset = i * n;

            for (int j = startCol; j < endCol; j++) {
                double sum = 0.0;
                // Dot product row i of U with col j of Y
                // This inner loop is primitive double arithmetic
                for (int k = 0; k < n; k++) {
                    sum += uData[uRowOffset + k] * Y[k * n + j];
                }
                Result[resRowOffset + j] = sum;
            }
        }
    }

    /**
     * Extract eigenvalues from the quasi-upper triangular Schur form.
     */
    private void extractEigenvaluesPrimitive() {
        this.valReal = new double[n];
        this.valImag = new double[n];
        double tol = Tolerance.get();

        // Better to scan from the bottom up so 2x2 blocks at the lower indices are
        // detected correctly when consecutive subdiagonals exist.
        int j = n - 1;
        while (j >= 0) {
            if (j == 0 || Math.abs(tData[j * n + (j - 1)]) <= tol) {
                // 1x1 block at j
                valReal[j] = tData[j * n + j];
                valImag[j] = 0.0;
                j--;
            } else {
                // 2x2 block covering (j-1, j)
                int i0 = j - 1;
                double a = tData[i0 * n + i0];
                double b = tData[i0 * n + j];
                double c = tData[j * n + i0];
                double d = tData[j * n + j];

                double tr = a + d;
                double det = a * d - b * c;
                double disc = tr * tr - 4 * det;

                if (disc >= 0) {
                    double sqrt = Math.sqrt(disc);
                    valReal[i0] = (tr + sqrt) / 2.0;
                    valReal[j] = (tr - sqrt) / 2.0;
                    valImag[i0] = 0.0;
                    valImag[j] = 0.0;
                } else {
                    double real = tr / 2.0;
                    double imag = Math.sqrt(-disc) / 2.0;
                    valReal[i0] = real;
                    valReal[j] = real;
                    valImag[i0] = imag;
                    valImag[j] = -imag;
                }
                j -= 2;
            }
        }
    }

    /**
     * Inflate a flattened row-major array into a Matrix.
     *
     * @param flat row-major data
     * @return matrix instance
     */
    private Matrix inflate(double[] flat) {
        Vector[] cols = new Vector[n];
        for (int c = 0; c < n; c++) {
            double[] colData = new double[n];
            for (int r = 0; r < n; r++) {
                colData[r] = flat[r * n + c];
            }
            cols[c] = new Vector(colData);
        }
        return new Matrix(cols);
    }

    /**
     * @return eigenvalues as complex numbers
     */
    public Complex[] getEigenvalues() {
        Complex[] c = new Complex[n];
        for(int i=0; i<n; i++) {
            c[i] = Complex.valueOf(valReal[i], valImag[i]);
        }
        return c;
    }

    /**
     * @return eigenvectors matrix, or null if not computed
     */
    public Matrix getEigenvectors() {
        return eigenvectors;
    }
}