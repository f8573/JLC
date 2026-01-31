package net.faulj.eigen.schur;

import java.util.Arrays;
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
                        // Use all basis vectors (up to count) to span the eigenspace properly.
                        // This ensures the eigenvector matrix has full rank when geometric
                        // multiplicity equals algebraic multiplicity (diagonalizable case).
                        java.util.Iterator<Vector> basisIter = basis.iterator();
                        for (int k = i; k < i + count; k++) {
                            double[] data;
                            if (basisIter.hasNext()) {
                                // Use distinct basis vectors for each column
                                data = basisIter.next().getData();
                            } else {
                                // If fewer basis vectors than algebraic multiplicity (defective),
                                // reuse the first basis vector for remaining columns
                                data = basis.iterator().next().getData();
                            }
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

        // Validate real eigenvectors
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

        // Validate complex eigenvectors
        // For complex eigenvector v = vr + i*vi with eigenvalue λ = λr + i*λi:
        // (T - λI)v = 0 expands to:
        //   (T - λr*I)vr + λi*vi = 0   (real part)
        //   (T - λr*I)vi - λi*vr = 0   (imag part)
        for (int j = 0; j < n; j++) {
            if (Math.abs(valImag[j]) <= tol) continue;
            if (valImag[j] < 0) continue; // Only process positive imaginary (first of pair)

            int colReal = j;
            int colImag = j + 1;
            if (colImag >= n) continue;

            // Extract vr and vi columns
            double[] vr = new double[n];
            double[] vi = new double[n];
            for (int r = 0; r < n; r++) {
                vr[r] = evecData[r * n + colReal];
                vi[r] = evecData[r * n + colImag];
            }

            double lambdaRe = valReal[j];
            double lambdaIm = valImag[j];

            // Compute norms
            double vrNorm = 0.0, viNorm = 0.0;
            for (int i = 0; i < n; i++) {
                vrNorm += vr[i] * vr[i];
                viNorm += vi[i] * vi[i];
            }
            double vNorm = Math.sqrt(vrNorm + viNorm);

            // Compute residual norms for both equations
            double resNormSq = 0.0;
            for (int i = 0; i < n; i++) {
                int rowOff = i * n;

                // (T - λr*I) * vr
                double sumRealVr = 0.0;
                for (int m = 0; m < n; m++) {
                    sumRealVr += tData[rowOff + m] * vr[m];
                }
                sumRealVr -= lambdaRe * vr[i];

                // (T - λr*I) * vi
                double sumRealVi = 0.0;
                for (int m = 0; m < n; m++) {
                    sumRealVi += tData[rowOff + m] * vi[m];
                }
                sumRealVi -= lambdaRe * vi[i];

                // Real equation residual: (T - λr*I)vr + λi*vi
                double resReal = sumRealVr + lambdaIm * vi[i];
                // Imag equation residual: (T - λr*I)vi - λi*vr
                double resImag = sumRealVi - lambdaIm * vr[i];

                resNormSq += resReal * resReal + resImag * resImag;
            }
            double resNorm = Math.sqrt(resNormSq);

            double allowed = Tolerance.get() * Math.max(1.0, vNorm) * Math.max(1.0, maxAbsT);
            if (vNorm < Tolerance.get() || resNorm > allowed * 1e3) {
                // Eigenvector is invalid, try to recompute from the 2x2 block directly
                int blockRow = findComplexBlockRow(colReal);
                if (blockRow >= 0 && blockRow + 1 < n) {
                    recomputeComplexEigenvector(blockRow, colReal, colImag, evecData);
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

        this.eigenvectors = buildEigenvectorMatrix(resultData);
    }

    private Matrix buildEigenvectorMatrix(double[] resultData) {
        double tol = Tolerance.get();
        double[] realData = Arrays.copyOf(resultData, resultData.length);
        double[] imagData = null;
        boolean hasComplex = false;

        for (int j = 0; j < n; j++) {
            if (Math.abs(valImag[j]) <= tol) {
                continue;
            }
            if (valImag[j] < 0) {
                continue; // handled by the positive-imag partner
            }
            int partner = j + 1;
            if (partner >= n) {
                continue;
            }
            hasComplex = true;
            if (imagData == null) {
                imagData = new double[n * n];
            }
            for (int r = 0; r < n; r++) {
                int base = r * n;
                double vr = realData[base + j];
                double vi = realData[base + partner];
                realData[base + j] = vr;
                imagData[base + j] = vi;
                realData[base + partner] = vr;
                imagData[base + partner] = -vi;
            }
        }

        return hasComplex ? Matrix.wrap(realData, imagData, n, n) : Matrix.wrap(realData, n, n);
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

        // First pass: identify 2x2 blocks and initialize complex eigenvectors at those blocks
        // For complex eigenvalues from 2x2 blocks, we need to set the initial eigenvector
        // values at the block position before back-substitution can proceed.
        for (int j = startCol; j < endCol; j++) {
            if (fixed != null && fixed[j]) continue;
            if (Math.abs(valImag[j]) <= tol) continue;
            if (valImag[j] < 0) continue; // Skip the conjugate partner

            int colReal = j;
            int colImag = j + 1;
            if (colImag >= endCol) continue;

            // Find the 2x2 block that generated this complex eigenvalue pair
            // The block is located where valReal[j] matches and there's a non-zero subdiagonal
            int blockRow = findComplexBlockRow(colReal);
            if (blockRow >= 0 && blockRow + 1 < n) {
                // Initialize eigenvector components at the 2x2 block
                // For a 2x2 block [a b; c d] with eigenvalue λ = α + iβ:
                // The eigenvector v = vr + i*vi satisfies (A - λI)v = 0
                // At the block rows, we compute the eigenvector from the 2x2 system

                int i0 = blockRow;
                int i1 = blockRow + 1;

                double a = tData[i0 * n + i0];
                double b = tData[i0 * n + i1];
                double c = tData[i1 * n + i0];
                double d = tData[i1 * n + i1];

                double lambdaRe = valReal[colReal];
                double lambdaIm = valImag[colReal];

                // For the eigenvector, we solve (A - λI)v = 0 where λ = lambdaRe + i*lambdaIm
                // Setting one component to a canonical value and solving for the other
                // From row 1 of 2x2: c*v0 + (d - λ)*v1 = 0
                // v1 = -c*v0 / (d - λ)
                // Let v0 = 1 (or a suitable scaling)

                // For v = vr + i*vi, v0 = (vr0, vi0), v1 = (vr1, vi1)
                // Set vr0 = 1, vi0 = 0 (real unit in first component)
                // Then: (vr1 + i*vi1) = -c / ((d - lambdaRe) - i*lambdaIm)

                double denom_re = d - lambdaRe;
                double denom_im = -lambdaIm;
                double denom_mag_sq = denom_re * denom_re + denom_im * denom_im;

                if (denom_mag_sq < tol * tol) {
                    // Degenerate case - use first row equation instead
                    // (a - λ)*v0 + b*v1 = 0 => v0 = -b*v1 / (a - λ)
                    // Set v1 = 1
                    double num_re = a - lambdaRe;
                    double num_im = -lambdaIm;
                    double num_mag_sq = num_re * num_re + num_im * num_im;

                    if (num_mag_sq < tol * tol) {
                        // Both degenerate - set simple values
                        Y[i0 * n + colReal] = 1.0;
                        Y[i0 * n + colImag] = 0.0;
                        Y[i1 * n + colReal] = 0.0;
                        Y[i1 * n + colImag] = 1.0;
                    } else {
                        // Use: v0 = -b / (a - λ), v1 = 1
                        // -b / (num_re + i*num_im) = -b * (num_re - i*num_im) / num_mag_sq
                        Y[i1 * n + colReal] = 1.0;
                        Y[i1 * n + colImag] = 0.0;
                        Y[i0 * n + colReal] = -b * num_re / num_mag_sq;
                        Y[i0 * n + colImag] = b * num_im / num_mag_sq;
                    }
                } else {
                    // Normal case: v0 = 1, v1 = -c / (d - λ)
                    // -c / (denom_re + i*denom_im) = -c * (denom_re - i*denom_im) / denom_mag_sq
                    Y[i0 * n + colReal] = 1.0;
                    Y[i0 * n + colImag] = 0.0;
                    Y[i1 * n + colReal] = -c * denom_re / denom_mag_sq;
                    Y[i1 * n + colImag] = c * denom_im / denom_mag_sq;
                }
            }
        }

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

                    int colReal = j;
                    int colImag = j + 1;

                    // Check if k is at or below the 2x2 block that generated this eigenvalue
                    int blockRow = findComplexBlockRow(colReal);
                    if (blockRow >= 0 && k >= blockRow) {
                        // Skip - already initialized by the first pass
                        continue;
                    }

                    // Current j is the Real part column, j+1 is the Imag part column
                    solveComplexPixel(k, colReal, colImag, rowOffset, tRowOffset, Y);
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
     * Find the row index of the 2x2 block that corresponds to a complex eigenvalue.
     * Returns the first row of the 2x2 block, or -1 if not found.
     *
     * @param eigenIndex the eigenvalue index (column in eigenvector matrix)
     * @return first row of the 2x2 block, or -1
     */
    private int findComplexBlockRow(int eigenIndex) {
        double tol = Tolerance.get();
        double targetRe = valReal[eigenIndex];
        double targetIm = Math.abs(valImag[eigenIndex]);

        // Scan for 2x2 blocks in T that match this eigenvalue
        int j = n - 1;
        while (j >= 0) {
            if (j == 0 || Math.abs(tData[j * n + (j - 1)]) <= tol) {
                // 1x1 block at j
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

                if (disc < 0) {
                    // Complex eigenvalue pair from this block
                    double blockRe = tr / 2.0;
                    double blockIm = Math.sqrt(-disc) / 2.0;

                    if (Math.abs(blockRe - targetRe) < tol && Math.abs(blockIm - targetIm) < tol) {
                        return i0;
                    }
                }
                j -= 2;
            }
        }
        return -1;
    }

    /**
     * Recompute the complex eigenvector from the 2x2 block directly.
     * This is used as a fallback when back-substitution produces invalid results.
     *
     * @param blockRow first row of the 2x2 block
     * @param colReal column index for real component
     * @param colImag column index for imaginary component
     * @param evecData eigenvector data array
     */
    private void recomputeComplexEigenvector(int blockRow, int colReal, int colImag, double[] evecData) {
        double tol = Tolerance.get();
        int i0 = blockRow;
        int i1 = blockRow + 1;

        double a = tData[i0 * n + i0];
        double b = tData[i0 * n + i1];
        double c = tData[i1 * n + i0];
        double d = tData[i1 * n + i1];

        double lambdaRe = valReal[colReal];
        double lambdaIm = valImag[colReal];

        // Clear the eigenvector columns first
        for (int r = 0; r < n; r++) {
            evecData[r * n + colReal] = 0.0;
            evecData[r * n + colImag] = 0.0;
        }

        // Compute eigenvector from the 2x2 block
        // For (A - λI)v = 0, we set v0 = 1 and solve for v1

        double denom_re = d - lambdaRe;
        double denom_im = -lambdaIm;
        double denom_mag_sq = denom_re * denom_re + denom_im * denom_im;

        if (denom_mag_sq < tol * tol) {
            // Use first row equation: (a - λ)*v0 + b*v1 = 0
            double num_re = a - lambdaRe;
            double num_im = -lambdaIm;
            double num_mag_sq = num_re * num_re + num_im * num_im;

            if (num_mag_sq < tol * tol) {
                // Both degenerate
                evecData[i0 * n + colReal] = 1.0;
                evecData[i0 * n + colImag] = 0.0;
                evecData[i1 * n + colReal] = 0.0;
                evecData[i1 * n + colImag] = 1.0;
            } else {
                evecData[i1 * n + colReal] = 1.0;
                evecData[i1 * n + colImag] = 0.0;
                evecData[i0 * n + colReal] = -b * num_re / num_mag_sq;
                evecData[i0 * n + colImag] = b * num_im / num_mag_sq;
            }
        } else {
            evecData[i0 * n + colReal] = 1.0;
            evecData[i0 * n + colImag] = 0.0;
            evecData[i1 * n + colReal] = -c * denom_re / denom_mag_sq;
            evecData[i1 * n + colImag] = c * denom_im / denom_mag_sq;
        }

        // Now back-substitute upward for rows above the block
        for (int k = i0 - 1; k >= 0; k--) {
            int tRowOffset = k * n;
            double t_kk = tData[tRowOffset + k];

            double sumReal = 0.0;
            double sumImag = 0.0;

            for (int m = k + 1; m < n; m++) {
                double T_km = tData[tRowOffset + m];
                sumReal += T_km * evecData[m * n + colReal];
                sumImag += T_km * evecData[m * n + colImag];
            }

            double diag = t_kk - lambdaRe;
            double li = lambdaIm;

            double det = diag * diag + li * li;
            if (det < tol) det = tol;

            double rhsR = -sumReal;
            double rhsI = -sumImag;

            evecData[k * n + colReal] = (diag * rhsR - li * rhsI) / det;
            evecData[k * n + colImag] = (li * rhsR + diag * rhsI) / det;
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