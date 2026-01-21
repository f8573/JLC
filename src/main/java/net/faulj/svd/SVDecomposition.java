package net.faulj.svd;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.SVDResult;

/**
 * Computes the Singular Value Decomposition (SVD) of a matrix.
 * <p>
 * This class performs the factorization A = UΣV<sup>T</sup> where:
 * </p>
 * <ul>
 * <li><b>A</b> - Original m×n rectangular real matrix</li>
 * <li><b>U</b> - m×m Orthogonal matrix (U<sup>T</sup>U = I) containing left singular vectors</li>
 * <li><b>Σ</b> - m×n Diagonal matrix containing singular values (σ)</li>
 * <li><b>V</b> - n×n Orthogonal matrix (V<sup>T</sup>V = I) containing right singular vectors</li>
 * </ul>
 *
 * <h2>Mathematical Definition:</h2>
 * <p>
 * For any real matrix A, there exist orthogonal matrices U and V such that:
 * </p>
 * <pre>
 * A = U * diag(σ₁, σ₂, ..., σᵣ) * Vᵀ
 * </pre>
 * <p>
 * Where singular values are non-negative and ordered: σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0.
 * </p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = MatrixFactory.createRandom(5, 3);
 *
 * // Perform SVD
 * SVDecomposition svd = new SVDecomposition();
 * SVDResult result = svd.decompose(A);
 *
 * // Access components
 * Matrix U = result.getU();       // Left singular vectors
 * double[] S = result.getSingularValues(); // Singular values vector
 * Matrix V = result.getV();       // Right singular vectors
 *
 * // Verify: A ≈ U * Σ * V^T
 * Matrix Sigma = MatrixFactory.diagonal(S, 5, 3);
 * Matrix reconstructed = U.multiply(Sigma).multiply(V.transpose());
 * }</pre>
 *
 * <h2>Computational Path:</h2>
 * <ol>
 * <li><b>Bidiagonalization:</b> Reduce A to bidiagonal form B using Householder reflections (Golub-Kahan).
 * <br>A = P * B * Qᵀ</li>
 * <li><b>Diagonalization:</b> Apply implicit QR iteration with Wilkinson shift to B to zero out the superdiagonal.
 * <br>B → Σ</li>
 * <li><b>Accumulation:</b> Accumulate Householder and Givens rotations to form U and V.</li>
 * </ol>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li><b>Dimensionality Reduction:</b> Principal Component Analysis (PCA)</li>
 * <li><b>Data Compression:</b> Low-rank matrix approximation (Eckart-Young-Mirsky theorem)</li>
 * <li><b>Pseudoinverse:</b> Computing the Moore-Penrose inverse</li>
 * <li><b>Noise Reduction:</b> Truncated SVD to remove small singular values</li>
 * <li><b>Condition Number:</b> Ratio of largest to smallest singular value</li>
 * </ul>
 *
 * <h2>Performance Notes:</h2>
 * <ul>
 * <li>Complexity is approximately O(mn² + n³) for m ≥ n</li>
 * <li>This implementation computes the "Full SVD" (U is m×m)</li>
 * <li>For m &gt;&gt; n, consider using {@link ThinSVD} for efficiency</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.result.SVDResult
 * @see ThinSVD
 * @see BidiagonalQR
 */
public class SVDecomposition {
	private static final double TOL = 1e-12;

	public SVDecomposition() {
	}

	public SVDResult decompose(Matrix A) {
		if (A == null) {
			throw new IllegalArgumentException("Matrix must not be null");
		}
		net.faulj.decomposition.bidiagonal.Bidiagonalization bidiagonalization =
				new net.faulj.decomposition.bidiagonal.Bidiagonalization();
		net.faulj.decomposition.result.BidiagonalizationResult bidiag = bidiagonalization.decompose(A);

		Matrix U0 = bidiag.getU();
		Matrix B = bidiag.getB();
		Matrix V0 = bidiag.getV();

		BidiagonalSVDResult bidiagSvd = computeBidiagonalSvd(B);

		Matrix U = U0.multiply(bidiagSvd.U);
		Matrix V = V0.multiply(bidiagSvd.V);
		double[] singularValues = bidiagSvd.singularValues;

		int r = Math.min(A.getRowCount(), A.getColumnCount());
		int[] order = sortIndicesDescending(singularValues);
		singularValues = reorderValues(singularValues, order);
		U = reorderColumns(U, order, r);
		V = reorderColumns(V, order, r);

		return new SVDResult(A, U, singularValues, V);
	}

	private static BidiagonalSVDResult computeBidiagonalSvd(Matrix B) {
		int m = B.getRowCount();
		int n = B.getColumnCount();
		if (m >= n) {
			double[] diag = extractDiagonal(B, n);
			double[] superDiag = extractSuperDiagonal(B, n);
			TridiagonalData t = buildTridiagonalFromUpper(diag, superDiag);
			EigenDecomposition eig = tridiagonalEigenDecompose(t.diagonal, t.offDiagonal);
			double[] sigma = toSingularValues(eig.values);
			Matrix V = eig.vectors;
			Matrix BV = B.multiply(V);
			Matrix Uthin = scaleColumns(BV, sigma);
			Matrix U = completeOrthonormalBasis(Uthin);
			return new BidiagonalSVDResult(U, V, sigma);
		}
		double[] diag = extractDiagonal(B, m);
		double[] subDiag = extractSubDiagonal(B, m);
		TridiagonalData t = buildTridiagonalFromLower(diag, subDiag);
		EigenDecomposition eig = tridiagonalEigenDecompose(t.diagonal, t.offDiagonal);
		double[] sigma = toSingularValues(eig.values);
		Matrix U = eig.vectors;
		Matrix BTU = B.transpose().multiply(U);
		Matrix Vthin = scaleColumns(BTU, sigma);
		Matrix V = completeOrthonormalBasis(Vthin);
		return new BidiagonalSVDResult(U, V, sigma);
	}

	private static double[] extractDiagonal(Matrix B, int len) {
		double[] diag = new double[len];
		for (int i = 0; i < len; i++) {
			diag[i] = B.get(i, i);
		}
		return diag;
	}

	private static double[] extractSuperDiagonal(Matrix B, int len) {
		if (len <= 1) return new double[0];
		double[] sup = new double[len - 1];
		for (int i = 0; i < len - 1; i++) {
			sup[i] = B.get(i, i + 1);
		}
		return sup;
	}

	private static double[] extractSubDiagonal(Matrix B, int len) {
		if (len <= 1) return new double[0];
		double[] sub = new double[len - 1];
		for (int i = 0; i < len - 1; i++) {
			sub[i] = B.get(i + 1, i);
		}
		return sub;
	}

	private static TridiagonalData buildTridiagonalFromUpper(double[] d, double[] e) {
		int n = d.length;
		double[] diag = new double[n];
		double[] off = new double[Math.max(0, n - 1)];
		for (int i = 0; i < n; i++) {
			double val = d[i] * d[i];
			if (i > 0) val += e[i - 1] * e[i - 1];
			diag[i] = val;
		}
		for (int i = 0; i < n - 1; i++) {
			off[i] = d[i] * e[i];
		}
		return new TridiagonalData(diag, off);
	}

	private static TridiagonalData buildTridiagonalFromLower(double[] d, double[] f) {
		int n = d.length;
		double[] diag = new double[n];
		double[] off = new double[Math.max(0, n - 1)];
		for (int i = 0; i < n; i++) {
			double val = d[i] * d[i];
			if (i > 0) val += f[i - 1] * f[i - 1];
			diag[i] = val;
		}
		for (int i = 0; i < n - 1; i++) {
			off[i] = d[i] * f[i];
		}
		return new TridiagonalData(diag, off);
	}

	private static double[] toSingularValues(double[] eigenvalues) {
		double[] sigma = new double[eigenvalues.length];
		for (int i = 0; i < eigenvalues.length; i++) {
			double val = eigenvalues[i];
			if (val < 0 && Math.abs(val) < 1e-12) {
				val = 0.0;
			}
			sigma[i] = Math.sqrt(Math.max(val, 0.0));
		}
		return sigma;
	}

	private static Matrix scaleColumns(Matrix M, double[] sigma) {
		int cols = M.getColumnCount();
		net.faulj.vector.Vector[] data = new net.faulj.vector.Vector[cols];
		for (int col = 0; col < cols; col++) {
			double scale = 0.0;
			if (col < sigma.length && Math.abs(sigma[col]) > TOL) {
				scale = 1.0 / sigma[col];
			}
			if (scale == 0.0) {
				data[col] = net.faulj.vector.VectorUtils.zero(M.getRowCount());
			} else {
				data[col] = M.getData()[col].multiplyScalar(scale);
			}
		}
		return new Matrix(data);
	}

	private static Matrix completeOrthonormalBasis(Matrix thin) {
		int rows = thin.getRowCount();
		int cols = thin.getColumnCount();
		if (rows == cols) {
			return thin;
		}
		net.faulj.vector.Vector[] basis = new net.faulj.vector.Vector[rows];
		java.util.List<net.faulj.vector.Vector> accepted = new java.util.ArrayList<>();
		for (int i = 0; i < cols; i++) {
			net.faulj.vector.Vector col = thin.getData()[i].copy();
			double norm = col.norm2();
			net.faulj.vector.Vector candidate = col;
			if (norm <= TOL) {
				candidate = null;
			}
			if (candidate != null) {
				candidate = orthogonalize(candidate, accepted);
			}
			double candNorm = candidate == null ? 0.0 : candidate.norm2();
			if (candNorm <= TOL) {
				for (int b = 0; b < rows; b++) {
					double[] data = new double[rows];
					data[b] = 1.0;
					net.faulj.vector.Vector fallback = new net.faulj.vector.Vector(data);
					fallback = orthogonalize(fallback, accepted);
					double fallbackNorm = fallback.norm2();
					if (fallbackNorm > TOL) {
						candidate = fallback;
						candNorm = fallbackNorm;
						break;
					}
				}
			}
			if (candidate != null && candNorm > TOL) {
				candidate = candidate.multiplyScalar(1.0 / candNorm);
			} else if (candidate == null) {
				candidate = net.faulj.vector.VectorUtils.zero(rows);
			}
			basis[i] = candidate;
			accepted.add(candidate);
		}
		int count = cols;
		for (int i = 0; i < rows && count < rows; i++) {
			double[] data = new double[rows];
			data[i] = 1.0;
			net.faulj.vector.Vector v = new net.faulj.vector.Vector(data);
			v = orthogonalize(v, accepted);
			double norm = v.norm2();
			if (norm > TOL) {
				v = v.multiplyScalar(1.0 / norm);
				basis[count++] = v;
				accepted.add(v);
			}
		}
		if (count < rows) {
			for (int i = 0; i < rows && count < rows; i++) {
				double[] data = new double[rows];
				data[i] = 1.0;
				basis[count++] = new net.faulj.vector.Vector(data);
			}
		}
		return new Matrix(basis);
	}

	private static net.faulj.vector.Vector orthogonalize(
			net.faulj.vector.Vector v,
			java.util.List<net.faulj.vector.Vector> basis) {
		net.faulj.vector.Vector result = v.copy();
		for (net.faulj.vector.Vector bj : basis) {
			double dot = 0.0;
			for (int r = 0; r < result.dimension(); r++) {
				dot += bj.get(r) * result.get(r);
			}
			for (int r = 0; r < result.dimension(); r++) {
				result.getData()[r] -= dot * bj.get(r);
			}
		}
		return result;
	}

	private static int[] sortIndicesDescending(double[] values) {
		Integer[] order = new Integer[values.length];
		for (int i = 0; i < values.length; i++) {
			order[i] = i;
		}
		java.util.Arrays.sort(order, (a, b) -> Double.compare(values[b], values[a]));
		int[] result = new int[values.length];
		for (int i = 0; i < values.length; i++) {
			result[i] = order[i];
		}
		return result;
	}

	private static double[] reorderValues(double[] values, int[] order) {
		double[] reordered = new double[values.length];
		for (int i = 0; i < order.length; i++) {
			reordered[i] = values[order[i]];
		}
		return reordered;
	}

	private static Matrix reorderColumns(Matrix M, int[] order, int reorderCount) {
		int cols = M.getColumnCount();
		net.faulj.vector.Vector[] data = new net.faulj.vector.Vector[cols];
		for (int i = 0; i < reorderCount; i++) {
			data[i] = M.getData()[order[i]].copy();
		}
		for (int i = reorderCount; i < cols; i++) {
			data[i] = M.getData()[i].copy();
		}
		return new Matrix(data);
	}

	private static EigenDecomposition tridiagonalEigenDecompose(double[] diag, double[] off) {
		int n = diag.length;
		if (n == 1) {
			return new EigenDecomposition(new double[]{diag[0]}, Matrix.Identity(1));
		}
		if (n == 2) {
			double d0 = diag[0];
			double d1 = diag[1];
			double e = off[0];
			if (Math.abs(e) < TOL) {
				Matrix V = Matrix.Identity(2);
				double[] vals = new double[]{Math.min(d0, d1), Math.max(d0, d1)};
				if (d0 > d1) {
					V = reorderColumns(V, new int[]{1, 0}, 2);
				}
				return new EigenDecomposition(vals, V);
			}
			double t = 0.5 * (d0 + d1);
			double diff = 0.5 * (d0 - d1);
			double root = Math.hypot(diff, e);
			double lambda1 = t - root;
			double lambda2 = t + root;
			double[] v1 = eigenvector2x2(d0, d1, e, lambda1);
			double[] v2 = eigenvector2x2(d0, d1, e, lambda2);
			Matrix V = new Matrix(new net.faulj.vector.Vector[]{
					new net.faulj.vector.Vector(v1),
					new net.faulj.vector.Vector(v2)
			});
			return new EigenDecomposition(new double[]{lambda1, lambda2}, V);
		}

		int split = n / 2;
		double beta = off[split - 1];
		double[] d1 = java.util.Arrays.copyOfRange(diag, 0, split);
		double[] e1 = java.util.Arrays.copyOfRange(off, 0, split - 1);
		double[] d2 = java.util.Arrays.copyOfRange(diag, split, n);
		double[] e2 = java.util.Arrays.copyOfRange(off, split, n - 1);
		d1[split - 1] -= beta;
		d2[0] -= beta;

		EigenDecomposition left = tridiagonalEigenDecompose(d1, e1);
		EigenDecomposition right = tridiagonalEigenDecompose(d2, e2);

		Matrix Q = blockDiagonal(left.vectors, right.vectors);
		double[] values = concat(left.values, right.values);
		double[] z = new double[n];
		for (int i = 0; i < split; i++) {
			z[i] = left.vectors.get(split - 1, i);
		}
		for (int i = 0; i < n - split; i++) {
			z[split + i] = right.vectors.get(0, i);
		}

		SortedEigenData sorted = sortEigenData(values, Q, z);
		if (Math.abs(beta) < TOL) {
			return new EigenDecomposition(sorted.values, sorted.vectors);
		}

		java.util.List<Integer> active = new java.util.ArrayList<>();
		java.util.List<EigenPair> deflated = new java.util.ArrayList<>();
		for (int i = 0; i < sorted.values.length; i++) {
			if (Math.abs(sorted.z[i]) <= TOL) {
				deflated.add(new EigenPair(sorted.values[i], sorted.vectors.getData()[i].copy()));
			} else {
				active.add(i);
			}
		}

		if (active.isEmpty()) {
			return new EigenDecomposition(sorted.values, sorted.vectors);
		}

		double[] dActive = new double[active.size()];
		double[] zActive = new double[active.size()];
		for (int i = 0; i < active.size(); i++) {
			int idx = active.get(i);
			dActive[i] = sorted.values[idx];
			zActive[i] = sorted.z[idx];
		}

		java.util.List<EigenPair> combined = new java.util.ArrayList<>(sorted.values.length);
		combined.addAll(deflated);
		for (int i = 0; i < dActive.length; i++) {
			double lambda = solveSecular(dActive, zActive, beta, i);
			double[] w = new double[dActive.length];
			double norm = 0.0;
			for (int j = 0; j < dActive.length; j++) {
				w[j] = zActive[j] / (dActive[j] - lambda);
				norm += w[j] * w[j];
			}
			norm = Math.sqrt(norm);
			for (int j = 0; j < dActive.length; j++) {
				w[j] /= norm;
			}
			double[] wFull = new double[n];
			for (int j = 0; j < active.size(); j++) {
				wFull[active.get(j)] = w[j];
			}
			double[] vec = multiplyMatrixVector(sorted.vectors, wFull);
			combined.add(new EigenPair(lambda, new net.faulj.vector.Vector(vec)));
		}

		combined.sort(java.util.Comparator.comparingDouble(p -> p.value));
		double[] finalValues = new double[n];
		net.faulj.vector.Vector[] finalVectors = new net.faulj.vector.Vector[n];
		for (int i = 0; i < combined.size(); i++) {
			EigenPair pair = combined.get(i);
			finalValues[i] = pair.value;
			finalVectors[i] = pair.vector;
		}
		return new EigenDecomposition(finalValues, new Matrix(finalVectors));
	}

	private static double[] eigenvector2x2(double d0, double d1, double e, double lambda) {
		double a = d0 - lambda;
		double b = e;
		double x;
		double y;
		if (Math.abs(a) > Math.abs(b)) {
			x = -b;
			y = a;
		} else {
			x = b;
			y = -a;
		}
		double norm = Math.hypot(x, y);
		if (norm < TOL) {
			return new double[]{1.0, 0.0};
		}
		return new double[]{x / norm, y / norm};
	}

	private static double solveSecular(double[] d, double[] z, double rho, int index) {
		int n = d.length;
		double eps = 1e-12;
		double left;
		double right;
		if (rho > 0) {
			if (index < n - 1) {
				left = d[index] + eps * (Math.abs(d[index]) + 1.0);
				right = d[index + 1] - eps * (Math.abs(d[index + 1]) + 1.0);
			} else {
				left = d[n - 1] + eps * (Math.abs(d[n - 1]) + 1.0);
				right = left + Math.max(1.0, Math.abs(left));
				while (secularValue(right, d, z, rho) < 0.0) {
					right += Math.max(1.0, Math.abs(right));
				}
			}
		} else {
			if (index == 0) {
				right = d[0] - eps * (Math.abs(d[0]) + 1.0);
				left = right - Math.max(1.0, Math.abs(right));
				while (secularValue(left, d, z, rho) < 0.0) {
					left -= Math.max(1.0, Math.abs(left));
				}
			} else {
				left = d[index - 1] + eps * (Math.abs(d[index - 1]) + 1.0);
				right = d[index] - eps * (Math.abs(d[index]) + 1.0);
			}
		}

		double fLeft = secularValue(left, d, z, rho);
		double fRight = secularValue(right, d, z, rho);
		for (int iter = 0; iter < 100; iter++) {
			double mid = 0.5 * (left + right);
			double fMid = secularValue(mid, d, z, rho);
			if (Math.abs(fMid) < 1e-14 || Math.abs(right - left) < 1e-12) {
				return mid;
			}
			if (fLeft * fMid < 0.0) {
				right = mid;
				fRight = fMid;
			} else {
				left = mid;
				fLeft = fMid;
			}
		}
		return 0.5 * (left + right);
	}

	private static double secularValue(double lambda, double[] d, double[] z, double rho) {
		double sum = 1.0;
		for (int i = 0; i < d.length; i++) {
			double denom = d[i] - lambda;
			sum += rho * (z[i] * z[i]) / denom;
		}
		return sum;
	}

	private static double[] multiplyMatrixVector(Matrix M, double[] v) {
		int rows = M.getRowCount();
		int cols = M.getColumnCount();
		double[] result = new double[rows];
		for (int col = 0; col < cols; col++) {
			double coeff = v[col];
			if (Math.abs(coeff) < 1e-15) {
				continue;
			}
			net.faulj.vector.Vector column = M.getData()[col];
			for (int row = 0; row < rows; row++) {
				result[row] += column.get(row) * coeff;
			}
		}
		return result;
	}

	private static Matrix blockDiagonal(Matrix A, Matrix B) {
		int rowsA = A.getRowCount();
		int rowsB = B.getRowCount();
		int colsA = A.getColumnCount();
		int colsB = B.getColumnCount();
		net.faulj.vector.Vector[] data = new net.faulj.vector.Vector[colsA + colsB];
		for (int col = 0; col < colsA; col++) {
			double[] colData = new double[rowsA + rowsB];
			net.faulj.vector.Vector v = A.getData()[col];
			for (int r = 0; r < rowsA; r++) {
				colData[r] = v.get(r);
			}
			data[col] = new net.faulj.vector.Vector(colData);
		}
		for (int col = 0; col < colsB; col++) {
			double[] colData = new double[rowsA + rowsB];
			net.faulj.vector.Vector v = B.getData()[col];
			for (int r = 0; r < rowsB; r++) {
				colData[rowsA + r] = v.get(r);
			}
			data[colsA + col] = new net.faulj.vector.Vector(colData);
		}
		return new Matrix(data);
	}

	private static double[] concat(double[] a, double[] b) {
		double[] result = new double[a.length + b.length];
		System.arraycopy(a, 0, result, 0, a.length);
		System.arraycopy(b, 0, result, a.length, b.length);
		return result;
	}

	private static SortedEigenData sortEigenData(double[] values, Matrix vectors, double[] z) {
		int n = values.length;
		Integer[] order = new Integer[n];
		for (int i = 0; i < n; i++) {
			order[i] = i;
		}
		java.util.Arrays.sort(order, java.util.Comparator.comparingDouble(i -> values[i]));
		double[] sortedValues = new double[n];
		net.faulj.vector.Vector[] sortedVectors = new net.faulj.vector.Vector[n];
		double[] sortedZ = new double[n];
		for (int i = 0; i < n; i++) {
			int idx = order[i];
			sortedValues[i] = values[idx];
			sortedVectors[i] = vectors.getData()[idx].copy();
			sortedZ[i] = z[idx];
		}
		return new SortedEigenData(sortedValues, new Matrix(sortedVectors), sortedZ);
	}

	private static final class BidiagonalSVDResult {
		private final Matrix U;
		private final Matrix V;
		private final double[] singularValues;

		private BidiagonalSVDResult(Matrix U, Matrix V, double[] singularValues) {
			this.U = U;
			this.V = V;
			this.singularValues = singularValues;
		}
	}

	private static final class TridiagonalData {
		private final double[] diagonal;
		private final double[] offDiagonal;

		private TridiagonalData(double[] diagonal, double[] offDiagonal) {
			this.diagonal = diagonal;
			this.offDiagonal = offDiagonal;
		}
	}

	private static final class EigenDecomposition {
		private final double[] values;
		private final Matrix vectors;

		private EigenDecomposition(double[] values, Matrix vectors) {
			this.values = values;
			this.vectors = vectors;
		}
	}

	private static final class SortedEigenData {
		private final double[] values;
		private final Matrix vectors;
		private final double[] z;

		private SortedEigenData(double[] values, Matrix vectors, double[] z) {
			this.values = values;
			this.vectors = vectors;
			this.z = z;
		}
	}

	private static final class EigenPair {
		private final double value;
		private final net.faulj.vector.Vector vector;

		private EigenPair(double value, net.faulj.vector.Vector vector) {
			this.value = value;
			this.vector = vector;
		}
	}
}
