package net.faulj.decomposition.cholesky;

import net.faulj.decomposition.result.CholeskyResult;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.NativeFactorizationSupport;

/**
 * Cholesky decomposition implementation.
 */
public class CholeskyDecomposition {
	private static final int DEFAULT_BLOCK_THRESHOLD = 96;
	private static final int DEFAULT_BLOCK_SIZE = 32;

	/**
	 * Decomposes a symmetric positive definite matrix A into L such that A = L * L^T.
	 * @param A symmetric positive definite matrix
	 * @return CholeskyResult containing lower triangular factor L
	 */
	public CholeskyResult decompose(Matrix A) {
		if (A == null) throw new IllegalArgumentException("Matrix A must not be null");
		if (!A.isSquare()) throw new IllegalArgumentException("Matrix must be square");

		int n = A.getRowCount();
		Matrix L = A.copy();
		double[] a = L.getRawData();
		if (NativeFactorizationSupport.tryCholesky(a, n)) {
			zeroUpperTriangle(a, n);
			return new CholeskyResult(A, L);
		}
		if (n >= blockThreshold()) {
			decomposeBlocked(a, n);
		} else {
			factorPanel(a, n, 0, n);
		}
		zeroUpperTriangle(a, n);
		return new CholeskyResult(A, L);
	}

	private static void decomposeBlocked(double[] a, int n) {
		int blockSize = Math.min(blockSize(), n);
		for (int blockStart = 0; blockStart < n; blockStart += blockSize) {
			int panelWidth = Math.min(blockSize, n - blockStart);
			factorPanel(a, n, blockStart, panelWidth);

			int trailingStart = blockStart + panelWidth;
			if (trailingStart >= n) {
				continue;
			}

			int trailingRows = n - trailingStart;
			// A22 -= L21 * L21^T. L21 is the block below the current diagonal block.
			Gemm.gemmStridedColMajorB(
				a, trailingStart * n + blockStart, n,
				a, trailingStart * n + blockStart, n,
				a, trailingStart * n + trailingStart, n,
				trailingRows, panelWidth, trailingRows,
				-1.0, 1.0, panelWidth
			);
		}
	}

	private static void factorPanel(double[] a, int n, int blockStart, int panelWidth) {
		int blockEnd = blockStart + panelWidth;
		for (int col = blockStart; col < blockEnd; col++) {
			int colBase = col * n;
			double sum = 0.0;
			for (int k = blockStart; k < col; k++) {
				double v = a[colBase + k];
				sum = Math.fma(v, v, sum);
			}

			double diag = a[colBase + col] - sum;
			if (!(diag > 0.0) || Double.isNaN(diag)) {
				throw new ArithmeticException("Matrix is not positive definite (non-positive pivot at " + col + ")");
			}

			double diagSqrt = Math.sqrt(diag);
			a[colBase + col] = diagSqrt;

			for (int row = col + 1; row < n; row++) {
				int rowBase = row * n;
				double dot = 0.0;
				for (int k = blockStart; k < col; k++) {
					dot = Math.fma(a[rowBase + k], a[colBase + k], dot);
				}
				a[rowBase + col] = (a[rowBase + col] - dot) / diagSqrt;
			}
		}
	}

	private static void zeroUpperTriangle(double[] a, int n) {
		for (int row = 0; row < n; row++) {
			int rowBase = row * n;
			for (int col = row + 1; col < n; col++) {
				a[rowBase + col] = 0.0;
			}
		}
	}

	private static int blockThreshold() {
		return integerProperty("net.faulj.decomposition.cholesky.blockThreshold", DEFAULT_BLOCK_THRESHOLD);
	}

	private static int blockSize() {
		return integerProperty("net.faulj.decomposition.cholesky.blockSize", DEFAULT_BLOCK_SIZE);
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
