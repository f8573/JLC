package net.faulj.decomposition.hessenberg;

import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.matrix.Matrix;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.stream.IntStream;

/**
 * Reduces a square matrix to upper Hessenberg form using orthogonal Householder transformations.
 * Parallelized implementation for better performance on multi-core systems.
 */
public class HessenbergReduction {
	private static final double SAFE_MIN = Double.MIN_NORMAL;
	private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
	private static final int BLOCK_SIZE = 32; // LAPACK: smaller blocks = better cache
	private static final int PARALLEL_THRESHOLD = 200; // Lower threshold for 512x512
	private static final ForkJoinPool pool = ForkJoinPool.commonPool();

	// ThreadLocal workspace to avoid per-block allocations in Q accumulation
	private static final ThreadLocal<QAccumulationWorkspace> Q_WS =
			ThreadLocal.withInitial(QAccumulationWorkspace::new);

	private static final class QAccumulationWorkspace {
		double[] V;
		double[] T;
		double[] W;
		double[] WT;
		double[] w;
		double[] tw;

		void ensureCapacity(int blockLen, int panelSize, int n) {
			int vSize = blockLen * panelSize;
			int tSize = panelSize * panelSize;
			int wSize = n * panelSize;

			if (V == null || V.length < vSize) V = new double[vSize];
			if (T == null || T.length < tSize) T = new double[tSize];
			if (W == null || W.length < wSize) W = new double[wSize];
			if (WT == null || WT.length < wSize) WT = new double[wSize];
			if (w == null || w.length < panelSize) w = new double[panelSize];
			if (tw == null || tw.length < panelSize) tw = new double[panelSize];
		}
	}

	public static HessenbergResult decompose(Matrix A) {
		if (A == null) {
			throw new IllegalArgumentException("Matrix must not be null");
		}
		if (!A.isReal()) {
			throw new UnsupportedOperationException("Hessenberg reduction requires a real-valued matrix");
		}
		if (!A.isSquare()) {
			throw new ArithmeticException("Matrix must be square to compute Hessenberg form");
		}
		int n = A.getRowCount();
		if (n <= 2) {
			Matrix H = A.copy();
			return new HessenbergResult(A, H, Matrix.Identity(n));
		}

		return decomposeOptimized(A);
	}

	/**
	 * Reduce a matrix to Hessenberg form without forming Q.
	 * Useful for benchmarking to reduce allocation pressure.
	 *
	 * @param A matrix to reduce
	 * @return Hessenberg matrix H
	 */
	public static Matrix reduceToHessenberg(Matrix A) {
		if (A == null) {
			throw new IllegalArgumentException("Matrix must not be null");
		}
		if (!A.isReal()) {
			throw new UnsupportedOperationException("Hessenberg reduction requires a real-valued matrix");
		}
		if (!A.isSquare()) {
			throw new ArithmeticException("Matrix must be square to compute Hessenberg form");
		}
		int n = A.getRowCount();
		if (n <= 2) {
			return A.copy();
		}

		ReductionState state = computeHessenberg(A, false);
		return state.H;
	}
	
	private static HessenbergResult decomposeOptimized(Matrix A) {
		ReductionState state = computeHessenberg(A, true);
		int n = state.H.getRowCount();
		double[] Q = accumulateQBlocked(n, state.tau, state.reflectors);
		return new HessenbergResult(A, state.H, Matrix.wrap(Q, n, n));
	}

	private static ReductionState computeHessenberg(Matrix A, boolean storeReflectors) {
		Matrix H = A.copy();
		int n = H.getRowCount();
		double[] h = H.getRawData();

		double[] allTau = storeReflectors ? new double[n - 2] : null;
		double[][] allReflectors = storeReflectors ? new double[n - 2][] : null;

		// LAPACK: pre-allocate workspace once to minimize allocation overhead
		double[] vExternal = new double[n];
		int vecLen = SPECIES.length();

		// LAPACK-style: use blocking for better cache performance
		final int blockSize = 32;

		for (int kb = 0; kb < n - 2; kb += blockSize) {
			int kend = Math.min(kb + blockSize, n - 2);

			for (int k = kb; k < kend; k++) {
			int start = k + 1;
			int len = n - start;

			// ===== LAPACK dlarfg-style Householder generation =====
			int base = start * n + k;

			// LAPACK: compute norm directly (fast path for most cases)
			double x0 = h[base];
			double scale = Math.abs(x0);
			double ssq = 1.0;

			for (int i = 1; i < len; i++) {
				double absxi = Math.abs(h[(start + i) * n + k]);
				if (absxi > scale) {
					double temp = scale / absxi;
					ssq = 1.0 + ssq * temp * temp;
					scale = absxi;
				} else if (absxi > 0.0) {
					double temp = absxi / scale;
					ssq += temp * temp;
				}
			}

			double xnorm = scale * Math.sqrt(ssq);
			if (xnorm < SAFE_MIN) {
				if (storeReflectors) allTau[k] = 0.0;
				continue;
			}

			// LAPACK: choose sign for stability
			double beta = (x0 >= 0.0) ? -xnorm : xnorm;
			double tau = (beta - x0) / beta;
			double invV0 = 1.0 / (x0 - beta);

			if (storeReflectors) allTau[k] = tau;
			h[base] = beta;

			// LAPACK: normalize Householder vector in-place for speed
			double[] v = vExternal;
			v[0] = 1.0;
			for (int i = 1; i < len; i++) {
				v[i] = h[(start + i) * n + k] * invV0;
			}

			if (storeReflectors) {
				double[] reflector = new double[len];
				reflector[0] = 1.0;
				for (int i = 1; i < len; i++) {
					reflector[i] = v[i];
				}
				allReflectors[k] = reflector;
			}

			int numCols = n - k - 1;
			final int fk = k, fstart = start, flen = len;
			final double ftau = tau;
			final double[] fv = v;

			// ===== LAPACK: Apply from LEFT with optimized loop structure =====
			if (numCols >= PARALLEL_THRESHOLD && n >= 400) {
				IntStream.range(0, numCols).parallel().forEach(colOff -> {
					int col = fk + 1 + colOff;
					double dot = h[fstart * n + col];
					int i = 1;
					int limit = flen - 3;
					// 4x unroll for ILP
					for (; i < limit; i += 4) {
						dot += fv[i] * h[(fstart + i) * n + col];
						dot += fv[i+1] * h[(fstart + i+1) * n + col];
						dot += fv[i+2] * h[(fstart + i+2) * n + col];
						dot += fv[i+3] * h[(fstart + i+3) * n + col];
					}
					for (; i < flen; i++) {
						dot += fv[i] * h[(fstart + i) * n + col];
					}
					dot *= ftau;
					h[fstart * n + col] -= dot;
					i = 1;
					for (; i < limit; i += 4) {
						h[(fstart + i) * n + col] -= dot * fv[i];
						h[(fstart + i+1) * n + col] -= dot * fv[i+1];
						h[(fstart + i+2) * n + col] -= dot * fv[i+2];
						h[(fstart + i+3) * n + col] -= dot * fv[i+3];
					}
					for (; i < flen; i++) {
						h[(fstart + i) * n + col] -= dot * fv[i];
					}
				});
			} else {
				for (int colOff = 0; colOff < numCols; colOff++) {
					int col = k + 1 + colOff;
					double dot = h[start * n + col];
					int i = 1;
					int limit = len - 3;
					for (; i < limit; i += 4) {
						dot += v[i] * h[(start + i) * n + col];
						dot += v[i+1] * h[(start + i+1) * n + col];
						dot += v[i+2] * h[(start + i+2) * n + col];
						dot += v[i+3] * h[(start + i+3) * n + col];
					}
					for (; i < len; i++) {
						dot += v[i] * h[(start + i) * n + col];
					}
					dot *= tau;
					h[start * n + col] -= dot;
					i = 1;
					for (; i < limit; i += 4) {
						h[(start + i) * n + col] -= dot * v[i];
						h[(start + i+1) * n + col] -= dot * v[i+1];
						h[(start + i+2) * n + col] -= dot * v[i+2];
						h[(start + i+3) * n + col] -= dot * v[i+3];
					}
					for (; i < len; i++) {
						h[(start + i) * n + col] -= dot * v[i];
					}
				}
			}

			// ===== LAPACK: Apply from RIGHT with SIMD + better memory access =====
			if (n >= PARALLEL_THRESHOLD && n >= 400) {
				IntStream.range(0, n).parallel().forEach(row -> {
					int idx = row * n + fstart;
					double dot = h[idx];
					int j = 1;
					int limit = flen - 3;
					// 4x unroll before SIMD
					for (; j < limit; j += 4) {
						dot += h[idx + j] * fv[j];
						dot += h[idx + j+1] * fv[j+1];
						dot += h[idx + j+2] * fv[j+2];
						dot += h[idx + j+3] * fv[j+3];
					}
					for (; j < flen; j++) {
						dot += h[idx + j] * fv[j];
					}
					dot *= ftau;
					h[idx] -= dot;
					j = 1;
					for (; j < limit; j += 4) {
						h[idx + j] -= dot * fv[j];
						h[idx + j+1] -= dot * fv[j+1];
						h[idx + j+2] -= dot * fv[j+2];
						h[idx + j+3] -= dot * fv[j+3];
					}
					for (; j < flen; j++) {
						h[idx + j] -= dot * fv[j];
					}
				});
			} else {
				// LAPACK: non-parallel path with 4x unrolling
				for (int row = 0; row < n; row++) {
					int idx = row * n + start;
					double dot = h[idx];
					int j = 1;
					int limit = len - 3;
					for (; j < limit; j += 4) {
						dot += h[idx + j] * v[j];
						dot += h[idx + j+1] * v[j+1];
						dot += h[idx + j+2] * v[j+2];
						dot += h[idx + j+3] * v[j+3];
					}
					for (; j < len; j++) {
						dot += h[idx + j] * v[j];
					}
					dot *= tau;
					h[idx] -= dot;
					j = 1;
					for (; j < limit; j += 4) {
						h[idx + j] -= dot * v[j];
						h[idx + j+1] -= dot * v[j+1];
						h[idx + j+2] -= dot * v[j+2];
						h[idx + j+3] -= dot * v[j+3];
					}
					for (; j < len; j++) {
						h[idx + j] -= dot * v[j];
					}
				}
			}
			}
		}
		
		// Zero out elements below subdiagonal
		for (int col = 0; col < n - 2; col++) {
			for (int row = col + 2; row < n; row++) {
				h[row * n + col] = 0.0;
			}
		}
		
		return new ReductionState(H, allTau, allReflectors);
	}

	private static final class ReductionState {
		final Matrix H;
		final double[] tau;
		final double[][] reflectors;

		ReductionState(Matrix H, double[] tau, double[][] reflectors) {
			this.H = H;
			this.tau = tau;
			this.reflectors = reflectors;
		}
	}
	
	private static double[] accumulateQBlocked(int n, double[] tau, double[][] reflectors) {
		double[] Q = new double[n * n];
		for (int i = 0; i < n; i++) {
			Q[i * n + i] = 1.0;
		}

		int vecLen = SPECIES.length();

		// Get workspace to avoid per-block allocations
		final QAccumulationWorkspace ws = Q_WS.get();

		for (int kBlock = 0; kBlock < n - 2; kBlock += BLOCK_SIZE) {
			int nbActual = Math.min(BLOCK_SIZE, n - 2 - kBlock);
			int blockStart = kBlock + 1;
			int blockLen = n - blockStart;

			int actualCount = 0;
			for (int j = 0; j < nbActual; j++) {
				if (tau[kBlock + j] != 0.0 && reflectors[kBlock + j] != null) {
					actualCount++;
				}
			}
			if (actualCount == 0) continue;

			// Use workspace arrays instead of allocating new ones
			ws.ensureCapacity(blockLen, actualCount, n);
			final double[] V = ws.V;
			final double[] T = ws.T;
			final double[] W = ws.W;
			final double[] WT = ws.WT;
			final double[] w = ws.w;
			final double[] tw = ws.tw;

			// Zero out V and T for this block
			java.util.Arrays.fill(V, 0, blockLen * actualCount, 0.0);
			java.util.Arrays.fill(T, 0, actualCount * actualCount, 0.0);

			int colIdx = 0;
			for (int j = 0; j < nbActual; j++) {
				int k = kBlock + j;
				if (tau[k] == 0.0 || reflectors[k] == null) continue;

				double[] v = reflectors[k];
				int vLen = v.length;
				int vOffset = k + 1 - blockStart;

				for (int i = 0; i < vLen; i++) {
					V[(vOffset + i) * actualCount + colIdx] = v[i];
				}

				T[colIdx * actualCount + colIdx] = tau[k];

				if (colIdx > 0) {
					// Compute w = V^T * v_colIdx using workspace array with 4x unrolling
					java.util.Arrays.fill(w, 0, colIdx, 0.0);
					for (int i = 0; i < colIdx; i++) {
						double dot = 0.0;
						int row = 0;
						int limit = blockLen - 3;

						// 4x unrolled loop for better ILP
						for (; row < limit; row += 4) {
							int base0 = row * actualCount;
							int base1 = (row + 1) * actualCount;
							int base2 = (row + 2) * actualCount;
							int base3 = (row + 3) * actualCount;
							dot += V[base0 + i] * V[base0 + colIdx];
							dot += V[base1 + i] * V[base1 + colIdx];
							dot += V[base2 + i] * V[base2 + colIdx];
							dot += V[base3 + i] * V[base3 + colIdx];
						}

						// Scalar remainder
						for (; row < blockLen; row++) {
							dot += V[row * actualCount + i] * V[row * actualCount + colIdx];
						}
						w[i] = dot;
					}

					// Compute tw = T * w (T is upper triangular) using workspace array with 4x unrolling
					java.util.Arrays.fill(tw, 0, colIdx, 0.0);
					for (int i = 0; i < colIdx; i++) {
						double sum = 0.0;
						int p = i;
						int limit = colIdx - 3;

						// 4x unrolled loop for better ILP
						for (; p < limit; p += 4) {
							int base = i * actualCount;
							sum += T[base + p] * w[p];
							sum += T[base + p + 1] * w[p + 1];
							sum += T[base + p + 2] * w[p + 2];
							sum += T[base + p + 3] * w[p + 3];
						}

						// Scalar remainder
						for (; p < colIdx; p++) {
							sum += T[i * actualCount + p] * w[p];
						}
						tw[i] = sum;
					}

					for (int i = 0; i < colIdx; i++) {
						T[i * actualCount + colIdx] = -tau[k] * tw[i];
					}
				}
				colIdx++;
			}

			final int fBlockStart = blockStart;
			final int fBlockLen = blockLen;
			final int fActualCount = actualCount;

			// W = Q(:, blockStart:) * V - PARALLEL
			// Zero out W for this computation
			java.util.Arrays.fill(W, 0, n * actualCount, 0.0);
			if (n >= PARALLEL_THRESHOLD) {
				final double[] fV = V;
				final double[] fW = W;
				IntStream.range(0, n).parallel().forEach(i -> {
					for (int j = 0; j < fActualCount; j++) {
						double sum = 0.0;
						for (int p = 0; p < fBlockLen; p++) {
							sum += Q[i * n + fBlockStart + p] * fV[p * fActualCount + j];
						}
						fW[i * fActualCount + j] = sum;
					}
				});
			} else {
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < actualCount; j++) {
						double sum = 0.0;
						for (int p = 0; p < blockLen; p++) {
							sum += Q[i * n + blockStart + p] * V[p * actualCount + j];
						}
						W[i * actualCount + j] = sum;
					}
				}
			}

			// WT = W * T (T is upper triangular)
			java.util.Arrays.fill(WT, 0, n * actualCount, 0.0);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < actualCount; j++) {
					double sum = 0.0;
					for (int p = 0; p <= j; p++) {
						sum += W[i * actualCount + p] * T[p * actualCount + j];
					}
					WT[i * actualCount + j] = sum;
				}
			}

			// Q(:, blockStart:) -= WT * V' - PARALLEL
			if (n >= PARALLEL_THRESHOLD) {
				final double[] fV = V;
				final double[] fWT = WT;
				IntStream.range(0, n).parallel().forEach(i -> {
					for (int j = 0; j < fBlockLen; j++) {
						double sum = 0.0;
						for (int p = 0; p < fActualCount; p++) {
							sum += fWT[i * fActualCount + p] * fV[j * fActualCount + p];
						}
						Q[i * n + fBlockStart + j] -= sum;
					}
				});
			} else {
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < blockLen; j++) {
						double sum = 0.0;
						for (int p = 0; p < actualCount; p++) {
							sum += WT[i * actualCount + p] * V[j * actualCount + p];
						}
						Q[i * n + blockStart + j] -= sum;
					}
				}
			}
		}

		return Q;
	}
}
