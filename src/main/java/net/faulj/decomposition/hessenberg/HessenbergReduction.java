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
	private static final int BLOCK_SIZE = 64;
	private static final int PARALLEL_THRESHOLD = 256;
	private static final ForkJoinPool pool = ForkJoinPool.commonPool();

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
	
	private static HessenbergResult decomposeOptimized(Matrix A) {
		Matrix H = A.copy();
		int n = H.getRowCount();
		double[] h = H.getRawData();
		
		double[] allTau = new double[n - 2];
		double[][] allReflectors = new double[n - 2][];
		
		int vecLen = SPECIES.length();
		
		for (int k = 0; k < n - 2; k++) {
			int start = k + 1;
			int len = n - start;
			
			// ===== Compute Householder vector =====
			int base = start * n + k;
			double x0 = h[base];
			
			double sigma = 0.0;
			for (int i = 1; i < len; i++) {
				double val = h[(start + i) * n + k];
				sigma += val * val;
			}
			
			if (sigma <= SAFE_MIN && Math.abs(x0) <= SAFE_MIN) {
				allTau[k] = 0.0;
				continue;
			}
			
			double mu = Math.sqrt(x0 * x0 + sigma);
			double beta = (x0 >= 0) ? -mu : mu;
			double v0 = x0 - beta;
			
			if (Math.abs(v0) <= SAFE_MIN) {
				allTau[k] = 0.0;
				continue;
			}
			
			double tau = 2.0 * v0 * v0 / (sigma + v0 * v0);
			allTau[k] = tau;
			
			h[base] = beta;
			double invV0 = 1.0 / v0;
			
			double[] v = new double[len];
			v[0] = 1.0;
			for (int i = 1; i < len; i++) {
				v[i] = h[(start + i) * n + k] * invV0;
			}
			
			allReflectors[k] = v;
			
			int numCols = n - k - 1;
			final int fk = k, fstart = start, flen = len;
			final double ftau = tau;
			final double[] fv = v;
			
			// ===== Apply from LEFT to H(start:n, k+1:n) - PARALLEL =====
			if (numCols >= PARALLEL_THRESHOLD) {
				IntStream.range(0, numCols).parallel().forEach(colOff -> {
					int col = fk + 1 + colOff;
					double dot = h[fstart * n + col];
					for (int i = 1; i < flen; i++) {
						dot += fv[i] * h[(fstart + i) * n + col];
					}
					dot *= ftau;
					h[fstart * n + col] -= dot;
					for (int i = 1; i < flen; i++) {
						h[(fstart + i) * n + col] -= dot * fv[i];
					}
				});
			} else {
				for (int colOff = 0; colOff < numCols; colOff++) {
					int col = k + 1 + colOff;
					double dot = h[start * n + col];
					for (int i = 1; i < len; i++) {
						dot += v[i] * h[(start + i) * n + col];
					}
					dot *= tau;
					h[start * n + col] -= dot;
					for (int i = 1; i < len; i++) {
						h[(start + i) * n + col] -= dot * v[i];
					}
				}
			}
			
			// ===== Apply from RIGHT to H(0:n, start:n) - SIMD + PARALLEL =====
			if (n >= PARALLEL_THRESHOLD) {
				IntStream.range(0, n).parallel().forEach(row -> {
					int idx = row * n + fstart;
					double dot = h[idx];
					int j = 1;
					int vl = SPECIES.length();
					for (; j + vl <= flen; j += vl) {
						DoubleVector hv = DoubleVector.fromArray(SPECIES, h, idx + j);
						DoubleVector vv = DoubleVector.fromArray(SPECIES, fv, j);
						dot += hv.mul(vv).reduceLanes(VectorOperators.ADD);
					}
					for (; j < flen; j++) {
						dot += h[idx + j] * fv[j];
					}
					dot *= ftau;
					h[idx] -= dot;
					j = 1;
					for (; j + vl <= flen; j += vl) {
						DoubleVector hv = DoubleVector.fromArray(SPECIES, h, idx + j);
						DoubleVector vv = DoubleVector.fromArray(SPECIES, fv, j);
						hv.sub(vv.mul(dot)).intoArray(h, idx + j);
					}
					for (; j < flen; j++) {
						h[idx + j] -= dot * fv[j];
					}
				});
			} else {
				for (int row = 0; row < n; row++) {
					int idx = row * n + start;
					double dot = h[idx];
					int j = 1;
					for (; j + vecLen <= len; j += vecLen) {
						DoubleVector hv = DoubleVector.fromArray(SPECIES, h, idx + j);
						DoubleVector vv = DoubleVector.fromArray(SPECIES, v, j);
						dot += hv.mul(vv).reduceLanes(VectorOperators.ADD);
					}
					for (; j < len; j++) {
						dot += h[idx + j] * v[j];
					}
					dot *= tau;
					h[idx] -= dot;
					j = 1;
					for (; j + vecLen <= len; j += vecLen) {
						DoubleVector hv = DoubleVector.fromArray(SPECIES, h, idx + j);
						DoubleVector vv = DoubleVector.fromArray(SPECIES, v, j);
						hv.sub(vv.mul(dot)).intoArray(h, idx + j);
					}
					for (; j < len; j++) {
						h[idx + j] -= dot * v[j];
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
		
		// Accumulate Q using blocked algorithm
		double[] Q = accumulateQBlocked(n, allTau, allReflectors);
		
		return new HessenbergResult(A, H, Matrix.wrap(Q, n, n));
	}
	
	private static double[] accumulateQBlocked(int n, double[] tau, double[][] reflectors) {
		double[] Q = new double[n * n];
		for (int i = 0; i < n; i++) {
			Q[i * n + i] = 1.0;
		}
		
		int vecLen = SPECIES.length();
		
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
			
			double[] V = new double[blockLen * actualCount];
			double[] T = new double[actualCount * actualCount];
			
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
					double[] w = new double[colIdx];
					for (int i = 0; i < colIdx; i++) {
						double dot = 0.0;
						for (int row = 0; row < blockLen; row++) {
							dot += V[row * actualCount + i] * V[row * actualCount + colIdx];
						}
						w[i] = dot;
					}
					
					double[] tw = new double[colIdx];
					for (int i = 0; i < colIdx; i++) {
						double sum = 0.0;
						for (int p = i; p < colIdx; p++) {
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
			final double[] fV = V;
			final double[] fT = T;
			
			// W = Q(:, blockStart:) * V - PARALLEL
			double[] W = new double[n * actualCount];
			if (n >= PARALLEL_THRESHOLD) {
				IntStream.range(0, n).parallel().forEach(i -> {
					for (int j = 0; j < fActualCount; j++) {
						double sum = 0.0;
						for (int p = 0; p < fBlockLen; p++) {
							sum += Q[i * n + fBlockStart + p] * fV[p * fActualCount + j];
						}
						W[i * fActualCount + j] = sum;
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
			double[] WT = new double[n * actualCount];
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
				IntStream.range(0, n).parallel().forEach(i -> {
					for (int j = 0; j < fBlockLen; j++) {
						double sum = 0.0;
						for (int p = 0; p < fActualCount; p++) {
							sum += WT[i * fActualCount + p] * fV[j * fActualCount + p];
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
