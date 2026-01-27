package net.faulj.matrix;

/**
 * Utility class for computing various matrix norms.
 */
public class MatrixNorms {

	/**
	 * Compute the Frobenius norm of a matrix.
	 *
	 * @param m matrix to measure
	 * @return Frobenius norm
	 */
	public static double frobeniusNorm(Matrix m) {
		double sum = 0.0;
		double[] real = m.getRawData();
		double[] imag = m.getRawImagData();
		if (imag == null) {
			for (double v : real) {
				sum += v * v;
			}
			return Math.sqrt(sum);
		}
		for (int i = 0; i < real.length; i++) {
			double re = real[i];
			double im = imag[i];
			sum += re * re + im * im;
		}
		return Math.sqrt(sum);
	}

	/**
	 * Compute the induced 1-norm (maximum column sum).
	 *
	 * @param m matrix to measure
	 * @return 1-norm
	 */
	public static double norm1(Matrix m) {
		double maxColSum = 0.0;
		int rows = m.getRowCount();
		int cols = m.getColumnCount();
		double[] real = m.getRawData();
		double[] imag = m.getRawImagData();
		for (int col = 0; col < cols; col++) {
			double colSum = 0.0;
			int idx = col;
			for (int row = 0; row < rows; row++) {
				if (imag == null) {
					colSum += Math.abs(real[idx]);
				} else {
					colSum += Math.hypot(real[idx], imag[idx]);
				}
				idx += cols;
			}
			if (colSum > maxColSum) {
				maxColSum = colSum;
			}
		}
		return maxColSum;
	}

	/**
	 * Compute the induced infinity norm (maximum row sum).
	 *
	 * @param m matrix to measure
	 * @return infinity norm
	 */
	public static double normInf(Matrix m) {
		double maxRowSum = 0.0;
		int rows = m.getRowCount();
		int cols = m.getColumnCount();
		double[] real = m.getRawData();
		double[] imag = m.getRawImagData();
		for (int row = 0; row < rows; row++) {
			double rowSum = 0.0;
			int offset = row * cols;
			for (int col = 0; col < cols; col++) {
				if (imag == null) {
					rowSum += Math.abs(real[offset + col]);
				} else {
					rowSum += Math.hypot(real[offset + col], imag[offset + col]);
				}
			}
			if (rowSum > maxRowSum) {
				maxRowSum = rowSum;
			}
		}
		return maxRowSum;
	}
}
