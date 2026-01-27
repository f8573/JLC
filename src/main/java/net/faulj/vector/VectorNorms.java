package net.faulj.vector;

/**
 * Utility class for computing vector norms.
 */
public class VectorNorms {

	/**
	 * Compute the 1-norm (sum of absolute values).
	 *
	 * @param v vector to measure
	 * @return 1-norm
	 */
	public static double norm1(Vector v) {
		double sum = 0.0;
		double[] real = v.getData();
		double[] imag = v.hasImag() ? v.getImagData() : null;
		for (int i = 0; i < real.length; i++) {
			double re = real[i];
			double im = imag == null ? 0.0 : imag[i];
			sum += Math.hypot(re, im);
		}
		return sum;
	}

	/**
	 * Compute the Euclidean norm.
	 *
	 * @param v vector to measure
	 * @return 2-norm
	 */
	public static double norm2(Vector v) {
		double sum = 0.0;
		double[] real = v.getData();
		double[] imag = v.hasImag() ? v.getImagData() : null;
		for (int i = 0; i < real.length; i++) {
			double re = real[i];
			double im = imag == null ? 0.0 : imag[i];
			sum += re * re + im * im;
		}
		return Math.sqrt(sum);
	}

	/**
	 * Compute the infinity norm (max absolute component).
	 *
	 * @param v vector to measure
	 * @return infinity norm
	 */
	public static double normInf(Vector v) {
		double max = 0.0;
		double[] real = v.getData();
		double[] imag = v.hasImag() ? v.getImagData() : null;
		for (int i = 0; i < real.length; i++) {
			double re = real[i];
			double im = imag == null ? 0.0 : imag[i];
			double ad = Math.hypot(re, im);
			if (ad > max) max = ad;
		}
		return max;
	}

	/**
	 * Normalize a vector to unit length.
	 *
	 * @param v vector to normalize
	 * @return new normalized vector
	 */
	public static Vector normalize(Vector v) {
		double n2 = norm2(v);
		if (n2 == 0.0) return v.copy();
		double[] real = v.getData();
		double[] imag = v.hasImag() ? v.getImagData() : null;
		for (int i = 0; i < real.length; i++) {
			real[i] /= n2;
			if (imag != null) {
				imag[i] /= n2;
			}
		}
		return new Vector(real, imag);
	}
}
