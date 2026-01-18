package net.faulj.vector;

/**
 * Utility class for computing vector norms.
 * <p>
 * This class provides static methods to calculate standard vector norms ($L_p$)
 * used to measure the "length" or "magnitude" of vectors in $\mathbb{R}^n$.
 * </p>
 *
 * <h2>Supported Norms:</h2>
 * <ul>
 * <li><b>$L_1$ (Taxicab/Manhattan):</b> Sum of absolute differences</li>
 * <li><b>$L_2$ (Euclidean):</b> Straight-line distance from origin</li>
 * <li><b>$L_\infty$ (Maximum/Supremum):</b> Maximum absolute component</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Vector v = new Vector(new double[]{-3.0, 4.0});
 *
 * double n1 = VectorNorms.norm1(v);   // |-3| + |4| = 7.0
 * double n2 = VectorNorms.norm2(v);   // sqrt((-3)^2 + 4^2) = 5.0
 * double nInf = VectorNorms.normInf(v); // max(|-3|, |4|) = 4.0
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.vector.Vector
 */
public class VectorNorms {

	/**
	 * Computes the $L_1$ norm (Taxicab norm).
	 * <p>
	 * Formula: $\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$
	 * </p>
	 *
	 * @param v The vector
	 * @return The sum of absolute values of the components
	 */
	public static double norm1(Vector v) {
		double sum = 0.0;
		for (double d : v.getData()) {
			sum += Math.abs(d);
		}
		return sum;
	}

	/**
	 * Computes the $L_2$ norm (Euclidean norm).
	 * <p>
	 * Formula: $\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$
	 * </p>
	 *
	 * @param v The vector
	 * @return The Euclidean length of the vector
	 */
	public static double norm2(Vector v) {
		double sum = 0.0;
		for (int i = 0; i < v.dimension(); i++) {
			double d = v.get(i);
			sum += d * d;
		}
		return Math.sqrt(sum);
	}

	/**
	 * Computes the $L_\infty$ norm (Maximum norm).
	 * <p>
	 * Formula: $\|\mathbf{v}\|_\infty = \max_{i} |v_i|$
	 * </p>
	 *
	 * @param v The vector
	 * @return The maximum absolute value among components
	 */
	public static double normInf(Vector v) {
		double max = 0.0;
		for (double d : v.getData()) {
			double ad = Math.abs(d);
			if (ad > max) max = ad;
		}
		return max;
	}

	/**
	 * Returns a normalized version of the vector.
	 * <p>
	 * Returns vector $\mathbf{u}$ such that $\mathbf{u} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$.
	 * If the norm is zero, returns a copy of the zero vector.
	 * </p>
	 *
	 * @param v The vector to normalize
	 * @return A unit vector in the direction of {@code v}
	 */
	public static Vector normalize(Vector v) {
		double n2 = norm2(v);
		if (n2 == 0.0) return v.copy();
		double[] out = new double[v.dimension()];
		for (int i = 0; i < v.dimension(); i++) out[i] = v.get(i) / n2;
		return new Vector(out);
	}
}