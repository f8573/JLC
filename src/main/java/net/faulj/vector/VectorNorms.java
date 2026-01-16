package net.faulj.vector;

public class VectorNorms {
	public static double norm1(Vector v) {
		double sum = 0.0;
		for (double d : v.getData()) {
			sum += Math.abs(d);
		}
		return sum;
	}

	public static double norm2(Vector v) {
		double sum = 0.0;
		for (int i = 0; i < v.dimension(); i++) {
			double d = v.get(i);
			sum += d * d;
		}
		return Math.sqrt(sum);
	}

	public static double normInf(Vector v) {
		double max = 0.0;
		for (double d : v.getData()) {
			double ad = Math.abs(d);
			if (ad > max) max = ad;
		}
		return max;
	}

	public static Vector normalize(Vector v) {
		double n2 = norm2(v);
		if (n2 == 0.0) return v.copy();
		double[] out = new double[v.dimension()];
		for (int i = 0; i < v.dimension(); i++) out[i] = v.get(i) / n2;
		return new Vector(out);
	}
}
