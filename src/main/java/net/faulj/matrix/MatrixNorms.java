package net.faulj.matrix;

public class MatrixNorms {
	public static double frobeniusNorm(Matrix m) {
		double sum = 0.0;
		for (int i = 0; i < m.getRowCount(); i++) {
			for (int j = 0; j < m.getColumnCount(); j++) {
				double val = m.get(i, j);
				sum += val * val;
			}
		}
		return Math.sqrt(sum);
	}

	public static double norm1(Matrix m) {
		double maxColSum = 0.0;
		for (int j = 0; j < m.getColumnCount(); j++) {
			double colSum = 0.0;
			for (int i = 0; i < m.getRowCount(); i++) {
				colSum += Math.abs(m.get(i, j));
			}
			if (colSum > maxColSum) {
				maxColSum = colSum;
			}
		}
		return maxColSum;
	}

	public static double normInf(Matrix m) {
		double maxRowSum = 0.0;
		for (int i = 0; i < m.getRowCount(); i++) {
			double rowSum = 0.0;
			for (int j = 0; j < m.getColumnCount(); j++) {
				rowSum += Math.abs(m.get(i, j));
			}
			if (rowSum > maxRowSum) {
				maxRowSum = rowSum;
			}
		}
		return maxRowSum;
	}
}
