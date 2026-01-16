package net.faulj.decomposition.hessenberg;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;

public class HessenbergReduction {
	public static Matrix[] decompose(Matrix A) {
		if (!A.isSquare()) {
			throw new ArithmeticException("Matrix must be square to compute Hessenberg form");
		}
		Matrix M = A.copy();
		java.util.ArrayList<Vector> vVectors = new java.util.ArrayList<>();
		for(int i = 0; i < M.getRowCount() - 2; i++) {
			Vector a = M.getData()[i];
			int j = i+2;
			Vector x = new Vector(java.util.Arrays.copyOfRange(a.getData(), j - 1, a.dimension()));
			double mag = x.norm2();
			if (mag > 1e-10) {
				Vector hh = VectorUtils.householder(x);
				double tau = hh.get(hh.dimension() - 1);
				Vector v = hh.resize(hh.dimension() - 1);
				vVectors.add(v);
				Matrix P = Matrix.Identity(v.dimension()).subtract(v.multiply(v.transpose()).multiplyScalar(tau));
				Matrix PHat = Matrix.diag(M.getColumnCount()-P.getColumnCount(), P);
				M = PHat.multiply(M.multiply(PHat));
			}
		}
		Matrix H = M;
		Vector[] V = vVectors.toArray(new Vector[0]);
		return new Matrix[]{H, new Matrix(V)};
	}
}
