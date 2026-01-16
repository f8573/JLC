package net.faulj.spaces;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class SubspaceBasis {
	public static Set<Vector> rowSpaceBasis(Matrix m) {
		Matrix copy = m.copy();
		copy.toReducedRowEchelonForm();
		copy = copy.transpose();
		Set<Vector> vectors = new HashSet<>();
		for (Vector v : copy.getData()) {
			if (!v.isZero()) {
				vectors.add(v);
			}
		}
		return vectors;
	}

	public static Set<Vector> columnSpaceBasis(Matrix m) {
		Matrix copy = m.copy();
		copy.toReducedRowEchelonForm();
		return new HashSet<>(copy.getPivotColumns());
	}

	public static Set<Vector> nullSpaceBasis(Matrix m) {
		Set<Vector> set = new HashSet<>();

		Matrix mm = m.copy();
		mm.toReducedRowEchelonForm();

		ArrayList<Integer> e = new ArrayList<>();
		ArrayList<Integer> free = new ArrayList<>();

		Matrix I = Matrix.Identity(mm.getData()[0].dimension());
		Vector[] mData = mm.getData();

		for (int i = 0; i < mm.getColumnCount(); i++) {
			if (mData[i].isUnitVector()) {
				e.add(i);
			} else {
				free.add(i);
			}
		}

		ArrayList<Vector> fList = new ArrayList<>();
		for (int i : free) {
			fList.add(mData[i]);
		}

		Matrix temp = new Matrix(fList.toArray(new Vector[0]));
		temp = temp.transpose();
		Vector[] tempData = Arrays.copyOf(temp.getData(), e.size());
		Matrix F = new Matrix(tempData);
		Vector[] fData = F.getData();
		for (int i = 0; i < fData.length; i++) {
			fData[i] = fData[i].negate();
		}
		F = F.transpose();
		Matrix B = F.AppendMatrix(Matrix.Identity(free.size()), "DOWN");
		ArrayList<Integer> permutation = new ArrayList<>();
		permutation.addAll(e);
		permutation.addAll(free);
		for (Vector v : B.getData()) {
			Vector vec = VectorUtils.zero(permutation.size());
			for (int i = 0; i < permutation.size(); i++) {
				vec.set(permutation.get(i), v.get(i));
			}
			set.add(vec);
		}
		return set;
	}
}
