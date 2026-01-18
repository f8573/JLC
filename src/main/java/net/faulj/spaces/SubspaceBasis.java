package net.faulj.spaces;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Utility class for computing bases of fundamental matrix subspaces.
 * <p>
 * This class provides static methods to determine the basis vectors for the three
 * fundamental subspaces associated with a matrix \( A \):
 * </p>
 * <ul>
 * <li><b>Row Space:</b> The span of the row vectors of \( A \).</li>
 * <li><b>Column Space:</b> The span of the column vectors of \( A \) (also known as the Image or Range).</li>
 * <li><b>Null Space:</b> The set of all vectors \( x \) such that \( Ax = 0 \) (also known as the Kernel).</li>
 * </ul>
 *
 * <h2>Fundamental Theorem of Linear Algebra:</h2>
 * <p>
 * The dimensions of these subspaces are related by the Rank-Nullity Theorem:
 * </p>
 * <ul>
 * <li>dim(Row Space) = dim(Column Space) = rank(A)</li>
 * <li>dim(Null Space) + rank(A) = n (where n is the number of columns)</li>
 * <li>Row Space \(\perp\) Null Space (in \(\mathbb{R}^n\))</li>
 * </ul>
 *
 * <h2>Algorithms:</h2>
 * <p>
 * Computations rely on Gaussian elimination to Reduced Row Echelon Form (RREF):
 * </p>
 * <ol>
 * <li><b>Row Basis:</b> Non-zero rows of the RREF (since row operations preserve row space).</li>
 * <li><b>Column Basis:</b> Columns of the <i>original</i> matrix corresponding to pivot columns in the RREF.</li>
 * <li><b>Null Basis:</b> Derived from the free variables in the solution to \( Ax = 0 \).</li>
 * </ol>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = ...; // Define matrix
 *
 * // Compute bases
 * Set<Vector> rowBasis = SubspaceBasis.rowSpaceBasis(A);
 * Set<Vector> colBasis = SubspaceBasis.columnSpaceBasis(A);
 * Set<Vector> nullBasis = SubspaceBasis.nullSpaceBasis(A);
 *
 * System.out.println("Rank: " + rowBasis.size());
 * System.out.println("Nullity: " + nullBasis.size());
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.matrix.Matrix
 * @see net.faulj.vector.Vector
 */
public class SubspaceBasis {

	/**
	 * Computes a basis for the row space of the given matrix.
	 * <p>
	 * The row space is the subspace spanned by the rows of the matrix. This method
	 * performs Gaussian elimination to transform the matrix into Reduced Row Echelon
	 * Form (RREF). The non-zero rows of the RREF form a basis for the row space
	 * of the original matrix.
	 * </p>
	 *
	 * <h3>Mathematical Properties:</h3>
	 * <ul>
	 * <li>The basis vectors are orthogonal to the Null Space.</li>
	 * <li>Elementary row operations do not change the row space.</li>
	 * <li>The size of the returned set equals the rank of the matrix.</li>
	 * </ul>
	 *
	 * @param m The matrix to analyze.
	 * @return A set of linearly independent {@link Vector} objects spanning the row space.
	 */
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

	/**
	 * Computes a basis for the column space (range) of the given matrix.
	 * <p>
	 * The column space is the subspace spanned by the columns of the matrix.
	 * This method identifies the pivot columns in the Reduced Row Echelon Form (RREF)
	 * of the matrix. The corresponding columns from the <b>original</b> matrix form
	 * the basis.
	 * </p>
	 *
	 * <h3>Relation to Linear Systems:</h3>
	 * <p>
	 * The system \( Ax = b \) is solvable if and only if \( b \) lies in the column space.
	 * </p>
	 *
	 * @param m The matrix to analyze.
	 * @return A set of linearly independent {@link Vector} objects spanning the column space.
	 */
	public static Set<Vector> columnSpaceBasis(Matrix m) {
		Matrix copy = m.copy();
		copy.toReducedRowEchelonForm();
		return new HashSet<>(copy.getPivotColumns());
	}

	/**
	 * Computes a basis for the null space (kernel) of the given matrix.
	 * <p>
	 * The null space consists of all vectors \( x \) such that \( Ax = 0 \).
	 * This method finds the general solution to the homogeneous system by identifying
	 * free variables in the RREF. Each free variable generates one basis vector.
	 * </p>
	 *
	 * <h3>Algorithm Details:</h3>
	 * <p>
	 * This implementation constructs the basis by separating pivot and free columns,
	 * effectively solving \( x_{pivot} = -F x_{free} \), where \( F \) is the
	 * matrix of coefficients for free variables in the equations for pivot variables.
	 * </p>
	 *
	 * @param m The matrix to analyze.
	 * @return A set of linearly independent {@link Vector} objects spanning the null space.
	 * Returns an empty set if the null space is trivial (contains only the zero vector).
	 */
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