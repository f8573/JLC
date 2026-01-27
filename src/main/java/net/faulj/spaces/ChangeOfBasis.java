package net.faulj.spaces;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

import java.util.Objects;

/**
 * Handles coordinate transformations between different vector space bases.
 * <p>
 * This class represents the change-of-basis matrix \( P_{C \leftarrow B} \) that converts
 * coordinate vectors relative to a basis \( B \) into coordinate vectors relative to
 * a basis \( C \).
 * </p>
 *
 * <h2>Mathematical Definition:</h2>
 * <p>
 * Let \( B = \{b_1, \dots, b_n\} \) and \( C = \{c_1, \dots, c_n\} \) be two bases for a
 * vector space \( V \). The change-of-basis matrix \( P \) satisfies:
 * </p>
 * <pre>
 * [x]_C = P * [x]_B
 * </pre>
 * <p>
 * Where \( [x]_B \) is the coordinate vector of \( x \) relative to \( B \).
 * </p>
 *
 * <h2>Construction:</h2>
 * <p>
 * The columns of \( P_{C \leftarrow B} \) are the coordinate vectors of the basis vectors
 * of \( B \) relative to \( C \):
 * </p>
 * <pre>
 * P = [ [b_1]_C  [b_2]_C  ...  [b_n]_C ]
 * </pre>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li><b>Invertibility:</b> Change-of-basis matrices are always invertible.</li>
 * <li><b>Inverse:</b> \( (P_{C \leftarrow B})^{-1} = P_{B \leftarrow C} \)</li>
 * <li><b>Composition:</b> \( P_{D \leftarrow B} = P_{D \leftarrow C} P_{C \leftarrow B} \)</li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Simplifying linear transformations (e.g., diagonalization).</li>
 * <li>Switching between standard coordinates and eigenbasis.</li>
 * <li>Transforming geometric descriptions.</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see CoordinateMapping
 * @see net.faulj.matrix.Matrix
 */
public final class ChangeOfBasis {

	/**
	 * Underlying change-of-basis matrix P_{C <- B}.
	 */
	private final Matrix P;

	/**
	 * Creates a change-of-basis mapping from basis B to basis C using the
	 * change-of-basis matrix P_{C <- B}.
	 *
	 * @param P change-of-basis matrix mapping coordinates from B to C
	 * @throws IllegalArgumentException if P is null, non-square, or non-invertible
	 */
	public ChangeOfBasis(Matrix P) {
		if (P == null) {
			throw new IllegalArgumentException("Change-of-basis matrix P cannot be null.");
		}
		if (P.getRowCount() != P.getColumnCount()) {
			throw new IllegalArgumentException(
					"Change-of-basis matrix must be square. Got " +
							P.getRowCount() + "x" + P.getColumnCount()
			);
		}

		// Per documentation: change-of-basis matrices are invertible.
		if (!P.isInvertible()) {
			throw new IllegalArgumentException("Change-of-basis matrix must be invertible.");
		}

		this.P = P.copy(); // defensive copy for immutability
	}

	/**
	 * Returns the underlying change-of-basis matrix P_{C <- B}.
	 * Returned value is a copy to preserve immutability.
	 */
	public Matrix matrix() {
		return P.copy();
	}

	/**
	 * Applies the change of basis to convert [x]_B into [x]_C:
	 *
	 * [x]_C = P_{C <- B} * [x]_B
	 *
	 * @param xB coordinates of x in basis B
	 * @return coordinates of x in basis C
	 * @throws IllegalArgumentException if dimensions mismatch
	 */
	public Vector toC(Vector xB) {
		Objects.requireNonNull(xB, "xB cannot be null.");
		if (xB.dimension() != P.getColumnCount()) {
			throw new IllegalArgumentException(
					"Dimension mismatch: xB has dimension " + xB.dimension() +
							" but expected " + P.getColumnCount()
			);
		}
		// Perform Matrix-Vector multiplication P * xB
		// Converts xB to a column matrix, multiplies, then extracts the column vector
		return P.multiply(xB.toMatrix()).getData()[0];
	}

	/**
	 * Converts [x]_C back into [x]_B:
	 *
	 * [x]_B = (P_{C <- B})^{-1} * [x]_C
	 *
	 * @param xC coordinates of x in basis C
	 * @return coordinates of x in basis B
	 */
	public Vector toB(Vector xC) {
		Objects.requireNonNull(xC, "xC cannot be null.");
		if (xC.dimension() != P.getRowCount()) {
			throw new IllegalArgumentException(
					"Dimension mismatch: xC has dimension " + xC.dimension() +
							" but expected " + P.getRowCount()
			);
		}
		// Perform Matrix-Vector multiplication P^-1 * xC
		return P.inverse().multiply(xC.toMatrix()).getData()[0];
	}

	/**
	 * Returns the inverse change-of-basis mapping:
	 *
	 * (P_{C <- B})^{-1} = P_{B <- C}
	 *
	 * @return ChangeOfBasis object representing P_{B <- C}
	 */
	public ChangeOfBasis inverse() {
		return new ChangeOfBasis(P.inverse());
	}

	/**
	 * Composition property:
	 *
	 * P_{D <- B} = P_{D <- C} * P_{C <- B}
	 *
	 * This method returns (this ∘ other) in the mathematical sense:
	 * If other maps B -> C, and this maps C -> D, then:
	 * result maps B -> D.
	 *
	 * @param other mapping P_{C <- B}
	 * @return mapping P_{D <- B}
	 */
	public ChangeOfBasis compose(ChangeOfBasis other) {
		Objects.requireNonNull(other, "other cannot be null.");

		// this: P_{D <- C}
		// other: P_{C <- B}
		// result: P_{D <- B} = this.P * other.P
		if (this.P.getColumnCount() != other.P.getRowCount()) {
			throw new IllegalArgumentException(
					"Composition dimension mismatch: " +
							"this is " + this.P.getRowCount() + "x" + this.P.getColumnCount() + ", " +
							"other is " + other.P.getRowCount() + "x" + other.P.getColumnCount()
			);
		}

		return new ChangeOfBasis(this.P.multiply(other.P));
	}

	/**
	 * Compare change-of-basis matrices for equality.
	 *
	 * @param o other object
	 * @return true if equal
	 */
	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (!(o instanceof ChangeOfBasis)) return false;
		ChangeOfBasis that = (ChangeOfBasis) o;

		// Verify dimensions first
		if (this.P.getRowCount() != that.P.getRowCount() ||
				this.P.getColumnCount() != that.P.getColumnCount()) {
			return false;
		}

		// Manual deep comparison of columns since Matrix may not implement value-based equals
		Vector[] theseCols = this.P.getData();
		Vector[] thoseCols = that.P.getData();

		for (int i = 0; i < theseCols.length; i++) {
			if (!theseCols[i].equals(thoseCols[i])) {
				return false;
			}
		}
		return true;
	}

	/**
	 * @return hash code derived from matrix contents
	 */
	@Override
	public int hashCode() {
		// Use toString hash as a stable fallback for value-based hashing
		return P.toString().hashCode();
	}

	/**
	 * @return string representation of the change-of-basis matrix
	 */
	@Override
	public String toString() {
		return "ChangeOfBasis(P_{C<-B}=\n" + P.toString() + ")";
	}
}