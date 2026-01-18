package net.faulj.spaces;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

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
public class ChangeOfBasis {
	public ChangeOfBasis() {
		throw new RuntimeException("Class unfinished");
	}
}