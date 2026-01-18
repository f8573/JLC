package net.faulj.matrix;

/**
 * Represents elementary matrices used for row and column operations.
 * <p>
 * An elementary matrix <b>E</b> differs from the identity matrix by a single elementary
 * row operation. Left-multiplying a matrix A by E (EA) performs the operation on the
 * rows of A. Right-multiplying (AE) performs the operation on the columns.
 * </p>
 *
 * <h2>Types of Elementary Matrices:</h2>
 * <ul>
 * <li><b>Row Interchange (Swap):</b> Permutation matrix that swaps row <i>i</i> and row <i>j</i>.
 * <pre>R<sub>i</sub> ↔ R<sub>j</sub></pre>
 * </li>
 * <li><b>Row Scaling (Dilatation):</b> Diagonal matrix that multiplies row <i>i</i> by non-zero scalar <i>k</i>.
 * <pre>R<sub>i</sub> ← k * R<sub>i</sub></pre>
 * </li>
 * <li><b>Row Addition (Shear):</b> Identity matrix with an off-diagonal entry that adds a multiple of row <i>j</i> to row <i>i</i>.
 * <pre>R<sub>i</sub> ← R<sub>i</sub> + k * R<sub>j</sub></pre>
 * </li>
 * </ul>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li><b>Gaussian Elimination:</b> Reducing a matrix to Row Echelon Form.</li>
 * <li><b>LU Decomposition:</b> Factoring A into Lower and Upper triangular matrices.</li>
 * <li><b>Inverse Calculation:</b> Gauss-Jordan elimination (multiplying [A|I] by sequence of E).</li>
 * </ul>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li><b>Invertibility:</b> All elementary matrices are invertible. The inverse is an elementary matrix of the same type.</li>
 * <li><b>Determinant:</b>
 * <ul>
 * <li>Swap: det(E) = -1</li>
 * <li>Scale: det(E) = k</li>
 * <li>Add: det(E) = 1</li>
 * </ul>
 * </li>
 * </ul>
 *
 * <h2>Usage Example (Hypothetical):</h2>
 * <pre>{@code
 * // Create a matrix that adds 2.5 * row 1 to row 3 for a 4x4 system
 * ElementaryMatrix E = ElementaryMatrix.rowAddition(4, 3, 1, 2.5);
 *
 * // Apply operation to A
 * Matrix processed = E.multiply(A);
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.decomposition.lu.LUDecomposition
 * @see net.faulj.inverse.GaussJordanInverse
 */
public class ElementaryMatrix {
}