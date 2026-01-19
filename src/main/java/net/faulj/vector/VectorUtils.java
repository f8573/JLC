package net.faulj.vector;

import net.faulj.matrix.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Static utility methods for vector creation and algorithms.
 * <p>
 * This class serves as a factory for common vector types (zero, unit, random)
 * and provides specialized algorithms like Householder vector computation used
 * in matrix decompositions (QR, Hessenberg, Tridiagonalization).
 * </p>
 *
 * <h2>Functionality:</h2>
 * <ul>
 * <li><b>Factories:</b> Create standard basis vectors, zero vectors, random vectors</li>
 * <li><b>Decomposition Tools:</b> Compute Householder reflection vectors</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create a 3D standard basis vector e2
 * Vector e2 = VectorUtils.unitVector(3, 1); // [0.0, 1.0, 0.0]
 *
 * // Create a random vector of size 5
 * Vector rand = VectorUtils.random(5);
 *
 * // Compute Householder vector for QR decomposition
 * Vector v = VectorUtils.householder(columnVector);
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 */
public class VectorUtils {

    /**
     * Creates a standard basis unit vector $\mathbf{e}_i$.
     * <p>
     * Creates a vector of size {@code dimension} with a 1.0 at index {@code number}
     * and 0.0 everywhere else.
     * </p>
     *
     * @param dimension The size of the vector ($n$)
     * @param number The index to set to 1 ($0 \le i < n$)
     * @return The unit vector $\mathbf{e}_i$
     * @throws ArithmeticException If the index is out of bounds
     */
    public static Vector unitVector(int dimension, int number) {
        if (number >= dimension) {
            throw new ArithmeticException("The index of the unit cannot be larger than the dimension of the vector");
        }
        Vector v = zero(dimension);
        v.set(number,1);
        return v;
    }

    /**
     * Creates a zero vector of the specified size.
     *
     * @param size The dimension of the vector
     * @return A vector containing all zeros
     */
    public static Vector zero (int size) {
        return new Vector(new double[size]);
    }

    /**
     * Creates a random vector with values in range $[0.0, 1.0)$.
     *
     * @param size The dimension of the vector
     * @return A vector populated with {@code Math.random()}
     */
    public static Vector random(int size) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
        }
        return new Vector(data);
    }

    /**
     * Computes the Householder vector used for orthogonal transformations.
     * <p>
     * Given vector $\mathbf{x}$, this computes the Householder vector $\mathbf{v}$ and
     * stores the coefficient $\tau$ (tau) in the last extra component of the return vector.
     * This vector defines a reflection $P = I - \tau \mathbf{v}\mathbf{v}^T$ that annihilates
     * sub-diagonal elements in decompositions.
     * </p>
     *
     * <h2>Mechanism:</h2>
     * <ol>
     * <li>$\alpha = \|\mathbf{x}\|_2$</li>
     * <li>$\beta = -\text{sgn}(x_0) \alpha$</li>
     * <li>$v_0 = x_0 - \beta$</li>
     * <li>$\tau = 2 / (\mathbf{v} \cdot \mathbf{v})$</li>
     * </ol>
     *
     * @param x The input vector to be reflected
     * @return A vector of size $m+1$ where first $m$ are $\mathbf{v}$ and index $m$ is $\tau$
     */
    public static Vector householder(Vector x) {
        x = x.copy();
        int m = x.dimension();

        double alpha = x.norm2();
        double beta = -1*Math.copySign(alpha, x.get(0));
        Vector v = x;
        v.getData()[0] -= beta;
        double tau = 2.0 / v.dot(v);
        v = v.resize(m+1);
        v.set(m,tau);
        return v;
    }

    /**
     * Performs the Gram-Schmidt process on a list of vectors to create an orthonormal basis.
     *
     * @param vectors The input list of vectors
     * @return A new list containing the orthonormalized basis vectors
     */
    public static List<Vector> gramSchmidt(List<Vector> vectors) {
        List<Vector> orthogonalBasis = new ArrayList<>();
        for (Vector vector : vectors) {
            Vector projection = null;
            if (!orthogonalBasis.isEmpty()) {
                for (Vector basisVector : orthogonalBasis) {
                    projection = basisVector.project(vector);
                    vector = vector.subtract(projection); // Update the current vector
                }
            }
            orthogonalBasis.add(vector.normalize()); // Normalize the updated vector and add to the basis
        }
        return orthogonalBasis;
    }

    /**
     * Projects a list of vectors onto another list of vectors using Gram-Schmidt.
     *
     * @param projectors List of vectors used for projection (the vectors in this list should form an orthonormal basis)
     * @param sourceVectors The list of vectors to be projected
     * @return A new matrix where each column is the orthogonal projection of the corresponding input vector onto the projector vectors
     */
    public static Matrix projectOnto(List<Vector> projectors, List<Vector> sourceVectors) {
        if (projectors.size() != sourceVectors.size()) {
            throw new IllegalArgumentException("Number of projectors must match the number of source vectors");
        }

        // Ensure projectors form an orthonormal basis
        List<Vector> orthogonalBasis = gramSchmidt(projectors);

        Matrix resultMatrix = new Matrix(new Vector[sourceVectors.size()]);
        for (int i = 0; i < sourceVectors.size(); i++) {
            resultMatrix.setColumn(i, orthogonalBasis.get(i).project(sourceVectors.get(i)));
        }
        return resultMatrix;
    }

    /**
     * Normalizes the orthonormalized basis to create a matrix.
     *
     * @param basis The list of orthonormal vectors
     * @return A matrix where each column is the normalized basis vector
     */
    public static Matrix normalizeBasis(List<Vector> basis) {
        Matrix resultMatrix = new Matrix(basis.getFirst().dimension(),basis.size());
        for (int i = 0; i < basis.size(); i++) {
            resultMatrix.setColumn(i, basis.get(i));
        }
        return resultMatrix;
    }

    /**
     * Converts a list of vectors to a matrix where each column is a vector from the list.
     *
     * @param vectors The input list of vectors
     * @return A new matrix with vectors as columns
     */
    public static Matrix toMatrix(List<Vector> vectors) {
        return new Matrix(vectors.toArray(new Vector[0]));
    }
}