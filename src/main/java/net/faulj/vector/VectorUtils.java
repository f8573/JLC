package net.faulj.vector;

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
}