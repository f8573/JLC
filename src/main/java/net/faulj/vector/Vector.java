package net.faulj.vector;

import net.faulj.core.Tolerance;
import net.faulj.matrix.Matrix;

import java.util.Arrays;
import java.util.List;

/**
 * Represents a mathematical vector of real numbers.
 * <p>
 * This class provides a mutable implementation of a vector $\mathbf{v} \in \mathbb{R}^n$
 * backed by a double array. It supports fundamental linear algebra operations including
 * vector addition, subtraction, scalar multiplication, and dot products.
 * </p>
 *
 * <h2>Data Representation:</h2>
 * <ul>
 * <li><b>Storage:</b> Dense representation using {@code double[]}</li>
 * <li><b>Indexing:</b> 0-based indexing (0 to n-1)</li>
 * <li><b>Mutability:</b> The underlying data can be modified via setters</li>
 * </ul>
 *
 * <h2>Mathematical Operations:</h2>
 * <p>
 * Given vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ and scalar $\alpha$:
 * </p>
 * <ul>
 * <li><b>Addition:</b> $\mathbf{w} = \mathbf{u} + \mathbf{v}$ where $w_i = u_i + v_i$</li>
 * <li><b>Scalar Mult:</b> $\mathbf{w} = \alpha \mathbf{v}$ where $w_i = \alpha v_i$</li>
 * <li><b>Dot Product:</b> $d = \mathbf{u} \cdot \mathbf{v} = \sum u_i v_i$</li>
 * <li><b>Norms:</b> $L_1, L_2$ (Euclidean), and $L_\infty$ norms</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create vectors
 * Vector v1 = new Vector(new double[]{1.0, 2.0, 3.0});
 * Vector v2 = new Vector(new double[]{4.0, 5.0, 6.0});
 *
 * // Operations
 * Vector sum = v1.add(v2);           // [5.0, 7.0, 9.0]
 * double dot = v1.dot(v2);           // 32.0
 * Vector scaled = v1.multiplyScalar(2.0); // [2.0, 4.0, 6.0]
 *
 * // Norms
 * double length = v1.norm2();        // sqrt(14)
 * Vector unit = v1.normalize();      // Unit vector direction
 * }</pre>
 *
 * <h2>Auto-Sizing:</h2>
 * <p>
 * Some binary operations (like {@code add} and {@code subtract}) attempt to {@code equalize}
 * dimensions. If vectors differ in size, the smaller vector is effectively padded or the
 * larger is truncated depending on the internal implementation of {@code equalize}, though
 * mathematically vectors usually must share dimensions.
 * </p>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.vector.VectorNorms
 * @see net.faulj.vector.VectorUtils
 */
public class Vector {
    private double[] data;
    private int size;

    /**
     * Constructs a new Vector from the given array of doubles.
     * <p>
     * The array is stored by reference in this constructor (not deeply copied initially),
     * so external changes to the array may affect the vector.
     * </p>
     *
     * @param data The double array representing vector components
     */
    public Vector(double[] data) {
        this.data = data;
        this.size = data.length;
    }

    /**
     * Returns the dimension (size) of the vector.
     *
     * @return The number of components in the vector (n)
     */
    public int dimension() {
        return size;
    }

    /**
     * Creates a deep copy of this vector.
     *
     * @return A new Vector instance with copied data independent of the original
     */
    public Vector copy() {
        return new Vector(Arrays.copyOf(data, size));
    }

    /**
     * Retrieves the component value at the specified index.
     *
     * @param index The 0-based index of the component
     * @return The value at the specified index
     * @throws ArrayIndexOutOfBoundsException if the index is invalid
     */
    public double get(int index) {
        return data[index];
    }

    /**
     * Returns the direct backing array of this vector.
     * <p>
     * <b>Warning:</b> This exposes internal state. Modifying the returned array
     * directly affects the vector.
     * </p>
     *
     * @return The backing double array
     */
    public double[] getData() {
        return data;
    }

    /**
     * Replaces the backing array of this vector.
     *
     * @param data The new data array
     */
    public void setData(double[] data) {
        this.data = data;
    }

    /**
     * Sets the value of a specific component.
     *
     * @param index The 0-based index to modify
     * @param value The new value to assign
     */
    public void set(int index, double value) {
        data[index] = value;
    }

    /**
     * Adds another vector to this vector.
     * <p>
     * Performs component-wise addition: $\mathbf{w} = \mathbf{this} + \mathbf{other}$.
     * This operation modifies the calling vector (or the equalized copy) depending on size.
     * It handles dimension mismatches via {@link #equalize(Vector)}.
     * </p>
     *
     * @param vector The vector to add
     * @return The result of the addition (this instance, or a new instance if resizing occurred)
     */
    public Vector add(Vector vector) {
        Vector[] vectors = equalize(vector);
        double[] data = vectors[0].getData();
        for (int i = 0; i < vectors[0].size; i++) {
            data[i] += vectors[1].get(i);
        }
        return vectors[0];
    }

    /**
     * Subtracts another vector from this vector.
     * <p>
     * Performs component-wise subtraction: $\mathbf{w} = \mathbf{this} - \mathbf{other}$.
     * </p>
     *
     * @param vector The vector to subtract
     * @return The result of the subtraction
     */
    public Vector subtract(Vector vector) {
        Vector[] vectors = equalize(vector);
        double[] data = vectors[0].getData();
        for (int i = 0; i < vectors[0].size; i++) {
            data[i] -= vectors[1].get(i);
        }
        return vectors[0];
    }

    /**
     * Computes the standard dot product (scalar product).
     * <p>
     * Calculation: $\sum_{i=0}^{n-1} a_i b_i$
     * </p>
     *
     * @param vector The vector to multiply with
     * @return The scalar result of the dot product
     */
    public double dot(Vector vector) {
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += data[i] * vector.get(i);
        }
        return sum;
    }

    /**
     * Computes the Taxicab norm ($L_1$).
     *
     * @return $\sum |v_i|$
     * @see VectorNorms#norm1(Vector)
     */
    public double norm1() {
        return VectorNorms.norm1(this);
    }

    /**
     * Computes the Euclidean norm ($L_2$).
     *
     * @return $\sqrt{\sum v_i^2}$
     * @see VectorNorms#norm2(Vector)
     */
    public double norm2() {
        return VectorNorms.norm2(this);
    }

    /**
     * Computes the Maximum norm ($L_\infty$).
     *
     * @return $\max(|v_i|)$
     * @see VectorNorms#normInf(Vector)
     */
    public double normInf() {
        return VectorNorms.normInf(this);
    }

    /**
     * Returns a normalized version of this vector (unit vector).
     * <p>
     * $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$
     * </p>
     *
     * @return A new Vector with norm 1 pointing in the same direction
     */
    public Vector normalize() {
        return VectorNorms.normalize(this);
    }

    /**
     * Equalizes the dimensions of this vector and another vector.
     * <p>
     * This method ensures both vectors have the same dimension, usually by expanding
     * the smaller vector.
     * </p>
     *
     * @param other The other vector to compare dimensions with
     * @return An array containing two vectors of equal dimension
     */
    public Vector[] equalize(Vector other) {
        Vector v = this.copy();
        other = other.copy();
        if (other.dimension() > v.dimension()) {
            data = Arrays.copyOf(v.getData(), other.dimension());
        } else {
            other.setData(Arrays.copyOf(other.getData(), v.dimension()));
        }
        return new Vector[]{v, other};
    }

    /**
     * Checks if this is a zero vector.
     *
     * @return {@code true} if all components are zero, {@code false} otherwise
     */
    public boolean isZero() {
        return this.equals(new Vector(new double[size]));
    }

    /**
     * Checks for equality with another vector.
     * <p>
     * Equality requires identical dimensions and exactly equal floating-point values
     * for all components.
     * </p>
     *
     * @param other The vector to compare
     * @return {@code true} if vectors are identical
     */
    public boolean equals(Vector other) {
        for(int i = 0; i < size; i++) {
            if (data[i] != other.getData()[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Checks if this is a unit vector.
     * <p>
     * Checks if exactly one component is 1 and all others are 0 (Standard Basis Vector).
     * Note: This does <i>not</i> generally check if norm == 1.
     * </p>
     *
     * @return {@code true} if vector is a standard basis vector (e.g., [0, 1, 0])
     */
    public boolean isUnitVector() {
        double sum = 0;
        for(double d : data) {
            if (d != 0 && d != 1) {
                return false;
            }
            sum += d;
        }
        return sum == 1;
    }

    /**
     * Returns the negation of this vector.
     *
     * @return A new vector $-\mathbf{v}$ where every component is negated
     */
    public Vector negate() {
        Vector v = this.copy();
        double[] vData = v.getData();
        for (int i = 0; i < v.dimension(); i++) {
            vData[i] *= -1;
        }
        return v;
    }

    /**
     * Resizes the vector to a new dimension $n$.
     * <p>
     * If $n$ is smaller, the vector is truncated. If $n$ is larger, it is padded with zeros.
     * </p>
     *
     * @param n The new dimension
     * @return A new Vector of size $n$
     */
    public Vector resize(int n) {
        double[] d = new double[n];
        Vector v = this.copy();
        double[] vData = v.getData();
        if (Math.min(v.dimension(), n) >= 0) System.arraycopy(vData, 0, d, 0, Math.min(v.dimension(), n));
        return new Vector(d);
    }

    @Override
    public String toString() {
        return Arrays.toString(data);
    }

    /**
     * Multiplies the vector by a scalar.
     *
     * @param scalar The scalar value $\alpha$
     * @return A new vector equal to $\alpha \mathbf{v}$
     */
    public Vector multiplyScalar(double scalar) {
        Vector v = this.copy();
        double[] vData = v.getData();
        for (int i = 0; i < v.dimension(); i++) {
            vData[i] *= scalar;
        }
        return v;
    }

    /**
     * Returns the transpose of this vector as a Matrix.
     * <p>
     * If this is treated as a column vector ($n \times 1$), this returns a row matrix ($1 \times n$).
     * </p>
     *
     * @return The transposed Matrix
     */
    public Matrix transpose() {
        return new Matrix(new Vector[]{this}).transpose();
    }

    /**
     * Performs vector-matrix multiplication.
     * <p>
     * Treats this vector as a row vector and multiplies by the given matrix:
     * $\mathbf{result} = \mathbf{v}^T A$
     * </p>
     *
     * @param matrix The matrix to multiply
     * @return The resulting Matrix (row vector)
     * @throws ArithmeticException if dimensions are incompatible
     */
    public Matrix multiply(Matrix matrix) {
        if (1 != matrix.getRowCount()) {
            throw new ArithmeticException("Vector dimension must match matrix row count for multiplication");
        }
        Matrix m = new Matrix(new Vector[]{this});
        return m.multiply(matrix);
    }

    /**
     * Converts this vector to a Matrix.
     *
     * @return A Matrix containing this vector as a row/column
     */
    public Matrix toMatrix() {
        return new Matrix(new Vector[]{this});
    }

    /**
     * Converts negligible values to zeroes.
     *
     * @return a Vector with negligible values set to zero
     */
    public Vector round() {
        double[] data = this.copy().getData();
        for (int i = 0; i < data.length; i++) {
            if (data[i] < Tolerance.get()) {
                data[i] = 0;
            }
        }
        return new Vector(data);
    }
}