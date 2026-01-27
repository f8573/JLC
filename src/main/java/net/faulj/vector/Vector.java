package net.faulj.vector;

import net.faulj.core.Tolerance;
import net.faulj.matrix.Matrix;

import java.util.Arrays;

/**
 * Represents a mathematical vector of real or complex numbers.
 * <p>
 * This class is a lightweight view over a Matrix with a single column.
 * </p>
 */
public class Vector {
    private final Matrix matrix;
    private final int column;
    private final int size;
    private final int stride;

    /**
     * Constructs a new real Vector from the given array of doubles.
     * The array is stored by reference (not copied).
     *
     * @param data The real-part array representing vector components
     */
    public Vector(double[] data) {
        if (data == null) {
            throw new IllegalArgumentException("Vector data must not be null");
        }
        this.matrix = Matrix.wrap(data, data.length, 1);
        this.column = 0;
        this.size = data.length;
        this.stride = 1;
    }

    /**
     * Constructs a new complex Vector from real and imaginary arrays.
     * The arrays are stored by reference (not copied).
     *
     * @param real The real-part array
     * @param imag The imaginary-part array (may be null to imply zeros)
     */
    public Vector(double[] real, double[] imag) {
        if (real == null) {
            throw new IllegalArgumentException("Vector data must not be null");
        }
        if (imag != null && imag.length != real.length) {
            throw new IllegalArgumentException("Imaginary data length must match real data length");
        }
        this.matrix = Matrix.wrap(real, imag, real.length, 1);
        this.column = 0;
        this.size = real.length;
        this.stride = 1;
    }

    /**
     * Creates a vector view over a matrix column.
     *
     * @param matrix The backing matrix
     * @param column The column index to view
     */
    public Vector(Matrix matrix, int column) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (column < 0 || column >= matrix.getColumnCount()) {
            throw new IllegalArgumentException("Invalid column index");
        }
        this.matrix = matrix;
        this.column = column;
        this.size = matrix.getRowCount();
        this.stride = matrix.getColumnCount();
    }

    /**
     * Get the number of elements in this vector.
     *
     * @return vector dimension
     */
    public int dimension() {
        return size;
    }

    /**
     * Create a deep copy of this vector.
     *
     * @return copied vector
     */
    public Vector copy() {
        double[] real = getData();
        double[] imag = hasImag() ? getImagData() : null;
        return new Vector(Arrays.copyOf(real, size), imag == null ? null : Arrays.copyOf(imag, size));
    }

    /**
     * Get the real part at the given index.
     *
     * @param index element index
     * @return real value
     */
    public double get(int index) {
        return matrix.get(index, column);
    }

    /**
     * Get the imaginary part at the given index.
     *
     * @param index element index
     * @return imaginary value
     */
    public double getImag(int index) {
        return matrix.getImag(index, column);
    }

    /**
     * Get the real data as a dense array (copied if needed).
     *
     * @return real data array
     */
    public double[] getData() {
        if (stride == 1 && column == 0) {
            return matrix.getRawData();
        }
        double[] out = new double[size];
        double[] data = matrix.getRawData();
        int idx = column;
        for (int i = 0; i < size; i++) {
            out[i] = data[idx];
            idx += stride;
        }
        return out;
    }

    /**
     * Get the imaginary data as a dense array (zeros if none).
     *
     * @return imaginary data array
     */
    public double[] getImagData() {
        double[] imag = matrix.getRawImagData();
        if (imag == null) {
            return new double[size];
        }
        if (stride == 1 && column == 0) {
            return imag;
        }
        double[] out = new double[size];
        int idx = column;
        for (int i = 0; i < size; i++) {
            out[i] = imag[idx];
            idx += stride;
        }
        return out;
    }

    /**
     * Set the real data for this vector.
     *
     * @param data real data array
     */
    public void setData(double[] data) {
        if (data == null) {
            throw new IllegalArgumentException("Vector data must not be null");
        }
        if (data.length != size) {
            throw new IllegalArgumentException("Vector dimension mismatch");
        }
        double[] dest = matrix.getRawData();
        if (stride == 1 && column == 0) {
            System.arraycopy(data, 0, dest, 0, size);
            return;
        }
        int idx = column;
        for (int i = 0; i < size; i++) {
            dest[idx] = data[i];
            idx += stride;
        }
    }

    /**
     * Set the imaginary data for this vector.
     *
     * @param imag imaginary data array
     */
    public void setImagData(double[] imag) {
        if (imag == null) {
            return;
        }
        if (imag.length != size) {
            throw new IllegalArgumentException("Vector dimension mismatch");
        }
        if (stride == 1 && column == 0) {
            matrix.setImagData(imag);
            return;
        }
        for (int i = 0; i < size; i++) {
            setImag(i, imag[i]);
        }
    }

    /**
     * Set the real part at an index.
     *
     * @param index element index
     * @param value real value
     */
    public void set(int index, double value) {
        matrix.set(index, column, value);
    }

    /**
     * Set the imaginary part at an index.
     *
     * @param index element index
     * @param value imaginary value
     */
    public void setImag(int index, double value) {
        matrix.setImag(index, column, value);
    }

    /**
     * Set both real and imaginary parts at an index.
     *
     * @param index element index
     * @param real real value
     * @param imag imaginary value
     */
    public void setComplex(int index, double real, double imag) {
        matrix.setComplex(index, column, real, imag);
    }

    /**
     * Add another vector to this vector.
     *
     * @param vector vector to add
     * @return sum vector
     */
    public Vector add(Vector vector) {
        Vector[] vectors = equalize(vector);
        Vector left = vectors[0];
        Vector right = vectors[1];
        double[] leftData = left.matrix.getRawData();
        double[] rightData = right.matrix.getRawData();
        double[] leftImag = left.matrix.getRawImagData();
        double[] rightImag = right.matrix.getRawImagData();
        if (rightImag != null || leftImag != null) {
            leftImag = left.matrix.ensureImagData();
        }
        int leftStride = left.stride;
        int rightStride = right.stride;
        int li = left.column;
        int ri = right.column;
        if (leftImag == null) {
            for (int i = 0; i < left.size; i++) {
                leftData[li] += rightData[ri];
                li += leftStride;
                ri += rightStride;
            }
            return left;
        }
        for (int i = 0; i < left.size; i++) {
            leftData[li] += rightData[ri];
            leftImag[li] += rightImag == null ? 0.0 : rightImag[ri];
            li += leftStride;
            ri += rightStride;
        }
        return left;
    }

    /**
     * Subtract another vector from this vector.
     *
     * @param vector vector to subtract
     * @return difference vector
     */
    public Vector subtract(Vector vector) {
        Vector[] vectors = equalize(vector);
        Vector left = vectors[0];
        Vector right = vectors[1];
        double[] leftData = left.matrix.getRawData();
        double[] rightData = right.matrix.getRawData();
        double[] leftImag = left.matrix.getRawImagData();
        double[] rightImag = right.matrix.getRawImagData();
        if (rightImag != null || leftImag != null) {
            leftImag = left.matrix.ensureImagData();
        }
        int leftStride = left.stride;
        int rightStride = right.stride;
        int li = left.column;
        int ri = right.column;
        if (leftImag == null) {
            for (int i = 0; i < left.size; i++) {
                leftData[li] -= rightData[ri];
                li += leftStride;
                ri += rightStride;
            }
            return left;
        }
        for (int i = 0; i < left.size; i++) {
            leftData[li] -= rightData[ri];
            leftImag[li] -= rightImag == null ? 0.0 : rightImag[ri];
            li += leftStride;
            ri += rightStride;
        }
        return left;
    }

    /**
     * Compute the dot product with another vector.
     *
     * @param vector other vector
     * @return dot product
     */
    public double dot(Vector vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        if (vector.dimension() != size) {
            throw new IllegalArgumentException("Vector dimensions must match for dot product");
        }
        double sum = 0.0;
        double[] leftData = matrix.getRawData();
        double[] rightData = vector.matrix.getRawData();
        double[] leftImag = matrix.getRawImagData();
        double[] rightImag = vector.matrix.getRawImagData();
        int leftStride = stride;
        int rightStride = vector.stride;
        int li = column;
        int ri = vector.column;
        if (leftImag == null && rightImag == null) {
            for (int i = 0; i < size; i++) {
                sum += leftData[li] * rightData[ri];
                li += leftStride;
                ri += rightStride;
            }
            return sum;
        }
        for (int i = 0; i < size; i++) {
            double lr = leftData[li];
            double rr = rightData[ri];
            double liVal = leftImag == null ? 0.0 : leftImag[li];
            double riVal = rightImag == null ? 0.0 : rightImag[ri];
            sum += lr * rr + liVal * riVal;
            li += leftStride;
            ri += rightStride;
        }
        return sum;
    }

    /**
     * Compute the 1-norm of this vector.
     *
     * @return 1-norm
     */
    public double norm1() {
        return VectorNorms.norm1(this);
    }

    /**
     * Compute the Euclidean norm of this vector.
     *
     * @return 2-norm
     */
    public double norm2() {
        return VectorNorms.norm2(this);
    }

    /**
     * Compute the infinity norm of this vector.
     *
     * @return infinity norm
     */
    public double normInf() {
        return VectorNorms.normInf(this);
    }

    /**
     * Normalize this vector to unit length.
     *
     * @return normalized vector
     */
    public Vector normalize() {
        return VectorNorms.normalize(this);
    }

    /**
     * Ensure both vectors share complex storage when required.
     *
     * @param other other vector
     * @return array of equalized vectors
     */
    public Vector[] equalize(Vector other) {
        if (other == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        Vector v = this.copy();
        Vector o = other.copy();
        int max = Math.max(v.dimension(), o.dimension());
        if (v.dimension() != max) {
            v = v.resize(max);
        }
        if (o.dimension() != max) {
            o = o.resize(max);
        }
        return new Vector[]{v, o};
    }

    /**
     * Check whether the vector is (near) zero within tolerance.
     *
     * @return true if zero
     */
    public boolean isZero() {
        double[] real = getData();
        double[] imag = hasImag() ? getImagData() : null;
        for (int i = 0; i < real.length; i++) {
            if (real[i] != 0.0) {
                return false;
            }
            if (imag != null && imag[i] != 0.0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Compare with another vector using tolerance.
     *
     * @param other other vector
     * @return true if equal within tolerance
     */
    public boolean equals(Vector other) {
        if (other == null || other.dimension() != size) {
            return false;
        }
        double[] left = getData();
        double[] right = other.getData();
        double[] leftImag = hasImag() ? getImagData() : null;
        double[] rightImag = other.hasImag() ? other.getImagData() : null;
        for (int i = 0; i < size; i++) {
            if (left[i] != right[i]) {
                return false;
            }
            double li = leftImag == null ? 0.0 : leftImag[i];
            double ri = rightImag == null ? 0.0 : rightImag[i];
            if (li != ri) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check whether this vector is a unit vector.
     *
     * @return true if unit length
     */
    public boolean isUnitVector() {
        if (!isReal()) {
            return false;
        }
        double sum = 0;
        double[] data = getData();
        for (double d : data) {
            if (d != 0 && d != 1) {
                return false;
            }
            sum += d;
        }
        return sum == 1;
    }

    /**
     * Negate this vector.
     *
     * @return negated vector
     */
    public Vector negate() {
        Vector v = this.copy();
        double[] real = v.matrix.getRawData();
        for (int i = 0; i < real.length; i++) {
            real[i] *= -1;
        }
        double[] imag = v.matrix.getRawImagData();
        if (imag != null) {
            for (int i = 0; i < imag.length; i++) {
                imag[i] *= -1;
            }
        }
        return v;
    }

    /**
     * Resize this vector to a new dimension.
     *
     * @param n new dimension
     * @return resized vector
     */
    public Vector resize(int n) {
        double[] real = new double[n];
        double[] imag = hasImag() ? new double[n] : null;
        double[] src = getData();
        int len = Math.min(src.length, n);
        if (len > 0) {
            System.arraycopy(src, 0, real, 0, len);
        }
        if (imag != null) {
            double[] srcImag = getImagData();
            System.arraycopy(srcImag, 0, imag, 0, Math.min(srcImag.length, n));
        }
        return new Vector(real, imag);
    }

    /**
     * Render the vector as a string.
     *
     * @return string representation
     */
    @Override
    public String toString() {
        if (isReal()) {
            return Arrays.toString(getData());
        }
        double[] real = getData();
        double[] imag = getImagData();
        StringBuilder s = new StringBuilder("[");
        for (int i = 0; i < size; i++) {
            if (i > 0) {
                s.append(", ");
            }
            double re = real[i];
            double im = imag[i];
            if (im == 0) {
                s.append(re);
            } else if (im > 0) {
                s.append(re).append(" + ").append(im).append("i");
            } else {
                s.append(re).append(" - ").append(-im).append("i");
            }
        }
        s.append("]");
        return s.toString();
    }

    /**
     * Multiply this vector by a scalar.
     *
     * @param scalar scalar value
     * @return scaled vector
     */
    public Vector multiplyScalar(double scalar) {
        Vector v = this.copy();
        double[] real = v.matrix.getRawData();
        for (int i = 0; i < real.length; i++) {
            real[i] *= scalar;
        }
        double[] imag = v.matrix.getRawImagData();
        if (imag != null) {
            for (int i = 0; i < imag.length; i++) {
                imag[i] *= scalar;
            }
        }
        return v;
    }

    /**
     * Convert this vector into a row matrix.
     *
     * @return transposed matrix
     */
    public Matrix transpose() {
        return toMatrix().transpose();
    }

    /**
     * Multiply this vector (as a row) by a matrix.
     *
     * @param matrix matrix to multiply
     * @return resulting matrix
     */
    public Matrix multiply(Matrix matrix) {
        if (1 != matrix.getRowCount()) {
            throw new ArithmeticException("Vector dimension must match matrix row count for multiplication");
        }
        Matrix m = new Matrix(new Vector[]{this});
        return m.multiply(matrix);
    }

    /**
     * Convert this vector into a column matrix.
     *
     * @return column matrix
     */
    public Matrix toMatrix() {
        return new Matrix(new Vector[]{this});
    }

    /**
     * Round components close to zero based on tolerance.
     *
     * @return rounded vector
     */
    public Vector round() {
        double[] real = getData();
        double[] imag = hasImag() ? getImagData() : null;
        double tol = Tolerance.get();
        for (int i = 0; i < real.length; i++) {
            if (real[i] < tol) {
                real[i] = 0;
            }
            if (imag != null && imag[i] < tol) {
                imag[i] = 0;
            }
        }
        return new Vector(real, imag);
    }

    /**
     * Project this vector onto another vector.
     *
     * @param u target vector
     * @return projection vector
     */
    public Vector project(Vector u) {
        Vector v = this.copy();
        u = u.copy();
        double n = v.dot(u);
        double d = u.dot(u);
        return u.multiplyScalar(n / d);
    }

    /**
     * Conjugate the vector (negate imaginary parts).
     *
     * @return conjugated vector
     */
    public Vector conjugate() {
        if (!hasImag()) {
            return copy();
        }
        Vector v = this.copy();
        double[] imag = v.matrix.getRawImagData();
        for (int i = 0; i < imag.length; i++) {
            imag[i] *= -1;
        }
        return v;
    }

    /**
     * Check if this vector has no imaginary part.
     *
     * @return true if real
     */
    public boolean isReal() {
        double[] imag = matrix.getRawImagData();
        if (imag == null) {
            return true;
        }
        int idx = column;
        for (int i = 0; i < size; i++) {
            if (imag[idx] != 0.0) {
                return false;
            }
            idx += stride;
        }
        return true;
    }

    /**
     * Check if this vector stores any imaginary data.
     *
     * @return true if complex storage exists
     */
    public boolean hasImag() {
        return matrix.hasImagData();
    }
}
