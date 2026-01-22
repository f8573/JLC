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

    public int dimension() {
        return size;
    }

    public Vector copy() {
        double[] real = getData();
        double[] imag = hasImag() ? getImagData() : null;
        return new Vector(Arrays.copyOf(real, size), imag == null ? null : Arrays.copyOf(imag, size));
    }

    public double get(int index) {
        return matrix.get(index, column);
    }

    public double getImag(int index) {
        return matrix.getImag(index, column);
    }

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

    public void set(int index, double value) {
        matrix.set(index, column, value);
    }

    public void setImag(int index, double value) {
        matrix.setImag(index, column, value);
    }

    public void setComplex(int index, double real, double imag) {
        matrix.setComplex(index, column, real, imag);
    }

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

    public double norm1() {
        return VectorNorms.norm1(this);
    }

    public double norm2() {
        return VectorNorms.norm2(this);
    }

    public double normInf() {
        return VectorNorms.normInf(this);
    }

    public Vector normalize() {
        return VectorNorms.normalize(this);
    }

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

    public Matrix transpose() {
        return toMatrix().transpose();
    }

    public Matrix multiply(Matrix matrix) {
        if (1 != matrix.getRowCount()) {
            throw new ArithmeticException("Vector dimension must match matrix row count for multiplication");
        }
        Matrix m = new Matrix(new Vector[]{this});
        return m.multiply(matrix);
    }

    public Matrix toMatrix() {
        return new Matrix(new Vector[]{this});
    }

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

    public Vector project(Vector u) {
        Vector v = this.copy();
        u = u.copy();
        double n = v.dot(u);
        double d = u.dot(u);
        return u.multiplyScalar(n / d);
    }

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

    public boolean hasImag() {
        return matrix.hasImagData();
    }
}
