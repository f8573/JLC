package net.faulj.vector;

import net.faulj.matrix.Matrix;

import java.util.Arrays;
import java.util.List;

public class Vector {
    private double[] data;
    private int size;

    public Vector(double[] data) {
        this.data = data;
        this.size = data.length;
    }

    public int dimension() {
        return size;
    }

    public Vector copy() {
        return new Vector(Arrays.copyOf(data, size));
    }

    public double get(int index) {
        return data[index];
    }

    public double[] getData() {
        return data;
    }

    public void setData(double[] data) {
        this.data = data;
    }

    public void set(int index, double value) {
        data[index] = value;
    }

    public Vector add(Vector vector) {
        Vector[] vectors = equalize(vector);
        double[] data = vectors[0].getData();
        for (int i = 0; i < vectors[0].size; i++) {
            data[i] += vectors[1].get(i);
        }
        return vectors[0];
    }

    public Vector subtract(Vector vector) {
        Vector[] vectors = equalize(vector);
        double[] data = vectors[0].getData();
        for (int i = 0; i < vectors[0].size; i++) {
            data[i] -= vectors[1].get(i);
        }
        return vectors[0];
    }

    public double dot(Vector vector) {
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += data[i] * vector.get(i);
        }
        return sum;
    }

    public double magnitude() {
        return Math.sqrt(dot(this));
    }

    public Vector normalize() {
        return new Vector(Arrays.stream(data).map(d -> d / magnitude()).toArray());
    }

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

    public boolean isZero() {
        return this.equals(new Vector(new double[size]));
    }

    public boolean equals(Vector other) {
        for(int i = 0; i < size; i++) {
            if (data[i] != other.getData()[i]) {
                return false;
            }
        }
        return true;
    }

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

    public void negate() {
        for (int i = 0; i < size; i++) {
            data[i] *= -1;
        }
    }

    public String toString() {
        return Arrays.toString(data);
    }

    public Vector multiplyScalar(double scalar) {
        Vector v = this.copy();
        double[] vData = v.getData();
        for (int i = 0; i < v.dimension(); i++) {
            vData[i] *= scalar;
        }
        return v;
    }

    public Matrix transpose() {
        return new Matrix(new Vector[]{this}).transpose();
    }

    public Matrix multiply(Matrix matrix) {
        if (1 != matrix.getRowCount()) {
            throw new ArithmeticException("Vector dimension must match matrix row count for multiplication");
        }
        Matrix m = new Matrix(new Vector[]{this});
        return m.multiply(matrix);
    }
}