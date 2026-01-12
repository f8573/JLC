package net.faulj.vector;

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

    public void add(Vector vector) {
        equalize(vector);
        for (int i = 0; i < size; i++) {
            data[i] += vector.get(i);
        }
    }

    public void subtract(Vector vector) {
        equalize(vector);
        for (int i = 0; i < size; i++) {
            data[i] -= vector.get(i);
        }
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

    public void equalize(Vector other) {
        if (other.dimension() > this.dimension()) {
            data = Arrays.copyOf(data, other.dimension());
        } else {
            other.setData(Arrays.copyOf(other.getData(), this.dimension()));
        }
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
}