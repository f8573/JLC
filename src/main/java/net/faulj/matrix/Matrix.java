package net.faulj.matrix;

import net.faulj.vector.Vector;

public class Matrix {
    private Vector[] data;
    private int columns;

    public Matrix(Vector[] data) {
        this.data = data;
        this.columns = data.length;
    }

    public Vector[] getData() {
        return data;
    }

    public double get(int row, int column) {
        return data[column].get(row);
    }

    public void set(int row, int column, double value) {
        data[column].set(row, value);
    }

    public void setData(Vector[] data) {
        this.data = data;
    }


}