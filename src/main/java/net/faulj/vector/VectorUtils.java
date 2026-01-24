package net.faulj.vector;

import net.faulj.matrix.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Static utility methods for vector creation and algorithms.
 */
public class VectorUtils {

    public static Vector unitVector(int dimension, int number) {
        if (number >= dimension) {
            throw new ArithmeticException("The index of the unit cannot be larger than the dimension of the vector");
        }
        Vector v = zero(dimension);
        v.set(number, 1);
        return v;
    }

    public static Vector zero(int size) {
        return new Vector(new double[size]);
    }

    public static Vector random(int size) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
        }
        return new Vector(data);
    }

    public static Vector householder(Vector x) {
        if (!x.isReal()) {
            throw new UnsupportedOperationException("Householder vectors require real input");
        }
        Vector v = x.copy();
        int m = v.dimension();

        double alpha = v.norm2();
        double beta = -1 * Math.copySign(alpha, v.get(0));
        v.set(0, v.get(0) - beta);
        double tau = 2.0 / v.dot(v);
        v = v.resize(m + 1);
        v.set(m, tau);
        return v;
    }

    public static List<Vector> gramSchmidt(List<Vector> vectors) {
        List<Vector> orthogonalBasis = new ArrayList<>();
        for (Vector vector : vectors) {
            Vector projection = null;
            if (!orthogonalBasis.isEmpty()) {
                for (Vector basisVector : orthogonalBasis) {
                    projection = vector.project(basisVector);
                    vector = vector.subtract(projection);
                }
            }
            orthogonalBasis.add(vector.normalize());
        }
        return orthogonalBasis;
    }

    public static Matrix projectOnto(List<Vector> projectors, List<Vector> sourceVectors) {
        if (projectors == null || sourceVectors == null) {
            throw new IllegalArgumentException("Projectors and source vectors must not be null");
        }
        if (projectors.isEmpty()) {
            throw new IllegalArgumentException("Projector basis must not be empty");
        }
        if (sourceVectors.isEmpty()) {
            throw new IllegalArgumentException("Source vectors must not be empty");
        }

        List<Vector> orthogonalBasis = gramSchmidt(projectors);

        int dimension = sourceVectors.getFirst().dimension();
        for (Vector v : orthogonalBasis) {
            if (v.dimension() != dimension) {
                throw new IllegalArgumentException("Projector basis dimension mismatch");
            }
        }
        for (Vector v : sourceVectors) {
            if (v.dimension() != dimension) {
                throw new IllegalArgumentException("Source vector dimension mismatch");
            }
        }

        Matrix resultMatrix = new Matrix(dimension, sourceVectors.size());
        for (int i = 0; i < sourceVectors.size(); i++) {
            Vector source = sourceVectors.get(i);
            Vector projection = new Vector(new double[dimension]);
            for (Vector basisVector : orthogonalBasis) {
                double coeff = source.dot(basisVector);
                projection = projection.add(basisVector.multiplyScalar(coeff));
            }
            resultMatrix.setColumn(i, projection);
        }
        return resultMatrix;
    }

    public static Matrix normalizeBasis(List<Vector> basis) {
        Matrix resultMatrix = new Matrix(basis.getFirst().dimension(), basis.size());
        for (int i = 0; i < basis.size(); i++) {
            resultMatrix.setColumn(i, basis.get(i));
        }
        return resultMatrix;
    }

    public static Matrix toMatrix(List<Vector> vectors) {
        return new Matrix(vectors.toArray(new Vector[0]));
    }
}
