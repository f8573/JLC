package net.faulj.vector;

import net.faulj.matrix.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Static utility methods for vector creation and algorithms.
 */
public class VectorUtils {

    /**
     * Create a unit basis vector of given dimension.
     *
     * @param dimension vector dimension
     * @param number index of the unit entry
     * @return unit vector
     */
    public static Vector unitVector(int dimension, int number) {
        if (number >= dimension) {
            throw new ArithmeticException("The index of the unit cannot be larger than the dimension of the vector");
        }
        Vector v = zero(dimension);
        v.set(number, 1);
        return v;
    }

    /**
     * Create a zero vector.
     *
     * @param size vector dimension
     * @return zero vector
     */
    public static Vector zero(int size) {
        return new Vector(new double[size]);
    }

    /**
     * Create a vector with random entries in $[0,1)$.
     *
     * @param size vector dimension
     * @return random vector
     */
    public static Vector random(int size) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
        }
        return new Vector(data);
    }

    /**
     * Build the Householder vector for a real input vector.
     *
     * @param x input vector
     * @return Householder vector with tau stored as last entry
     */
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

    /**
     * Perform Gram-Schmidt orthonormalization.
     *
     * @param vectors input vectors
     * @return orthonormal basis list
     */
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

    /**
     * Project source vectors onto the span of projector vectors.
     *
     * @param projectors basis vectors used to define the subspace
     * @param sourceVectors vectors to project
     * @return matrix of projected vectors
     */
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

    /**
     * Convert a basis list into a matrix with vectors as columns.
     *
     * @param basis basis vectors
     * @return matrix of basis vectors
     */
    public static Matrix normalizeBasis(List<Vector> basis) {
        Matrix resultMatrix = new Matrix(basis.getFirst().dimension(), basis.size());
        for (int i = 0; i < basis.size(); i++) {
            resultMatrix.setColumn(i, basis.get(i));
        }
        return resultMatrix;
    }

    /**
     * Convert a list of vectors into a matrix with columns as vectors.
     *
     * @param vectors vectors to convert
     * @return matrix with vectors as columns
     */
    public static Matrix toMatrix(List<Vector> vectors) {
        return new Matrix(vectors.toArray(new Vector[0]));
    }
}
