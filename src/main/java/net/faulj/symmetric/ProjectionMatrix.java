package net.faulj.symmetric;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Represents a symmetric orthogonal projection matrix.
 * <p>
 * A square matrix P is an orthogonal projection matrix onto a subspace S if:
 * </p>
 * <ul>
 * <li><b>Idempotent:</b> P<sup>2</sup> = P (Projection property)</li>
 * <li><b>Symmetric:</b> P<sup>T</sup> = P (Orthogonality property)</li>
 * </ul>
 *
 * <h2>Construction:</h2>
 * <p>
 * Given a matrix A whose columns form a basis for subspace S:
 * </p>
 * <pre>
 * P = A(A<sup>T</sup>A)⁻¹A<sup>T</sup>
 * </pre>
 * <p>
 * If the columns of A are orthonormal (Q), this simplifies to:
 * </p>
 * <pre>
 * P = QQ<sup>T</sup> = &sum; u<sub>i</sub>u<sub>i</sub><sup>T</sup>
 * </pre>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li>For any vector v, Pv is the vector in S closest to v.</li>
 * <li>The vector (v - Pv) is orthogonal to S.</li>
 * <li>Eigenvalues are either 0 or 1.</li>
 * <li>Trace(P) = dimension of subspace S (Rank of P).</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Basis vectors for a plane
 * Vector u1 = ...;
 * Vector u2 = ...;
 *
 * // Create projection onto span{u1, u2}
 * ProjectionMatrix P = ProjectionMatrix.fromBasis(u1, u2);
 *
 * Vector v = new Vector(10, 5, 2);
 * Vector projection = P.apply(v); // The "shadow" of v on the plane
 * Vector error = v.subtract(projection); // The orthogonal component
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see net.faulj.spaces.Projection
 * @see SpectralDecomposition
 */
public class ProjectionMatrix {
    
    private final Matrix P;
    private final int dimension;
    private final int rank;
    private static final double EPS = 2.220446049250313e-16;
    
    /**
     * Construct a projection matrix from a given matrix.
     * 
     * @param P Projection matrix (should be symmetric and idempotent)
     */
    private ProjectionMatrix(Matrix P) {
        this.P = P;
        this.dimension = P.getRowCount();
        this.rank = computeRank(P);
    }
    
    /**
     * Create an orthogonal projection matrix onto the column space of A.
     * Formula: P = A(A^T A)^(-1) A^T
     * 
     * @param A Matrix whose columns span the subspace
     * @return ProjectionMatrix onto column space of A
     * @throws IllegalArgumentException if columns are linearly dependent
     */
    public static ProjectionMatrix fromBasis(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        
        int m = A.getRowCount();
        int n = A.getColumnCount();
        
        if (n == 0 || m == 0) {
            throw new IllegalArgumentException("Matrix must have non-zero dimensions");
        }
        
        // Compute A^T * A
        Matrix AtA = A.transpose().multiply(A);
        
        // Check if columns are linearly independent
        double det = (n == m && n <= 3) ? AtA.determinant() : Double.NaN;
        if (!Double.isNaN(det) && Math.abs(det) < EPS) {
            throw new IllegalArgumentException("Columns of A are linearly dependent");
        }
        
        // Compute (A^T A)^(-1)
        Matrix AtAinv;
        try {
            AtAinv = AtA.inverse();
        } catch (Exception e) {
            throw new IllegalArgumentException("Columns of A are linearly dependent or nearly so", e);
        }
        
        // P = A * (A^T A)^(-1) * A^T
        Matrix P = A.multiply(AtAinv).multiply(A.transpose());
        
        return new ProjectionMatrix(P);
    }
    
    /**
     * Create an orthogonal projection matrix from a set of basis vectors.
     * 
     * @param vectors Basis vectors for the subspace
     * @return ProjectionMatrix onto span of the vectors
     */
    public static ProjectionMatrix fromBasis(Vector... vectors) {
        if (vectors == null || vectors.length == 0) {
            throw new IllegalArgumentException("Must provide at least one vector");
        }
        
        Matrix A = new Matrix(vectors);
        return fromBasis(A);
    }
    
    /**
     * Create an orthogonal projection matrix from orthonormal basis vectors.
     * For orthonormal columns Q: P = Q * Q^T
     * This is more efficient than the general case.
     * 
     * @param Q Matrix with orthonormal columns
     * @return ProjectionMatrix
     */
    public static ProjectionMatrix fromOrthonormalBasis(Matrix Q) {
        if (Q == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        
        Matrix P = Q.multiply(Q.transpose());
        return new ProjectionMatrix(P);
    }
    
    /**
     * Create an orthogonal projection matrix from orthonormal basis vectors.
     * 
     * @param vectors Orthonormal basis vectors
     * @return ProjectionMatrix
     */
    public static ProjectionMatrix fromOrthonormalBasis(Vector... vectors) {
        if (vectors == null || vectors.length == 0) {
            throw new IllegalArgumentException("Must provide at least one vector");
        }
        
        Matrix Q = new Matrix(vectors);
        return fromOrthonormalBasis(Q);
    }
    
    /**
     * Create projection onto a single vector (1-dimensional subspace).
     * P = (v * v^T) / (v^T * v)
     * 
     * @param v Vector defining the subspace
     * @return ProjectionMatrix onto span{v}
     */
    public static ProjectionMatrix fromVector(Vector v) {
        if (v == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        
        int n = v.dimension();
        double normSquared = v.dot(v);
        
        if (normSquared < EPS) {
            throw new IllegalArgumentException("Vector must be non-zero");
        }
        
        Matrix P = new Matrix(n, n);
        double[] vData = v.getData();
        
        // P = (v * v^T) / ||v||^2
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                P.set(i, j, (vData[i] * vData[j]) / normSquared);
            }
        }
        
        return new ProjectionMatrix(P);
    }
    
    /**
     * Create the identity projection (projects onto entire space).
     * 
     * @param dimension Dimension of the space
     * @return Identity projection matrix
     */
    public static ProjectionMatrix identity(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive");
        }
        
        Matrix I = Matrix.Identity(dimension);
        return new ProjectionMatrix(I);
    }
    
    /**
     * Create the zero projection (projects everything to zero).
     * 
     * @param dimension Dimension of the space
     * @return Zero projection matrix
     */
    public static ProjectionMatrix zero(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive");
        }
        
        Matrix Z = new Matrix(dimension, dimension);
        return new ProjectionMatrix(Z);
    }
    
    /**
     * Get the projection matrix.
     * 
     * @return The matrix P
     */
    public Matrix getMatrix() {
        return P;
    }
    
    /**
     * Apply the projection to a vector: result = P * v
     * 
     * @param v Vector to project
     * @return Projected vector (in the subspace)
     */
    public Vector apply(Vector v) {
        if (v == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        if (v.dimension() != dimension) {
            throw new IllegalArgumentException("Vector dimension must match projection dimension");
        }
        
        double[] result = new double[dimension];
        double[] vData = v.getData();
        
        for (int i = 0; i < dimension; i++) {
            double sum = 0.0;
            for (int j = 0; j < dimension; j++) {
                sum += P.get(i, j) * vData[j];
            }
            result[i] = sum;
        }
        
        return new Vector(result);
    }
    
    /**
     * Compute the orthogonal complement projection: I - P
     * This projects onto the orthogonal complement of the subspace.
     * 
     * @return Complementary projection matrix
     */
    public ProjectionMatrix complement() {
        Matrix I = Matrix.Identity(dimension);
        Matrix complement = I.subtract(P);
        return new ProjectionMatrix(complement);
    }
    
    /**
     * Compute the error (orthogonal component) when projecting v.
     * error = v - P*v = (I - P)*v
     * 
     * @param v Vector to analyze
     * @return Orthogonal component (error vector)
     */
    public Vector error(Vector v) {
        Vector projection = apply(v);
        return v.subtract(projection);
    }
    
    /**
     * Get the dimension of the ambient space.
     * 
     * @return Dimension of the full space
     */
    public int getDimension() {
        return dimension;
    }
    
    /**
     * Get the rank of the projection (dimension of the subspace).
     * 
     * @return Rank of P
     */
    public int getRank() {
        return rank;
    }
    
    /**
     * Verify that this is indeed a projection matrix: P^2 = P
     * 
     * @param tolerance Tolerance for comparison
     * @return true if idempotent within tolerance
     */
    public boolean isIdempotent(double tolerance) {
        Matrix P2 = P.multiply(P);
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                double diff = Math.abs(P.get(i, j) - P2.get(i, j));
                if (diff > tolerance) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /**
     * Verify that this is an orthogonal projection: P^T = P
     * 
     * @param tolerance Tolerance for comparison
     * @return true if symmetric within tolerance
     */
    public boolean isOrthogonal(double tolerance) {
        for (int i = 0; i < dimension; i++) {
            for (int j = i + 1; j < dimension; j++) {
                double diff = Math.abs(P.get(i, j) - P.get(j, i));
                if (diff > tolerance) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /**
     * Compute the rank of the projection matrix (trace equals rank for projections).
     *
     * @param P projection matrix
     * @return rank estimate
     */
    private static int computeRank(Matrix P) {
        double trace = 0.0;
        int n = P.getRowCount();
        for (int i = 0; i < n; i++) {
            trace += P.get(i, i);
        }
        return (int) Math.round(trace);
    }
    
    /**
     * Compute the distance from a point to the subspace.
     * distance = ||v - P*v|| = ||error||
     * 
     * @param v Vector to measure from
     * @return Distance to subspace
     */
    public double distance(Vector v) {
        Vector err = error(v);
        return err.norm2();
    }
    
    /**
     * Check if a vector is in the subspace (P*v ≈ v).
     * 
     * @param v Vector to test
     * @param tolerance Tolerance for comparison
     * @return true if v is in the subspace
     */
    public boolean contains(Vector v, double tolerance) {
        return distance(v) <= tolerance;
    }
    
    /**
     * Apply projection to a matrix (project each column).
     * 
     * @param M Matrix to project
     * @return Projected matrix
     */
    public Matrix apply(Matrix M) {
        if (M == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (M.getRowCount() != dimension) {
            throw new IllegalArgumentException("Matrix row count must match projection dimension");
        }
        
        return P.multiply(M);
    }
    
    /**
     * @return string summary of the projection matrix
     */
    @Override
    public String toString() {
        return String.format("ProjectionMatrix[dimension=%d, rank=%d]", dimension, rank);
    }
}