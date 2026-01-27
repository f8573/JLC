package net.faulj.symmetric;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Solves constrained optimization problems involving symmetric matrices.
 * <p>
 * This class primarily deals with the optimization of the <b>Rayleigh Quotient</b>:
 * </p>
 * <div align="center">R(<b>x</b>) = (<b>x</b><sup>T</sup>A<b>x</b>) / (<b>x</b><sup>T</sup><b>x</b>)</div>
 *
 * <h2>Min-Max Theorem (Courant-Fischer):</h2>
 * <p>
 * The extrema of the Rayleigh quotient are determined by the eigenvalues of A (ordered &lambda;₁ &ge; ... &ge; &lambda;ₙ):
 * </p>
 * <ul>
 * <li><b>Maximum:</b> max R(x) = &lambda;₁ (attained when x is the first eigenvector)</li>
 * <li><b>Minimum:</b> min R(x) = &lambda;ₙ (attained when x is the last eigenvector)</li>
 * <li><b>Saddle Points:</b> Intermediate eigenvalues correspond to saddle points.</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = ...;
 *
 * // Find vector x maximizing x^T*A*x subject to |x|=1
 * OptimizationResult max = ConstrainedOptimization.maximize(A);
 * double maxVal = max.getValue(); // equals largest eigenvalue
 * Vector maxVec = max.getVector(); // equals corresponding eigenvector
 *
 * // Find vector minimizing x^T*A*x subject to |x|=1
 * OptimizationResult min = ConstrainedOptimization.minimize(A);
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Vibration analysis (Finding resonant frequencies)</li>
 * <li>Quantum mechanics (Ground state energy)</li>
 * <li>Signal processing (Principal components)</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see SymmetricEigenDecomposition
 * @see QuadraticForm
 */
public class ConstrainedOptimization {
    
    /**
     * Result of a constrained optimization containing the optimal value and vector.
     */
    public static class OptimizationResult {
        private final double value;
        private final Vector vector;
        private final String type;
        
        /**
         * Create an optimization result container.
         *
         * @param value optimal Rayleigh quotient value
         * @param vector optimal unit vector
         * @param type extremum type (maximum, minimum, saddle)
         */
        public OptimizationResult(double value, Vector vector, String type) {
            this.value = value;
            this.vector = vector;
            this.type = type;
        }
        
        /**
         * @return The optimal value of the Rayleigh quotient
         */
        public double getValue() {
            return value;
        }
        
        /**
         * @return The optimal vector (normalized)
         */
        public Vector getVector() {
            return vector;
        }
        
        /**
         * @return Type of extremum ("maximum" or "minimum")
         */
        public String getType() {
            return type;
        }
        
        /**
         * @return string summary of the optimization result
         */
        @Override
        public String toString() {
            return String.format("OptimizationResult[%s = %.6f]", type, value);
        }
    }
    
    /**
     * Compute the Rayleigh quotient: R(x) = (x^T * A * x) / (x^T * x)
     * 
     * @param A Symmetric matrix
     * @param x Vector
     * @return Rayleigh quotient value
     */
    public static double rayleighQuotient(Matrix A, Vector x) {
        if (A == null || x == null) {
            throw new IllegalArgumentException("Arguments must not be null");
        }
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        if (x.dimension() != A.getRowCount()) {
            throw new IllegalArgumentException("Vector dimension must match matrix dimension");
        }
        
        double normSquared = x.dot(x);
        if (normSquared < 1e-14) {
            throw new IllegalArgumentException("Vector must be non-zero");
        }
        
        // Compute x^T * A * x
        double numerator = 0.0;
        double[] xData = x.getData();
        int n = x.dimension();
        
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A.get(i, j) * xData[j];
            }
            numerator += xData[i] * sum;
        }
        
        return numerator / normSquared;
    }
    
    /**
     * Find the vector x that maximizes x^T * A * x subject to ||x|| = 1.
     * By the Courant-Fischer theorem, this is the eigenvector corresponding
     * to the largest eigenvalue.
     * 
     * @param A Symmetric matrix
     * @return OptimizationResult containing the maximum value and optimal vector
     * @throws IllegalArgumentException if A is not square or not symmetric
     */
    public static OptimizationResult maximize(Matrix A) {
        SpectralDecomposition spectral = SymmetricEigenDecomposition.decompose(A);
        
        // Largest eigenvalue (first in sorted array)
        double[] eigenvalues = spectral.getEigenvalues();
        double maxValue = eigenvalues[0];
        
        // Corresponding eigenvector (first column)
        Matrix eigenvectors = spectral.getEigenvectors();
        double[] vecData = eigenvectors.getColumn(0);
        Vector maxVector = new Vector(vecData);
        
        return new OptimizationResult(maxValue, maxVector, "maximum");
    }
    
    /**
     * Find the vector x that minimizes x^T * A * x subject to ||x|| = 1.
     * By the Courant-Fischer theorem, this is the eigenvector corresponding
     * to the smallest eigenvalue.
     * 
     * @param A Symmetric matrix
     * @return OptimizationResult containing the minimum value and optimal vector
     * @throws IllegalArgumentException if A is not square or not symmetric
     */
    public static OptimizationResult minimize(Matrix A) {
        SpectralDecomposition spectral = SymmetricEigenDecomposition.decompose(A);
        
        // Smallest eigenvalue (last in sorted array)
        double[] eigenvalues = spectral.getEigenvalues();
        double minValue = eigenvalues[eigenvalues.length - 1];
        
        // Corresponding eigenvector (last column)
        Matrix eigenvectors = spectral.getEigenvectors();
        double[] vecData = eigenvectors.getColumn(eigenvalues.length - 1);
        Vector minVector = new Vector(vecData);
        
        return new OptimizationResult(minValue, minVector, "minimum");
    }
    
    /**
     * Find all stationary points (critical points) of the Rayleigh quotient.
     * These are all the eigenvectors of A, with corresponding eigenvalues as values.
     * 
     * @param A Symmetric matrix
     * @return Array of optimization results for each stationary point
     */
    public static OptimizationResult[] stationaryPoints(Matrix A) {
        SpectralDecomposition spectral = SymmetricEigenDecomposition.decompose(A);
        
        double[] eigenvalues = spectral.getEigenvalues();
        Matrix eigenvectors = spectral.getEigenvectors();
        int n = eigenvalues.length;
        
        OptimizationResult[] results = new OptimizationResult[n];
        
        for (int i = 0; i < n; i++) {
            double[] vecData = eigenvectors.getColumn(i);
            Vector vec = new Vector(vecData);
            
            String type;
            if (i == 0) {
                type = "maximum";
            } else if (i == n - 1) {
                type = "minimum";
            } else {
                type = "saddle";
            }
            
            results[i] = new OptimizationResult(eigenvalues[i], vec, type);
        }
        
        return results;
    }
    
    /**
     * Perform one step of the power iteration method to find the dominant eigenvector.
     * Useful for iterative approximation when full decomposition is too expensive.
     * 
     * @param A Symmetric matrix
     * @param x Current approximation vector
     * @return Updated approximation (normalized)
     */
    public static Vector powerIterationStep(Matrix A, Vector x) {
        if (A == null || x == null) {
            throw new IllegalArgumentException("Arguments must not be null");
        }
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        if (x.dimension() != A.getRowCount()) {
            throw new IllegalArgumentException("Vector dimension must match matrix dimension");
        }
        
        int n = x.dimension();
        double[] xData = x.getData();
        double[] result = new double[n];
        
        // Compute A * x
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A.get(i, j) * xData[j];
            }
            result[i] = sum;
        }
        
        // Normalize
        Vector v = new Vector(result);
        return v.normalize();
    }
    
    /**
     * Perform one step of inverse iteration to find the smallest eigenvector.
     * This requires solving A * y = x.
     * 
     * @param A Symmetric matrix
     * @param x Current approximation vector
     * @return Updated approximation (normalized)
     * @throws IllegalStateException if A is singular
     */
    public static Vector inverseIterationStep(Matrix A, Vector x) {
        if (A == null || x == null) {
            throw new IllegalArgumentException("Arguments must not be null");
        }
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        if (x.dimension() != A.getRowCount()) {
            throw new IllegalArgumentException("Vector dimension must match matrix dimension");
        }
        
        // Solve A * y = x
        Vector y = A.solve(x);
        
        // Normalize
        return y.normalize();
    }
    
    /**
     * Estimate the condition number using Rayleigh quotients.
     * Condition number = max(|λ|) / min(|λ|)
     * 
     * @param A Symmetric matrix
     * @return Condition number estimate
     */
    public static double estimateConditionNumber(Matrix A) {
        SpectralDecomposition spectral = SymmetricEigenDecomposition.decompose(A);
        double[] eigenvalues = spectral.getEigenvalues();
        
        double maxAbs = 0.0;
        double minAbs = Double.POSITIVE_INFINITY;
        
        for (double lambda : eigenvalues) {
            double abs = Math.abs(lambda);
            maxAbs = Math.max(maxAbs, abs);
            if (abs > 1e-14) {
                minAbs = Math.min(minAbs, abs);
            }
        }
        
        if (minAbs == Double.POSITIVE_INFINITY || minAbs < 1e-14) {
            return Double.POSITIVE_INFINITY;
        }
        
        return maxAbs / minAbs;
    }
}