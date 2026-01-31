package net.faulj.symmetric;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Represents a quadratic form associated with a symmetric matrix.
 * <p>
 * A quadratic form on R<sup>n</sup> is a function Q: R<sup>n</sup> &rarr; R defined by:
 * </p>
 * <div align="center">Q(<b>x</b>) = <b>x</b><sup>T</sup>A<b>x</b> = &sum; a<sub>ij</sub>x<sub>i</sub>x<sub>j</sub></div>
 * <p>
 * where A is a symmetric n&times;n matrix.
 * </p>
 *
 * <h2>Definiteness Classification:</h2>
 * <p>
 * The nature of the quadratic form is determined by the eigenvalues of A:
 * </p>
 * <table border="1">
 * <tr><th>Type</th><th>Condition</th><th>Eigenvalues (&lambda;)</th><th>Geometry</th></tr>
 * <tr><td>Positive Definite</td><td>x<sup>T</sup>Ax &gt; 0 for all x &ne; 0</td><td>All &lambda; &gt; 0</td><td>Elliptical Paraboloid (Bowl)</td></tr>
 * <tr><td>Positive Semidefinite</td><td>x<sup>T</sup>Ax &ge; 0 for all x</td><td>All &lambda; &ge; 0</td><td>Valley</td></tr>
 * <tr><td>Negative Definite</td><td>x<sup>T</sup>Ax &lt; 0 for all x &ne; 0</td><td>All &lambda; &lt; 0</td><td>Inverted Bowl</td></tr>
 * <tr><td>Indefinite</td><td>Positive and negative values exist</td><td>Mixed signs</td><td>Saddle Point</td></tr>
 * </table>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = ...; // Symmetric matrix
 * QuadraticForm q = new QuadraticForm(A);
 *
 * Vector x = new Vector(1.0, 2.0, 3.0);
 * double value = q.evaluate(x); // Computes x^T * A * x
 *
 * if (q.isPositiveDefinite()) {
 * System.out.println("Global minimum at origin");
 * } else if (q.isIndefinite()) {
 * System.out.println("Saddle point detected");
 * }
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Multivariate calculus (Hessian matrix test)</li>
 * <li>Physics (Kinetic and Potential Energy)</li>
 * <li>Statistics (Mahalanobis distance, covariance)</li>
 * <li>Optimization (Convexity analysis)</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see PrincipalAxes
 * @see ConstrainedOptimization
 */
public class QuadraticForm {
    
    private final Matrix A;
    private final int n;
    private final boolean isSquare;
    private SpectralDecomposition spectral;
    private static final double EPS = 2.220446049250313e-16;
    
    /**
     * Classification of quadratic form definiteness based on eigenvalues.
     */
    public enum Definiteness {
        POSITIVE_DEFINITE,      // All λ > 0
        POSITIVE_SEMIDEFINITE,  // All λ ≥ 0, at least one λ = 0
        NEGATIVE_DEFINITE,      // All λ < 0
        NEGATIVE_SEMIDEFINITE,  // All λ ≤ 0, at least one λ = 0
        INDEFINITE,             // Mixed signs
        UNDEFINED               // Not defined for rectangular matrices
    }
    
    /**
     * Construct a quadratic form from a symmetric matrix.
     * 
     * @param A Symmetric matrix defining the quadratic form Q(x) = x^T * A * x
     * @throws IllegalArgumentException if A is not symmetric
     */
    public QuadraticForm(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        this.isSquare = A.isSquare();

        if (this.isSquare) {
            // Symmetrize if nearly symmetric
            this.A = symmetrize(A);
            this.n = A.getRowCount();
        } else {
            // Keep original rectangular matrix; definiteness will be UNDEFINED
            this.A = A;
            this.n = A.getRowCount();
        }
    }
    
    /**
     * Ensure matrix is exactly symmetric by averaging with its transpose.
     *
     * @param A input matrix
     * @return symmetrized matrix
     */
    private static Matrix symmetrize(Matrix A) {
        int n = A.getRowCount();
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Symmetrize requires a square matrix");
        }
        Matrix sym = new Matrix(n, n);
        
        for (int i = 0; i < n; i++) {
            sym.set(i, i, A.get(i, i));
            for (int j = i + 1; j < n; j++) {
                double avg = (A.get(i, j) + A.get(j, i)) / 2.0;
                sym.set(i, j, avg);
                sym.set(j, i, avg);
            }
        }
        
        return sym;
    }
    
    /**
     * @return The symmetric matrix defining this quadratic form
     */
    public Matrix getMatrix() {
        return A;
    }
    
    /**
     * Evaluate the quadratic form at a point: Q(x) = x^T * A * x
     * 
     * @param x Vector at which to evaluate
     * @return Scalar value Q(x)
     * @throws IllegalArgumentException if x has wrong dimension
     */
    public double evaluate(Vector x) {
        if (x == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        if (x.dimension() != n) {
            throw new IllegalArgumentException("Vector dimension must match matrix dimension");
        }
        
        // Compute x^T * A * x efficiently
        double result = 0.0;
        double[] xData = x.getData();
        
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A.get(i, j) * xData[j];
            }
            result += xData[i] * sum;
        }
        
        return result;
    }
    
    /**
     * Compute the gradient of Q at x: ∇Q(x) = 2*A*x
     * (For symmetric A, gradient is exactly 2*A*x)
     * 
     * @param x Point at which to compute gradient
     * @return Gradient vector
     */
    public Vector gradient(Vector x) {
        if (x == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        if (x.dimension() != n) {
            throw new IllegalArgumentException("Vector dimension must match matrix dimension");
        }
        
        double[] grad = new double[n];
        double[] xData = x.getData();
        
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A.get(i, j) * xData[j];
            }
            grad[i] = 2.0 * sum;
        }
        
        return new Vector(grad);
    }
    
    /**
     * The Hessian of a quadratic form is constant: H = 2*A
     * 
     * @return Hessian matrix (2*A)
     */
    public Matrix hessian() {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Hessian is defined only for square matrices");
        }
        return A.multiplyScalar(2.0);
    }
    
    /**
     * Classify the definiteness of this quadratic form.
     * 
     * @return Definiteness classification
     */
    public Definiteness classify() {
        if (!A.isSquare()) {
            return Definiteness.UNDEFINED;
        }

        ensureSpectral();
        double[] eigenvalues = spectral.getEigenvalues();
        
        // Count positive, negative, and zero eigenvalues
        int numPositive = 0;
        int numNegative = 0;
        int numZero = 0;
        
        double maxAbs = 0.0;
        for (double lambda : eigenvalues) {
            maxAbs = Math.max(maxAbs, Math.abs(lambda));
        }
        double tol = Math.max(EPS * maxAbs * n, 1e-12);
        
        for (double lambda : eigenvalues) {
            if (lambda > tol) {
                numPositive++;
            } else if (lambda < -tol) {
                numNegative++;
            } else {
                numZero++;
            }
        }
        
        // Determine definiteness
        if (numPositive == n) {
            return Definiteness.POSITIVE_DEFINITE;
        } else if (numNegative == n) {
            return Definiteness.NEGATIVE_DEFINITE;
        } else if (numPositive > 0 && numNegative > 0) {
            return Definiteness.INDEFINITE;
        } else if (numPositive > 0 && numZero > 0 && numNegative == 0) {
            return Definiteness.POSITIVE_SEMIDEFINITE;
        } else if (numNegative > 0 && numZero > 0 && numPositive == 0) {
            return Definiteness.NEGATIVE_SEMIDEFINITE;
        } else {
            // All zeros - technically positive semidefinite
            return Definiteness.POSITIVE_SEMIDEFINITE;
        }
    }
    
    /**
     * @return true if the form is positive definite (all eigenvalues > 0)
     */
    public boolean isPositiveDefinite() {
        return classify() == Definiteness.POSITIVE_DEFINITE;
    }
    
    /**
     * @return true if the form is positive semidefinite (all eigenvalues ≥ 0)
     */
    public boolean isPositiveSemidefinite() {
        Definiteness d = classify();
        return d == Definiteness.POSITIVE_DEFINITE || d == Definiteness.POSITIVE_SEMIDEFINITE;
    }
    
    /**
     * @return true if the form is negative definite (all eigenvalues < 0)
     */
    public boolean isNegativeDefinite() {
        return classify() == Definiteness.NEGATIVE_DEFINITE;
    }
    
    /**
     * @return true if the form is negative semidefinite (all eigenvalues ≤ 0)
     */
    public boolean isNegativeSemidefinite() {
        Definiteness d = classify();
        return d == Definiteness.NEGATIVE_DEFINITE || d == Definiteness.NEGATIVE_SEMIDEFINITE;
    }
    
    /**
     * @return true if the form is indefinite (mixed sign eigenvalues)
     */
    public boolean isIndefinite() {
        return classify() == Definiteness.INDEFINITE;
    }
    
    /**
     * Get the spectral decomposition (computed lazily).
     * 
     * @return SpectralDecomposition of the matrix A
     */
    public SpectralDecomposition getSpectralDecomposition() {
        ensureSpectral();
        return spectral;
    }
    
    /**
     * Ensure spectral decomposition is computed.
     */
    private void ensureSpectral() {
        if (!A.isSquare()) {
            throw new IllegalStateException("Spectral decomposition is only available for square matrices");
        }

        if (spectral == null) {
            // Use symmetric part H = (A + A^T)/2 for eigenvalue computation
            Matrix H = symmetrize(A);
            spectral = SymmetricEigenDecomposition.decompose(H);
        }
    }
    
    /**
     * Find critical points: where ∇Q(x) = 0 subject to ||x|| = 1
     * These are the eigenvectors of A.
     * 
     * @return Matrix whose columns are the critical points (eigenvectors)
     */
    public Matrix getCriticalPoints() {
        ensureSpectral();
        return spectral.getEigenvectors();
    }
    
    /**
     * Get the values of Q at all critical points (the eigenvalues).
     * 
     * @return Array of critical values
     */
    public double[] getCriticalValues() {
        ensureSpectral();
        return spectral.getEigenvalues();
    }
    
    /**
     * Get the maximum value of Q(x) subject to ||x|| = 1
     * 
     * @return Maximum value (largest eigenvalue)
     */
    public double getMaximum() {
        ensureSpectral();
        return spectral.getEigenvalues()[0]; // First eigenvalue (largest)
    }
    
    /**
     * Get the minimum value of Q(x) subject to ||x|| = 1
     * 
     * @return Minimum value (smallest eigenvalue)
     */
    public double getMinimum() {
        ensureSpectral();
        double[] eigenvalues = spectral.getEigenvalues();
        return eigenvalues[eigenvalues.length - 1]; // Last eigenvalue (smallest)
    }
    
    /**
     * @return string summary of the quadratic form
     */
    @Override
    public String toString() {
        Definiteness d = classify();
        return String.format("QuadraticForm[%dx%d, %s]", n, n, d);
    }
}