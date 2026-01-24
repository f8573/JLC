package net.faulj.symmetric;

import net.faulj.matrix.Matrix;

/**
 * Encapsulates the Spectral Decomposition of a symmetric matrix.
 * <p>
 * For a real symmetric matrix A, the spectral decomposition is the factorization:
 * </p>
 * <div align="center">A = Q &Lambda; Q<sup>T</sup></div>
 * <p>where:</p>
 * <ul>
 * <li><b>Q</b> - Orthogonal matrix of eigenvectors (Q<sup>T</sup>Q = I)</li>
 * <li><b>&Lambda;</b> - Diagonal matrix of real eigenvalues</li>
 * <li><b>Q<sup>T</sup></b> - Transpose of Q (which is also its inverse)</li>
 * </ul>
 *
 * <h2>Fundamental Theorem:</h2>
 * <p>
 * The Spectral Theorem states that every real symmetric matrix is diagonalizable by an
 * orthogonal matrix. Furthermore, all its eigenvalues are real.
 * </p>
 * <pre>
 * A = λ₁u₁u₁ᵀ + λ₂u₂u₂ᵀ + ... + λₙuₙuₙᵀ
 * </pre>
 * <p>
 * This represents A as a weighted sum of orthogonal projections onto its principal axes.
 * </p>
 *
 * <h2>Properties:</h2>
 * <ul>
 * <li><b>Real Eigenvalues:</b> All entries of &Lambda; are real numbers.</li>
 * <li><b>Orthogonal Eigenvectors:</b> Eigenvectors corresponding to distinct eigenvalues are orthogonal.</li>
 * <li><b>Completeness:</b> The eigenvectors form an orthonormal basis for R<sup>n</sup>.</li>
 * <li><b>Best Approximation:</b> Truncating the sum gives the best low-rank approximation (in Frobenius norm).</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Matrix A = MatrixFactory.createSymmetric(data);
 * SymmetricEigenDecomposition eig = new SymmetricEigenDecomposition();
 * SpectralDecomposition spectral = eig.decompose(A);
 *
 * // Access components
 * Matrix Q = spectral.getEigenvectors();
 * double[] lambda = spectral.getEigenvalues();
 *
 * // Reconstruct A
 * Matrix D = MatrixFactory.diagonal(lambda);
 * Matrix reconstructed = Q.multiply(D).multiply(Q.transpose());
 *
 * // Compute power: A^k = Q * D^k * Q^T
 * Matrix Ak = spectral.function(val -> Math.pow(val, k));
 * }</pre>
 *
 * <h2>Applications:</h2>
 * <ul>
 * <li>Principal Component Analysis (PCA)</li>
 * <li>Solving quadratic optimization problems</li>
 * <li>Graph theory (spectral clustering)</li>
 * <li>Mechanical vibrations (modal analysis)</li>
 * <li>Image compression</li>
 * </ul>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see SymmetricEigenDecomposition
 * @see PrincipalAxes
 */
public class SpectralDecomposition {
    private final Matrix Q;           // Orthogonal matrix of eigenvectors
    private final double[] eigenvalues; // Real eigenvalues (sorted descending)
    private final Matrix originalMatrix;
    
    /**
     * Constructs a spectral decomposition result.
     * 
     * @param Q Orthogonal matrix of eigenvectors (columns)
     * @param eigenvalues Array of real eigenvalues corresponding to columns of Q
     * @param originalMatrix The original symmetric matrix
     */
    public SpectralDecomposition(Matrix Q, double[] eigenvalues, Matrix originalMatrix) {
        if (Q == null || eigenvalues == null || originalMatrix == null) {
            throw new IllegalArgumentException("Arguments must not be null");
        }
        if (Q.getColumnCount() != eigenvalues.length) {
            throw new IllegalArgumentException("Number of eigenvectors must equal number of eigenvalues");
        }
        this.Q = Q;
        this.eigenvalues = eigenvalues.clone();
        this.originalMatrix = originalMatrix;
    }
    
    /**
     * @return Orthogonal matrix Q where columns are eigenvectors
     */
    public Matrix getEigenvectors() {
        return Q;
    }
    
    /**
     * @return Array of eigenvalues (real, sorted in descending order)
     */
    public double[] getEigenvalues() {
        return eigenvalues.clone();
    }
    
    /**
     * @return The original matrix that was decomposed
     */
    public Matrix getOriginalMatrix() {
        return originalMatrix;
    }
    
    /**
     * Reconstruct the original matrix: A = Q * Λ * Q^T
     * 
     * @return Reconstructed matrix
     */
    public Matrix reconstruct() {
        int n = eigenvalues.length;
        Matrix Lambda = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            Lambda.set(i, i, eigenvalues[i]);
        }
        return Q.multiply(Lambda).multiply(Q.transpose());
    }
    
    /**
     * Compute matrix power: A^k = Q * Λ^k * Q^T
     * Works for any real k (negative values compute inverse powers)
     * 
     * @param k Power exponent
     * @return A raised to the k-th power
     * @throws IllegalStateException if k < 0 and matrix has zero eigenvalues
     */
    public Matrix power(double k) {
        int n = eigenvalues.length;
        Matrix LambdaK = new Matrix(n, n);
        
        for (int i = 0; i < n; i++) {
            double lambda = eigenvalues[i];
            if (k < 0 && Math.abs(lambda) < 1e-14) {
                throw new IllegalStateException("Cannot compute negative power: matrix has zero eigenvalue");
            }
            LambdaK.set(i, i, Math.pow(lambda, k));
        }
        
        return Q.multiply(LambdaK).multiply(Q.transpose());
    }
    
    /**
     * Apply arbitrary function to eigenvalues: f(A) = Q * f(Λ) * Q^T
     * 
     * @param function Function to apply to each eigenvalue
     * @return Matrix with function applied to eigenvalue spectrum
     */
    public Matrix function(java.util.function.DoubleUnaryOperator function) {
        int n = eigenvalues.length;
        Matrix fLambda = new Matrix(n, n);
        
        for (int i = 0; i < n; i++) {
            fLambda.set(i, i, function.applyAsDouble(eigenvalues[i]));
        }
        
        return Q.multiply(fLambda).multiply(Q.transpose());
    }
    
    /**
     * Compute low-rank approximation using top k eigenvalues/eigenvectors.
     * This is the best rank-k approximation in Frobenius norm.
     * 
     * @param k Number of components to keep
     * @return Rank-k approximation of original matrix
     */
    public Matrix lowRankApproximation(int k) {
        if (k < 1 || k > eigenvalues.length) {
            throw new IllegalArgumentException("k must be between 1 and " + eigenvalues.length);
        }
        
        Matrix result = new Matrix(originalMatrix.getRowCount(), originalMatrix.getColumnCount());
        double[] resultData = result.getRawData();
        
        // Sum: λᵢ * uᵢ * uᵢᵀ for i = 0 to k-1
        for (int i = 0; i < k; i++) {
            double lambda = eigenvalues[i];
            double[] ui = Q.getColumn(i);
            
            // Compute outer product: λ * u * u^T
            for (int row = 0; row < result.getRowCount(); row++) {
                for (int col = 0; col < result.getColumnCount(); col++) {
                    resultData[row * result.getColumnCount() + col] += lambda * ui[row] * ui[col];
                }
            }
        }
        
        return result;
    }
    
    /**
     * @return Sum of eigenvalues (equals trace of original matrix)
     */
    public double trace() {
        double sum = 0.0;
        for (double lambda : eigenvalues) {
            sum += lambda;
        }
        return sum;
    }
    
    /**
     * @return Product of eigenvalues (equals determinant of original matrix)
     */
    public double determinant() {
        double product = 1.0;
        for (double lambda : eigenvalues) {
            product *= lambda;
        }
        return product;
    }
    
    /**
     * @return Condition number (ratio of largest to smallest absolute eigenvalue)
     */
    public double conditionNumber() {
        if (eigenvalues.length == 0) return 1.0;
        
        double maxAbs = Math.abs(eigenvalues[0]);
        double minAbs = Math.abs(eigenvalues[0]);
        
        for (int i = 1; i < eigenvalues.length; i++) {
            double abs = Math.abs(eigenvalues[i]);
            maxAbs = Math.max(maxAbs, abs);
            minAbs = Math.min(minAbs, abs);
        }
        
        if (minAbs < 1e-14) return Double.POSITIVE_INFINITY;
        return maxAbs / minAbs;
    }
}