package net.faulj.symmetric;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

/**
 * Handles the Principal Axis Theorem and coordinate transformations for symmetric matrices.
 * <p>
 * The Principal Axis Theorem states that any quadratic form Q(x) = x<sup>T</sup>Ax can be
 * transformed into a diagonal form with no "cross-product" terms (x<sub>i</sub>x<sub>j</sub>) by a suitable
 * orthogonal change of variables <b>x</b> = P<b>y</b>.
 * </p>
 *
 * <h2>Transformation:</h2>
 * <p>
 * Substituting <b>x</b> = P<b>y</b> (where P is the matrix of eigenvectors):
 * </p>
 * <pre>
 * Q(x) = (Py)<sup>T</sup> A (Py)
 * = y<sup>T</sup> (P<sup>T</sup>AP) y
 * = y<sup>T</sup> &Lambda; y
 * = &lambda;₁y₁² + &lambda;₂y₂² + ... + &lambda;ₙyₙ²
 * </pre>
 *
 * <h2>Geometric Interpretation:</h2>
 * <ul>
 * <li><b>Principal Axes:</b> The directions defined by the eigenvectors of A.</li>
 * <li><b>Lengths:</b> The lengths of the semi-axes of the quadric surface (e.g., ellipsoid) are related to 1/&radic;|&lambda;<sub>i</sub>|.</li>
 * <li><b>Rotation:</b> The matrix P represents a rotation (and possibly reflection) aligning the standard basis with the principal axes.</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * QuadraticForm q = new QuadraticForm(A);
 * PrincipalAxes axes = new PrincipalAxes(q);
 *
 * // Get the rotation matrix (eigenvectors)
 * Matrix P = axes.getRotationMatrix();
 *
 * // Get the new coefficients (eigenvalues)
 * double[] coeffs = axes.getPrincipalCoefficients();
 *
 * System.out.println("Equation in new coords: " +
 * coeffs[0] + "y1^2 + " + coeffs[1] + "y2^2 = C");
 * }</pre>
 *
 * @author JLC Development Team
 * @version 1.0
 * @since 1.0
 * @see QuadraticForm
 * @see SpectralDecomposition
 */
public class PrincipalAxes {
    
    private final QuadraticForm form;
    private final SpectralDecomposition spectral;
    private final Matrix P;  // Rotation matrix (eigenvectors)
    private final double[] coefficients;  // Principal coefficients (eigenvalues)
    
    /**
     * Construct principal axes from a quadratic form.
     * 
     * @param form QuadraticForm to analyze
     */
    public PrincipalAxes(QuadraticForm form) {
        if (form == null) {
            throw new IllegalArgumentException("QuadraticForm must not be null");
        }
        this.form = form;
        this.spectral = form.getSpectralDecomposition();
        this.P = spectral.getEigenvectors();
        this.coefficients = spectral.getEigenvalues();
    }
    
    /**
     * Construct principal axes directly from a symmetric matrix.
     * 
     * @param A Symmetric matrix
     */
    public PrincipalAxes(Matrix A) {
        this(new QuadraticForm(A));
    }
    
    /**
     * Get the rotation matrix P that transforms to principal coordinates.
     * Columns of P are the principal axes (eigenvectors).
     * 
     * @return Orthogonal matrix P
     */
    public Matrix getRotationMatrix() {
        return P;
    }
    
    /**
     * Get the principal coefficients (eigenvalues).
     * In the transformed coordinates y = P^T * x, the quadratic form becomes:
     * Q(x) = λ₁y₁² + λ₂y₂² + ... + λₙyₙ²
     * 
     * @return Array of principal coefficients
     */
    public double[] getPrincipalCoefficients() {
        return coefficients.clone();
    }
    
    /**
     * Transform a point from standard coordinates to principal coordinates.
     * y = P^T * x
     * 
     * @param x Point in standard coordinates
     * @return Point in principal coordinates
     */
    public Vector toPrincipalCoordinates(Vector x) {
        if (x == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        if (x.dimension() != P.getRowCount()) {
            throw new IllegalArgumentException("Vector dimension must match matrix dimension");
        }
        
        int n = x.dimension();
        double[] y = new double[n];
        double[] xData = x.getData();
        
        // y = P^T * x
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += P.get(j, i) * xData[j];
            }
            y[i] = sum;
        }
        
        return new Vector(y);
    }
    
    /**
     * Transform a point from principal coordinates back to standard coordinates.
     * x = P * y
     * 
     * @param y Point in principal coordinates
     * @return Point in standard coordinates
     */
    public Vector fromPrincipalCoordinates(Vector y) {
        if (y == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        if (y.dimension() != P.getColumnCount()) {
            throw new IllegalArgumentException("Vector dimension must match matrix dimension");
        }
        
        int n = y.dimension();
        double[] x = new double[n];
        double[] yData = y.getData();
        
        // x = P * y
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += P.get(i, j) * yData[j];
            }
            x[i] = sum;
        }
        
        return new Vector(x);
    }
    
    /**
     * Evaluate the quadratic form in principal coordinates.
     * In principal coordinates: Q(y) = λ₁y₁² + λ₂y₂² + ... + λₙyₙ²
     * 
     * @param y Point in principal coordinates
     * @return Value of quadratic form
     */
    public double evaluateInPrincipalCoordinates(Vector y) {
        if (y == null) {
            throw new IllegalArgumentException("Vector must not be null");
        }
        if (y.dimension() != coefficients.length) {
            throw new IllegalArgumentException("Vector dimension must match number of coefficients");
        }
        
        double sum = 0.0;
        double[] yData = y.getData();
        
        for (int i = 0; i < coefficients.length; i++) {
            sum += coefficients[i] * yData[i] * yData[i];
        }
        
        return sum;
    }
    
    /**
     * Get the i-th principal axis (i-th eigenvector).
     * 
     * @param i Index of principal axis (0-based)
     * @return Principal axis as a unit vector
     */
    public Vector getPrincipalAxis(int i) {
        if (i < 0 || i >= P.getColumnCount()) {
            throw new IllegalArgumentException("Invalid axis index");
        }
        return new Vector(P.getColumn(i));
    }
    
    /**
     * Get the length of the i-th semi-axis of the quadric surface Q(x) = 1.
     * For an ellipsoid: semi-axis length = 1/√|λᵢ|
     * 
     * @param i Index of axis (0-based)
     * @return Semi-axis length
     * @throws IllegalStateException if eigenvalue is zero
     */
    public double getSemiAxisLength(int i) {
        if (i < 0 || i >= coefficients.length) {
            throw new IllegalArgumentException("Invalid axis index");
        }
        
        double lambda = coefficients[i];
        if (Math.abs(lambda) < 1e-14) {
            throw new IllegalStateException("Cannot compute semi-axis length: eigenvalue is zero");
        }
        
        return 1.0 / Math.sqrt(Math.abs(lambda));
    }
    
    /**
     * Get all semi-axis lengths for the quadric surface Q(x) = 1.
     * 
     * @return Array of semi-axis lengths
     * @throws IllegalStateException if any eigenvalue is zero
     */
    public double[] getAllSemiAxisLengths() {
        double[] lengths = new double[coefficients.length];
        for (int i = 0; i < coefficients.length; i++) {
            lengths[i] = getSemiAxisLength(i);
        }
        return lengths;
    }
    
    /**
     * Generate a string representation of the quadratic form in principal coordinates.
     * Example: "2.5*y1^2 + 1.3*y2^2 - 0.7*y3^2"
     * 
     * @return String representation
     */
    public String getCanonicalForm() {
        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < coefficients.length; i++) {
            double coeff = coefficients[i];
            
            if (i > 0) {
                if (coeff >= 0) {
                    sb.append(" + ");
                } else {
                    sb.append(" - ");
                    coeff = -coeff;
                }
            } else if (coeff < 0) {
                sb.append("-");
                coeff = -coeff;
            }
            
            sb.append(String.format("%.4f", coeff));
            sb.append("*y").append(i + 1).append("^2");
        }
        
        return sb.toString();
    }
    
    /**
     * Classify the type of quadric surface defined by Q(x) = c.
     * 
     * @param c Constant value
     * @return Description of quadric type
     */
    public String classifyQuadric(double c) {
        QuadraticForm.Definiteness def = form.classify();
        int n = coefficients.length;
        
        if (n == 2) {
            if (def == QuadraticForm.Definiteness.POSITIVE_DEFINITE) {
                if (c > 0) return "Ellipse";
                if (c == 0) return "Point";
                return "Empty set";
            } else if (def == QuadraticForm.Definiteness.NEGATIVE_DEFINITE) {
                if (c < 0) return "Ellipse";
                if (c == 0) return "Point";
                return "Empty set";
            } else if (def == QuadraticForm.Definiteness.INDEFINITE) {
                if (c != 0) return "Hyperbola";
                return "Pair of intersecting lines";
            }
        } else if (n == 3) {
            if (def == QuadraticForm.Definiteness.POSITIVE_DEFINITE) {
                if (c > 0) return "Ellipsoid";
                if (c == 0) return "Point";
                return "Empty set";
            } else if (def == QuadraticForm.Definiteness.NEGATIVE_DEFINITE) {
                if (c < 0) return "Ellipsoid";
                if (c == 0) return "Point";
                return "Empty set";
            } else if (def == QuadraticForm.Definiteness.INDEFINITE) {
                // Count positive and negative eigenvalues
                int numPos = 0;
                int numNeg = 0;
                for (double lambda : coefficients) {
                    if (lambda > 1e-14) numPos++;
                    else if (lambda < -1e-14) numNeg++;
                }
                
                if (numPos == 2 && numNeg == 1) {
                    if (c != 0) return "Hyperboloid of one sheet";
                    return "Double cone";
                } else if (numPos == 1 && numNeg == 2) {
                    if (c != 0) return "Hyperboloid of two sheets";
                    return "Double cone";
                }
            }
        }
        
        return "Quadric surface";
    }
    
    /**
     * @return string summary of the principal axes
     */
    @Override
    public String toString() {
        return String.format("PrincipalAxes[n=%d, canonical: %s]", 
            coefficients.length, getCanonicalForm());
    }
}