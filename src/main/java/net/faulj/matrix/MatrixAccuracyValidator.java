package net.faulj.matrix;

/**
 * Comprehensive validation of matrix accuracy for decompositions and transformations.
 * Uses adaptive thresholds based on matrix size and condition number to classify
 * numerical accuracy with realistic expectations for floating-point arithmetic.
 * 
 * <p>Thresholds are based on O(sqrt(n) * eps * log(kappa)) for norm residuals and
 * O(n * eps * log(kappa)) for element-wise residuals, where:
 * <ul>
 *   <li>n = matrix dimension</li>
 *   <li>eps = machine epsilon (~ 2.22e-16 for double)</li>
 *   <li>kappa = condition number estimate</li>
 * </ul>
 * 
 * @author JLC
 */
public class MatrixAccuracyValidator {
    
    /** Machine epsilon for double precision floating-point arithmetic */
    private static final double EPS = 2.220446049250313e-16;
    
    /**
     * Classification levels for numerical accuracy
     */
    public enum AccuracyLevel {
        /** Excellent accuracy: better than expected for the problem size */
        EXCELLENT,
        
        /** Good accuracy: within expected bounds for well-conditioned problems */
        GOOD,
        
        /** Acceptable accuracy: slightly elevated but still reasonable */
        ACCEPTABLE,
        
        /** Warning: concerning error levels, verify conditioning */
        WARNING,
        
        /** Poor accuracy: likely numerical instability or ill-conditioning */
        POOR,
        
        /** Critical failure: severe accuracy loss, results unreliable */
        CRITICAL
    }
    
    /**
     * Result of accuracy validation containing both norm-based and element-wise metrics
     */
    public static class ValidationResult {
        /** Accuracy level based on Frobenius norm residual */
        public final AccuracyLevel normLevel;
        
        /** Accuracy level based on element-wise maximum relative error */
        public final AccuracyLevel elementLevel;
        
        /** Relative Frobenius norm residual: ||A - Ahat||_F / ||A||_F */
        public final double normResidual;
        
        /** Maximum element-wise relative error */
        public final double elementResidual;
        
        /** Diagnostic message with interpretation */
        public final String message;
        
        /** Whether this result passes validation (not CRITICAL) */
        public final boolean passes;
        
        /** Whether to emit a warning (WARNING or worse) */
        public final boolean shouldWarn;
        
        /**
         * Create a validation result instance.
         *
         * @param normLevel accuracy level based on norm residual
         * @param elementLevel accuracy level based on element residual
         * @param normResidual Frobenius norm residual
         * @param elementResidual element-wise residual
         * @param message summary message
         */
        public ValidationResult(AccuracyLevel normLevel, AccuracyLevel elementLevel,
                               double normResidual, double elementResidual, String message) {
            this.normLevel = normLevel;
            this.elementLevel = elementLevel;
            this.normResidual = normResidual;
            this.elementResidual = elementResidual;
            this.message = message;
            this.passes = (normLevel != AccuracyLevel.CRITICAL && 
                          elementLevel != AccuracyLevel.CRITICAL);
            this.shouldWarn = (normLevel.ordinal() >= AccuracyLevel.WARNING.ordinal() ||
                              elementLevel.ordinal() >= AccuracyLevel.WARNING.ordinal());
        }
        
        /**
         * Get the worst-case accuracy level between norm and element metrics.
         *
         * @return overall accuracy level
         */
        public AccuracyLevel getOverallLevel() {
            return (normLevel.ordinal() > elementLevel.ordinal()) ? normLevel : elementLevel;
        }
        
        /**
         * @return diagnostic message
         */
        @Override
        public String toString() {
            return message;
        }
    }
    
    /**
     * Validate accuracy of matrix reconstruction with default condition estimate
     * 
     * @param A Original matrix
     * @param Ahat Reconstructed/computed matrix
     * @param decompositionType Description of the operation (e.g., "QR Decomposition")
     * @return Validation result with accuracy classification
     */
    public static ValidationResult validate(Matrix A, Matrix Ahat, String decompositionType) {
        return validate(A, Ahat, decompositionType, 1.0);
    }
    
    /**
     * Comprehensive accuracy validation with adaptive thresholds
     * 
     * @param A Original matrix
     * @param Ahat Reconstructed/computed matrix  
     * @param decompositionType Description of the operation (e.g., "QR Decomposition")
     * @param conditionEstimate Estimated condition number (use 1.0 if unknown)
     * @return Validation result with accuracy classification and diagnostics
     */
    public static ValidationResult validate(Matrix A, Matrix Ahat, 
                                           String decompositionType,
                                           double conditionEstimate) {
        int n = Math.max(A.getRowCount(), A.getColumnCount());
        double[] normThresholds = getNormThresholds(n, conditionEstimate);
        double[] elemThresholds = getElementThresholds(n, conditionEstimate);
        
        // Compute residuals
        double normRes = computeNormResidual(A, Ahat);
        double elemRes = computeElementResidual(A, Ahat);
        
        // Classify accuracy
        AccuracyLevel normLevel = classifyError(normRes, normThresholds);
        AccuracyLevel elemLevel = classifyError(elemRes, elemThresholds);
        
        // Generate diagnostic message
        String message = generateMessage(normLevel, elemLevel, normRes, elemRes, 
                                        decompositionType, n, conditionEstimate);
        
        return new ValidationResult(normLevel, elemLevel, normRes, elemRes, message);
    }
    
    /**
     * Quick validation that throws exception on critical failure
     * 
     * @param A Original matrix
     * @param Ahat Reconstructed matrix
     * @param decompositionType Description of the operation
     * @throws IllegalStateException if accuracy is CRITICAL
     */
    public static void validateOrThrow(Matrix A, Matrix Ahat, String decompositionType) {
        ValidationResult result = validate(A, Ahat, decompositionType);
        if (!result.passes) {
            throw new IllegalStateException("CRITICAL accuracy failure in " + decompositionType + 
                                          "\n" + result.message);
        }
    }
    
    /**
     * Compute adaptive error thresholds based on matrix size and conditioning
     * 
     * @param n Matrix dimension (use max of rows/cols for rectangular)
     * @param conditionNumber Condition number estimate
     * @return Array of thresholds: [excellent, good, acceptable, warning, poor]
     */
    private static double[] getNormThresholds(int n, double conditionNumber) {
        // Base error expectation: O(sqrt(n) * epsilon)
        double base = Math.sqrt(n) * EPS;
        
        // Condition factor: log scale to avoid overly permissive thresholds
        double condFactor = Math.max(1.0, Math.log10(Math.max(conditionNumber, 1.0)));
        
        return new double[] {
            base * 10,                        // Excellent: ~sqrt(n) * 10 * eps
            base * 100 * condFactor,          // Good: ~sqrt(n) * 100 * eps * log(cond)
            base * 1000 * condFactor,         // Acceptable
            base * 10000 * condFactor,        // Warning
            base * 100000 * condFactor        // Poor (above this is critical)
        };
    }

    /**
     * Compute element-wise thresholds based on matrix size and conditioning.
     *
     * @param n matrix dimension
     * @param conditionNumber condition estimate
     * @return thresholds array
     */
    private static double[] getElementThresholds(int n, double conditionNumber) {
        // Element-wise max error grows faster with size than norm-based residuals.
        double base = n * EPS;
        double condFactor = Math.max(1.0, Math.log10(Math.max(conditionNumber, 1.0)));

        return new double[] {
            base * 10,
            base * 100 * condFactor,
            base * 1000 * condFactor,
            base * 10000 * condFactor,
            base * 100000 * condFactor
        };
    }
    
    /**
     * Classify error magnitude into accuracy level
     */
    /**
     * Classify an error magnitude against thresholds.
     *
     * @param error measured error
     * @param thresholds threshold array
     * @return accuracy level
     */
    private static AccuracyLevel classifyError(double error, double[] thresholds) {
        if (error <= thresholds[0]) return AccuracyLevel.EXCELLENT;
        if (error <= thresholds[1]) return AccuracyLevel.GOOD;
        if (error <= thresholds[2]) return AccuracyLevel.ACCEPTABLE;
        if (error <= thresholds[3]) return AccuracyLevel.WARNING;
        if (error <= thresholds[4]) return AccuracyLevel.POOR;
        return AccuracyLevel.CRITICAL;
    }
    
    /**
     * Compute relative Frobenius norm residual: ||A - Ahat||_F / ||A||_F
     * 
     * @param A Original matrix
     * @param Ahat Reconstructed matrix
     * @return Relative residual (0.0 for zero matrices)
     */
    private static double computeNormResidual(Matrix A, Matrix Ahat) {
        double normA = A.frobeniusNorm();
        if (normA < EPS) return 0.0;
        return A.subtract(Ahat).frobeniusNorm() / normA;
    }
    
    /**
     * Compute maximum element-wise relative error: max_ij |A_ij - Ahat_ij| / scale_ij
     * where scale_ij = max(|A_ij|, |Ahat_ij|) for a conservative, scale-invariant denominator
     * 
     * @param A Original matrix
     * @param Ahat Reconstructed matrix
     * @return Maximum relative error across all elements
     */
    private static double computeElementResidual(Matrix A, Matrix Ahat) {
        int m = A.getRowCount();
        int n = A.getColumnCount();
        double maxError = 0.0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double a = A.get(i, j);
                double ahat = Ahat.get(i, j);
                double error = Math.abs(a - ahat);
                
                // Use max absolute value for scale (avoids overly harsh relative error)
                double scale = Math.max(Math.abs(a), Math.abs(ahat));
                if (scale < EPS) {
                    // Both near zero: use absolute error instead of relative
                    maxError = Math.max(maxError, error);
                } else {
                    double relError = error / scale;
                    maxError = Math.max(maxError, relError);
                }
            }
        }
        return maxError;
    }
    
    /**
     * Generate comprehensive diagnostic message
     */
    /**
     * Build a detailed validation message for the results.
     *
     * @param normLevel norm-based accuracy level
     * @param elemLevel element-based accuracy level
     * @param normRes norm residual
     * @param elemRes element residual
     * @param decompositionType label for the operation
     * @param n matrix dimension
     * @param condition condition estimate
     * @return formatted message
     */
    private static String generateMessage(AccuracyLevel normLevel, 
                                         AccuracyLevel elemLevel,
                                         double normRes, 
                                         double elemRes,
                                         String decompositionType,
                                         int n,
                                         double condition) {
        StringBuilder msg = new StringBuilder();
        msg.append(String.format("%s [%dx%d, cond~%.2e]:\n", 
                                decompositionType, n, n, condition));
        msg.append(String.format("  Norm residual: %.2e (%s)\n", normRes, normLevel));
        msg.append(String.format("  Elem residual: %.2e (%s)\n", elemRes, elemLevel));
        
        // Specific diagnostics based on error pattern
        if (normLevel == AccuracyLevel.EXCELLENT && elemLevel == AccuracyLevel.EXCELLENT) {
            msg.append("  ✓ Excellent numerical accuracy");
        } 
        else if (normLevel == AccuracyLevel.CRITICAL || elemLevel == AccuracyLevel.CRITICAL) {
            msg.append("  ✗ CRITICAL: Severe accuracy loss - results unreliable!");
            msg.append("\n  → Check for singular/near-singular matrices");
            msg.append("\n  → Consider higher precision or regularization");
        }
        else if (normLevel.ordinal() >= AccuracyLevel.POOR.ordinal() || 
                 elemLevel.ordinal() >= AccuracyLevel.POOR.ordinal()) {
            msg.append("  ✗ POOR accuracy - possible ill-conditioning or instability");
            msg.append("\n  → Check condition number and matrix properties");
        }
        else if (normLevel.ordinal() <= AccuracyLevel.ACCEPTABLE.ordinal() && 
                 elemLevel.ordinal() >= AccuracyLevel.WARNING.ordinal()) {
            msg.append("  ⚠ Localized errors: few elements have elevated error");
            msg.append("\n  → Global accuracy good, but some entries are problematic");
        }
        else if (normLevel.ordinal() >= AccuracyLevel.WARNING.ordinal() && 
                 elemLevel.ordinal() <= AccuracyLevel.ACCEPTABLE.ordinal()) {
            msg.append("  ⚠ Global error accumulation detected");
            msg.append("\n  → Check algorithm stability and error propagation");
        }
        else if (normLevel.ordinal() >= AccuracyLevel.WARNING.ordinal() || 
                 elemLevel.ordinal() >= AccuracyLevel.WARNING.ordinal()) {
            msg.append("  ⚠ WARNING: Elevated errors - verify matrix conditioning");
        }
        else if (normLevel.ordinal() <= AccuracyLevel.ACCEPTABLE.ordinal() && 
                 elemLevel.ordinal() <= AccuracyLevel.ACCEPTABLE.ordinal()) {
            msg.append("  ✓ Acceptable accuracy for this problem");
        }
        else {
            msg.append("  → Mixed accuracy levels detected");
        }
        
        return msg.toString();
    }
    
    /**
     * Validate orthogonality of a matrix Q: ||Q^T*Q - I||_F
     * 
     * @param Q Matrix to check for orthogonality
     * @param context Description for error messages
     * @return Validation result
     */
    public static ValidationResult validateOrthogonality(Matrix Q, String context) {
        int n = Q.getColumnCount();
        Matrix QtQ = Q.transpose().multiply(Q);
        Matrix I = Matrix.Identity(n);
        
        return validate(I, QtQ, context + " orthogonality", 1.0);
    }
    
    /**
     * Estimate condition number using simple max/min diagonal ratio
     * (rough estimate, not precise)
     * 
     * @param A Matrix to estimate
     * @return Rough condition number estimate
     */
    public static double estimateCondition(Matrix A) {
        int n = Math.min(A.getRowCount(), A.getColumnCount());
        double maxDiag = 0.0;
        double minDiag = Double.MAX_VALUE;
        
        for (int i = 0; i < n; i++) {
            double val = Math.abs(A.get(i, i));
            if (val > EPS) {  // Avoid near-zero diagonals
                maxDiag = Math.max(maxDiag, val);
                minDiag = Math.min(minDiag, val);
            }
        }
        
        if (minDiag < EPS || minDiag == Double.MAX_VALUE) {
            return 1e16;  // Nearly singular
        }
        
        return maxDiag / minDiag;
    }
}
