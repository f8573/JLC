package net.faulj.core;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import net.faulj.condition.ReciprocalCondition;
import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.decomposition.result.CholeskyResult;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.decomposition.result.LUResult;
import net.faulj.decomposition.result.PolarResult;
import net.faulj.decomposition.result.QRResult;
import net.faulj.decomposition.result.SVDResult;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.Diagonalization;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.inverse.LUInverse;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixAccuracyValidator;
import net.faulj.matrix.MatrixUtils;
import net.faulj.polar.PolarDecomposition;
import net.faulj.scalar.Complex;
import net.faulj.spaces.SubspaceBasis;
import net.faulj.svd.RankEstimation;
import net.faulj.svd.SVDecomposition;
import net.faulj.symmetric.QuadraticForm;
import net.faulj.symmetric.SpectralDecomposition;
import net.faulj.symmetric.SymmetricEigenDecomposition;
import net.faulj.vector.Vector;

/**
 * Collects and reports diagnostic metrics for algorithm performance and numerical behavior.
 * <p>
 * This class also provides a comprehensive matrix diagnostic workflow that computes
 * available decompositions and fundamental properties, with validation, warnings,
 * and errors when results are unreliable or impossible.
 * </p>
 */
public class DiagnosticMetrics {

    public enum Status {
        OK,
        WARNING,
        ERROR
    }

    public static final class DiagnosticItem<T> {
        private final String name;
        private final Status status;
        private final String message;
        private final T value;
        private final MatrixAccuracyValidator.ValidationResult validation;

        /**
         * Create a diagnostic item.
         *
         * @param name metric name
         * @param status status flag
         * @param message message or context
         * @param value metric value
         * @param validation validation result
         */
        public DiagnosticItem(String name,
                              Status status,
                              String message,
                              T value,
                              MatrixAccuracyValidator.ValidationResult validation) {
            this.name = name;
            this.status = status;
            this.message = message;
            this.value = value;
            this.validation = validation;
        }

        /**
         * @return metric name
         */
        public String getName() {
            return name;
        }

        /**
         * @return status flag
         */
        public Status getStatus() {
            return status;
        }

        /**
         * @return message string
         */
        public String getMessage() {
            return message;
        }

        /**
         * @return metric value
         */
        public T getValue() {
            return value;
        }

        /**
         * @return validation details
         */
        public MatrixAccuracyValidator.ValidationResult getValidation() {
            return validation;
        }
    }

    public static final class MatrixDiagnostics {
        private Matrix matrix;
        private int rows;
        private int cols;
        private int columns;
        private boolean square;
        private boolean real;
        private boolean symmetric;
        private double symmetryError;
        private double norm1;
        private double normInf;
        private double frobeniusNorm;
        private String domain;
        private double density;
        private double[][] matrixData;
        private double[][] matrixImagData;
        private Double trace;
        private Double determinant;
        private Double conditionNumber;
        private Double reciprocalConditionNumber;
        private Integer rank;
        private Integer nullity;
        private Boolean invertible;
        private Boolean singular;
        private Boolean fullRank;
        private Boolean rankDeficient;
        private Boolean leftInvertible;
        private Boolean rightInvertible;
        private Boolean wellConditioned;
        private Boolean illConditioned;
        private Boolean nearlySingular;
        private Complex[] eigenvalues;
        private Matrix eigenvectors;
        private Matrix eigenspace;
        private int[] algebraicMultiplicity;
        private int[] geometricMultiplicity;
        private Complex[] characteristicPolynomial;
        private double[] singularValues;
        private Double spectralRadius;
        private Double operatorNorm;
        private Boolean defective;
        private Boolean diagonalizable;
        private String definiteness;

        private Boolean orthogonal;
        private Boolean unitary;
        private Boolean orthonormal;
        private Boolean skewSymmetric;
        private Boolean diagonal;
        private Boolean bidiagonal;
        private Boolean tridiagonal;
        private Boolean upperTriangular;
        private Boolean lowerTriangular;
        private Boolean sparse;
        private Boolean zero;
        private Boolean identity;
        private Boolean scalar;
        private Boolean antidiagonal;
        private Boolean hermitian;
        private Boolean persymmetric;
        private Boolean rotation;
        private Boolean reflection;
        private Boolean normal;
        private Boolean nonNormal;
        private Boolean spectral;
        private Boolean nilpotent;
        private Boolean involutory;
        private Boolean idempotent;
        private Boolean companion;
        private Boolean block;
        private Boolean hessenberg;
        private Boolean lowerHessenberg;
        private Boolean upperHessenberg;
        private Boolean schur;
        private Boolean rowEchelon;
        private Boolean reducedRowEchelon;

        private Complex[] characteristicPolynomials;

        private DiagnosticItem<Matrix> rref;
        private DiagnosticItem<Matrix> inverse;
        private DiagnosticItem<Set<Vector>> rowSpaceBasis;
        private DiagnosticItem<Set<Vector>> columnSpaceBasis;
        private DiagnosticItem<Set<Vector>> nullSpaceBasis;
        private List<DiagnosticItem<Set<Vector>>> eigenbasisList;
        private List<DiagnosticItem<Set<Vector>>> eigenspaceList;

        private DiagnosticItem<QRResult> qr;
        private DiagnosticItem<LUResult> lu;
        private DiagnosticItem<CholeskyResult> cholesky;
        private DiagnosticItem<SVDResult> svd;
        private DiagnosticItem<HessenbergResult> hessenbergDecomposition;
        private DiagnosticItem<SchurResult> schurDecomposition;
        private DiagnosticItem<Diagonalization> diagonalization;
        private DiagnosticItem<SpectralDecomposition> symmetricSpectral;
        private DiagnosticItem<BidiagonalizationResult> bidiagonalization;
        private DiagnosticItem<PolarResult> polar;

        /**
         * @return source matrix
         */
        public Matrix getMatrix() {
            return matrix;
        }

        /**
         * @return row count
         */
        public int getRows() {
            return rows;
        }

        /**
         * @return column count
         */
        public int getCols() {
            return cols;
        }

        /**
         * @return column count (alias)
         */
        public int getColumns() {
            return columns;
        }

        /**
         * @return true if matrix is square
         */
        public boolean isSquare() {
            return square;
        }

        /**
         * @return true if matrix is real-valued
         */
        public boolean isReal() {
            return real;
        }

        /**
         * @return true if matrix is symmetric
         */
        public boolean isSymmetric() {
            return symmetric;
        }

        /**
         * @return symmetry error metric
         */
        public double getSymmetryError() {
            return symmetryError;
        }

        /**
         * @return 1-norm
         */
        public double getNorm1() {
            return norm1;
        }

        /**
         * @return infinity norm
         */
        public double getNormInf() {
            return normInf;
        }

        /**
         * @return Frobenius norm
         */
        public double getFrobeniusNorm() {
            return frobeniusNorm;
        }

        /**
         * @return domain label (e.g., R or C)
         */
        public String getDomain() {
            return domain;
        }

        /**
         * @return density estimate
         */
        public double getDensity() {
            return density;
        }

        /**
         * @return real matrix data
         */
        public double[][] getMatrixData() {
            return matrixData;
        }

        /**
         * @return imaginary matrix data
         */
        public double[][] getMatrixImagData() {
            return matrixImagData;
        }

        /**
         * @return trace
         */
        public Double getTrace() {
            return trace;
        }

        /**
         * @return determinant
         */
        public Double getDeterminant() {
            return determinant;
        }

        /**
         * @return condition number estimate
         */
        public Double getConditionNumber() {
            return conditionNumber;
        }

        /**
         * @return reciprocal condition number
         */
        public Double getReciprocalConditionNumber() {
            return reciprocalConditionNumber;
        }

        /**
         * @return rank
         */
        public Integer getRank() {
            return rank;
        }

        /**
         * @return nullity
         */
        public Integer getNullity() {
            return nullity;
        }

        /**
         * @return invertibility flag
         */
        public Boolean getInvertible() {
            return invertible;
        }

        

        /**
         * @return singularity flag
         */
        public Boolean getSingular() {
            return singular;
        }

        /**
         * @return full-rank flag
         */
        public Boolean getFullRank() {
            return fullRank;
        }

        /**
         * @return rank-deficient flag
         */
        public Boolean getRankDeficient() {
            return rankDeficient;
        }

        /**
         * @return left-invertible flag
         */
        public Boolean getLeftInvertible() {
            return leftInvertible;
        }

        /**
         * @return right-invertible flag
         */
        public Boolean getRightInvertible() {
            return rightInvertible;
        }

        /**
         * @return well-conditioned flag
         */
        public Boolean getWellConditioned() {
            return wellConditioned;
        }

        /**
         * @return ill-conditioned flag
         */
        public Boolean getIllConditioned() {
            return illConditioned;
        }

        /**
         * @return nearly singular flag
         */
        public Boolean getNearlySingular() {
            return nearlySingular;
        }

        /**
         * @return eigenvalues
         */
        public Complex[] getEigenvalues() {
            return eigenvalues;
        }

        /**
         * @return eigenvector matrix
         */
        public Matrix getEigenvectors() {
            return eigenvectors;
        }

        /**
         * @return eigenspace basis matrix
         */
        public Matrix getEigenspace() {
            return eigenspace;
        }

        /**
         * @return algebraic multiplicities
         */
        public int[] getAlgebraicMultiplicity() {
            return algebraicMultiplicity;
        }

        /**
         * @return geometric multiplicities
         */
        public int[] getGeometricMultiplicity() {
            return geometricMultiplicity;
        }

        /**
         * @return characteristic polynomial coefficients
         */
        public Complex[] getCharacteristicPolynomial() {
            return characteristicPolynomial;
        }

        /**
         * @return singular values
         */
        public double[] getSingularValues() {
            return singularValues;
        }

        /**
         * @return spectral radius
         */
        public Double getSpectralRadius() {
            return spectralRadius;
        }

        /**
         * @return operator norm
         */
        public Double getOperatorNorm() {
            return operatorNorm;
        }

        /**
         * @return defective flag
         */
        public Boolean getDefective() {
            return defective;
        }

        /**
         * @return diagonalizable flag
         */
        public Boolean getDiagonalizable() {
            return diagonalizable;
        }

        /**
         * @return definiteness label
         */
        public String getDefiniteness() {
            return definiteness;
        }

        /**
         * @return orthogonal flag
         */
        public Boolean getOrthogonal() {
            return orthogonal;
        }

        /**
         * @return unitary flag
         */
        public Boolean getUnitary() {
            return unitary;
        }

        /**
         * @return orthonormal flag
         */
        public Boolean getOrthonormal() {
            return orthonormal;
        }

        /**
         * @return skew-symmetric flag
         */
        public Boolean getSkewSymmetric() {
            return skewSymmetric;
        }

        /**
         * @return diagonal flag
         */
        public Boolean getDiagonal() {
            return diagonal;
        }

        /**
         * @return bidiagonal flag
         */
        public Boolean getBidiagonal() {
            return bidiagonal;
        }

        /**
         * @return tridiagonal flag
         */
        public Boolean getTridiagonal() {
            return tridiagonal;
        }

        /**
         * @return upper-triangular flag
         */
        public Boolean getUpperTriangular() {
            return upperTriangular;
        }

        /**
         * @return lower-triangular flag
         */
        public Boolean getLowerTriangular() {
            return lowerTriangular;
        }

        /**
         * @return sparse flag
         */
        public Boolean getSparse() {
            return sparse;
        }

        /**
         * @return zero-matrix flag
         */
        public Boolean getZero() {
            return zero;
        }

        /**
         * @return identity flag
         */
        public Boolean getIdentity() {
            return identity;
        }

        /**
         * @return scalar-matrix flag
         */
        public Boolean getScalar() {
            return scalar;
        }

        /**
         * @return antidiagonal flag
         */
        public Boolean getAntidiagonal() {
            return antidiagonal;
        }

        /**
         * @return Hermitian flag
         */
        public Boolean getHermitian() {
            return hermitian;
        }

        /**
         * @return persymmetric flag
         */
        public Boolean getPersymmetric() {
            return persymmetric;
        }

        

        /**
         * @return rotation flag
         */
        public Boolean getRotation() {
            return rotation;
        }

        /**
         * @return reflection flag
         */
        public Boolean getReflection() {
            return reflection;
        }

        /**
         * @return normal flag
         */
        public Boolean getNormal() {
            return normal;
        }

        /**
         * @return non-normal flag
         */
        public Boolean getNonNormal() {
            return nonNormal;
        }

        /**
         * @return spectral flag
         */
        public Boolean getSpectral() {
            return spectral;
        }

        /**
         * @return nilpotent flag
         */
        public Boolean getNilpotent() {
            return nilpotent;
        }

        /**
         * @return involutory flag
         */
        public Boolean getInvolutory() {
            return involutory;
        }

        /**
         * @return idempotent flag
         */
        public Boolean getIdempotent() {
            return idempotent;
        }

        /**
         * @return companion flag
         */
        public Boolean getCompanion() {
            return companion;
        }

        /**
         * @return block-diagonal flag
         */
        public Boolean getBlock() {
            return block;
        }

        /**
         * @return Hessenberg flag
         */
        public Boolean getHessenberg() {
            return hessenberg;
        }

        /**
         * @return lower-Hessenberg flag
         */
        public Boolean getLowerHessenberg() {
            return lowerHessenberg;
        }

        /**
         * @return upper-Hessenberg flag
         */
        public Boolean getUpperHessenberg() {
            return upperHessenberg;
        }

        /**
         * @return Schur-form flag
         */
        public Boolean getSchur() {
            return schur;
        }

        /**
         * @return row-echelon flag
         */
        public Boolean getRowEchelon() {
            return rowEchelon;
        }

        /**
         * @return reduced row-echelon flag
         */
        public Boolean getReducedRowEchelon() {
            return reducedRowEchelon;
        }

        /**
         * @return characteristic polynomial coefficients
         */
        public Complex[] getCharacteristicPolynomials() {
            return characteristicPolynomials;
        }

        /**
         * @return RREF diagnostic item
         */
        public DiagnosticItem<Matrix> getRref() {
            return rref;
        }

        /**
         * @return inverse diagnostic item
         */
        public DiagnosticItem<Matrix> getInverse() {
            return inverse;
        }

        /**
         * @return row space basis diagnostic item
         */
        public DiagnosticItem<Set<Vector>> getRowSpaceBasis() {
            return rowSpaceBasis;
        }

        /**
         * @return row space basis diagnostic item (alias)
         */
        public DiagnosticItem<Set<Vector>> getRowSpace() {
            return rowSpaceBasis;
        }

        /**
         * @return column space basis diagnostic item
         */
        public DiagnosticItem<Set<Vector>> getColumnSpaceBasis() {
            return columnSpaceBasis;
        }

        /**
         * @return column space basis diagnostic item (alias)
         */
        public DiagnosticItem<Set<Vector>> getColumnSpace() {
            return columnSpaceBasis;
        }

        /**
         * @return null space basis diagnostic item
         */
        public DiagnosticItem<Set<Vector>> getNullSpaceBasis() {
            return nullSpaceBasis;
        }

        /**
         * @return eigenbasis diagnostic list
         */
        public List<DiagnosticItem<Set<Vector>>> getEigenbasisList() {
            return eigenbasisList;
        }

        /**
         * @return eigenspace diagnostic list
         */
        public List<DiagnosticItem<Set<Vector>>> getEigenspaceList() {
            return eigenspaceList;
        }

        /**
         * @return null space diagnostic item (alias)
         */
        public DiagnosticItem<Set<Vector>> getNullSpace() {
            return nullSpaceBasis;
        }

        /**
         * @return QR diagnostic item
         */
        public DiagnosticItem<QRResult> getQr() {
            return qr;
        }

        /**
         * @return LU diagnostic item
         */
        public DiagnosticItem<LUResult> getLu() {
            return lu;
        }

        /**
         * @return Cholesky diagnostic item
         */
        public DiagnosticItem<CholeskyResult> getCholesky() {
            return cholesky;
        }

        /**
         * @return SVD diagnostic item
         */
        public DiagnosticItem<SVDResult> getSvd() {
            return svd;
        }

        /**
         * @return Hessenberg diagnostic item
         */
        public DiagnosticItem<HessenbergResult> getHessenbergDecomposition() {
            return hessenbergDecomposition;
        }

        /**
         * @return Schur diagnostic item
         */
        public DiagnosticItem<SchurResult> getSchurDecomposition() {
            return schurDecomposition;
        }

        /**
         * @return diagonalization diagnostic item
         */
        public DiagnosticItem<Diagonalization> getDiagonalization() {
            return diagonalization;
        }

        /**
         * @return symmetric spectral diagnostic item
         */
        public DiagnosticItem<SpectralDecomposition> getSymmetricSpectral() {
            return symmetricSpectral;
        }

        /**
         * @return bidiagonalization diagnostic item
         */
        public DiagnosticItem<BidiagonalizationResult> getBidiagonalization() {
            return bidiagonalization;
        }

        /**
         * @return polar diagnostic item
         */
        public DiagnosticItem<PolarResult> getPolar() {
            return polar;
        }
    }

    /**
     * Compute diagnostics for a matrix.
     *
     * @param A matrix to analyze
     * @return diagnostics object
     */
    public static MatrixDiagnostics analyze(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }

        MatrixDiagnostics diag = new MatrixDiagnostics();
        diag.matrix = A;
        diag.rows = A.getRowCount();
        diag.cols = A.getColumnCount();
        diag.columns = diag.cols;
        diag.square = A.isSquare();
        diag.real = A.isReal();
        diag.domain = diag.real ? "R" : "C";
        diag.norm1 = A.norm1();
        diag.normInf = A.normInf();
        diag.frobeniusNorm = A.frobeniusNorm();
        diag.matrixData = toMatrixData(A);
        diag.matrixImagData = toMatrixImagData(A);

        double tol = Tolerance.get();
        diag.density = computeDensity(A, tol);
        diag.sparse = diag.density < 0.1;
        diag.zero = isZeroMatrix(A, tol);
        diag.identity = diag.square && isIdentity(A, tol);
        diag.scalar = diag.square && isScalar(A, tol);
        diag.diagonal = diag.square && isDiagonal(A, tol);
        diag.upperTriangular = diag.square && isUpperTriangular(A, tol);
        diag.lowerTriangular = diag.square && isLowerTriangular(A, tol);
        diag.bidiagonal = diag.square && isBidiagonal(A, tol);
        diag.tridiagonal = diag.square && isTridiagonal(A, tol);
        diag.antidiagonal = diag.square && isAntidiagonal(A, tol);
        diag.skewSymmetric = diag.square && diag.real && isSkewSymmetric(A, tol);
        diag.hermitian = diag.square && isHermitian(A, tol);
        diag.persymmetric = diag.square && isPersymmetric(A, tol);
        diag.lowerHessenberg = diag.square && isLowerHessenberg(A, tol);
        diag.upperHessenberg = diag.square && isUpperHessenberg(A, tol);
        diag.hessenberg = Boolean.TRUE.equals(diag.lowerHessenberg) || Boolean.TRUE.equals(diag.upperHessenberg);
        diag.schur = diag.upperHessenberg;
        if (diag.square) {
            SymmetryResult symmetry = computeSymmetry(A, tol);
            diag.symmetric = symmetry.symmetric;
            diag.symmetryError = symmetry.error;
        } else {
            diag.symmetric = false;
            diag.symmetryError = Double.NaN;
        }

        if (diag.square && diag.real) {
            diag.trace = A.trace();
            diag.determinant = A.determinant();
        }

        if (diag.real) {
            Matrix rrefMatrix = A.copy();
            MatrixUtils.toReducedRowEchelonForm(rrefMatrix);
            diag.rref = new DiagnosticItem<>("rref", Status.OK, "Computed reduced row echelon form",
                    rrefMatrix, null);
        } else {
            diag.rref = errorItem("rref", "Reduced row echelon form requires a real-valued matrix", null);
        }

        if (diag.real) {
            diag.rowSpaceBasis = new DiagnosticItem<>("rowSpaceBasis", Status.OK, null,
                    SubspaceBasis.rowSpaceBasis(A), null);
            diag.columnSpaceBasis = new DiagnosticItem<>("columnSpaceBasis", Status.OK, null,
                    SubspaceBasis.columnSpaceBasis(A), null);
            diag.nullSpaceBasis = new DiagnosticItem<>("nullSpaceBasis", Status.OK, null,
                    SubspaceBasis.nullSpaceBasis(A), null);
        } else {
            diag.rowSpaceBasis = errorItem("rowSpaceBasis", "Row space basis requires a real-valued matrix", null);
            diag.columnSpaceBasis = errorItem("columnSpaceBasis", "Column space basis requires a real-valued matrix", null);
            diag.nullSpaceBasis = errorItem("nullSpaceBasis", "Null space basis requires a real-valued matrix", null);
        }

        SVDResult svdResult = null;
        if (diag.real) {
            try {
                svdResult = new SVDecomposition().decompose(A);
                double cond = safeConditionEstimate(svdResult.getConditionNumber());
                MatrixAccuracyValidator.ValidationResult validation =
                        MatrixAccuracyValidator.validate(A, svdResult.reconstruct(), "SVD", cond);
                diag.svd = itemFromValidation("svd", svdResult, validation, null);
                diag.singularValues = svdResult.getSingularValues();
                diag.rank = RankEstimation.effectiveRank(svdResult.getSingularValues(), diag.rows, diag.cols);
                diag.nullity = diag.cols - diag.rank;
                diag.conditionNumber = cond;
            } catch (RuntimeException ex) {
                diag.svd = errorItem("svd", "SVD failed", ex);
            }
        } else {
            diag.svd = errorItem("svd", "SVD requires a real-valued matrix", null);
        }

        if (diag.rank == null && diag.rref != null && diag.rref.getStatus() == Status.OK) {
            diag.rank = rankFromRref(diag.rref.getValue(), tol);
            diag.nullity = diag.cols - diag.rank;
        }

        // If SVD gave a higher rank but RREF (with global tolerance) suggests a smaller
        // rank (e.g., obvious exact linear dependence in small integer matrices), prefer
        // the RREF result which uses the global Tolerance. This helps detect simple
        // singular cases that SVD's machine-epsilon-based threshold might miss.
        if (diag.rank != null && diag.rref != null && diag.rref.getStatus() == Status.OK) {
            int rrefRank = rankFromRref(diag.rref.getValue(), tol);
            if (rrefRank < diag.rank) {
                diag.rank = rrefRank;
                diag.nullity = diag.cols - diag.rank;
            }
        }

        if (diag.square && diag.real && diag.conditionNumber == null) {
            diag.conditionNumber = MatrixAccuracyValidator.estimateCondition(A);
        }

        double conditionEstimate = diag.conditionNumber != null ? diag.conditionNumber : 1.0;

        if (diag.real) {
            try {
                QRResult qr = HouseholderQR.decompose(A);
                MatrixAccuracyValidator.ValidationResult validation =
                        MatrixAccuracyValidator.validate(A, qr.reconstruct(), "QR", safeConditionEstimate(conditionEstimate));
                MatrixAccuracyValidator.ValidationResult orth =
                        MatrixAccuracyValidator.validateOrthogonality(qr.getQ(), "QR");
                diag.qr = itemFromValidation("qr", qr, validation, orthogonalityMessage(orth));
                diag.qr = elevateIfNeeded(diag.qr, orth);
            } catch (RuntimeException ex) {
                diag.qr = errorItem("qr", "QR decomposition failed", ex);
            }
        } else {
            diag.qr = errorItem("qr", "QR decomposition requires a real-valued matrix", null);
        }

        if (diag.square && diag.real) {
            try {
                LUResult lu = new LUDecomposition().decompose(A);
                Matrix pa = lu.getP().asMatrix().multiply(A);
                MatrixAccuracyValidator.ValidationResult validation =
                        MatrixAccuracyValidator.validate(pa, lu.reconstruct(), "LU", safeConditionEstimate(conditionEstimate));
                Status status = statusFromValidation(validation);
                String message = validation == null ? null : validation.message;
                if (lu.isSingular()) {
                    status = worse(status, Status.WARNING);
                    message = joinMessages(message, "Matrix is singular or nearly singular");
                }
                diag.lu = new DiagnosticItem<>("lu", status, message, lu, validation);
                diag.determinant = lu.getDeterminant();
                if (!lu.isSingular()) {
                    diag.reciprocalConditionNumber = ReciprocalCondition.estimateFromLU(lu);
                } else {
                    diag.reciprocalConditionNumber = 0.0;
                }
            } catch (RuntimeException ex) {
                diag.lu = errorItem("lu", "LU decomposition failed", ex);
            }
        } else {
            diag.lu = errorItem("lu", "LU decomposition requires a real square matrix", null);
        }

        if (diag.square && diag.real && diag.symmetric) {
            try {
                CholeskyResult chol = new CholeskyDecomposition().decompose(A);
                MatrixAccuracyValidator.ValidationResult validation =
                        MatrixAccuracyValidator.validate(A, chol.reconstruct(), "Cholesky", safeConditionEstimate(conditionEstimate));
                diag.cholesky = itemFromValidation("cholesky", chol, validation, null);
            } catch (RuntimeException ex) {
                diag.cholesky = errorItem("cholesky", "Cholesky decomposition failed", ex);
            }
        } else if (!diag.square || !diag.real) {
            diag.cholesky = errorItem("cholesky", "Cholesky decomposition requires a real square matrix", null);
        } else {
            diag.cholesky = errorItem("cholesky", "Cholesky decomposition requires a symmetric positive definite matrix", null);
        }

        if (diag.square && diag.real) {
            try {
                HessenbergResult hess = HessenbergReduction.decompose(A);
                MatrixAccuracyValidator.ValidationResult validation =
                        MatrixAccuracyValidator.validate(A, hess.reconstruct(), "Hessenberg", safeConditionEstimate(conditionEstimate));
                MatrixAccuracyValidator.ValidationResult orth =
                        MatrixAccuracyValidator.validateOrthogonality(hess.getQ(), "Hessenberg");
                diag.hessenbergDecomposition = itemFromValidation("hessenberg", hess, validation, orthogonalityMessage(orth));
                diag.hessenbergDecomposition = elevateIfNeeded(diag.hessenbergDecomposition, orth);
            } catch (RuntimeException ex) {
                diag.hessenbergDecomposition = errorItem("hessenberg", "Hessenberg reduction failed", ex);
            }
        } else {
            diag.hessenbergDecomposition = errorItem("hessenberg", "Hessenberg reduction requires a real square matrix", null);
        }

        if (diag.square && diag.real) {
            try {
                SchurResult schur = RealSchurDecomposition.decompose(A);
                Matrix reconstructed = schur.getU().multiply(schur.getT()).multiply(schur.getU().transpose());
                MatrixAccuracyValidator.ValidationResult validation =
                        MatrixAccuracyValidator.validate(A, reconstructed, "Schur", safeConditionEstimate(conditionEstimate));
                MatrixAccuracyValidator.ValidationResult orth =
                        MatrixAccuracyValidator.validateOrthogonality(schur.getU(), "Schur");
                diag.schurDecomposition = itemFromValidation("schur", schur, validation, orthogonalityMessage(orth));
                diag.schurDecomposition = elevateIfNeeded(diag.schurDecomposition, orth);
                diag.eigenvalues = schur.getEigenvalues();
                if (schur.getEigenvectors() != null) {
                    diag.eigenvectors = schur.getEigenvectors();
                }
            } catch (RuntimeException ex) {
                diag.schurDecomposition = errorItem("schur", "Schur decomposition failed", ex);
            }
        } else {
            diag.schurDecomposition = errorItem("schur", "Schur decomposition requires a real square matrix", null);
        }

        if (diag.square && diag.real) {
            try {
                Diagonalization diagResult = Diagonalization.decompose(A);
                Matrix P = diagResult.getP();
                Matrix D = diagResult.getD();
                Matrix pinv;
                try {
                    pinv = P.inverse();
                } catch (RuntimeException ex) {
                    diag.diagonalization = errorItem("diagonalization",
                            "Eigenvector matrix is singular; matrix is not diagonalizable", ex);
                    pinv = null;
                }
                if (pinv != null) {
                    Matrix reconstructed = P.multiply(D).multiply(pinv);
                    MatrixAccuracyValidator.ValidationResult validation =
                            MatrixAccuracyValidator.validate(A, reconstructed, "Diagonalization",
                                    safeConditionEstimate(conditionEstimate));
                    diag.diagonalization = itemFromValidation("diagonalization", diagResult, validation, null);
                    diag.eigenvalues = diagResult.getEigenvalues();
                    diag.eigenvectors = P;
                }
            } catch (RuntimeException ex) {
                diag.diagonalization = errorItem("diagonalization", "Diagonalization failed", ex);
            }
        } else {
            diag.diagonalization = errorItem("diagonalization", "Diagonalization requires a real square matrix", null);
        }

        if (diag.square && diag.real && diag.symmetric) {
            try {
                SpectralDecomposition spectral = SymmetricEigenDecomposition.decompose(A);
                MatrixAccuracyValidator.ValidationResult validation =
                        MatrixAccuracyValidator.validate(A, spectral.reconstruct(), "Symmetric eigen",
                                safeConditionEstimate(conditionEstimate));
                diag.symmetricSpectral = itemFromValidation("symmetricEigen", spectral, validation, null);
                diag.eigenvectors = spectral.getEigenvectors();
                diag.eigenvalues = toComplex(spectral.getEigenvalues());
            } catch (RuntimeException ex) {
                diag.symmetricSpectral = errorItem("symmetricEigen", "Symmetric eigen decomposition failed", ex);
            }
        } else if (!diag.square || !diag.real) {
            diag.symmetricSpectral = errorItem("symmetricEigen", "Symmetric eigen decomposition requires a real square matrix", null);
        } else {
            diag.symmetricSpectral = errorItem("symmetricEigen", "Matrix is not symmetric", null);
        }

        if (diag.real) {
            try {
                BidiagonalizationResult bidiag = new Bidiagonalization().decompose(A);
                MatrixAccuracyValidator.ValidationResult validation =
                        MatrixAccuracyValidator.validate(A, bidiag.reconstruct(), "Bidiagonalization",
                                safeConditionEstimate(conditionEstimate));
                MatrixAccuracyValidator.ValidationResult orthU =
                        MatrixAccuracyValidator.validateOrthogonality(bidiag.getU(), "Bidiagonal U");
                MatrixAccuracyValidator.ValidationResult orthV =
                        MatrixAccuracyValidator.validateOrthogonality(bidiag.getV(), "Bidiagonal V");
                String extra = joinMessages(orthogonalityMessage(orthU), orthogonalityMessage(orthV));
                diag.bidiagonalization = itemFromValidation("bidiagonalization", bidiag, validation, extra);
                diag.bidiagonalization = elevateIfNeeded(diag.bidiagonalization, orthU);
                diag.bidiagonalization = elevateIfNeeded(diag.bidiagonalization, orthV);
            } catch (RuntimeException ex) {
                diag.bidiagonalization = errorItem("bidiagonalization", "Bidiagonalization failed", ex);
            }
        } else {
            diag.bidiagonalization = errorItem("bidiagonalization", "Bidiagonalization requires a real-valued matrix", null);
        }

        if (diag.real) {
            try {
                PolarResult polar = PolarDecomposition.decompose(A);
                MatrixAccuracyValidator.ValidationResult validation =
                        MatrixAccuracyValidator.validate(A, polar.reconstruct(), "Polar", safeConditionEstimate(conditionEstimate));
                MatrixAccuracyValidator.ValidationResult orth =
                        MatrixAccuracyValidator.validateOrthogonality(polar.getU(), "Polar U");
                diag.polar = itemFromValidation("polar", polar, validation, orthogonalityMessage(orth));
                diag.polar = elevateIfNeeded(diag.polar, orth);
            } catch (RuntimeException ex) {
                diag.polar = errorItem("polar", "Polar decomposition failed", ex);
            }
        } else {
            diag.polar = errorItem("polar", "Polar decomposition requires a real-valued matrix", null);
        }

        if (diag.square && diag.real) {
            try {
                Matrix inverse = LUInverse.compute(A);
                Matrix identity = Matrix.Identity(diag.rows);
                MatrixAccuracyValidator.ValidationResult left =
                        MatrixAccuracyValidator.validate(identity, A.multiply(inverse), "Inverse A*inv",
                                safeConditionEstimate(conditionEstimate));
                MatrixAccuracyValidator.ValidationResult right =
                        MatrixAccuracyValidator.validate(identity, inverse.multiply(A), "Inverse inv*A",
                                safeConditionEstimate(conditionEstimate));
                Status status = worse(statusFromValidation(left), statusFromValidation(right));
                String message = joinMessages(left == null ? null : left.message,
                        right == null ? null : right.message);
                diag.inverse = new DiagnosticItem<>("inverse", status, message, inverse, left);
            } catch (RuntimeException ex) {
                diag.inverse = errorItem("inverse", "Matrix inversion failed", ex);
            }
        } else {
            diag.inverse = errorItem("inverse", "Inverse requires a real square matrix", null);
        }

        if (diag.eigenvalues == null && diag.symmetricSpectral != null
                && diag.symmetricSpectral.getStatus() == Status.OK) {
            diag.eigenvalues = toComplex(diag.symmetricSpectral.getValue().getEigenvalues());
        }

        if (diag.eigenvectors == null && diag.symmetricSpectral != null
                && diag.symmetricSpectral.getStatus() == Status.OK) {
            diag.eigenvectors = diag.symmetricSpectral.getValue().getEigenvectors();
        }

        if (diag.real) {
            diag.rowEchelon = isRowEchelon(A, tol);
            diag.reducedRowEchelon = isReducedRowEchelon(A, tol);
        }

        int minDim = Math.min(diag.rows, diag.cols);
        if (diag.rank != null) {
            diag.fullRank = diag.rank == minDim;
            diag.rankDeficient = diag.rank < minDim;
            diag.leftInvertible = diag.rows >= diag.cols && diag.rank == diag.cols;
            diag.rightInvertible = diag.cols >= diag.rows && diag.rank == diag.rows;
        }

        if (diag.square) {
            if (diag.rank != null) {
                diag.invertible = diag.rank == diag.rows;
            } else if (diag.determinant != null) {
                diag.invertible = !Tolerance.isZero(diag.determinant);
            }
            if (diag.invertible != null) {
                diag.singular = !diag.invertible;
            }
        }

        if (diag.conditionNumber != null) {
            double cond = diag.conditionNumber;
            diag.wellConditioned = cond <= 1e6;
            diag.illConditioned = cond >= 1e12;
            diag.nearlySingular = diag.illConditioned;
        }

        if (diag.reciprocalConditionNumber != null) {
            double rcond = diag.reciprocalConditionNumber;
            if (Double.isFinite(rcond)) {
                if (diag.nearlySingular == null) {
                    diag.nearlySingular = rcond <= 1e-12;
                } else {
                    diag.nearlySingular = diag.nearlySingular || rcond <= 1e-12;
                }
            }
        }

        if (diag.singularValues != null && diag.singularValues.length > 0) {
            diag.operatorNorm = max(diag.singularValues);
        }

        if (diag.eigenvalues != null && diag.eigenvalues.length > 0) {
            diag.spectralRadius = maxAbs(diag.eigenvalues);
            MultiplicityResult mult = computeMultiplicities(A, diag.eigenvalues, tol);
            diag.algebraicMultiplicity = mult.algebraicMultiplicity;
            diag.geometricMultiplicity = mult.geometricMultiplicity;
            diag.characteristicPolynomial = characteristicPolynomial(diag.eigenvalues);
            diag.characteristicPolynomials = diag.characteristicPolynomial;
        }

        // If sum of geometric multiplicities equals n, then an eigenbasis exists and the matrix is diagonalizable
        if (diag.geometricMultiplicity != null) {
            int sumGeom = 0;
            for (int gm : diag.geometricMultiplicity) {
                sumGeom += gm;
            }
            if (diag.rows > 0) {
                diag.diagonalizable = sumGeom == diag.rows;
            }
        }

        // Eigenspace matrix (convenience): set to eigenvectors matrix only when a full
        // eigenbasis exists (i.e., the matrix is diagonalizable). If the matrix is
        // defective there is no global eigenspace matrix — do not expose `eigenspace`.
        if (diag.eigenvectors != null && diag.diagonalizable != null && diag.diagonalizable) {
            diag.eigenspace = diag.eigenvectors;
        } else {
            diag.eigenspace = null;
        }

        // Per-eigenvalue eigenspace/eigenbasis
        diag.eigenspaceList = new ArrayList<>();
        diag.eigenbasisList = new ArrayList<>();
        if (diag.eigenvalues != null && diag.eigenvalues.length > 0) {
            int nEv = diag.eigenvalues.length;
            boolean[] visited = new boolean[nEv];
            for (int i = 0; i < nEv; i++) {
                if (visited[i]) continue;
                net.faulj.scalar.Complex lambda = diag.eigenvalues[i];
                int count = 0;
                for (int j = i; j < nEv; j++) {
                    if (!visited[j] && close(lambda, diag.eigenvalues[j], tol)) {
                        visited[j] = true;
                        count++;
                    }
                }

                // Eigenspace: kernel of (A - lambda I). Use a single representative vector for display.
                if (diag.real && isRealEigenvalue(lambda, tol)) {
                    Matrix shifted = A.copy();
                    int m = shifted.getRowCount();
                    for (int r = 0; r < m; r++) {
                        shifted.set(r, r, shifted.get(r, r) - lambda.real);
                    }
                    try {
                        Set<Vector> basis = SubspaceBasis.nullSpaceBasis(shifted);
                        if (basis == null || basis.isEmpty()) {
                            diag.eigenspaceList.add(errorItem("eigenspace-" + i, "Eigenspace is empty", null));
                        } else {
                            // store the full eigenspace basis
                            diag.eigenspaceList.add(new DiagnosticItem<>("eigenspace-" + i, Status.OK, null, basis, null));
                        }
                    } catch (RuntimeException ex) {
                        diag.eigenspaceList.add(errorItem("eigenspace-" + i, "Failed to compute eigenspace", ex));
                    }
                } else {
                    diag.eigenspaceList.add(errorItem("eigenspace-" + i, "Eigenspace computation requires a real eigenvalue and real matrix", null));
                }

                // Eigenbasis: only exists when matrix is diagonalizable. Provide the full set
                // of eigenvector columns corresponding to this eigenvalue (one per algebraic multiplicity).
                if (diag.diagonalizable != null && diag.diagonalizable && diag.eigenvectors != null) {
                    try {
                        Vector[] cols = diag.eigenvectors.getData();
                        Set<Vector> basis = new LinkedHashSet<>();
                        for (int k = 0; k < nEv; k++) {
                            if (close(lambda, diag.eigenvalues[k], tol)) {
                                Vector v = cols[k];
                                if (v != null) {
                                    // Check if vector is linearly independent from existing basis vectors
                                    boolean isIndependent = true;
                                    for (Vector existing : basis) {
                                        // Two vectors are dependent if one is a scalar multiple of the other
                                        // Check if v = c * existing for some scalar c
                                        double ratio = 0.0;
                                        boolean ratioSet = false;
                                        boolean dependent = true;
                                        for (int idx = 0; idx < Math.min(v.dimension(), existing.dimension()); idx++) {
                                            double vVal = v.get(idx);
                                            double eVal = existing.get(idx);
                                            if (Math.abs(vVal) > tol || Math.abs(eVal) > tol) {
                                                if (!ratioSet && Math.abs(eVal) > tol) {
                                                    ratio = vVal / eVal;
                                                    ratioSet = true;
                                                } else if (ratioSet) {
                                                    double expectedVal = ratio * eVal;
                                                    if (Math.abs(vVal - expectedVal) > tol) {
                                                        dependent = false;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if (dependent && ratioSet) {
                                            isIndependent = false;
                                            break;
                                        }
                                    }
                                    if (isIndependent) {
                                        basis.add(v);
                                    }
                                }
                            }
                        }
                        if (basis.isEmpty()) {
                            diag.eigenbasisList.add(errorItem("eigenbasis-" + i, "No eigenvector found for eigenvalue", null));
                        } else {
                            diag.eigenbasisList.add(new DiagnosticItem<>("eigenbasis-" + i, Status.OK, null, basis, null));
                        }
                    } catch (RuntimeException ex) {
                        diag.eigenbasisList.add(errorItem("eigenbasis-" + i, "Failed to extract eigenbasis from eigenvector matrix", ex));
                    }
                } else {
                    diag.eigenbasisList.add(errorItem("eigenbasis-" + i, "Eigenbasis exists only when matrix is diagonalizable and eigenvectors are available", null));
                }

                // (no generalized eigenspace computation — not needed)
            }
        }

        if (diag.square && diag.real) {
            diag.orthonormal = isOrthonormal(A, tol);
            diag.orthogonal = diag.orthonormal && diag.rows == diag.cols;
            diag.unitary = diag.orthogonal;
            diag.normal = isNormal(A, tol);
            diag.nonNormal = diag.normal == null ? null : !diag.normal;
            if (diag.orthogonal != null && diag.orthogonal && diag.determinant != null) {
                double det = diag.determinant;
                diag.rotation = Math.abs(det - 1.0) <= tol * Math.max(1.0, Math.abs(det));
                diag.reflection = Math.abs(det + 1.0) <= tol * Math.max(1.0, Math.abs(det));
            }
            diag.idempotent = isIdempotent(A, tol);
            diag.involutory = isInvolutory(A, tol);
            diag.nilpotent = isNilpotent(A, tol);
        }

        if (diag.square && diag.real && diag.symmetric) {
            QuadraticForm form = new QuadraticForm(A);
            diag.definiteness = form.classify().name();
        }

        if (diag.square && diag.eigenvectors != null && diag.eigenvectors.isReal()) {
            Integer eigRank = rankEstimate(diag.eigenvectors, tol);
            if (eigRank != null) {
                diag.defective = eigRank < diag.rows;
            }
        }

        if (diag.defective != null) {
            diag.diagonalizable = !diag.defective;
        }

        diag.spectral = diag.normal;
        diag.companion = diag.square && isCompanion(A, tol);
        diag.block = diag.square && isBlockDiagonal(A, tol);

        return diag;
    }

    private static <T> DiagnosticItem<T> errorItem(String name, String message, Throwable ex) {
        String detail = ex != null ? ex.getMessage() : null;
        String combined = joinMessages(message, detail);
        return new DiagnosticItem<>(name, Status.ERROR, combined, null, null);
    }

    private static <T> DiagnosticItem<T> itemFromValidation(String name,
                                                            T value,
                                                            MatrixAccuracyValidator.ValidationResult validation,
                                                            String extraMessage) {
        Status status = statusFromValidation(validation);
        String message = validation == null ? null : validation.message;
        message = joinMessages(message, extraMessage);
        return new DiagnosticItem<>(name, status, message, value, validation);
    }

    private static <T> DiagnosticItem<T> elevateIfNeeded(DiagnosticItem<T> item,
                                                        MatrixAccuracyValidator.ValidationResult validation) {
        if (validation == null) {
            return item;
        }
        Status elevated = worse(item.getStatus(), statusFromValidation(validation));
        if (elevated == item.getStatus()) {
            return item;
        }
        return new DiagnosticItem<>(item.getName(), elevated, item.getMessage(), item.getValue(), item.getValidation());
    }

    private static Status statusFromValidation(MatrixAccuracyValidator.ValidationResult result) {
        if (result == null) {
            return Status.OK;
        }
        if (!result.passes) {
            return Status.ERROR;
        }
        if (result.shouldWarn) {
            return Status.WARNING;
        }
        return Status.OK;
    }

    private static Status worse(Status a, Status b) {
        return a.ordinal() >= b.ordinal() ? a : b;
    }

    private static double safeConditionEstimate(double condition) {
        if (Double.isNaN(condition) || Double.isInfinite(condition) || condition <= 0.0) {
            return 1e16;
        }
        return condition;
    }

    private static String joinMessages(String a, String b) {
        if (a == null || a.isBlank()) {
            return (b == null || b.isBlank()) ? null : b;
        }
        if (b == null || b.isBlank()) {
            return a;
        }
        return a + " | " + b;
    }

    private static int rankFromRref(Matrix rref, double tol) {
        if (rref == null) {
            return 0;
        }
        int rows = rref.getRowCount();
        int cols = rref.getColumnCount();
        int rank = 0;
        for (int i = 0; i < rows; i++) {
            boolean nonzero = false;
            for (int j = 0; j < cols; j++) {
                if (Math.abs(rref.get(i, j)) > tol) {
                    nonzero = true;
                    break;
                }
            }
            if (nonzero) {
                rank++;
            }
        }
        return rank;
    }

    private static String orthogonalityMessage(MatrixAccuracyValidator.ValidationResult result) {
        if (result == null) {
            return null;
        }
        return "Orthogonality check: " + result.message;
    }

    private static SymmetryResult computeSymmetry(Matrix A, double tol) {
        int n = A.getRowCount();
        double sumSq = 0.0;
        double maxDiff = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dr = A.get(i, j) - A.get(j, i);
                double di = A.getImag(i, j) - A.getImag(j, i);
                double diff = Math.hypot(dr, di);
                sumSq += diff * diff;
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }
        double error = Math.sqrt(sumSq);
        double scale = Math.max(1.0, A.frobeniusNorm());
        boolean symmetric = maxDiff <= tol * scale;
        return new SymmetryResult(symmetric, error);
    }

    private static Complex[] toComplex(double[] values) {
        if (values == null) {
            return null;
        }
        Complex[] out = new Complex[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = new Complex(values[i], 0.0);
        }
        return out;
    }

    private static double[][] toMatrixData(Matrix A) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = A.get(i, j);
            }
        }
        return out;
    }

    private static double[][] toMatrixImagData(Matrix A) {
        if (!A.hasImagData()) {
            return null;
        }
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = A.getImag(i, j);
            }
        }
        return out;
    }

    private static double computeDensity(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        int total = rows * cols;
        if (total == 0) {
            return 0.0;
        }
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        int nonzero = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double real = A.get(i, j);
                double imag = A.getImag(i, j);
                if (Math.hypot(real, imag) > threshold) {
                    nonzero++;
                }
            }
        }
        return ((double) nonzero) / total;
    }

    private static boolean isZeroMatrix(Matrix A, double tol) {
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isIdentity(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        if (rows != cols) {
            return false;
        }
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double real = A.get(i, j);
                double imag = A.getImag(i, j);
                if (i == j) {
                    if (Math.hypot(real - 1.0, imag) > threshold) {
                        return false;
                    }
                } else if (Math.hypot(real, imag) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isScalar(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        if (rows != cols) {
            return false;
        }
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        double diag = A.get(0, 0);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double real = A.get(i, j);
                double imag = A.getImag(i, j);
                if (i == j) {
                    if (Math.hypot(real - diag, imag) > threshold) {
                        return false;
                    }
                } else if (Math.hypot(real, imag) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isDiagonal(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == j) {
                    continue;
                }
                if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isUpperTriangular(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < Math.min(i, cols); j++) {
                if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isLowerTriangular(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < cols; j++) {
                if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isBidiagonal(Matrix A, double tol) {
        if (!isTridiagonal(A, tol)) {
            return false;
        }
        int n = A.getRowCount();
        boolean hasSuper = false;
        boolean hasSub = false;
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && Math.hypot(A.get(i, i + 1), A.getImag(i, i + 1)) > threshold) {
                hasSuper = true;
            }
            if (i - 1 >= 0 && Math.hypot(A.get(i, i - 1), A.getImag(i, i - 1)) > threshold) {
                hasSub = true;
            }
        }
        return !(hasSuper && hasSub);
    }

    private static boolean isTridiagonal(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(i - j) <= 1) {
                    continue;
                }
                if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isAntidiagonal(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        if (rows != cols) {
            return false;
        }
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        int n = rows;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i + j == n - 1) {
                    continue;
                }
                if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isSkewSymmetric(Matrix A, double tol) {
        int n = A.getRowCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double real = A.get(i, j) + A.get(j, i);
                double imag = A.getImag(i, j) + A.getImag(j, i);
                if (Math.hypot(real, imag) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isHermitian(Matrix A, double tol) {
        int n = A.getRowCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double realDiff = A.get(i, j) - A.get(j, i);
                double imagDiff = A.getImag(i, j) + A.getImag(j, i);
                if (Math.hypot(realDiff, imagDiff) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isPersymmetric(Matrix A, double tol) {
        int n = A.getRowCount();
        if (n != A.getColumnCount()) {
            return false;
        }
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int ii = n - 1 - j;
                int jj = n - 1 - i;
                double realDiff = A.get(i, j) - A.get(ii, jj);
                double imagDiff = A.getImag(i, j) - A.getImag(ii, jj);
                if (Math.hypot(realDiff, imagDiff) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isLowerHessenberg(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < rows; i++) {
            for (int j = i + 2; j < cols; j++) {
                if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isUpperHessenberg(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < Math.min(cols, i - 1); j++) {
                if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isRowEchelon(Matrix A, double tol) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        int lastLead = -1;
        boolean zeroRowSeen = false;
        for (int i = 0; i < rows; i++) {
            int lead = -1;
            for (int j = 0; j < cols; j++) {
                if (Math.abs(A.get(i, j)) > threshold) {
                    lead = j;
                    break;
                }
            }
            if (lead == -1) {
                zeroRowSeen = true;
                continue;
            }
            if (zeroRowSeen || lead <= lastLead) {
                return false;
            }
            for (int r = i + 1; r < rows; r++) {
                if (Math.abs(A.get(r, lead)) > threshold) {
                    return false;
                }
            }
            lastLead = lead;
        }
        return true;
    }

    private static boolean isReducedRowEchelon(Matrix A, double tol) {
        if (!isRowEchelon(A, tol)) {
            return false;
        }
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < rows; i++) {
            int lead = -1;
            for (int j = 0; j < cols; j++) {
                if (Math.abs(A.get(i, j)) > threshold) {
                    lead = j;
                    break;
                }
            }
            if (lead == -1) {
                continue;
            }
            if (Math.abs(A.get(i, lead) - 1.0) > threshold) {
                return false;
            }
            for (int r = 0; r < rows; r++) {
                if (r == i) {
                    continue;
                }
                if (Math.abs(A.get(r, lead)) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean isOrthonormal(Matrix A, double tol) {
        MatrixAccuracyValidator.ValidationResult validation =
                MatrixAccuracyValidator.validateOrthogonality(A, "Orthonormal");
        if (validation == null) {
            return false;
        }
        double scale = Math.max(1.0, A.frobeniusNorm());
        return validation.normResidual <= tol * scale && validation.passes;
    }

    private static boolean isNormal(Matrix A, double tol) {
        Matrix left = A.transpose().multiply(A);
        Matrix right = A.multiply(A.transpose());
        double diff = left.subtract(right).frobeniusNorm();
        double scale = Math.max(1.0, A.frobeniusNorm());
        return diff <= tol * scale;
    }

    private static boolean isIdempotent(Matrix A, double tol) {
        Matrix sq = A.multiply(A);
        return isClose(A, sq, tol);
    }

    private static boolean isInvolutory(Matrix A, double tol) {
        Matrix sq = A.multiply(A);
        Matrix I = Matrix.Identity(A.getRowCount());
        return isClose(I, sq, tol);
    }

    private static boolean isNilpotent(Matrix A, double tol) {
        int n = A.getRowCount();
        Matrix power = A.copy();
        for (int k = 1; k <= n; k++) {
            if (isZeroMatrix(power, tol)) {
                return true;
            }
            power = power.multiply(A);
        }
        return false;
    }

    private static boolean isCompanion(Matrix A, double tol) {
        int n = A.getRowCount();
        if (n != A.getColumnCount() || n < 2) {
            return false;
        }
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                boolean onSub = i == j + 1;
                boolean lastCol = j == n - 1;
                if (i == 0 && j < n - 1) {
                    if (Math.abs(A.get(i, j)) > threshold) {
                        return false;
                    }
                } else if (onSub) {
                    if (Math.abs(A.get(i, j) - 1.0) > threshold) {
                        return false;
                    }
                } else if (!lastCol) {
                    if (Math.abs(A.get(i, j)) > threshold) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    private static boolean isBlockDiagonal(Matrix A, double tol) {
        int n = A.getRowCount();
        if (n != A.getColumnCount()) {
            return false;
        }
        double scale = Math.max(1.0, A.frobeniusNorm());
        double threshold = tol * scale;
        for (int split = 1; split < n; split++) {
            boolean zeroUpperRight = true;
            boolean zeroLowerLeft = true;
            for (int i = 0; i < split; i++) {
                for (int j = split; j < n; j++) {
                    if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                        zeroUpperRight = false;
                        break;
                    }
                }
                if (!zeroUpperRight) {
                    break;
                }
            }
            for (int i = split; i < n; i++) {
                for (int j = 0; j < split; j++) {
                    if (Math.hypot(A.get(i, j), A.getImag(i, j)) > threshold) {
                        zeroLowerLeft = false;
                        break;
                    }
                }
                if (!zeroLowerLeft) {
                    break;
                }
            }
            if (zeroUpperRight && zeroLowerLeft) {
                return true;
            }
        }
        return false;
    }

    private static boolean isClose(Matrix A, Matrix B, double tol) {
        double diff = A.subtract(B).frobeniusNorm();
        double scale = Math.max(1.0, A.frobeniusNorm());
        return diff <= tol * scale;
    }

    private static int rankEstimate(Matrix A, double tol) {
        if (!A.isReal()) {
            return 0;
        }
        Matrix rref = A.copy();
        MatrixUtils.toReducedRowEchelonForm(rref);
        return rankFromRref(rref, tol);
    }

    private static double max(double[] values) {
        double max = values[0];
        for (int i = 1; i < values.length; i++) {
            max = Math.max(max, values[i]);
        }
        return max;
    }

    private static double maxAbs(Complex[] values) {
        double max = 0.0;
        for (Complex c : values) {
            if (c != null) {
                max = Math.max(max, c.abs());
            }
        }
        return max;
    }

    private static MultiplicityResult computeMultiplicities(Matrix A, Complex[] eigenvalues, double tol) {
        int n = eigenvalues.length;
        int[] alg = new int[n];
        int[] geo = new int[n];
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (visited[i]) {
                continue;
            }
            int count = 0;
            for (int j = i; j < n; j++) {
                if (!visited[j] && close(eigenvalues[i], eigenvalues[j], tol)) {
                    visited[j] = true;
                    count++;
                }
            }
            int geoCount = -1;
            if (A != null) {
                try {
                    geoCount = geometricMultiplicity(A, eigenvalues[i], tol);
                } catch (RuntimeException ex) {
                    geoCount = -1;
                }
            }
            for (int j = 0; j < n; j++) {
                if (close(eigenvalues[i], eigenvalues[j], tol)) {
                    alg[j] = count;
                    geo[j] = geoCount;
                }
            }
        }
        return new MultiplicityResult(alg, geo);
    }

    private static boolean close(Complex a, Complex b, double tol) {
        double scale = Math.max(1.0, Math.max(a.abs(), b.abs()));
        return Math.hypot(a.real - b.real, a.imag - b.imag) <= tol * scale;
    }

    private static boolean isRealEigenvalue(Complex eigenvalue, double tol) {
        double scale = Math.max(1.0, Math.abs(eigenvalue.real));
        return Math.abs(eigenvalue.imag) <= tol * scale;
    }

    private static int geometricMultiplicity(Matrix A, double eigenvalue, double tol) {
        return geometricMultiplicity(A, new Complex(eigenvalue, 0.0), tol);
    }

    private static int geometricMultiplicity(Matrix A, Complex eigenvalue, double tol) {
        int n = A.getRowCount();
        Matrix shifted = A.copy();
        // Subtract eigenvalue from diagonal (handle complex shifts)
        for (int i = 0; i < n; i++) {
            double realPart = shifted.get(i, i) - eigenvalue.real;
            double imagPart = shifted.getImag(i, i) - eigenvalue.imag;
            shifted.setComplex(i, i, realPart, imagPart);
        }

        // If shifted matrix is real-valued, use direct rank estimate
        if (shifted.isReal()) {
            int rank = rankEstimate(shifted, tol);
            return n - rank;
        }

        // For complex matrices, convert to real block matrix [[Re, -Im],[Im, Re]]
        int m = n * 2;
        double[][] block = new double[m][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double re = shifted.get(i, j);
                double im = shifted.getImag(i, j);
                block[i][j] = re;
                block[i][j + n] = -im;
                block[i + n][j] = im;
                block[i + n][j + n] = re;
            }
        }
        Matrix realBlock = new Matrix(block);
        int rankR = rankEstimate(realBlock, tol);
        int rankC = (int) Math.round(rankR / 2.0);
        return n - rankC;
    }

    private static Complex[] characteristicPolynomial(Complex[] eigenvalues) {
        Complex[] coeffs = new Complex[] { Complex.ONE };
        for (Complex eigenvalue : eigenvalues) {
            Complex[] next = new Complex[coeffs.length + 1];
            for (int i = 0; i < next.length; i++) {
                next[i] = Complex.ZERO;
            }
            Complex neg = new Complex(-eigenvalue.real, -eigenvalue.imag);
            for (int i = 0; i < coeffs.length; i++) {
                next[i] = next[i].add(coeffs[i]);
                next[i + 1] = next[i + 1].add(coeffs[i].multiply(neg));
            }
            coeffs = next;
        }
        return coeffs;
    }

    private static final class MultiplicityResult {
        private final int[] algebraicMultiplicity;
        private final int[] geometricMultiplicity;

        private MultiplicityResult(int[] algebraicMultiplicity, int[] geometricMultiplicity) {
            this.algebraicMultiplicity = algebraicMultiplicity;
            this.geometricMultiplicity = geometricMultiplicity;
        }
    }

    private static final class SymmetryResult {
        private final boolean symmetric;
        private final double error;

        private SymmetryResult(boolean symmetric, double error) {
            this.symmetric = symmetric;
            this.error = error;
        }
    }
}
