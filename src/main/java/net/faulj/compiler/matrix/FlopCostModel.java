package net.faulj.compiler.matrix;

/**
 * M1 arithmetic cost model for dense matrix expressions.
 *
 * <p>For {@code (m x k) * (k x n)}, the scalar multiplication estimate is
 * {@code m * k * n}, evaluated with saturating long arithmetic. Output bytes
 * use {@code rows * columns * 8} and are likewise saturation-safe.</p>
 */
public final class FlopCostModel implements MatrixCostModel {
    @Override
    public CostEstimate estimateMatMul(MatrixShape lhs, MatrixShape rhs) {
        if (lhs == null || rhs == null) {
            throw new IllegalArgumentException("MatMul shapes must not be null");
        }
        if (lhs.columns() != rhs.rows()) {
            throw new IllegalArgumentException(
                "Cannot estimate multiplication for incompatible shapes " + lhs + " and " + rhs);
        }
        long scalarMultiplications = SaturatingMath.multiply(
            SaturatingMath.multiply((long) lhs.rows(), (long) lhs.columns()),
            (long) rhs.columns());
        MatrixShape output = new MatrixShape(lhs.rows(), rhs.columns());
        return new CostEstimate(scalarMultiplications, outputBytes(output));
    }
}
