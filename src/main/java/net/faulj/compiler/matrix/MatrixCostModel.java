package net.faulj.compiler.matrix;

/**
 * Cost model used by the matrix planner.
 *
 * <p>The interface keeps arithmetic and future hardware policy out of the
 * optimizer. M1's {@link FlopCostModel} is intentionally small, while later
 * models can account for measured throughput, materialization traffic,
 * allocation, backend availability, or device placement.</p>
 */
public interface MatrixCostModel {
    /**
     * Estimate a matrix multiplication. Implementations must return a
     * non-negative, saturation-safe estimate.
     */
    CostEstimate estimateMatMul(MatrixShape lhs, MatrixShape rhs);

    default CostEstimate estimateInput(MatrixShape shape) {
        requireShape(shape, "Input shape");
        return CostEstimate.ZERO;
    }

    default CostEstimate estimateAdd(MatrixShape shape) {
        return outputOnly(shape);
    }

    default CostEstimate estimateScale(MatrixShape shape) {
        requireShape(shape, "Scale shape");
        return new CostEstimate(elementCount(shape), outputBytes(shape));
    }

    default CostEstimate estimateTranspose(MatrixShape shape) {
        return outputOnly(shape);
    }

    /**
     * Convenience alias for callers that describe the operation as a cost.
     */
    default long matMulCost(MatrixShape lhs, MatrixShape rhs) {
        return estimateMatMul(lhs, rhs).scalarMultiplications();
    }

    /**
     * Estimated dense row-major bytes for one matrix result.
     */
    default long outputBytes(MatrixShape shape) {
        requireShape(shape, "Shape");
        return SaturatingMath.multiply(
            SaturatingMath.multiply((long) shape.rows(), (long) shape.columns()),
            (long) Double.BYTES);
    }

    default long estimatedOutputBytes(MatrixShape shape) {
        return outputBytes(shape);
    }

    private static CostEstimate outputOnly(MatrixShape shape) {
        requireShape(shape, "Output shape");
        return new CostEstimate(0L, SaturatingMath.multiply(
            SaturatingMath.multiply((long) shape.rows(), (long) shape.columns()),
            (long) Double.BYTES));
    }

    private static long elementCount(MatrixShape shape) {
        requireShape(shape, "Shape");
        return SaturatingMath.multiply((long) shape.rows(), (long) shape.columns());
    }

    private static void requireShape(MatrixShape shape, String role) {
        if (shape == null) {
            throw new IllegalArgumentException(role + " must not be null");
        }
    }
}
