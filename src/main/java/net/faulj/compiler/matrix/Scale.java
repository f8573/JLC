package net.faulj.compiler.matrix;

/**
 * Scalar multiplication expression node.
 */
public final class Scale implements MatrixExpr {
    private final double factor;
    private final MatrixExpr operand;

    public Scale(double factor, MatrixExpr operand) {
        MatrixExpr.requireExpression(operand, "Scale operand");
        this.factor = factor;
        this.operand = operand;
    }

    public Scale(MatrixExpr operand, double factor) {
        this(factor, operand);
    }

    public double factor() {
        return factor;
    }

    public double scalar() {
        return factor;
    }

    public MatrixExpr operand() {
        return operand;
    }

    @Override
    public MatrixShape shape() {
        return operand.shape();
    }
}
