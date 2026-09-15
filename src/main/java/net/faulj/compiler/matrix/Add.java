package net.faulj.compiler.matrix;

/**
 * Element-wise matrix addition expression node.
 */
public final class Add implements MatrixExpr {
    private final MatrixExpr lhs;
    private final MatrixExpr rhs;
    private final MatrixShape shape;

    public Add(MatrixExpr lhs, MatrixExpr rhs) {
        MatrixExpr.requireExpression(lhs, "Add left operand");
        MatrixExpr.requireExpression(rhs, "Add right operand");
        if (!lhs.shape().equals(rhs.shape())) {
            throw new IllegalArgumentException(
                "Cannot add matrices with shapes " + lhs.shape() + " and " + rhs.shape()
                    + ": shapes must be identical");
        }
        this.lhs = lhs;
        this.rhs = rhs;
        this.shape = lhs.shape();
    }

    public MatrixExpr lhs() {
        return lhs;
    }

    public MatrixExpr rhs() {
        return rhs;
    }

    public MatrixExpr left() {
        return lhs;
    }

    public MatrixExpr right() {
        return rhs;
    }

    @Override
    public MatrixShape shape() {
        return shape;
    }
}
