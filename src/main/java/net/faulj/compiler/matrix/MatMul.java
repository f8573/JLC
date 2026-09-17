package net.faulj.compiler.matrix;

/**
 * Matrix multiplication expression node.
 */
public final class MatMul implements MatrixExpr {
    private final MatrixExpr lhs;
    private final MatrixExpr rhs;
    private final MatrixShape shape;

    public MatMul(MatrixExpr lhs, MatrixExpr rhs) {
        MatrixExpr.requireExpression(lhs, "MatMul left operand");
        MatrixExpr.requireExpression(rhs, "MatMul right operand");
        if (lhs.shape().columns() != rhs.shape().rows()) {
            throw new IllegalArgumentException(
                "Cannot multiply matrices with shapes " + lhs.shape() + " and " + rhs.shape()
                    + ": left columns (" + lhs.shape().columns()
                    + ") must equal right rows (" + rhs.shape().rows() + ")");
        }
        this.lhs = lhs;
        this.rhs = rhs;
        this.shape = new MatrixShape(lhs.shape().rows(), rhs.shape().columns());
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
