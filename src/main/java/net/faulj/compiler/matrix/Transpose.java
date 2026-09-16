package net.faulj.compiler.matrix;

/**
 * Matrix transpose expression node.
 */
public final class Transpose implements MatrixExpr {
    private final MatrixExpr operand;
    private final MatrixShape shape;

    public Transpose(MatrixExpr operand) {
        MatrixExpr.requireExpression(operand, "Transpose operand");
        this.operand = operand;
        this.shape = new MatrixShape(operand.shape().columns(), operand.shape().rows());
    }

    public MatrixExpr operand() {
        return operand;
    }

    @Override
    public MatrixShape shape() {
        return shape;
    }
}
