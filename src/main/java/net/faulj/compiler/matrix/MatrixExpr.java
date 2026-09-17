package net.faulj.compiler.matrix;

import net.faulj.matrix.Matrix;

/**
 * Typed, immutable matrix-expression DAG node.
 *
 * <p>{@code MatrixExpr} is deliberately separate from the mutable JLC
 * {@link Matrix} API. Building an expression is opt-in and does not change
 * the behavior of existing eager matrix code.</p>
 */
public sealed interface MatrixExpr
    permits Input, SymbolicInput, MatMul, Add, Scale, Transpose {

    /**
     * Shape of this expression without executing it.
     */
    MatrixShape shape();

    default MatrixShape outputShape() {
        return shape();
    }

    default int rows() {
        return shape().rows();
    }

    default int columns() {
        return shape().columns();
    }

    default int getRowCount() {
        return rows();
    }

    default int getColumnCount() {
        return columns();
    }

    default MatMul matmul(MatrixExpr right) {
        return new MatMul(this, right);
    }

    default Add add(MatrixExpr right) {
        return new Add(this, right);
    }

    default MatrixExpr scale(double factor) {
        return scale(factor, this);
    }

    default MatrixExpr transpose() {
        return transpose(this);
    }

    static Input input(Matrix matrix) {
        return new Input(matrix);
    }

    static Input input(String name, Matrix matrix) {
        return new Input(name, matrix);
    }

    /**
     * Create a shape-only input for planner tests and future front ends.
     * Symbolic inputs cannot be evaluated.
     */
    static SymbolicInput symbolicInput(String name, MatrixShape shape) {
        return new SymbolicInput(name, shape);
    }

    static MatMul matmul(MatrixExpr left, MatrixExpr right) {
        return new MatMul(left, right);
    }

    static Add add(MatrixExpr left, MatrixExpr right) {
        return new Add(left, right);
    }

    /**
     * Construct a scale node, removing only the exact identity scale.
     */
    static MatrixExpr scale(double factor, MatrixExpr operand) {
        requireExpression(operand, "Scale operand");
        if (factor == 1.0) {
            return operand;
        }
        return new Scale(factor, operand);
    }

    static MatrixExpr scale(MatrixExpr operand, double factor) {
        return scale(factor, operand);
    }

    /**
     * Construct a transpose node, removing a transpose pair.
     */
    static MatrixExpr transpose(MatrixExpr operand) {
        requireExpression(operand, "Transpose operand");
        if (operand instanceof Transpose transpose) {
            return transpose.operand();
        }
        return new Transpose(operand);
    }

    static void requireExpression(MatrixExpr expression, String role) {
        if (expression == null) {
            throw new IllegalArgumentException(role + " must not be null");
        }
    }
}
